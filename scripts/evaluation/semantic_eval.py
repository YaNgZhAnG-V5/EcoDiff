import os

import torch
from open_clip import IMAGENET_CLASSNAMES, SIMPLE_IMAGENET_TEMPLATES, create_model_and_transforms, get_tokenizer
from open_clip.zero_shot_classifier import build_zero_shot_classifier
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

import argparse
from sdib.data import ImageNetDataset
from sdib.utils import create_pipeline, get_clip_encoders, load_pipeline


# CLIP evaluation
def accuracy(output, target, topk=(1, 5)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_classifier(model_name, pretrained, device):
    model, _, _ = create_model_and_transforms(model_name=model_name, pretrained=pretrained)
    tokenizer = get_tokenizer(model_name=model_name)

    classifier = build_zero_shot_classifier(
        model.to(device),
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
        templates=SIMPLE_IMAGENET_TEMPLATES,
        use_tqdm=True,
        device=device,
    )
    return classifier


def get_accuracy_score(model, data, classifier):
    image = data[0].cuda()  # image
    target = data[1].cuda()  # label
    classifier = classifier.cuda()

    image_features = model.visual(image)
    logits = 100.0 * image_features @ classifier
    return accuracy(logits, target)


def zero_shot_classification(model, classifier, dataloader, batch_size):
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for data in tqdm(dataloader, unit_scale=batch_size):
            # measure accuracy
            acc1, acc5 = get_accuracy_score(model, data, classifier)
            top1 += acc1
            top5 += acc5
            n += data[0].size(0)

    top1 = top1 / n
    top5 = top5 / n
    return top1, top5


def semantic_eval(args):
    if args.task not in ["gen", "clip", "fid", "all"]:
        raise ValueError(f"task {args.task} not supported")

    # Generating the synthetic images with sdxl model
    # TODO for sd3 model
    device = args.device
    if args.mix_precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"torch dtype {args.mix_precision} not supported")

    pipe = load_pipeline(args.model, torch_dtype, disable_progress_bar=True)
    pipe.to(args.device)

    ds = ImageNetDataset(
        save_dir=args.save_dir,
        device=args.device,
        pipe=pipe,
        num_inference_steps=args.num_intervention_steps,
        seed=args.seed,
    )
    # if args.task == "gen":  # overwrite the previous images
    #     print("Generating synthetic images with sdxl model...")
    #     ds.prepare_data()

    pipe = create_pipeline(
        args.model, device, torch_dtype, save_pt=args.save_pth, lambda_threshold=args.lambda_threshold
    )

    masked_ds = ImageNetDataset(
        save_dir=args.save_dir,
        device=args.device,
        dataset_name="masked_imagenet",
        pipe=pipe,
        seed=args.seed,
        num_inference_steps=args.num_intervention_steps,
        task=args.save_pth.split(os.sep)[-2],
    )
    if args.task == "gen":
        print("Generating synthetic images with masked sdxl model...")
        masked_ds.prepare_data()

    if args.task == "fid":
        # if not ds.image_path:
        ds.load_data()
        if not ds.image_path:
            ds.prepare_data()

        masked_ds.load_data()
        if not masked_ds.image_path:
            print("Generating synthetic images with masked sdxl model")
            masked_ds.prepare_data()

    # save the pth folder
    log_txt = os.path.join(os.path.dirname(args.save_pth), "semantic_eval.txt")

    if args.task == "clip" or args.task == "all":
        # define model and classifier
        clip_model = get_clip_encoders(backbone=args.clip_backbone, pretrained=args.clip_pretrained)["clip_model"].to(
            device
        )
        classifier = get_classifier(args.clip_backbone, args.clip_pretrained, device)

        # evaluate the generated images from orginal sd model
        dataloader = DataLoader(ds, batch_size=100, shuffle=False, num_workers=4)
        top1, top5 = zero_shot_classification(clip_model, classifier, dataloader, batch_size=100)
        print(f"Orignal SDXL Top1: {top1}, Top5: {top5}")
        # evaluate the generated images from masked sd model
        dataloader = DataLoader(masked_ds, batch_size=100, shuffle=False, num_workers=4)
        mask_top1, mask_top5 = zero_shot_classification(clip_model, classifier, dataloader, batch_size=100)
        print(f"Masked SDXL Top1: {mask_top1}, Top5: {mask_top5}")
        with open(log_txt, "a") as f:
            f.write(f"Orignal SDXL Top1: {top1}, Top5: {top5}\n")
            f.write(f"Masked SDXL Top1: {mask_top1}, Top5: {mask_top5}\n")

    if args.task == "fid" or args.task == "all":
        fid = FrechetInceptionDistance(feature=64)
        total_length = len(ds)
        print("Computing FID score...")
        ds.is_transform = False
        masked_ds.is_transform = False
        with tqdm(total=total_length) as pbar:
            original_list, mask_list = [], []
            for idx in range(total_length):
                original_data, mask_data = ds[idx], masked_ds[idx]
                # if original_data[1] != mask_data[1]:
                #     raise ValueError("original image and masked image should have the same label")
                original_img = original_data[0].unsqueeze(0) * 255
                mask_img = mask_data[0].unsqueeze(0) * 255
                original_list.append(original_img.to(torch.uint8))
                mask_list.append(mask_img.to(torch.uint8))
                if len(original_list) % 1000 == 0 and len(original_list) > 0:
                    fid.update(torch.cat(original_list, dim=0), real=True)
                    fid.update(torch.cat(mask_list, dim=0), real=False)
                    original_list, mask_list = [], []
                pbar.update()

        print(f"FID score:{fid.compute()}")
        with open(log_txt, "a") as f:
            f.write(f"FID score:{fid.compute()}\n")


def main(args):
    semantic_eval(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Image semantic evaluation")
    parser.add_argument("--task", type=str, default="clip", help="options: gen | clip | fid | all")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to run the model")
    parser.add_argument("--seed", type=list, default=[44], help="random seed")
    parser.add_argument("--save_dir", "-s", type=str, default="./datasets", help="store generated images")
    parser.add_argument("--mix_precision", type=str, default="bf16", help="mixed precision, available bf16")
    parser.add_argument("--num_intervention_steps", type=int, default=50, help="number of intervention steps")
    parser.add_argument("--model", type=str, default="sdxl", help="model type, available sdxl, sd2")
    parser.add_argument(
        "--save_pth",
        "-sp",
        type=str,
        default=None,
        help="path to save the model",
    )
    parser.add_argument(
        "--clip_backbone", type=str, default="ViT-B-16", help="clip model type, available ViT-B-16, ViT-L-14"
    )
    parser.add_argument("--clip_pretrained", type=str, default="laion400m_e32", help="clip pretrained model")
    parser.add_argument("--lambda_threshold", "-lt", type=float, default=0.01, help="threshold for lambda")

    args = parser.parse_args()
    main(args)

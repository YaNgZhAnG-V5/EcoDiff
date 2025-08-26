import os
import shutil

import torch
from tqdm import tqdm

import argparse
from sdib.utils import calculate_mask_sparsity, create_pipeline, save_image_seed


def binary_mask_eval(args):
    device = args.device
    torch_dtype = torch.bfloat16 if args.mix_precision == "bf16" else torch.float32
    pipe = create_pipeline(args.model, device, torch_dtype)

    mask_pipe, hookers = create_pipeline(
        args.model,
        device,
        torch_dtype,
        args.ckpt,
        binary=args.binary,
        lambda_threshold=args.lambda_threshold,
        epsilon=args.epsilon,
        masking=args.masking,
        return_hooker=True,
        scope=args.scope,
        ratio=args.ratio,
    )

    for hooker in hookers:
        if hooker.binary:
            assert hooker.masking == "binary", "masking must be binary when using binary mask"
        else:
            assert hooker.masking != "binary", "masking must be not binary when using continuous mask"

    # get mask sparsity
    threshold = None if args.binary else args.lambda_threshold
    threshold = None if args.scope is not None else threshold
    name = ["ff", "attn"]
    if len(hookers) == 3:
        name.append("norm")
    if args.binary:
        print("Use binary mask")
        if not threshold:
            print(f"binarize with threshold: {threshold}")
        else:
            print(f"binarize with scope: {args.scope}, ratio: {args.ratio}")
    else:
        print("Use continuous mask")
    for n, hooker in zip(name, hookers):
        total_num_heads, num_activate_heads, mask_sparsity = calculate_mask_sparsity(hooker, threshold)
        print(f"model: {args.model}, {n} masking: {args.masking}")
        print(
            f"total num heads: {total_num_heads},"
            + f"num activate heads: {num_activate_heads}, mask sparsity: {mask_sparsity}"
        )

    # prepare save dir, if not exist, create it, if not keep_old, delete it
    dir_name = os.path.dirname(args.ckpt)
    dst = os.path.join(dir_name, "binary_mask_eval")
    if not os.path.exists(dst):
        os.mkdir(dst)
    elif not args.keep_old:
        shutil.rmtree(dst)
        os.mkdir(dst)

    # perform evaluation
    for seed, p in tqdm([(s, p) for s in args.seed for p in args.prompt]):
        # pipeline without masking
        save_image_seed(pipe, p, args.num_intervention_steps, device, seed, dst, width=args.width, height=args.height)

        # with masking
        save_image_seed(
            mask_pipe, p, args.num_intervention_steps, device, seed, dst, width=args.width, height=args.height
        )


def main(args):
    binary_mask_eval(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("binary lambda mask for quantative analysis")
    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
        help="path to lambda ckpt path",
    )
    parser.add_argument("--device", type=str, default="0", help="device to run the model")
    parser.add_argument("--seed", type=int, nargs="+", default=[44], help="random seed")
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=[
            "A clock tower floating in a sea of clouds",
            "A cozy library with a roaring fireplace",
            "A dragon sleeping on a pile of gold",
            "A floating city above the clouds at sunset",
            "A fox standing in a misty meadow",
            "A magical tree with glowing fruit in a dark forest",
            "A medieval castle surrounded by fog",
            "A mystical wolf howling under a glowing aurora",
            "A panda playing the guitar under a cherry blossom tree",
            "A polar bear enjoying a hot drink in an igloo",
            "A robot dog exploring an abandoned spaceship",
            "A snowy forest with trees covered in sparkling frost",
            "A steampunk airship flying over a bustling city",
            "A tiger walking through a glowing jungle",
            "An underwater palace illuminated by bioluminescent creatures",
        ],
        help="prompts for the model eval",
    )
    parser.add_argument("--mix_precision", type=str, default="bf16", help="mixed precision, available bf16")
    parser.add_argument("--num_intervention_steps", "-ni", type=int, default=20, help="number of intervention steps")
    parser.add_argument("--model", type=str, default="sdxl", help="model type, available sdxl, sd2")
    parser.add_argument(
        "--binary", action="store_true", help="whether to use binary mask"
    )
    parser.add_argument(
        "--masking", type=str, default="hard_discrete", help="masking type, available binary, hard_discrete, sigmoid"
    )
    parser.add_argument("--scope", type=str, default=None, help="scope for lambda binary mask")
    parser.add_argument(
        "--ratio", type=float, nargs="+", default=None, help="sparsity ratio for local global lambda mask"
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--epsilon", "-e", type=float, default=0.0, help="epsilon for lambda")
    parser.add_argument("--lambda_threshold", "-lt", type=float, default=0.001, help="threshold for lambda")
    parser.add_argument("--keep_old", action="store_true", help="keep the old images")
    parser.add_argument("--output_dir", type=str, default="./results/")
    args = parser.parse_args()
    args.device = f"cuda:{args.device}"
    main(args)

import os

import torch
from tqdm import tqdm

import argparse
from src.utils import (
    create_pipeline,
    save_img,
    calculate_mask_sparsity,
    ffn_linear_layer_pruning,
    linear_layer_pruning,
)
from diffusers import StableDiffusionXLPipeline


def binary_mask_eval(args):
    # load sdxl model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
    ).to(args.device)

    device = args.device
    torch_dtype = torch.bfloat16 if args.mix_precision == "bf16" else torch.float32
    mask_pipe, hookers = create_pipeline(
        pipe,
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
    # get mask sparsity
    threshold = None if args.binary else args.lambda_threshold
    threshold = None if args.scope is not None else threshold
    name = ["ff", "attn"]
    for n, hooker in zip(name, hookers):
        total_num_heads, num_activate_heads, mask_sparsity = calculate_mask_sparsity(hooker, threshold)
        print(f"model: {args.model}, {n} masking: {args.masking}")
        print(
            f"total num heads: {total_num_heads},"
            + f"num activate heads: {num_activate_heads}, mask sparsity: {mask_sparsity}"
        )

    # remove parameters in attention blocks
    cross_attn_hooker = hookers[0]
    for name in tqdm(cross_attn_hooker.hook_dict.keys(), desc="Pruning"):
        # make it compatible with both unet and transformer (FLUX and SD3)
        if getattr(pipe, "unet", None):
            module = pipe.unet.get_submodule(name)
        else:
            module = pipe.transformer.get_submodule(name)
        lamb = cross_attn_hooker.lambs[cross_attn_hooker.lambs_module_names.index(name)]
        assert module.heads == lamb.shape[0]
        # perform pruning on the linear layers
        module = linear_layer_pruning(module, lamb)

        # replace module in pipeline
        parent_module_name, child_name = name.rsplit(".", 1)

        if getattr(pipe, "unet", None):
            parent_module = pipe.unet.get_submodule(parent_module_name)
        else:
            parent_module = pipe.transformer.get_submodule(parent_module_name)
        setattr(parent_module, child_name, module)

    # remove parameters in ffn blocks
    ffn_hook = hookers[1]
    for name in tqdm(ffn_hook.hook_dict.keys(), desc="Pruning on FFN linear lazer"):
        if getattr(pipe, "unet", None):
            module = pipe.unet.get_submodule(name)
        else:
            module = pipe.transformer.get_submodule(name)
        lamb = ffn_hook.lambs[ffn_hook.lambs_module_names.index(name)]

        # perform pruning on the ffn layers
        module = ffn_linear_layer_pruning(module, lamb)

        # replace module in pipeline
        parent_module_name, child_name = name.rsplit(".", 1)

        if getattr(pipe, "unet", None):
            parent_module = pipe.unet.get_submodule(parent_module_name)
        else:
            parent_module = pipe.transformer.get_submodule(parent_module_name)
        setattr(parent_module, child_name, module)

    cross_attn_hooker.clear_hooks()
    ffn_hook.clear_hooks()
    del cross_attn_hooker
    del lamb
    del module
    del parent_module

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # perform evaluation
    for seed, p in tqdm([(s, p) for s in args.seed for p in args.prompt]):
        g_cpu = torch.Generator(device).manual_seed(seed)
        # diffusion image
        save_img(
            pipe=pipe,
            g_cpu=g_cpu,
            steps=args.num_intervention_steps,
            prompt=p,
            save_path=os.path.join(args.output_dir, f"original_seed_{seed}_prompt_{p}.png"),
        )

        save_img(
            pipe=mask_pipe,
            g_cpu=g_cpu,
            steps=args.num_intervention_steps,
            prompt=p,
            save_path=os.path.join(args.output_dir, f"ecodiff_seed_{seed}_prompt_{p}.png"),
        )


def main(args):
    binary_mask_eval(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("binary lambda mask for quantative analysis")
    parser.add_argument(
        "--ckpt",
        default="./mask/ff.pt",
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
    parser.add_argument("--num_intervention_steps", "-ni", type=int, default=50, help="number of intervention steps")
    parser.add_argument("--model", type=str, default="sdxl", help="model type, available sdxl, sd2")
    parser.add_argument("--binary", action="store_true", help="whether to use binary mask")
    parser.add_argument("--masking", type=str, default="binary", help="masking type")
    parser.add_argument("--scope", type=str, default="global", help="scope for lambda binary mask")
    parser.add_argument(
        "--ratio", type=float, nargs="+", default=[0.68, 0.88], help="sparsity ratio for local global lambda mask"
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

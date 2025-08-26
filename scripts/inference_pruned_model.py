import pickle
import os

import argparse
import torch
from tqdm import tqdm
from diffusers.models import UNet2DConditionModel, FluxTransformer2DModel, SD3Transformer2DModel

from sdib.evaluation import show_model_param_summary
from sdib.utils import (
    load_pipeline,
    get_precision,
)

@torch.no_grad()
def main(args):
    pipe = load_pipeline(args.model, get_precision(args.precision), True)
    pipe.to(args.device)

    os.makedirs(args.dst, exist_ok=True)

    # generate images
    for idx, prompt in enumerate(tqdm(args.prompts, desc="Generating Original")):
        g_cpu = torch.Generator(args.device).manual_seed(args.seed)
        image = pipe(prompt=prompt, generator=g_cpu, num_inference_steps=args.num_intervention_steps).images[0]
        image.save(os.path.join(args.dst, f"original_{idx}.png"))

    with open(args.pruned_model_pt, "rb") as f:
        model = pickle.load(f)
    model.to(get_precision(args.precision))

    if hasattr(pipe, "unet"): 
        assert isinstance(model, UNet2DConditionModel), "Model must be UNet2DConditionModel"
        pipe.unet = model
    else:
        assert isinstance(model, (FluxTransformer2DModel, SD3Transformer2DModel)), "Model must be either FluxTransformer2DModel or SD3Transformer2DModel"
        pipe.transformer = model
    
    pipe.to(args.device)

    modules_of_interest = ["attn", "ff", "conv", "norm", "overall"]
    show_model_param_summary(model, modules_of_interest)

    # generate images
    for idx, prompt in enumerate(tqdm(args.prompts, desc="Generating")):
        g_cpu = torch.Generator(args.device).manual_seed(args.seed)
        image = pipe(prompt=prompt, generator=g_cpu, num_inference_steps=args.num_intervention_steps).images[0]
        image.save(os.path.join(args.dst, f"pruned_{idx}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "A clock tower floating in a sea of clouds",
            "A cozy library with a roaring fireplace",
            "A cat playing football",
        ],
        help="prompts for the model",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--model", type=str, default="sdxl")
    parser.add_argument("--num_intervention_steps", "-ni", type=int, default=50)
    parser.add_argument("--dst", type=str, default="./demo/sdxl_pruned")
    pruned_model_pt = "./results/sdxl_debug/" \
        + "model_sdxl_eps_0.5_sample_100_beta_0.5_epochs_4_" \
        + "lr_0.150.15_batch_size_4_loss_21_regex_.*_masking_hard_discrete" \
        + "/lambda/prune_results/pruned_model_30.pkl"

    parser.add_argument("--pruned_model_pt", type=str, default=pruned_model_pt, help="path to the pruned model")
    parser.add_argument("--device", type=str, default="cuda:0", help="device for the model")
    parser.add_argument(
        "--precision", type=str, default="bf16", help="precision for the model, bf16, fp32, fp16 available"
    )
    args = parser.parse_args()

    main(args)

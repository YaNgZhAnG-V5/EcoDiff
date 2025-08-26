import pickle
import os

import argparse
import torch
from tqdm import tqdm
from diffusers.models import UNet2DConditionModel, FluxTransformer2DModel, SD3Transformer2DModel
from diffusers.utils import convert_unet_state_dict_to_peft, SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME

from peft import LoraConfig, set_peft_model_state_dict

from safetensors.torch import load_file
from sdib.evaluation import show_model_param_summary
from sdib.utils import (
    load_pipeline,
    get_precision,
)

def load_lora(args, pipe, model):
    # Load LoRA weights
    if not args.lora_pt.endswith('.pkl'):
        print(f"Loading LoRA weights using diffusers method")
        pipe.load_lora_weights(args.lora_pt)

    else:
        # Only support SDXL lora
        print(f"Loading LoRA weights using custom method")

        # load the lora state dict
        with open(args.lora_pt, "rb") as f:
            lora_state_dict = pickle.load(f)

        one_key = list(lora_state_dict.keys())[0]
        rank = lora_state_dict[one_key].shape[0]

        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        if args.full_lora:
            target_modules.extend(["ff.net.0.proj", "ff.net.2"])

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model.add_adapter(lora_config)

        unet_state_dict = convert_unet_state_dict_to_peft(lora_state_dict)
        incompatible_keys = set_peft_model_state_dict(model, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )


def load_finetuned_model(args, model):
    if not args.finetuned_model_pt.endswith('.pkl'):
        filepath = os.path.join(args.finetuned_model_pt, "unet", SAFETENSORS_WEIGHTS_NAME)
        model_state_dict = load_file(filepath)
    else:
        with open(args.finetuned_model_pt, "rb") as f:
            model_state_dict = pickle.load(f)
    
    model.load_state_dict(model_state_dict)    


@torch.no_grad()
def main(args):
    pipe = load_pipeline(args.model, get_precision(args.precision), True)
    pipe.to(args.device)

    os.makedirs(args.dst, exist_ok=True)

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
    for idx, prompt in enumerate(tqdm(args.prompts, desc="Generating Pruned")):
        g_cpu = torch.Generator(args.device).manual_seed(args.seed)
        image = pipe(prompt=prompt, generator=g_cpu, num_inference_steps=args.num_intervention_steps).images[0]
        image.save(os.path.join(args.dst, f"pruned_{idx}.png"))

    if args.load_finetuned:
        load_finetuned_model(args, model)
    else:
        load_lora(args, pipe, model)

    # perform inference
    pipe.to(get_precision(args.precision))
    pipe.to(args.device)

    for idx, prompt in enumerate(tqdm(args.prompts, desc="Generating Finetuned")):
        img_name = f"finetuned_pruned_{idx}.png" if args.load_finetuned else f"lora_pruned_{idx}.png"

        g_cpu = torch.Generator(args.device).manual_seed(args.seed)
        image = pipe(prompt=prompt, generator=g_cpu, num_inference_steps=args.num_intervention_steps).images[0]
        image.save(os.path.join(args.dst, img_name))


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
    parser.add_argument("--dst", type=str, default="./demo/sdxl_retrain")
    pruned_model_pt = "./results/sdxl_debug/" \
        + "model_sdxl_eps_0.5_sample_100_beta_0.5_epochs_4_" \
        + "lr_0.150.15_batch_size_4_loss_21_regex_.*_masking_hard_discrete" \
        + "/lambda/prune_results/pruned_model_35.pkl"

    parser.add_argument("--pruned_model_pt", type=str, default=pruned_model_pt, help="path to the pruned model")
    lora_pt = "./results/sdxl_debug/" \
        + "model_sdxl_eps_0.5_sample_100_beta_0.5_epochs_4_" \
        + "lr_0.150.15_batch_size_4_loss_21_regex_.*_masking_hard_discrete" \
        + "/lambda/prune_results/lora_pruned_model_35.pkl"
    
    parser.add_argument("--lora_pt", type=str, default=lora_pt, help="path to the saved lora state dict")
    
    finetuned_model_pt = "./results/sdxl_debug/" \
        + "model_sdxl_eps_0.5_sample_100_beta_0.5_epochs_4_" \
        + "lr_0.150.15_batch_size_4_loss_21_regex_.*_masking_hard_discrete" \
        + "/lambda/prune_results/finetune_pruned_model_35.pkl"
    
    parser.add_argument("--finetuned_model_pt", type=str, default=finetuned_model_pt, help="path to the finetuned model")
    parser.add_argument("--device", type=str, default="cuda:0", help="device for the model")
    parser.add_argument(
        "--precision", type=str, default="bf16", help="precision for the model, bf16, fp32, fp16 available"
    )
    parser.add_argument(
        "--load_finetuned", action="store_true", help="Whether to use full finetuned model or LoRA."
    )
    parser.add_argument(
        "--full_lora", action="store_true", help="Whether to use LoRA on all pruned layers, otherwise attention layers only."
    )
    args = parser.parse_args()

    main(args)


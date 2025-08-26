import gc
import os
import pickle

import torch
from tqdm import tqdm

import argparse
from sdib.evaluation import show_model_memory_consumption_summary, show_model_param_summary
from sdib.utils import (
    calculate_mask_sparsity,
    create_pipeline,
    ffn_linear_layer_pruning,
    get_precision,
    linear_layer_pruning,
    update_flux_single_transformer_projection,
    load_pipeline,
    norm_layer_pruning,
    save_image_seed,
)


@torch.no_grad()
def main(args):
    # assign arguments
    prompts = args.prompts
    seed = args.seed
    num_intervention_steps = args.num_intervention_steps
    dst = args.dst
    save_pt = args.save_pt
    lambda_threshold = args.lambda_threshold
    device = args.device
    memory_usage_device = args.memory_device
    model = args.model
    precision = args.precision

    # save it in the same folder as checkpoint (easier to track)
    dst = os.path.join(os.path.dirname(save_pt), dst)

    # initialize pipeline and hooker
    torch_dtype = get_precision(precision)

    # defaults
    modules_of_interest = ["attn", "ff", "conv", "norm", "proj_", "overall"]

    # perform inference on original model
    original_pipe = load_pipeline(model, torch_dtype, disable_progress_bar=True)
    original_pipe.to("cpu")

    if hasattr(original_pipe, "unet"):
        pruning_module = original_pipe.unet
    else:
        pruning_module = original_pipe.transformer

    show_model_param_summary(pruning_module, modules_of_interest)
    show_model_memory_consumption_summary(pruning_module, memory_usage_device, modules_of_interest)

    original_pipe.to(device)
    for prompt in prompts:
        save_image_seed(
            original_pipe, prompt, num_intervention_steps, device, seed, save_dir=os.path.join(dst, "original")
        )
    del original_pipe
    torch.cuda.empty_cache()

    # get masked model
    pipe, hookers = create_pipeline(
        args.model,
        device,
        torch_dtype,
        args.save_pt,
        binary=args.binary,
        lambda_threshold=args.lambda_threshold,
        epsilon=args.epsilon,
        masking=args.masking,
        return_hooker=True,
        scope=args.scope,
        ratio=args.ratio,
    )
    
    print("calculating mask sparsity for attn")
    total_num_heads, num_activate_heads, mask_sparsity = calculate_mask_sparsity(hookers[0])
    print(f"model: {model}")
    print(
        f"total num heads: {total_num_heads}, num activate heads: {num_activate_heads}, mask sparsity: {mask_sparsity}"
    )

    print("calculating mask sparsity for ffn")
    total_num_heads, num_activate_heads, mask_sparsity = calculate_mask_sparsity(hookers[1])
    print(f"model: {model}")
    print(
        f"total num heads: {total_num_heads}, num activate heads: {num_activate_heads}, mask sparsity: {mask_sparsity}"
    )

    # perform a dummy forward pass to get module names for each lambda
    g_cpu = torch.Generator(device=device).manual_seed(0)
    _ = pipe("...", generator=g_cpu, num_inference_steps=1)

    # perform inference on masked model
    for prompt in prompts:
        save_image_seed(pipe, prompt, num_intervention_steps, device, seed, os.path.join(dst, "masked"))

    # remove parameters in attention blocks
    cross_attn_hooker = hookers[0]
    for name in tqdm(cross_attn_hooker.hook_dict.keys(), desc="Pruning on Attention layer"):
        # make it compatible with both unet and transformer (FLUX and SD3)
        if getattr(pipe, "unet", None):
            module = pipe.unet.get_submodule(name)
        else:
            module = pipe.transformer.get_submodule(name)
        lamb = cross_attn_hooker.lambs[cross_attn_hooker.lambs_module_names.index(name)]
        assert module.heads == lamb.shape[0]
        
        old_inner_dim = module.inner_dim

        # perform pruning on the linear layers
        module = linear_layer_pruning(module, lamb, model)

        # replace module in pipeline
        parent_module_name, child_name = name.rsplit(".", 1)

        if getattr(pipe, "unet", None):
            parent_module = pipe.unet.get_submodule(parent_module_name)
        else:
            parent_module = pipe.transformer.get_submodule(parent_module_name)
            parent_module = update_flux_single_transformer_projection(parent_module, module, lamb, old_inner_dim)
        setattr(parent_module, child_name, module)

    # remove parameters in ffn blocks
    ffn_hook = hookers[1]
    for name in tqdm(ffn_hook.hook_dict.keys(), desc="Pruning on FFN linear layer"):
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

    # remove parameters in norm blocks so far only for flux model
    if not hasattr(pipe, "unet"):
        norm_hook = hookers[2]
        for name in tqdm(norm_hook.hook_dict.keys(), desc="Pruning on Norm layer"):
            module = pipe.transformer.get_submodule(name)
            lamb = norm_hook.lambs[norm_hook.lambs_module_names.index(name)]
            module = norm_layer_pruning(module, lamb)
            parent_module_name, child_name = name.rsplit(".", 1)
            parent_module = pipe.transformer.get_submodule(parent_module_name)
            setattr(parent_module, child_name, module)
        norm_hook.clear_hooks()

    # to get memory usage, need to empty the current GPU
    cross_attn_hooker.clear_hooks()
    ffn_hook.clear_hooks()
    del cross_attn_hooker
    del ffn_hook
    del g_cpu
    del lamb
    del module
    del parent_module
    del _
    pipe.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    network = pipe.unet if hasattr(pipe, "unet") else pipe.transformer
    show_model_param_summary(network, modules_of_interest)
    show_model_memory_consumption_summary(network, memory_usage_device, modules_of_interest)
    pipe.to(device)

    # perform inference on pruned model
    for prompt in prompts:
        save_image_seed(pipe, prompt, num_intervention_steps, device, seed, save_dir=os.path.join(dst, "pruned"))

    # save the pruned model
    if args.save_pruned_model:
        if args.ratio is None:
            raise ValueError("Need to provide ratio to save the pruned model") 
        
        if isinstance(args.ratio, float):
            ratio = [args.ratio]
        elif len(set(args.ratio)) == 1:
            ratio = [args.ratio[0]]
        else:
            ratio = args.ratio
        save_name = f"pruned_model_{'_'.join(f'{(1 - r) * 100:.0f}' for r in ratio)}.pkl"

        with open(os.path.join(dst, save_name), "wb") as f:
            pruned_model = network
            pickle.dump(pruned_model, f)

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
        help="prompts for the model eval",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--model", type=str, default="sdxl")
    parser.add_argument("--num_intervention_steps", "-ni", type=int, default=50)
    parser.add_argument("--dst", type=str, default="./results/prune_results")
    save_pt = None  # Must be provided by user
    parser.add_argument("--save_pt", type=str, default=save_pt, help="path to the saved model")
    parser.add_argument("--lambda_threshold", "-lt", type=float, default=-10, help="lambda threshold for pruning")
    parser.add_argument("--device", type=str, default="cuda:0", help="device for the model")
    parser.add_argument(
        "--memory_device", type=str, default="cuda:1", help="device for evaluation memory usage the model"
    )
    parser.add_argument(
        "--precision", type=str, default="bf16", help="precision for the model, bf16, fp32, fp16 available"
    )
    parser.add_argument(
        "--binary", action="store_true", help="whether to use binary mask"
    )
    parser.add_argument(
        "--masking", type=str, default="binary", help="masking type, available binary, hard_discrete, sigmoid"
    )
    parser.add_argument("--scope", type=str, default=None, help="scope for lambda binary mask")
    parser.add_argument(
        "--ratio", type=float, nargs="+", default=None, help="sparsity ratio for local global lambda mask"
    )
    parser.add_argument("--epsilon", "-e", type=float, default=0.0, help="epsilon for lambda")
    parser.add_argument("--save_pruned_model", "-spm", action="store_true", help="whether to save the pruned model")
    args = parser.parse_args()

    main(args)

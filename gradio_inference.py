import gradio as gr
from dataclasses import dataclass

import torch
from tqdm import tqdm

from src.utils import (
    create_pipeline,
    calculate_mask_sparsity,
    ffn_linear_layer_pruning,
    linear_layer_pruning,
)
from diffusers import StableDiffusionXLPipeline


def get_model_param_summary(model, verbose=False):
    params_dict = dict()
    overall_params = 0
    for name, params in model.named_parameters():
        num_params = params.numel()
        overall_params += num_params
        if verbose:
            print(f"GPU Memory Requirement for {name}: {params} MiB")
        params_dict.update({name: num_params})
    params_dict.update({"overall": overall_params})
    return params_dict


@dataclass
class GradioArgs:
    ckpt: str = "./mask/ff.pt"
    device: str = "cuda:0"
    seed: list = None
    prompt: str = None
    mix_precision: str = "bf16"
    num_intervention_steps: int = 50
    model: str = "sdxl"
    binary: bool = False
    masking: str = "binary"
    scope: str = "global"
    ratio: list = None
    width: int = None
    height: int = None
    epsilon: float = 0.0
    lambda_threshold: float = 0.001

    def __post_init__(self):
        if self.seed is None:
            self.seed = [44]
        if self.ratio is None:
            self.ratio = [0.68, 0.88]


def prune_model(pipe, hookers):
    # remove parameters in attention blocks
    cross_attn_hooker = hookers[0]
    for name in tqdm(cross_attn_hooker.hook_dict.keys(), desc="Pruning attention layers"):
        if getattr(pipe, "unet", None):
            module = pipe.unet.get_submodule(name)
        else:
            module = pipe.transformer.get_submodule(name)
        lamb = cross_attn_hooker.lambs[cross_attn_hooker.lambs_module_names.index(name)]
        assert module.heads == lamb.shape[0]
        module = linear_layer_pruning(module, lamb)

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
        module = ffn_linear_layer_pruning(module, lamb)

        parent_module_name, child_name = name.rsplit(".", 1)
        if getattr(pipe, "unet", None):
            parent_module = pipe.unet.get_submodule(parent_module_name)
        else:
            parent_module = pipe.transformer.get_submodule(parent_module_name)
        setattr(parent_module, child_name, module)

    cross_attn_hooker.clear_hooks()
    ffn_hook.clear_hooks()
    return pipe


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

    # Print mask sparsity info
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

    # Prune the model
    pruned_pipe = prune_model(mask_pipe, hookers)

    # reload the original model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
    ).to(args.device)

    # get model param summary
    print(f"original model param: {get_model_param_summary(pipe.unet)['overall']}")
    print(f"pruned model param: {get_model_param_summary(pruned_pipe.unet)['overall']}")
    print("prune complete")
    return pipe, pruned_pipe


def generate_images(prompt, seed, steps, pipe, pruned_pipe):
    # Run the model and return images directly
    g_cpu = torch.Generator("cuda:0").manual_seed(seed)
    original_image = pipe(prompt=prompt, generator=g_cpu, num_inference_steps=steps).images[0]
    g_cpu = torch.Generator("cuda:0").manual_seed(seed)
    ecodiff_image = pruned_pipe(prompt=prompt, generator=g_cpu, num_inference_steps=steps).images[0]
    return original_image, ecodiff_image


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Image Generation with EcoDiff Pruned Model")
        with gr.Row():
            model_choice = gr.Dropdown(choices=["SDXL"], value="SDXL", label="Model", scale=1.2)
            pruning_ratio = gr.Dropdown(choices=["20%"], value="20%", label="Pruning Ratio", scale=1.2)
            status_label = gr.Label(value="Model Not Initialized", scale=1)
            prune_btn = gr.Button("Initialize Original and Pruned Models", variant="primary", scale=1)
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="A clock tower floating in a sea of clouds", scale=3)
            seed = gr.Number(label="Seed", value=44, precision=0, scale=1)
            steps = gr.Slider(label="Number of Steps", minimum=1, maximum=100, value=50, step=1, scale=1)
            generate_btn = gr.Button("Generate Images")
        with gr.Row():
            original_output = gr.Image(label="Original Output")
            ecodiff_output = gr.Image(label="EcoDiff Output")

        pipe_state = gr.State(None)
        pruned_pipe_state = gr.State(None)

        def on_prune_click(prompt, seed, steps):
            args = GradioArgs(prompt=prompt, seed=[seed], num_intervention_steps=steps)
            pipe, pruned_pipe = binary_mask_eval(args)
            return pipe, pruned_pipe, "Model Initialized"

        prune_btn.click(
            fn=on_prune_click,
            inputs=[prompt, seed, steps],
            outputs=[pipe_state, pruned_pipe_state, status_label],
        )

        def on_generate_click(prompt, seed, steps, pipe, pruned_pipe):
            original_image, ecodiff_image = generate_images(prompt, seed, steps, pipe, pruned_pipe)
            return original_image, ecodiff_image

        generate_btn.click(
            fn=on_generate_click,
            inputs=[prompt, seed, steps, pipe_state, pruned_pipe_state],
            outputs=[original_output, ecodiff_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)

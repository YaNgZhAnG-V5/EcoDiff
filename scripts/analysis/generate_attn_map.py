import os

import matplotlib.pyplot as plt
import torch

import argparse
from sdib.hooks import CrossAttentionExtractionHook
from sdib.models import SDXLDiffusionPipeline
from sdib.utils import MODEL_ID, get_cfg, save_image_seed


def get_args(save_pt):
    basename = os.path.basename(save_pt)
    basename_list = basename.split("_")
    arg_dict = {}
    config_length = len(basename_list) // 2
    for i in range(config_length):
        key = basename_list[2 * i]
        value = basename_list[2 * i + 1]
        arg_dict[key] = value
    return arg_dict


def attn_map_eval(args):
    seed = args.seed[0]
    if args.torch_dtype == "bf16":
        torch_dtype = torch.bfloat16
    save_pt = args.save_pt
    prompt = args.prompt
    pipe = SDXLDiffusionPipeline.from_pretrained(
        MODEL_ID[args.model],
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    pipe.set_progress_bar_config(disable=True)
    args.device = f"cuda:{args.device}"
    pipe.to(args.device)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    device = torch.device(args.device)
    g_cpu = torch.Generator(device.type).manual_seed(seed)
    # initialize cross attention hooks
    cfg_dict = get_cfg(save_pt)
    cross_attn_hooker = CrossAttentionExtractionHook(
        pipe,
        regex=cfg_dict["regex"],
        dtype=torch_dtype,
        head_num_filter=1,
        masking=cfg_dict["masking"],
        dst=save_pt,
        epsilon=0.0,
        binary=True,
        return_attention=True,
    )

    cross_attn_hooker.add_hooks(init_value=1)
    _ = pipe(" ", generator=g_cpu, num_inference_steps=1)
    cross_attn_hooker.load(device=args.device, threshold=args.lambda_threshold)

    if not args.load_lambda:
        # activate all heads
        num_blocks = len(cross_attn_hooker.lambs)
        for i in range(num_blocks):
            cross_attn_hooker.lambs[i] = torch.ones_like(cross_attn_hooker.lambs[i])

    cross_attn_hooker.clean_cross_attn()

    for prompt in args.prompt:
        cross_attn_hooker.clean_cross_attn()
        _ = save_image_seed(pipe, prompt, 50, device, seed, None)

        # use tokenizier for tokenizing the prompt
        tokenizer = pipe.tokenizer
        prompt_list = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt_list = [tokenizer.convert_ids_to_tokens(i) for i in prompt_list]
        text_sequence_length = len(prompt_list)
        model_dir = os.path.dirname(save_pt)

        # Plot each heatmap using a loop
        if args.all_heatmap:
            for t in args.timestep:
                output_dict = cross_attn_hooker.get_process_cross_attn_result(text_sequence_length, timestep=t)
                for block_name, block_data in output_dict.items():
                    attn_map = block_data["attn_map"]
                    lambs = block_data["lambda"]
                    text_sequence_length = len(prompt_list)

                    num_heads = len(attn_map) // text_sequence_length
                    fig, axs = plt.subplots(
                        num_heads, text_sequence_length, figsize=(5 * text_sequence_length, 5 * num_heads)
                    )
                    # Flatten the axs array for easy iteration
                    axs = axs.flatten()

                    for i, d in enumerate(attn_map):
                        head_num, text_token_id = divmod(i, text_sequence_length)
                        cax = axs[i].imshow(d, cmap="viridis")
                        axs[i].set_title(
                            f"head:{head_num}," + f"token:{prompt_list[text_token_id]}" + f"lambda:{lambs[head_num]}"
                        )
                        fig.colorbar(cax, ax=axs[i])

                    # Adjust layout and display the plot
                    plt.tight_layout()
                    output_dir = os.path.join(
                        args.output_dir, model_dir, "attn_heatmap_per_head", prompt, f"timestep_{t}"
                    )
                    if os.path.exists(output_dir) is False:
                        os.makedirs(output_dir)

                    output_path = os.path.join(output_dir, f"{block_name}.jpg")
                    fig.savefig(output_path)

        else:
            # generate attn map for each timestep (intervention step t)
            for t in args.timestep:
                # get the cross attn result for the timestep t
                output_dict = cross_attn_hooker.get_process_cross_attn_result(text_sequence_length, timestep=t)

                # get the average attention map for each block
                for block_name, block_data in output_dict.items():
                    attn_map = block_data["attn_map"]
                    lambs = block_data["lambda"]
                    text_sequence_length = len(prompt_list)

                    # generate attn map for averaging out the heads
                    average_attn_map_dict_mask = {}
                    average_attn_map_dict_nomask = {}

                    num_heads = len(attn_map) // text_sequence_length

                    if args.softmax:
                        # softmax per head across the text tokens
                        for head_idx in range(num_heads):
                            attn_map_per_head = attn_map[
                                head_idx * text_sequence_length : (head_idx + 1) * text_sequence_length
                            ]
                            att_map = torch.cat([torch.tensor(attn) for attn in attn_map_per_head], dim=-1)
                            att_map = att_map.softmax(dim=-1)
                            for i in range(text_sequence_length):
                                attn_map[head_idx * text_sequence_length + i] = att_map[..., i : i + 1].numpy()

                    for i, d in enumerate(attn_map):
                        head_num, text_token_id = divmod(i, text_sequence_length)
                        lambda_val = lambs[head_num].detach().to(torch.float32).cpu().numpy()

                        if prompt_list[text_token_id] not in average_attn_map_dict_mask:
                            average_attn_map_dict_mask[prompt_list[text_token_id]] = d * lambda_val
                            average_attn_map_dict_nomask[prompt_list[text_token_id]] = d
                        else:
                            average_attn_map_dict_mask[prompt_list[text_token_id]] += d * lambda_val
                            average_attn_map_dict_nomask[prompt_list[text_token_id]] += d

                    for k, v in average_attn_map_dict_mask.items():
                        average_attn_map_dict_mask[k] /= num_heads
                        average_attn_map_dict_nomask[k] /= num_heads

                    fig, axs = plt.subplots(1, text_sequence_length, figsize=(5 * text_sequence_length, 5 * 1))

                    if args.load_lambda:
                        for i, (k, v) in enumerate(average_attn_map_dict_mask.items()):
                            # v = np.array(v*255).astype('uint8')
                            # v = cv.equalizeHist(v)
                            # v  = (v - v.min()) / (v.max() - v.min() + v.min() / 10) # normalization
                            cax = axs[i].imshow(v, cmap="viridis")
                            axs[i].set_title(f"masked token:{k}")
                            fig.colorbar(cax, ax=axs[i])
                    else:
                        for idx, (k, v) in enumerate(average_attn_map_dict_nomask.items()):
                            # v = np.array(v*255).astype('uint8')
                            # v = cv.equalizeHist(v)
                            # v = (v - v.min()) / (v.max() - v.min() + v.min() / 10)  # normalization
                            cax = axs[idx].imshow(v, cmap="viridis")
                            axs[idx].set_title(f"no mask token:{k}")
                            fig.colorbar(cax, ax=axs[idx])

                    plt.tight_layout()

                    status = "load_lambda" if args.load_lambda else "no_lambda"
                    output_dir = os.path.join(args.output_dir, model_dir, "attn_heatmap", f"timestep_{t}", status)
                    if os.path.exists(output_dir) is False:
                        os.makedirs(output_dir)

                    output_path = os.path.join(output_dir, f"{block_name}_timestep_{t}_average.jpg")
                    fig.savefig(output_path, dpi=100)


def main(args):
    attn_map_eval(args)


if __name__ == "__main__":
    # define default arguments
    save_pt = None  # Must be provided by user
    model = "sdxl"
    device = "0"
    seed = [44]
    prompt = ["cat"]  # , "one blue horse and a red dog", "girl with umbrella"]
    torch_dtype = "bf16"

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep", "-ts", nargs="+", default=[15])
    parser.add_argument("--output_dir", type=str, default="./results/")
    parser.add_argument("--save_pt", "-sp", type=str, default=save_pt)
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("--device", "-d", type=str, default=device)
    parser.add_argument("--seed", default=seed)
    parser.add_argument("--prompt", type=str, nargs="+", default=prompt)
    parser.add_argument("--torch_dtype", default=torch_dtype)
    parser.add_argument("--lambda_threshold", "-lt", type=float, default=1)
    parser.add_argument("--all_heatmap", "-ah", action="store_true")
    parser.add_argument("--softmax", "-sm", action="store_true")
    parser.add_argument("--load_lambda", "-ll", action="store_true")
    args = parser.parse_args()
    main(args)

import os
from copy import deepcopy
from typing import Optional

import torch
from diffusers.models.activations import GEGLU, GELU
from cross_attn_hook import CrossAttentionExtractionHook
from ffn_hooker import FeedForwardHooker

# create dummy module for skip connection
class SkipConnection(torch.nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()

    def forward(*args, **kwargs):
        return args[1]

def calculate_mask_sparsity(hooker, threshold: Optional[float] = None):
    total_num_lambs = 0
    num_activate_lambs = 0
    binary = getattr(hooker, "binary", None)  # if binary is not present, it will return None for ff_hooks
    for lamb in hooker.lambs:
        total_num_lambs += lamb.size(0)
        if binary:
            assert threshold is None, "threshold should be None for binary mask"
            num_activate_lambs += lamb.sum().item()
        else:
            assert threshold is not None, "threshold must be provided for non-binary mask"
            num_activate_lambs += (lamb >= threshold).sum().item()
    return total_num_lambs, num_activate_lambs, num_activate_lambs / total_num_lambs

    
def create_pipeline(
    pipe,
    model_id,
    device,
    torch_dtype,
    save_pt=None,
    lambda_threshold: float = 1,
    binary=True,
    epsilon=0.0,
    masking="binary",
    attn_name="attn",
    return_hooker=False,
    scope=None,
    ratio=None,
):
    """
    create the pipeline and optionally load the saved mask
    """
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "unet"):
        pipe.unet.requires_grad_(False)
    else:
        pipe.transformer.requires_grad_(False)
    if save_pt:
        # TODO should merge all the hooks checkpoint into one
        if "ff.pt" in save_pt or "attn.pt" in save_pt:
            save_pts = get_save_pts(save_pt)

            cross_attn_hooker = CrossAttentionExtractionHook(
                pipe,
                model_name=model_id,
                regex=".*",
                dtype=torch_dtype,
                head_num_filter=1,
                masking=masking,  # need to change to binary during inference
                dst=save_pts["attn"],
                epsilon=epsilon,
                attn_name=attn_name,
                binary=binary,
            )
            cross_attn_hooker.add_hooks(init_value=1)
            ff_hooker = FeedForwardHooker(
                pipe,
                regex=".*",
                dtype=torch_dtype,
                masking=masking,
                dst=save_pts["ff"],
                epsilon=epsilon,
                binary=binary,
            )
            ff_hooker.add_hooks(init_value=1)
            norm_hooker = None

            g_cpu = torch.Generator(torch.device(device)).manual_seed(1)
            _ = pipe("abc", generator=g_cpu, num_inference_steps=1)
            cross_attn_hooker.load(device=device, threshold=lambda_threshold)
            ff_hooker.load(device=device, threshold=lambda_threshold)
            if norm_hooker:
                norm_hooker.load(device=device, threshold=lambda_threshold)
            if scope == "local" or scope == "global":
                if isinstance(ratio, float):
                    attn_hooker_ratio = ratio
                    ff_hooker_ratio = ratio
                else:
                    attn_hooker_ratio, ff_hooker_ratio = ratio[0], ratio[1]

                    if norm_hooker:
                        if len(ratio) < 3:
                            raise ValueError("Need to provide ratio for norm layer")
                        norm_hooker_ratio = ratio[2]

                cross_attn_hooker.binarize(scope, attn_hooker_ratio)
                ff_hooker.binarize(scope, ff_hooker_ratio)
                if norm_hooker:
                    norm_hooker.binarize(scope, norm_hooker_ratio)
            hookers = [cross_attn_hooker, ff_hooker]
            if norm_hooker:
                hookers.append(norm_hooker)

    if return_hooker:
        return pipe, hookers
    else:
        return pipe


def linear_layer_pruning(module, lamb):
    heads_to_keep = torch.nonzero(lamb).squeeze()
    if len(heads_to_keep.shape) == 0:
        # if only one head is kept, or none
        heads_to_keep = heads_to_keep.unsqueeze(0)

    modules_to_remove = [module.to_k, module.to_q, module.to_v]
    new_heads = int(lamb.sum().item())

    if new_heads == 0:
        return SkipConnection()

    for module_to_remove in modules_to_remove:
        # get head dimension
        inner_dim = module_to_remove.out_features // module.heads
        # place holder for the rows to keep
        rows_to_keep = torch.zeros(
            module_to_remove.out_features, dtype=torch.bool, device=module_to_remove.weight.device
        )

        for idx in heads_to_keep:
            rows_to_keep[idx * inner_dim : (idx + 1) * inner_dim] = True

        # overwrite the inner projection with masked projection
        module_to_remove.weight.data = module_to_remove.weight.data[rows_to_keep, :]
        if module_to_remove.bias is not None:
            module_to_remove.bias.data = module_to_remove.bias.data[rows_to_keep]
        module_to_remove.out_features = int(sum(rows_to_keep).item())

    # Also update the output projection layer if available, (for FLUXSingleAttnProcessor2_0)
    # with column masking, dim 1
    if getattr(module, "to_out", None) is not None:
        module.to_out[0].weight.data = module.to_out[0].weight.data[:, rows_to_keep]
        module.to_out[0].in_features = int(sum(rows_to_keep).item())

    # update parameters in the attention module
    module.inner_dim = module.inner_dim // module.heads * new_heads
    try:
        module.query_dim = module.query_dim // module.heads * new_heads
        module.inner_kv_dim = module.inner_kv_dim // module.heads * new_heads
    except:
        pass
    module.cross_attention_dim = module.cross_attention_dim // module.heads * new_heads
    module.heads = new_heads
    return module


def ffn_linear_layer_pruning(module, lamb):
    lambda_to_keep = torch.nonzero(lamb).squeeze()
    if len(lambda_to_keep) == 0:
        return SkipConnection()

    num_lambda = len(lambda_to_keep)

    if isinstance(module.net[0], GELU):
        # linear layer weight remove before activation
        module.net[0].proj.weight.data = module.net[0].proj.weight.data[lambda_to_keep, :]
        module.net[0].proj.out_features = num_lambda
        if module.net[0].proj.bias is not None:
            module.net[0].proj.bias.data = module.net[0].proj.bias.data[lambda_to_keep]

        update_act = GELU(module.net[0].proj.in_features, num_lambda)
        update_act.proj = module.net[0].proj
        module.net[0] = update_act
    elif isinstance(module.net[0], GEGLU):
        output_feature = module.net[0].proj.out_features
        module.net[0].proj.weight.data = torch.cat(
            [
                module.net[0].proj.weight.data[: output_feature // 2, :][lambda_to_keep, :],
                module.net[0].proj.weight.data[output_feature // 2 :][lambda_to_keep, :],
            ],
            dim=0,
        )
        module.net[0].proj.out_features = num_lambda * 2
        if module.net[0].proj.bias is not None:
            module.net[0].proj.bias.data = torch.cat(
                [
                    module.net[0].proj.bias.data[: output_feature // 2][lambda_to_keep],
                    module.net[0].proj.bias.data[output_feature // 2 :][lambda_to_keep],
                ]
            )

        update_act = GEGLU(module.net[0].proj.in_features, num_lambda * 2)
        update_act.proj = module.net[0].proj
        module.net[0] = update_act

    # proj weight after activation
    module.net[2].weight.data = module.net[2].weight.data[:, lambda_to_keep]
    module.net[2].in_features = num_lambda

    return module

def get_save_pts(save_pt):
    if "ff.pt" in save_pt:
        ff_save_pt = deepcopy(save_pt)  # avoid in-place operation
        attn_save_pt = save_pt.split(os.sep)
        attn_save_pt[-1] = attn_save_pt[-1].replace("ff", "attn")
        attn_save_pt_output = os.sep.join(attn_save_pt)
        attn_save_pt[-1] = attn_save_pt[-1].replace("attn", "norm")
        norm_save_pt = os.sep.join(attn_save_pt)

        return {
            "ff": ff_save_pt,
            "attn": attn_save_pt_output,
            "norm": norm_save_pt,
        }
    else:
        attn_save_pt = deepcopy(save_pt)
        ff_save_pt = save_pt.split(os.sep)
        ff_save_pt[-1] = ff_save_pt[-1].replace("attn", "ff")
        ff_save_pt_output = os.sep.join(ff_save_pt)
        ff_save_pt[-1] = ff_save_pt[-1].replace("ff", "norm")
        norm_save_pt = os.sep.join(attn_save_pt)

        return {
            "ff": ff_save_pt_output,
            "attn": attn_save_pt,
            "norm": norm_save_pt,
        }

def save_img(pipe, g_cpu, steps, prompt, save_path):
    image = pipe(prompt, generator=g_cpu, num_inference_steps=steps)
    image["images"][0].save(save_path)
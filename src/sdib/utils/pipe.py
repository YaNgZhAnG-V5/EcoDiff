import os

import torch
from diffusers import EulerDiscreteScheduler

from copy import deepcopy
from sdib.hooks import CrossAttentionExtractionHook, FeedForwardHooker, NormHooker
from sdib.models import (
    DiTIBPipeline,
    FluxIBPipeline,
    SDIBDiffusion3Pipeline,
    SDIBDiffusionPipeline,
    SDXLDiffusionPipeline,
)
from sdib.scheduler import ReverseDPMSolverMultistepScheduler


def get_cfg(save_pt):
    pos = -3 if "lambda" in save_pt else -2
    cfg = save_pt.split(os.sep)[pos]
    if "down_blocks" in cfg:
        cfg = cfg.replace("down_blocks", "down-blocks")
    if "up_blocks" in cfg:
        cfg = cfg.replace("up_blocks", "up-blocks")
    if "mid_block" in cfg:
        cfg = cfg.replace("mid_block", "mid-block")
    cfg = cfg.replace("batch_size", "batch-size")
    cfg = cfg.replace("hard_discrete", "hard-discrete")
    if "flux_dev" in cfg:
        cfg = cfg.replace("flux_dev", "flux-dev")
    cfg = cfg.split("_")

    # list insert
    if "loss" not in cfg:
        cfg.insert(10, "loss")
    cfg_length = len(cfg)
    cfg_dict = {}
    for i in range(cfg_length // 2):
        key = cfg[2 * i]
        value = cfg[2 * i + 1]
        cfg_dict[key] = value

    # correct hard_discrete
    if cfg_dict["masking"] == "hard-discrete":
        cfg_dict["masking"] = "hard_discrete"
        
    if cfg_dict["model"] == "flux-dev":
        cfg_dict["model"] = "flux_dev"
    cfg_dict["regex"] = cfg_dict["regex"].replace("-", "_")
    return cfg_dict


def load_pipeline(model_str: str, torch_dtype: torch.dtype, disable_progress_bar: bool):
    """load a diffusion pipeline"""
    if model_str == "sd1":
        pipe = SDIBDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", include_entities=False)
    elif model_str == "sd2":
        model_id = "stabilityai/stable-diffusion-2-base"
        # Use the Euler scheduler here instead
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = SDIBDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
        )
    elif model_str == "sdxl":
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = SDXLDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    elif model_str == "sd3":
        pipe = SDIBDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch_dtype)
    elif model_str == "dit":
        pipe = DiTIBPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch_dtype)
        pipe.scheduler = ReverseDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_str == "flux":
        pipe = FluxIBPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch_dtype)
    elif model_str == "flux_dev":
        pipe = FluxIBPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Model {model_str} not supported")
    pipe.set_progress_bar_config(disable=disable_progress_bar)
    return pipe


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
        norm_save_pt = os.sep.join(ff_save_pt)

        return {
            "ff": ff_save_pt_output,
            "attn": attn_save_pt,
            "norm": norm_save_pt,
        }


def create_pipeline(
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
    legacy_mode=False,
):
    """
    create the pipeline and optionally load the saved mask
    """
    pipe = load_pipeline(model_id, torch_dtype, disable_progress_bar=True)
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "unet"):
        pipe.unet.requires_grad_(False)
    else:
        pipe.transformer.requires_grad_(False)
    if save_pt:
        if "ff.pt" in save_pt or "attn.pt" in save_pt:
            save_pts = get_save_pts(save_pt)

            # use one to get the config
            cfg_dict = get_cfg(save_pts["ff"])
            cross_attn_hooker = CrossAttentionExtractionHook(
                pipe,
                model_name=model_id,
                regex=cfg_dict["regex"],
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
                regex=cfg_dict["regex"],
                dtype=torch_dtype,
                masking=masking,
                dst=save_pts["ff"],
                epsilon=epsilon,
                binary=binary,
                legacy_mode=legacy_mode
            )
            ff_hooker.add_hooks(init_value=1)

            if os.path.exists(save_pts["norm"]):
                norm_hooker = NormHooker(
                    pipe,
                    regex=cfg_dict["regex"],
                    dtype=torch_dtype,
                    masking=masking,
                    dst=save_pts["norm"],
                    epsilon=epsilon,
                    binary=binary,
                    legacy_mode=legacy_mode
                )
                norm_hooker.add_hooks(init_value=1)
            else:
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

                    if norm_hooker:
                        norm_hooker_ratio = ratio
                elif len(ratio) == 1:
                    attn_hooker_ratio = ratio[0]
                    ff_hooker_ratio = ratio[0]

                    if norm_hooker:
                        norm_hooker_ratio = ratio[0]
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
        else:
            hookers = None
    else:
        hookers = None
    if return_hooker:
        return pipe, hookers
    else:
        return pipe

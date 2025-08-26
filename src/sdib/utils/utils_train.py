import os

import omegaconf
import torch

import time


def get_file_name(save_dir: str, prompt: str = None, seed: int = 44):
    # get current time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # get file name
    name = f"{prompt}_seed_{seed}_{timestr}.png"
    out_path = os.path.join(save_dir, name)
    return out_path


@torch.no_grad()
def save_image(
    pipe,
    prompts: str,
    g_cpu: torch.Generator,
    steps: int,
    seed: int,
    save_dir=None,
    save_path=None,
    width=None,
    height=None,
):
    image = pipe(prompts, generator=g_cpu, num_inference_steps=steps, width=width, height=height)

    if save_path is not None:
        image["images"][0].save(save_path)
        return

    if save_dir is None:
        return image["images"]
    else:
        if isinstance(prompts, str):
            prompts = [prompts]
        for img, prompt in zip(image["images"], prompts):
            name = get_file_name(save_dir, prompt=prompt, seed=seed)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img.save(name)
        return None


def save_image_seed(
    pipe,
    prompts: str,
    steps: int,
    device: torch.device,
    seed: int,
    save_dir=None,
    save_path=None,
    width=None,
    height=None,
):
    g_cpu = torch.Generator(device).manual_seed(seed)
    return save_image(
        pipe, prompts, g_cpu, steps, seed=seed, save_dir=save_dir, save_path=save_path, width=width, height=height
    )


def save_image_binarize_seed(
    pipe,
    hookers: list,
    prompts: str,
    steps: int,
    device: torch.device,
    seed: int,
    save_dir=None,
    save_path=None,
    width=None,
    height=None,
):
    if not isinstance(hookers, list):
        hookers = [hookers]
    previous_masking = []
    for h in hookers:
        previous_masking.append(h.masking)
        h.masking = "continues2binary"
    g_cpu = torch.Generator(device).manual_seed(seed)
    img = save_image(
        pipe, prompts, g_cpu, steps, seed=seed, save_dir=save_dir, save_path=save_path, width=width, height=height
    )
    for h, pm in zip(hookers, previous_masking):
        h.masking = pm
    return img


def overwrite_debug_cfg(cfg):
    # overwrite the cfg for debug, use few image for training, and use few steps for generating image
    # get overwriteed cfg attribute list
    overwrite_list = cfg.debug_cfg
    for key, value in overwrite_list.items():
        key_list = key.split(".")
        attr = cfg[key_list[0]]
        if len(key_list) > 2:
            for k in key_list[1:-1]:
                attr = attr[k]
        attr[key_list[-1]] = value
    print(f"overwriting these config parameter: {overwrite_list}, updated cfg: {cfg}")


def load_config(cfg_path: str):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    if cfg.debug:
        overwrite_debug_cfg(cfg)
    return cfg


def save_model_hook(model, save_dir):
    pass


def load_model_hook(model, save_dir):
    pass

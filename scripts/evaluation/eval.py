# this scripts includes the evaluation functions for each masked stable diffusion model
# including:
# quantative analysis of masked diffusion model: binary mask evaluation and mask attention map generation
# qualitative analysis of masked diffusion model: semantic eval with clip contrastive score and fid score

import glob
import os

from binary_mask_eval import binary_mask_eval
from ..analysis.generate_attn_map import attn_map_eval
from omegaconf import OmegaConf
from semantic_eval import semantic_eval

import argparse


def get_configs(exp_path):
    # sample path
    # ./results/experiment_name/latest_lambda.pt
    config_list = exp_path.split(os.sep)
    experiment_name = config_list[-2]
    save_pt = os.path.join(exp_path, "latest_lambda.pt")
    return {
        "experiment_name": experiment_name,
        "save_pt": save_pt,
    }


def main(args):
    cfg = OmegaConf.load(args.config)
    overrides_cfg = get_configs(args.exp_path)
    cfg.base.experiment_name = overrides_cfg["experiment_name"]
    cfg.base.save_pt = overrides_cfg["save_pt"]
    OmegaConf.resolve(cfg)
    print("evaluation cfg", cfg)

    print("Start semantic evaluation...")
    semantic_eval(cfg.semantic_eval)

    print("Start binary mask evaluation...")
    binary_mask_eval(cfg.binary_mask_eval)

    print("Start binary mask evaluation...")
    attn_map_eval(cfg.attn_map_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Image semantic evaluation")
    parser.add_argument(
        "--config",
        type=str,
        help="path to config file",
        default="./configs/eval.yaml",
    )
    base_dir = "./results/"
    parser.add_argument(
        "--exp_path",
        "-ep",
        type=str,
        help="experiment path",
        default=os.path.join(
            base_dir, "sample_100_beta_7_epochs_3_lr_0.5_batch_size_4_l0_regex_.*_masking_hard_discrete"
        ),
    )
    parser.add_argument("--run_all", "-ra", action="store_true", help="run all experiments")
    args = parser.parse_args()

    if args.run_all:
        all_exp_paths = glob.glob(os.path.join(base_dir, "*"))
        all_exp_paths.sort()
        for exp_path in all_exp_paths:
            print("start evaluation for", exp_path)
            args.exp_path = exp_path
            main(args)
    else:
        main(args)

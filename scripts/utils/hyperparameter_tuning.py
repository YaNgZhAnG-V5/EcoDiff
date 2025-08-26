# this file for generating hyperparameters config files for the hyperparameter tuning

import concurrent.futures
import glob
import os
from functools import partial
from itertools import product

import omegaconf

import argparse


def generate_configs(args):
    base_cfg = omegaconf.OmegaConf.load(args.config)
    for alr, flr, nlr, masking, re, loss_reg_norm, loss_recons_norm, beta, num_intervention, prompt, eps, ds in product(
        args.attn_learning_rate,
        args.ffn_learning_rate,
        args.n_learning_rate,
        args.masking,
        args.regex,
        args.loss_reg,
        args.loss_recons,
        args.beta,
        args.num_intervention,
        args.prompts,
        args.eps,
        args.data_size,
    ):
        cfg = base_cfg
        # if not args.train_task == "deconcept":
        cfg.data.size = ds
        # check default base config file for formating
        cfg.trainer.attn_lr = alr
        cfg.trainer.n_lr = nlr
        cfg.trainer.ff_lr = flr
        cfg.trainer.masking = masking
        cfg.trainer.masking_eps = eps
        cfg.trainer.regex = re
        cfg.trainer.beta = beta
        cfg.trainer.device = f"cuda:{args.device}"
        cfg.trainer.attn_name = args.attn_name
        cfg.loss.reg = loss_reg_norm
        cfg.loss.reconstruct = loss_recons_norm
        cfg.debug_cfg["trainer.num_intervention_steps"] = num_intervention
        cfg.logger.project = args.project_name
        cfg.trainer.grad_checkpointing = args.ncheckpointing
        cfg.debug = False  # default
        if args.train_task == "general":
            # lr with attn_lr and ffn_lr, avoid too long file name
            output_file = os.path.join(
                args.output_dir,
                f"lr_{alr}{flr}{nlr}_masking_{masking}_eps_{eps}_re_{re}_loss_reg_{loss_reg_norm}"
                + f"_recon_{loss_recons_norm}_beta{beta}_ds_{ds}.yaml",
            )
        elif args.train_task == "debug":
            cfg.debug = args.debug
            cfg.debug_cfg["logger.type"] = "wandb"
            checkpointing = "checkpointing" if args.ncheckpointing else "no_checkpointing"
            output_file = os.path.join(args.output_dir, f"num_inter{num_intervention}_{checkpointing}.yaml")
        elif args.train_task == "concept":
            cfg.data.prompt = prompt
            output_file = os.path.join(
                args.output_dir,
                f"prompt_{prompt},lr_{alr}{flr}_masking_{masking}_re_{re}_loss_reg_"
                + f"{loss_reg_norm}_recon_{loss_recons_norm}_beta{beta}.yaml",
            )
        elif args.train_task == "deconcept":
            output_file = os.path.join(
                args.output_dir,
                f"lr_{alr}{flr}_masking_{masking}_re_{re}_loss_" + f"{loss_reg_norm}{loss_recons_norm}_beta{beta}.yaml",
            )
        else:
            raise ValueError("Invalid task, should be general, debug, concept or deconcept")
        omegaconf.OmegaConf.save(cfg, output_file)


def run_experiment(config_file, task="general"):
    base_cmd = f"""python ./scripts/train.py --cfg \"{config_file}/\" --task {task}"""
    os.system(base_cmd)


def main(args):
    if args.task == "gen":
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        generate_configs(args)
    elif args.task == "run":
        # get all the config files in the output directory
        cfg_path_list = glob.glob(os.path.join(args.output_dir, "*.yaml"))

        # run the experiments with multithreads processes with max running job of 5 at a time (default)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_job) as executor:
            # submit the jobs to executor
            func_run_list = [
                partial(run_experiment, config_file=cfg_path, task=args.train_task) for cfg_path in cfg_path_list
            ]
            futures = [executor.submit(func) for func in func_run_list]
            for future in concurrent.futures.as_completed(futures):
                print("finish this task", future)  # might not need this

        print("All tasks are done")
    else:
        raise ValueError("Invalid task, should be gen or run")


if __name__ == "__main__":
    # grid search for hyperparameters learning rate
    LR = [5e-1, 1e-1]
    # masking methods
    MASKING = ["hard_discrete"]
    RE = [
        "^(up_blocks).*",
        ".*",
        "^(down_blocks).*",
        "^(mid_block).*",
    ]
    L = [1, 0]
    BETA = [3, 4, 5, 6, 7]
    EPS = [1e-8, 1e-3, 1e-1, 1]
    NUM_INTERVENTION = [50]
    DEVICE = "cuda:0"
    DEBUG = False
    PROJECT_NAME = "sdxledit_vram_runtime"
    CHECKPOINTING = True
    PROMPTS = ["dog", "cat", "bird", "car", "plane", "tree", "flower", "house", "person", "apple", "person"]

    parser = argparse.ArgumentParser("Hyperparameter tuning, generate hyperparameters config files")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="config generation, gen for generating config files, run for running the experiments",
    )
    parser.add_argument("--config", type=str, default="configs/sdxl.yaml", help="Path to the basic config file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs/param_tuning_vram_runtime_no_checkpoint",
        help="Path to the output directory",
    )
    parser.add_argument("--max_job", type=int, default=4, help="max running job at a time")
    parser.add_argument("--data_size", "-ds", type=int, nargs="+", default=[100])
    parser.add_argument("--attn_learning_rate", "-alr", type=float, nargs="+", default=LR)
    parser.add_argument("--ffn_learning_rate", "-flr", type=float, nargs="+", default=LR)
    parser.add_argument("--n_learning_rate", "-nlr", type=float, nargs="+", default=[0])
    parser.add_argument("--masking", "-mask", type=str, nargs="+", default=MASKING)
    parser.add_argument("--regex", "-re", type=str, nargs="+", default=RE)
    parser.add_argument("--loss_reg", "-lreg", type=int, nargs="+", default=[1], help="loss regularization")
    parser.add_argument("--loss_recons", "-lrec", type=int, nargs="+", default=[2], help="loss reconstruction")
    parser.add_argument("--beta", "-b", type=float, nargs="+", default=BETA)
    parser.add_argument("--num_intervention", "-ni", type=int, nargs="+", default=NUM_INTERVENTION)
    parser.add_argument("--device", "-d", type=int, default=1)
    parser.add_argument("--attn_name", "-an", type=str, default="attn")
    parser.add_argument("--eps", type=float, nargs="+", default=EPS, help="epsilon for hard-discrete masking")
    parser.add_argument(
        "--train_task",
        "-ts",
        type=str,
        default="general",
        help="task to run, general for general training with gcc3m concepts,"
        + "debug for time profiling, concept for concept training"
        + "deconcept for deconcept training",
    )
    parser.add_argument("--prompts", "-p", default=PROMPTS, nargs="+", help="prompts for concept training")
    parser.add_argument("--project_name", "-pn", type=str, default=PROJECT_NAME)
    parser.add_argument("--ncheckpointing", "-ncp", action="store_false")
    args = parser.parse_args()
    main(args)

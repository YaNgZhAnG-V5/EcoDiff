import torch

from argparse import ArgumentParser
from sdib.evaluation import show_model_memory_consumption_summary, show_model_param_summary
from sdib.utils import load_pipeline


def parse_args():
    parser = ArgumentParser("Get number of parameters and memory consumption")
    parser.add_argument("--model-id", type=str, default="sdxl")
    parser.add_argument("--gpu-id", type=int, default=3, help="GPU ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}")
    torch_dtype = torch.float32

    # initialize pipeline
    pipe = load_pipeline(args.model_id, torch_dtype, disable_progress_bar=True)
    pipe.to("cpu")
    model = pipe.unet
    modules_of_interest = ["attn", "ff", "conv", "norm", "overall"]

    # get model parameters summary
    show_model_param_summary(model, modules_of_interest)

    # get memory consumption summary for interested modules
    show_model_memory_consumption_summary(model, device, modules_of_interest)


if __name__ == "__main__":
    main()

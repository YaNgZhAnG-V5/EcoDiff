import os
import csv
import pandas as pd

import argparse
import torch
from tqdm import tqdm

from sdib.utils import (
    load_pipeline,
    get_precision,
)

@torch.no_grad()
def main(args):
    pipe = load_pipeline(args.model, get_precision(args.precision), True)
    pipe.to(args.device)

    df = pd.read_csv(args.dataset_pt, sep="\t", header=None)
    prompts = df.iloc[:, 0][:args.size]

    # Create output directory if it doesn't exist
    os.makedirs(args.dst, exist_ok=True)

    # Create CSV file with prompts and image filenames
    with open(os.path.join(args.dst, "metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])

        for idx, p in enumerate(tqdm(prompts, desc="Generating")):
            writer.writerow([f"{idx:05d}.png", p])

            g_cpu = torch.Generator(args.device).manual_seed(args.seed)
            image = pipe(
                prompt=p, 
                height=args.resolution,
                width=args.resolution,
                generator=g_cpu, 
                num_inference_steps=args.num_intervention_steps
            ).images[0]
            image.save(os.path.join(args.dst, f"{idx:05d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--model", type=str, default="sdxl")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num_intervention_steps", "-ni", type=int, default=50)
    parser.add_argument("--dst", type=str, default="./datasets/sdxl/finetune_dataset")
    dataset_pt = "./datasets/gcc3m/Validation_GCC-1.1.0-Validation.tsv"

    parser.add_argument("--dataset_pt", type=str, default=dataset_pt, help="path to the GCC dataset tsv file")
    parser.add_argument("--size", type=int, default=100, help="size of the dataset")

    parser.add_argument("--device", type=str, default="cuda:0", help="device for the model")
    parser.add_argument(
        "--precision", type=str, default="bf16", help="precision for the model, bf16, fp32, fp16 available"
    )
    args = parser.parse_args()

    main(args)

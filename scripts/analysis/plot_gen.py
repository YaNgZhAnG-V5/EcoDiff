import matplotlib.pyplot as plt
from matplotlib import rcParams

import argparse

# Set Matplotlib to use a CVPR-like style
rcParams.update(
    {
        "font.family": "serif",  # Use serif font for text (similar to CVPR)
        "axes.titlesize": 12,  # Font size of the axes title
        "axes.labelsize": 10,  # Font size of the x and y labels
        "xtick.labelsize": 9,  # Font size of x tick labels
        "ytick.labelsize": 9,  # Font size of y tick labels
        "legend.fontsize": 9,  # Font size for the legend
        "figure.figsize": (8, 4),  # Figure size to resemble CVPR figure width
        "lines.linewidth": 1.5,  # Line width for plot lines
        "lines.markersize": 5,  # Marker size
        "grid.alpha": 0.3,  # Make grid lines lighter
        "axes.grid": False,  # Disable grid by default
        "legend.frameon": False,  # Remove the frame around the legend
    }
)

DATA = {
    "vram_usage": {
        "sd2": {
            "checkpointing": [4959, 4960, 4968, 4968, 4969, 4969],
            "checkpointing_runtime": [41, 77, 122, 153, 183, 212],
            "no_checkpointing": [18558, 32228, 46371, 59842, 73362, "N/A"],
            "no_checkpointing_runtime": [23, 41, 65, 87, 112, "N/A"],
            "num_intervention": [10, 20, 30, 40, 50, 60],
        },
        "sdxl": {
            "checkpointing": [17425, 17425, 17425, 17325, 17485, 17433],
            "checkpointing_runtime": [438, 475, 495, 525, 553, 585],
            "no_checkpointing": [59946, "N/A", "N/A", "N/A", "N/A", "N/A"],
            "no_checkpointing_runtime": [397, "N/A", "N/A", "N/A", "N/A", "N/A"],
            "num_intervention": [5, 10, 20, 30, 40, 50],
        },
    }
}


def main(args):
    # Extracting data
    num_interventions = DATA["vram_usage"][args.model]["num_intervention"]
    checkpointing_vram = DATA["vram_usage"][args.model]["checkpointing"]
    checkpointing_vram = [val / 1000 if val != "N/A" else None for val in checkpointing_vram]
    no_checkpointing_vram = [
        val / 1000 if val != "N/A" else None for val in DATA["vram_usage"][args.model]["no_checkpointing"]
    ]
    checkpointing_runtime = DATA["vram_usage"][args.model]["checkpointing_runtime"]
    no_checkpointing_runtime = [
        val if val != "N/A" else None for val in DATA["vram_usage"][args.model]["no_checkpointing_runtime"]
    ]

    # Replacing None values with zero for plotting purposes
    no_checkpointing_vram_cleaned = [val if val is not None else 0 for val in no_checkpointing_vram]

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # Adjusted for CVPR-style paper

    # Bar width
    bar_width = 0.35
    index = range(len(num_interventions))

    # Plotting bars for checkpointing and no checkpointing (with 'Out of Memory' for N/A)
    checkpointing_bars = axs[0].bar(
        [i - bar_width / 2 for i in index], checkpointing_vram, bar_width, label="Checkpointing", color="b"
    )
    no_checkpointing_bars = axs[0].bar(
        [i + bar_width / 2 for i in index],
        no_checkpointing_vram_cleaned,
        bar_width,
        label="No Checkpointing",
        color="r",
    )

    # Adding OOM (Out of Memory) annotation for N/A values in 'No Checkpointing'
    for i, v in enumerate(no_checkpointing_vram):
        if v is None:  # Denotes N/A (Out of Memory)
            axs[0].text(
                i + bar_width / 2, 25, "OOM", ha="center", va="bottom", color="red", fontsize=10, fontweight="bold"
            )
            break

    # Adding labels and title
    axs[0].set_xlabel("Number of Interventions")
    axs[0].set_ylabel("VRAM Usage (GB)")
    axs[0].set_title("VRAM Usage vs Number of Interventions")
    axs[0].set_xticks(index)
    axs[0].set_xticklabels(num_interventions)
    axs[0].legend()

    axs[1].text(10, 400, "OOM", ha="center", va="bottom", color="red", fontsize=10, fontweight="bold")
    # Runtime subplot
    axs[1].plot(num_interventions, checkpointing_runtime, marker="o", label="Checkpointing", color="b")
    axs[1].plot(num_interventions, no_checkpointing_runtime, marker="o", label="No Checkpointing", color="r")
    axs[1].set_xlabel("Number of Interventions")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].set_title("Runtime vs Number of Interventions")
    axs[1].legend(loc="upper left")

    plt.tight_layout(pad=0.5)

    plt.savefig(f"model_{args.model}_checkpointing_vs_no_checkpointing.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot VRAM usage and runtime")
    parser.add_argument("--output_dir", "-o", type=str, help="Output directory", default=".")
    parser.add_argument("--model", "-m", type=str, default="sd2", help="Model to plot (sd2 or sdxl)")
    args = parser.parse_args()
    main(args)

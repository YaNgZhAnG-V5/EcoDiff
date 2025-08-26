# Scripts Directory 📜⚡

Training, evaluation, and analysis scripts for diffusion model pruning.

## Purpose 🎯

Provides complete workflows for model pruning, hyperparameter optimization, and performance evaluation.

## Directory Structure 🏠

```
scripts/
├── train.py                    # Core training script
├── load_pruned_model.py        # Physical pruning and analysis
├── get_retrained_model.py      # Combine pruned + retrained models
├── inference_pruned_model.py   # Simple inference with pruned models
│
├── evaluation/                 # Model evaluation scripts
├── analysis/                   # Analysis and visualization scripts
├── utils/                      # Utility and helper scripts
└── retraining/                 # Fine-tuning scripts
```

## Core Scripts 📜

### Main Training and Model Management
- **`train.py`** - Main pruning training script that learns differentiable masks for attention and feed-forward layers across multiple diffusion architectures. Implements gradient-aware mask optimization with configurable regularization and supports multiple mask types with memory-efficient training.

- **`load_pruned_model.py`** - Converts learned masks into actual parameter removal by physically pruning model weights. Performs comprehensive analysis including memory usage comparison and handles architecture-specific requirements while maintaining pipeline compatibility.

- **`get_retrained_model.py`** - Combines pruned models with retrained weights, supporting both LoRA adapter integration and full fine-tuning weight loading. Provides inference capabilities for evaluating the combined pruned+retrained models.

- **`inference_pruned_model.py`** - Simple inference script for pickled pruned models

## Subdirectories 📁

### [`evaluation/`](./evaluation/) - Model Evaluation
- **`semantic_eval.py`** - Semantic quality evaluation using CLIP classification and FID metrics with multiple evaluation modes
- **`binary_mask_eval.py`** - Converts continuous masks to binary and analyzes sparsity patterns with visual comparisons
- **`image_quality_eval.py`** - Multi-metric image quality assessment using FID, SSIM, and CLIP scores
- **`eval.py`** - Unified evaluation pipeline combining multiple assessment types

### [`analysis/`](./analysis/) - Analysis and Visualization
- **`generate_attn_map.py`** - Visualizes attention patterns and learned pruning masks with heatmap generation
- **`get_num_of_parameters.py`** - Parameter count analysis by module type (attention, FFN, conv, norm)
- **`plot_gen.py`** - Plotting utilities for experimental results visualization

### [`utils/`](./utils/) - Utilities and Helpers
- **`hyperparameter_tuning.py`** - Grid search system for optimal pruning hyperparameters with two-stage operation: generation creates YAML configs for parameter combinations, execution runs experiments in parallel
- **`create_retraining_dataset.py`** - Training dataset generation for retraining pruned models using caption-image pairs

## Usage 💡

```bash
# Core training workflow
python scripts/train.py --cfg configs/sdxl.yaml

# Physical pruning and analysis
python scripts/load_pruned_model.py --save_pt <checkpoint> --scope "global" --ratio 0.9 --save_pruned_model

# Simple inference with pruned model
python scripts/inference_pruned_model.py --pruned_model_pt <pruned_model>

# Combine pruned model with retrained weights
python scripts/get_retrained_model.py --pruned_model_pt <pruned_model> --lora_pt <lora_weights>

# Evaluation
python scripts/evaluation/semantic_eval.py -sp <checkpoint> --task all
python scripts/evaluation/binary_mask_eval.py --ckpt <checkpoint> -lt 0.001
```

### [`retraining/`](./retraining/) - Retraining Scripts 🔄
- Fine-tuning scripts for different models.
- For detailed usage and examples, see the [README.md](./retraining/README.md) in the `retraining` directory.

> [!NOTE]
> For installation and detailed usage, see the [main README](../README.md).

# EcoDiff: Low-Cost Pruning of Diffusion Models ✂️🌀

<!-- Badges / decorative (adjust or remove if not desired) -->
<p align="center">
  <strong>🔥 Efficient • 🧠 Model-Agnostic • ⚡ Low-Cost • 🌱 Eco-Friendly</strong>
</p>

Official implementation of **"EcoDiff: Low-Cost Pruning of Diffusion Models"** - a novel approach for memory efficient diffusion model pruning.

<p align="center">
  <strong>📄 Paper: <a href="https://arxiv.org/abs/2412.02852">arXiv:2412.02852</a> • 🌐 Project Page: <a href="https://yangzhang-v5.github.io/EcoDiff">yangzhang-v5.github.io/EcoDiff</a></strong>
</p>

![teaser](images/teaser.png)

## Table of Contents 📚
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#supported-models">Supported Models</a></li>
    <li><a href="#advanced-usage">Advanced Usage</a>
      <ul>
        <li><a href="#pruning-training">Pruning Training</a></li>
        <li><a href="#hyperparameter-tuning">Hyperparameter Tuning</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
        <li><a href="#fine-tuning-after-pruning">Fine-tuning After Pruning</a></li>
      </ul>
    </li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#development">Development</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Overview 🚀
![method](images/method.png)

EcoDiff introduces a model-agnostic structural pruning framework that learns differentiable masks to sparsify diffusion models. Key innovations include:

- ✨ **Model-agnostic pruning** for various diffusion architectures
- 🧪 **Differentiable mask learning** allowing end-to-end optimization
- 🧵 **Time step gradient checkpointing** for memory-efficient training
- 📉 **Up to 20% parameter reduction** with minimal performance loss


## Installation ⚙️

### Requirements ✅
- 🐍 Python 3.10+
- 📦 Anaconda or Miniconda
- 🖥️ CUDA-compatible GPU

### Setup 🔧
```bash
# Create conda environment 🧬
conda create -n sdib python=3.10 -y
conda activate sdib

# Clone repository ⬇️
git clone https://github.com/your-repo/ecodiff.git
cd ecodiff

# Install dependencies 📦
pip install -e .[core,loggers,test]
```

### Environment Configuration 🗂️
Create a `.env` file:
```bash
PYTHON=/path/to/miniconda3/envs/sdib/bin/python
RESULTS_DIR=/path/to/ecodiff/results
CONFIG_DIR=/path/to/ecodiff/configs
```

> [!IMPORTANT]
> Ensure RESULTS_DIR has enough disk space for checkpoints and logs.

## Quick Start ⚡

### 1. Basic Pruning ✂️
```bash
# SDXL pruning
make visual cfg=sdxl

# FLUX pruning
make visual cfg=flux
```

### 2. Hyperparameter Tuning 🎯
```bash
# Generate configurations
python scripts/utils/hyperparameter_tuning.py --config configs/sdxl.yaml --task gen

# Run tuning
python scripts/utils/hyperparameter_tuning.py --task run --max_job 2
```

### 3. Evaluation 🧪
```bash
# Semantic evaluation
python scripts/evaluation/semantic_eval.py -sp <checkpoint_path> --task all

# Mask analysis
python scripts/evaluation/binary_mask_eval.py --ckpt <checkpoint_path> -lt 0.001
```

> [!CAUTION]
> Replace `<checkpoint_path>` with your actual checkpoint path before running evaluation commands.

## Advanced Usage 🧠

### Pruning Training 🏋️
```bash
# Direct training script
python scripts/train.py

# Development/debugging mode
make visual cfg=sdxl
make visual cfg=flux
```

### Hyperparameter Tuning 🔍
```bash
# Generate configuration files
python scripts/utils/hyperparameter_tuning.py \
  --config configs/sdxl.yaml \
  --output_dir configs/param_sdxl_tuning \
  -lr 0.1 0.2 \
  -mask "hard_discrete" \
  -re ".*" \
  -lreg 1 0 \
  -lrec 1 2 \
  -b 0.1 0.01 \
  -d 2 \
  -pn sdxl_pruning \
  --task gen

# Run tuning jobs
python scripts/utils/hyperparameter_tuning.py \
  --output_dir configs/param_sdxl_tuning \
  --task run \
  --max_job 2
```

### Evaluation 📊
```bash
# Generate semantic evaluation
python scripts/evaluation/semantic_eval.py -sp <checkpoint_path> --task gen

# Run all semantic evaluations
python scripts/evaluation/semantic_eval.py -sp <checkpoint_path> --task all

# Binary mask evaluation with threshold
python scripts/evaluation/binary_mask_eval.py --ckpt <checkpoint_path> -lt 0.001
```

### Fine-tuning After Pruning 🩹
```bash
# SDXL LoRA fine-tuning
bash scripts/retraining/train_text_to_image_lora_sdxl.sh 30 0

# FLUX LoRA fine-tuning
bash scripts/retraining/train_text_to_image_lora_flux.sh 30 0
```

### Load Pruned Models 📦
```bash
python scripts/load_pruned_model.py
```

## Configuration Files 🗃️

The framework uses YAML configuration files located in the `configs/` directory:

```
configs/
├── dit.yaml          # Diffusion Transformers configuration
├── flux.yaml         # FLUX.1 Schnell model configuration
├── flux_dev.yaml     # FLUX.1 Dev model configuration  
├── sd2.yaml          # Stable Diffusion v2 configuration
├── sd3.yaml          # Stable Diffusion 3 configuration
└── sdxl.yaml         # Stable Diffusion XL configuration
```

## Development 🛠️

For developers contributing to the project:

```bash
# Install development dependencies
pip install pre-commit && pre-commit install

# Run tests ✅
make test

# Format code 🧼
make format

# Clean generated files 🧹
make clean
```

## Repository Structure 🧭

- [`src/sdib/`](src/sdib/) - Core pruning framework
- [`scripts/`](scripts/) - Training and evaluation scripts  
- [`configs/`](configs/) - Model configuration files

## Supported Models 🤝

- **SDXL**: Stable Diffusion XL
- **FLUX**: FLUX diffusion models
- **SD3**: Stable Diffusion 3
- **DiT**: Diffusion Transformers
- **SD2**: Stable Diffusion v2

## Citation 📑
```bibtex
@article{zhang2024ecodiff,
  title={EcoDiff: Low-Cost Pruning of Diffusion Models},
  author={Zhang, Yang and Jin, Er and Dong, Yanfei and Khakzar, Ashkan and Torr, Philip and Stegmaier, Johannes and Kawaguchi, Kenji},
  journal={arXiv preprint arXiv:2412.02852},
  year={2024}
}
```

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏
- Built on [Diffusers](https://github.com/huggingface/diffusers) library
- Supports models from [Stability AI](https://stability.ai/) and [Black Forest Labs](https://blackforestlabs.ai/)
- ❤️ Community feedback welcome—open issues & PRs appreciated

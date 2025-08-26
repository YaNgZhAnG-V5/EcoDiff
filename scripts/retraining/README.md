# Retraining Scripts 💪🔄

Fine-tuning scripts for recovering performance after pruning and domain adaptation.

## Purpose 🎯

Provides LoRA and full fine-tuning scripts for different diffusion architectures to restore performance after aggressive pruning.

> [!NOTE]
> These scripts are modified from the official Diffusers examples:
> - [`train_dreambooth_lora_flux.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py)
> - [`train_dreambooth_flux.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_flux.py)
> - [`train_text_to_image_lora_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py)
> - [`train_text_to_image_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py)

## LoRA Fine-tuning Scripts 🌯

- **`train_text_to_image_lora_sdxl.py`** - SDXL LoRA fine-tuning
- **`train_text_to_image_lora_flux.py`** - FLUX LoRA fine-tuning

## Full Fine-tuning Scripts 🔄

- **`train_text_to_image_sdxl.py`** - Complete SDXL fine-tuning
- **`train_text_to_image_flux.py`** - FLUX fine-tuning (single-node)
- **`train_text_to_image_flux_fsdp.py`** - FLUX fine-tuning (multi-node FSDP)

> [!WARNING]
> **FLUX Full Fine-tune Limitations**: 
> - The full fine-tune scripts *do not work* with **single-node setups** that have **multiple GPUs** due to **NaN issues** during backpropagation when using Fully Sharded Data Parallel (FSDP).  
> - However, FSDP works **correctly** on **multi-node setups** with **single GPU** per node.  
> - Additionally, [issue #10925](https://github.com/huggingface/diffusers/issues/10925) highlights **incompatibilities** with *DeepSpeed ZeRO 3* in the above scripts.

## Usage 💡

Use the provided bash scripts for easy execution with proper parameters:

### SDXL Scripts

```bash
# SDXL LoRA fine-tuning - single validation prompt
bash scripts/retraining/train_text_to_image_lora_sdxl.sh <PRUNING_RATIO> <CUDA_DEVICE>
# Example: bash scripts/retraining/train_text_to_image_lora_sdxl.sh 30 0

# SDXL full fine-tuning - single validation prompt  
bash scripts/retraining/train_text_to_image_sdxl.sh <PRUNING_RATIO> <CUDA_DEVICE>
# Example: bash scripts/retraining/train_text_to_image_sdxl.sh 30 0
```

### FLUX Scripts

```bash
# FLUX LoRA fine-tuning - supports multiple validation prompts
bash scripts/retraining/train_text_to_image_lora_flux.sh <PRUNING_RATIO> <CUDA_DEVICE>
# Example: bash scripts/retraining/train_text_to_image_lora_flux.sh 30 0
```

> [!TIP]
> FLUX scripts support validation on multiple prompts simultaneously, while SDXL scripts validate on a single prompt.

> [!NOTE]
> For installation and setup, see the [main README](../../README.md).

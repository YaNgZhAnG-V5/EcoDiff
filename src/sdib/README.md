# SDIB: Core Pruning Framework

This directory contains the core implementation of the EcoDiff pruning framework for diffusion models.

## Purpose

Provides the main components for learning differentiable masks to prune diffusion model parameters with minimal performance loss.

## Structure 

- **[`data/`](./data/)** - Dataset classes for training and evaluation
- **[`models/`](./models/)** - Pipeline implementations (SDXL, FLUX, SD3, DiT)  
- **[`hooks/`](./hooks/)** - Pruning hooks that apply learnable masks
- **[`utils/`](./utils/)** - Utility functions and pipeline management
- **[`scheduler/`](./scheduler/)** - Custom schedulers for pruned models
- **[`evaluation/`](./evaluation/)** - Performance evaluation tools

## 🔍 Framework Overview 

SDIB implements learnable structural pruning for diffusion models through three key mechanisms:

### 1. Differentiable Masking System
- **Learnable masks** applied to attention heads, FFN layers, and normalization
- **Multiple mask types**: hard_discrete, sigmoid, gumbel_softmax
- **Gradient-aware optimization** with straight-through estimators

### 2. Memory-Efficient Training
- **Time step gradient checkpointing** to reduce memory consumption
- **Selective layer targeting** through regex-based module filtering  
- **Multi-architecture support** with unified interface

### 3. Progressive Pruning Pipeline
```python
# High-level workflow
import torch
from sdib.utils import create_pipeline
from sdib.hooks import CrossAttentionExtractionHook, FeedForwardHooker

# 1. Load model with pruning capabilities
pipeline = create_pipeline("sdxl", device="cuda", torch_dtype=torch.float16)

# 2. Apply learnable masks (see hooks/ for detailed usage)
# 3. Train masks with reconstruction + regularization loss
# 4. Convert learned masks to physical pruning
# 5. Optional: Fine-tune pruned model for performance recovery
```

### Supported Architectures
- **SDXL**: Stable Diffusion XL
- **FLUX**: FLUX.1 Schnell/Dev 
- **SD3**: Stable Diffusion 3
- **DiT**: Diffusion Transformers
- **SD2**: Stable Diffusion v2

> [!NOTE]
> For detailed installation and usage instructions, see the [main README](../../README.md).

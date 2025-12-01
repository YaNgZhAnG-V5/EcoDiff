# 🔧 Utils Module

Comprehensive utility functions for diffusion model pruning, pipeline management, and training workflows.

## Purpose

Provides the core infrastructure for EcoDiff's pruning framework, including model analysis, pipeline creation, training utilities, and evaluation tools.

## Key Components

### `utils.py` - Core Pruning and Model Operations

**Model Analysis Functions:**
- `get_total_params(model, trainable=True)` - Count model parameters
- `calculate_mask_sparsity(hooker, threshold=None)` - Analyze pruning mask sparsity
- `get_precision(precision)` - Convert precision strings to PyTorch dtypes

**Pruning Implementation:**
- `linear_layer_masking(module, lamb)` - Soft masking for attention layers (K, Q, V)
- `linear_layer_pruning(module, lamb, model_type)` - Hard pruning with weight removal
- `ffn_linear_layer_pruning(module, lamb)` - FFN layer pruning (GELU/GEGLU support)
- `norm_layer_pruning(module, lamb)` - Layer normalization pruning for FLUX
- `update_flux_single_transformer_projection()` - FLUX-specific projection updates

**Regularization and Loss Functions:**
- `hard_concrete_distribution()` - Learnable mask distribution
- `l0_complexity_loss()` - L0 regularization for sparsity
- `calculate_reg_loss()` - Unified regularization (L0, L1, L2)

**Custom Layers:**
- `SkipConnection` - Skip connections for pruned layers
- `AttentionSkipConnection` - Model-specific attention skip connections
- `SparsityLinear` - Sparse linear layer implementation

### `pipe.py` - Pipeline Management and Loading

**Configuration Parsing:**
- `get_cfg(save_pt)` - Extract config from checkpoint paths
- Automatic model type detection and parameter parsing

**Pipeline Creation:**
- `load_pipeline(model_str, torch_dtype, disable_progress_bar)` - Load diffusion pipelines
  - Supports: SD1, SD2, SDXL, SD3, DiT, FLUX models
  - Automatic model ID resolution and loading
- `create_pipeline()` - Comprehensive pipeline creation with pruning hooks
  - Integrated mask loading and application
  - Configurable pruning scope (local/global) and ratios

**Checkpoint Management:**
- `get_save_pts(save_pt)` - Generate attention/FFN/norm checkpoint paths
- Automatic hook state restoration

### `utils_train.py` - Training and Configuration Support

**Image Generation and Saving:**
- `save_image()` - Core image generation with seed control
- `save_image_seed()` - Wrapper with automatic generator seeding  
- `save_image_binarize_seed()` - Temporary mask binarization for inference
- `get_file_name()` - Timestamped filename generation

**Configuration Management:**
- `load_config(cfg_path)` - OmegaConf YAML configuration loading
- `overwrite_debug_cfg(cfg)` - Debug configuration overrides
- Support for hierarchical configurations and development modes

**Training Hooks (Extensible):**
- `save_model_hook()` - Model state saving interface
- `load_model_hook()` - Model state loading interface

### `clip.py` - CLIP Integration for Evaluation

**CLIP Model Loading:**
- `get_clip_encoders(backbone, pretrained, only_model)` - Load pretrained CLIP models
  - Default: ViT-B-16 with LAION-400M weights
  - Returns complete toolkit: model, preprocessor, tokenizer, logit scale
  - Flexible return format (model-only vs full configuration)

**Supported Architectures:**
- Multiple ViT and ResNet backbones via OpenCLIP
- Various pretrained weight options (LAION, OpenAI, etc.)
- Automatic error handling for invalid configurations

## 💡 Usage Examples 

### Basic Pipeline Creation
```python
import torch
from sdib.utils import create_pipeline, calculate_mask_sparsity

# Create SDXL pipeline with pruning hooks
pipeline = create_pipeline("sdxl", device="cuda", torch_dtype=torch.float16)

# Load pruned model from checkpoint
pruned_pipeline = create_pipeline(
    "sdxl", device="cuda", save_pt="./results/checkpoint.pt",
    lambda_threshold=0.5, binary=True
)
```

### Pruning Operations
```python
from sdib.utils import linear_layer_pruning, calculate_mask_sparsity

# Apply hard pruning to attention layers
linear_layer_pruning(attention_module, mask_values, model_type="sdxl")

# Calculate sparsity after pruning
total, active, ratio = calculate_mask_sparsity(hook, threshold=0.5)
print(f"Sparsity: {ratio:.2%} ({active}/{total} parameters active)")
```

### Training Utilities
```python
from sdib.utils import save_image_seed, load_config

# Load configuration
config = load_config("configs/sdxl.yaml")

# Generate and save validation images
save_image_seed(
    pipe=pipeline,
    prompts=["astronaut riding a horse"],
    steps=50,
    device="cuda",
    seed=42,
    save_dir="./validation",
    width=1024,
    height=1024
)
```

### CLIP Evaluation
```python
from sdib.utils import get_clip_encoders

# Load CLIP model for evaluation
clip_config = get_clip_encoders(
    backbone="ViT-B-16",
    pretrained="laion400m_e32",
    only_model=False
)
model, preprocess, tokenizer = clip_config['model'], clip_config['preprocess'], clip_config['tokenizer']
```

> [!NOTE]
> For installation and setup, see the [main README](../../../README.md).

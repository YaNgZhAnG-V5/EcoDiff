# Models Module

Pipeline implementations for different diffusion architectures with pruning support.

## Purpose

Provides specialized pipeline classes that extend standard Diffusers pipelines with gradient-aware inference for pruning mask training.

## Pipeline Classes

- **`SDIBDiffusionPipeline`** - Stable Diffusion v1/v2
- **`SDXLDiffusionPipeline`** - Stable Diffusion XL  
- **`SDIBDiffusion3Pipeline`** - Stable Diffusion 3
- **`DiTIBPipeline`** - Diffusion Transformers
- **`FluxIBPipeline`** - FLUX models

## 💡 Usage 

```python
from sdib.models import SDXLDiffusionPipeline

# Load pipeline with pruning support
pipeline = SDXLDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# Gradient-enabled inference for training pruning masks
result = pipeline.inference_with_grad(
    prompt="A beautiful landscape",
    num_inference_steps=50,
    guidance_scale=7.5
)
```

> [!NOTE]
> For installation and setup, see the [main README](../../../README.md).

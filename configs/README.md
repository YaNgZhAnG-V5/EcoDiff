# Configuration Files ⚙️📄

YAML configuration files for different diffusion model architectures and training scenarios.

## Purpose 🎯

Provides model-specific configurations for the EcoDiff pruning framework with support for various diffusion architectures.

## Configuration Files 📚

- **`sdxl.yaml`** - Stable Diffusion XL configuration
- **`flux.yaml`** - FLUX.1 Schnell model configuration  
- **`flux_dev.yaml`** - FLUX.1 Dev model configuration
- **`sd3.yaml`** - Stable Diffusion 3 configuration
- **`dit.yaml`** - Diffusion Transformers configuration
- **`sd2.yaml`** - Stable Diffusion v2 configuration
- **`eval.yaml`** - Evaluation settings
- **`validation_prompts.yaml`** - Standard validation prompts
- **`validation_prompts_small.yaml`** - Reduced validation prompt set

## Usage 💡

```bash
# Use specific configuration
python scripts/train.py --cfg configs/sdxl.yaml

# Override parameters
python scripts/train.py --cfg configs/flux.yaml --trainer.epochs 10 --data.size 500
```

## Key Sections 🗺️

Each config contains:
- **`data`** - Dataset parameters  
- **`trainer`** - Training and pruning settings
- **`loss`** - Loss function configuration
- **`logger`** - Logging and output settings

> [!NOTE]
> For setup and usage instructions, see the [main README](../README.md).

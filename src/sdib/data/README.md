# 📊 Data Module 

Dataset classes for training and evaluating diffusion model pruning.

## Purpose

Provides specialized dataset classes for different diffusion models and evaluation tasks.

## Dataset Classes

- **`PromptImageDataset`** - Text-to-image generation with metadata support
- **`ConceptDataset`** & **`DeConceptDataset`** - Concept-specific generation/removal
- **`DiTDataset`** - Diffusion Transformers with class conditioning  
- **`ImageNetDataset`** - ImageNet integration for evaluation
- **`EvalDataset`** - Standardized evaluation dataset

## 💡 Usage 

```python
from sdib.data import PromptImageDataset, DiTDataset

# Text-to-image dataset
dataset = PromptImageDataset(
    metadata="data/captions.tsv",
    pipe=pipeline,
    num_inference_steps=50,
    save_dir="./outputs",
    seed=42,
    device="cuda",
    size=1000
)

# DiT class-conditional dataset  
dit_dataset = DiTDataset(
    pipe=dit_pipeline,
    save_dir="./dit_outputs", 
    device="cuda",
    size=500,
    num_inference_steps=50,
    seed=44
)
```

> [!NOTE]
> For complete setup instructions, see the [main README](../../../README.md).

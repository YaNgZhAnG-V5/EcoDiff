# Evaluation Module 📊🏆

Performance evaluation tools for measuring the impact of pruning on diffusion models.

## Purpose 🎯

Provides tools for evaluating model performance, measuring computational efficiency, and analyzing pruning effectiveness.

## Components 🧩

- **`size_runtime.py`** - Model performance profiling (size, speed, memory usage)

## Usage 💡

```python
from sdib.evaluation import (
    get_model_memory_consumption_summary,
    get_model_param_summary,
    show_model_param_summary
)

# Get memory consumption summary
memory_summary = get_model_memory_consumption_summary(
    model=pruned_pipeline.unet,
    device="cuda",
    verbose=True
)

# Get parameter summary
param_summary = get_model_param_summary(
    model=pruned_pipeline.unet,
    verbose=True
)

# Show detailed summaries
show_model_param_summary(
    model=pruned_pipeline.unet,
    modules_of_interest=["attn", "ff", "conv"],
    verbose=True
)
```

> [!NOTE]
> For installation and setup, see the [main README](../../../README.md).

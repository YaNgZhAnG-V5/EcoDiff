# Scheduler Module 📅⚙️

Custom schedulers optimized for pruned diffusion models.

## Purpose 🎯

Provides enhanced sampling methods and noise scheduling strategies adapted for sparse model architectures.

## Components 🧩

- **`dpm.py`** - Custom DPM solver with pruning optimizations
  - `ReverseDPMSolverMultistepScheduler` - Enhanced DPM++ solver

## Usage 💡

```python
from sdib.scheduler import ReverseDPMSolverMultistepScheduler

# Create custom scheduler
scheduler = ReverseDPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    solver_order=2,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True
)

# Use with pipeline
pipeline.scheduler = scheduler
result = pipeline(
    prompt="A beautiful landscape",
    num_inference_steps=20
)
```

> [!NOTE]
> For installation and setup, see the [main README](../../../README.md).

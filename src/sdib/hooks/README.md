# Hooks Module 🎣✂️

Core pruning mechanism through learnable hooks that apply sparse masks to diffusion model layers.

## Purpose 🎯

Implements the main pruning functionality by applying learnable masks to attention and feed-forward layers during inference.

## Hook Classes 🔧

- **`CrossAttentionExtractionHook`** - Learnable masks for attention layers
- **`FeedForwardHooker`** - Structured pruning of MLP layers  
- **`NormHooker`** - Optional normalization layer pruning
- **Custom Attention Processors** - Modified attention with masking support

## Mask Types 🎭

- **`hard_discrete`** - Binary masks with straight-through gradients
- **`sigmoid`** - Continuous masks (0-1 range)
- **`gumbel_softmax`** - Differentiable discrete approximation

## Usage 💡

```python
import torch
from sdib.hooks import CrossAttentionExtractionHook, FeedForwardHooker

# Set up pruning hooks
cross_attn_hook = CrossAttentionExtractionHook(
    pipeline,
    regex=".*attn.*",
    dtype=torch.float32,
    head_num_filter=8,
    masking="hard_discrete",
    dst="./outputs",
    epsilon=0.0,
    model_name="sdxl",
    attn_name="attn"
)

ff_hook = FeedForwardHooker(
    pipeline,
    regex=".*ff.*", 
    dtype=torch.float32,
    masking="hard_discrete",
    dst="./outputs",
    epsilon=0.0
)

# Add hooks
cross_attn_hook.add_hooks(init_value=1.0)
ff_hook.add_hooks(init_value=1.0)

# Dummy generation to initialize the lambda
g_cpu = torch.Generator('cuda').manual_seed(42)
_ = pipe("...", generator=g_cpu, num_inference_steps=1)

# ... setup optimizer

# Training loop with hooks active
for batch in dataloader:
    output = pipeline.inference_with_grad(**batch)
    # ... loss computation and backprop
```

> [!NOTE]
> For complete setup, see the [main README](../../../README.md).

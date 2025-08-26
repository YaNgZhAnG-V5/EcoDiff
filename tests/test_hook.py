import pytest
import torch

from sdib.hooks import CrossAttentionExtractionHook, FeedForwardHooker
from sdib.utils import load_pipeline


def load_and_init_attn_hook(model, init_lambda):
    pipe = load_pipeline(model, torch_dtype=torch.float32, disable_progress_bar=True)
    hook = CrossAttentionExtractionHook(
        pipe,
        regex=".*",
        dtype=torch.float32,
        head_num_filter=1,
        masking="hard_discrete",
        dst="results",
        epsilon=0.0,
        model_name=model,
        attn_name="attn",
    )
    hook.add_hooks(init_value=init_lambda)

    # dumy generation to initialize the lambda
    _ = pipe("+++", num_inference_steps=1)
    return pipe, hook


@pytest.mark.parametrize("model", ["sd2"])
@pytest.mark.parametrize("init_lambda", [0.0, 1.0, 5.0, 10.0])
def test_cross_attn_hook_load(model, init_lambda):
    _, hook = load_and_init_attn_hook(model, init_lambda)

    # check init lambda
    for lamb in hook.lambs:
        assert torch.allclose(lamb - init_lambda, torch.zeros_like(lamb))

    # clear hooks
    hook.clear_hooks()


@pytest.mark.parametrize("scope", ["local", "global"])
@pytest.mark.parametrize("ratio", [0.5, 0.8])
def test_cross_attn_hook_binarize(scope: str, ratio: float):
    _, hook = load_and_init_attn_hook("sd2", 0.9)  # deliberately not use 1 for init
    hook.lambs[0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    hook.lambs[4] = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    hook.binarize(scope, ratio)
    assert hook.binary
    if scope == "local":
        for lamb in hook.lambs:
            mask_val = int(ratio * lamb.numel())
            assert torch.sum(lamb == 1.0).item() == mask_val
        if ratio == 0.5:
            assert torch.allclose(hook.lambs[0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
            assert torch.allclose(hook.lambs[4], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    else:
        mask_val = 0
        total_num = 0
        for lamb in hook.lambs:
            mask_val += int(ratio * lamb.numel())
            total_num += lamb.numel()
        assert abs(mask_val - int(total_num * ratio)) <= 10  # here allow a margin of error
        assert torch.allclose(hook.lambs[0], torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
        assert torch.allclose(hook.lambs[4], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.8])
def test_cross_attn_hook_bizarize_threshold(threshold: float):
    _, hook = load_and_init_attn_hook("sd2", 0.9)  # deliberately not use 1 for init
    hook.lambs[0] = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8])
    hook.lambs[4] = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    hook.bizarize_threshold(threshold)
    assert hook.binary
    target_tensor = (hook.lambs[0] >= threshold).float()
    assert torch.allclose(hook.lambs[0], target_tensor)


def test_ff_hook_add_hooks():
    pipe = load_pipeline("sd2", torch_dtype=torch.float32, disable_progress_bar=True)
    hook = FeedForwardHooker(
        pipe,
        regex=".*",
        dtype=torch.float32,
        masking="hard_discrete",
        epsilon=0.0,
    )
    hook.add_hooks()
    assert len(hook.hook_dict) == len(hook.lambs)

    # dummy forward to initialize the lambda
    _ = pipe("+++", num_inference_steps=1)
    hook.clear_hooks()
    assert len(hook.hook_dict) == 0

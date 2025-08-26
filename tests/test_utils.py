import pytest
import torch
from diffusers.models.attention_processor import AttnProcessor2_0

from sdib.hooks.attention_processor import AttnProcessor2_0_Masking
from sdib.utils import create_pipeline
from sdib.utils.utils import hard_concrete_distribution, linear_layer_masking, linear_layer_pruning


@pytest.mark.parametrize(
    "lamb, expected_mean",
    [
        (1.0, 0.5),
        (2.0, 0.6),
        (3.0, 0.7),
        (4.0, 0.7),
        (5.0, 0.8),
        (6.0, 0.8),
        (7.0, 0.8),
        (8.0, 0.8),
        (9.0, 0.9),
        (10.0, 0.9),
    ],
)
def test_hard_concrete_distribution(lamb, expected_mean):
    torch_lamb = torch.tensor([lamb])
    mean = 0.0
    for _ in range(500):
        mean += hard_concrete_distribution(torch_lamb, use_log=True)
    mean /= 500
    print(mean)
    assert mean.item() == pytest.approx(expected_mean, abs=0.1)


def get_dummy_input_output(module, lamb, attention_processor_masking, device, torch_dtype):
    dummy_dim = module.to_q.in_features
    dummy_input = torch.randn(1, 4096, dummy_dim, device=device, dtype=torch_dtype)
    dummy_output, _, _ = attention_processor_masking(module, dummy_input, lamb=lamb, masking="binary")
    return dummy_input, dummy_output


def get_test_pipeline(model, device, torch_dtype):
    save_pt = "./results/prune_results/epoch_2_step_20.pt"
    pipe, cross_attn_hooker = create_pipeline(
        model,
        device,
        torch_dtype,
        save_pt=save_pt,
        lambda_threshold=0,
        binary=True,
        epsilon=0.0,
        masking="binary",
        attn_name="attn",
        return_hooker=True,
    )
    return pipe, cross_attn_hooker


@pytest.mark.parametrize("model, device", [("sdxl", "cuda:1")])
def test_linear_layer_masking(model, device):
    # initialize pipeline and hooker
    torch_dtype = torch.float16
    pipe, cross_attn_hooker = get_test_pipeline(model, device, torch_dtype)

    # perform a dummy forward pass to get module names for each lambda
    g_cpu = torch.Generator(device=device).manual_seed(0)
    _ = pipe("...", generator=g_cpu, num_inference_steps=1)
    attention_processor = AttnProcessor2_0()
    attention_processor_masking = AttnProcessor2_0_Masking()

    # replace attention blocks with parameter reduced attention blocks
    for name in cross_attn_hooker.hook_dict.keys():
        if "attn2" in name:
            continue  # skip the cross attention
        module = pipe.unet.get_submodule(name)
        lamb = cross_attn_hooker.lambs[cross_attn_hooker.lambs_module_names.index(name)]
        assert module.heads == lamb.shape[0]

        # get reference output for sanity check
        dummy_input, original_dummy_output = get_dummy_input_output(
            module, lamb, attention_processor_masking, device, torch_dtype
        )

        # perform masking on the linear layers
        module = linear_layer_masking(module, lamb)

        # check if the output is the same as the original module
        dummy_output = attention_processor(module, dummy_input)
        assert torch.allclose(dummy_output, original_dummy_output, atol=1e-5)
        print(f"### Masking done for {name} ###")
    print("### Masking done ###")


@pytest.mark.parametrize("model, device", [("sdxl", "cuda:1")])
def test_linear_layer_pruning(model, device):
    # initialize pipeline and hooker
    torch_dtype = torch.float16
    pipe, cross_attn_hooker = get_test_pipeline(model, device, torch_dtype)

    # perform a dummy forward pass to get module names for each lambda
    g_cpu = torch.Generator(device=device).manual_seed(0)
    _ = pipe("...", generator=g_cpu, num_inference_steps=1)
    attention_processor = AttnProcessor2_0()
    attention_processor_masking = AttnProcessor2_0_Masking()

    # remove parameters in attention blocks
    for name in cross_attn_hooker.hook_dict.keys():
        if "attn2" in name:
            continue  # skip the cross attention
        module = pipe.unet.get_submodule(name)
        lamb = cross_attn_hooker.lambs[cross_attn_hooker.lambs_module_names.index(name)]
        assert module.heads == lamb.shape[0]

        # get reference output for sanity check
        dummy_input, original_dummy_output = get_dummy_input_output(
            module, lamb, attention_processor_masking, device, torch_dtype
        )

        # perform pruning on the linear layers
        module = linear_layer_pruning(module, lamb, model)

        # check if the output is the same as the original module
        dummy_output = attention_processor(module, dummy_input)
        assert torch.allclose(dummy_output, original_dummy_output, atol=5e-3)
        print(f"### Pruning done for {name} ###")
    print("### Pruning done ###")

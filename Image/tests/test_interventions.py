# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import tempfile

import numpy as np
import pytest
import torch
from hydra import compose, initialize

from act.hooks import get_hook
from act.hooks.kernel_steering import (
    KernelSteeringHook,
    compute_kernelized_steering_gradient,
    compute_median_heuristic_sigma,
)
from act.hooks.transport import GaussianOTHook
from act.scripts.learn_intervention import (
    InterventionsManager,
    learn_intervention,
)


@pytest.mark.parametrize(
    "intervention_name",
    [
        ("none"),
        ("aura"),
        ("mean_ot"),
        ("gaussian_ot"),
        ("linear_ot"),
        ("kernel"),
    ],
)
def test_hook_with_hook_args(intervention_name):
    # Assuming that the main function doesn't have any side effects and returns None when successful
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../act/configs"):
            # config is relative to a module
            cfg = compose(
                config_name="text_generation",
                overrides=[
                    "device=cpu",
                    "model.model_path=sshleifer/tiny-gpt2",
                    "model.module_names=['transformer.h.0.mlp.c_proj:0', 'transformer.h.1.mlp.c_proj:0']",
                    f"responses.tag=toxicity-responses",
                    f"intervention_params.name={intervention_name}",
                    f"intervention_params.incremental=atonce",
                    "intervention_params.hook_params.quantiles_src=q_all",
                    "data_dir=tests/data",
                    "cache_dir=tests/data",
                    f"interventions.cache_dir={tempfolder}",
                    "compute_responses=false",
                    "wandb.mode=disabled",
                ],
            )

        # interventions_manager = InterventionsManager(
        #     ResponsesManager.get_output_path(cfg.responses), cfg.interventions
        # )
        # # This call also tests the "fit()" method
        # interventions_manager.learn_intervention_all()
        learn_intervention(cfg)

        # Now create a new hook and load its state using the one learnt through "fit()"
        hook = get_hook(
            intervention_name,
            module_name="transformer.h.1.mlp.c_proj:0",
            device="cpu",
            std_eps=1e-7,  # <-- This one is not in statedict, will be updated.
            strength=0.9,
        )
        state_path = (
            InterventionsManager.get_output_path(cfg.interventions)
            / "transformer.h.1.mlp.c_proj:0.statedict"
        )
        hook.from_state_path(state_path)

        # Also testing the intervention forward
        zs = torch.randn(32, 3, 2)  # mu=0, std=1
        zs_post = hook(None, None, zs)
        assert torch.isnan(zs_post).sum() == 0
        assert torch.isinf(zs_post).sum() == 0

        if intervention_name == "none":
            assert torch.allclose(zs, zs_post)
            return

        # Check that values in statedict override those in constructor
        if intervention_name == "gaussian_ot":
            assert hook.onlymean is False
        elif intervention_name == "mean_ot":
            assert hook.onlymean is True
        if hasattr(hook, "hook_std_eps"):
            assert hook.std_eps == 1e-7

        assert hook.strength == 0.9


def test_gaussian_function():
    b, d = 1000, 5
    mu1_gt = 1.0
    std1_gt = 1.0
    mu2_gt = 5.0
    std2_gt = 0.5
    zs = torch.randn(b, d) * std1_gt + mu1_gt  # mu=1, std=1
    zd = torch.randn(b, d) * std2_gt + mu2_gt  # mu=5.0, std=0.5

    # Test in forward mode
    hook = GaussianOTHook(
        module_name="test",
        dtype=torch.float32,
        intervention_position="all",
    )

    labels = torch.cat([torch.ones([b]), torch.zeros([b])]).to(torch.int64)
    hook.fit(
        responses=torch.cat([zs, zd], 0),
        labels=labels,
    )

    d = hook.state_dict()
    assert np.allclose(d["mu1"], mu1_gt, atol=0.1), f"mean1: {d['mu1']} !!"
    assert np.allclose(d["mu2"], mu2_gt, atol=0.1), f"mean2: {d['mu2']} !!"
    assert np.allclose(d["std1"], std1_gt, atol=0.1), f"std1: {d['std1']} !!"
    assert np.allclose(d["std2"], std2_gt, atol=0.1), f"std2: {d['std2']} !!"
    print(d["quantiles_dict_src"])
    print(zs.shape)
    z_transport = hook(None, None, zs)
    assert np.allclose(z_transport.mean(0), mu2_gt, atol=0.1), f"mean1: {d['mu1']} !!"
    assert np.allclose(z_transport.std(0), std2_gt, atol=0.1), f"std1: {d['std1']} !!"

    hook2 = GaussianOTHook(
        module_name="test",
        dtype=torch.float32,
        intervention_position="all",
    )
    hook2.load_state_dict(d)
    d = hook2.state_dict()
    assert np.allclose(d["mu1"], 1.0, atol=0.1), f"mean: {d['mu1']} !!"
    assert np.allclose(d["mu2"], 5.0, atol=0.1), f"mean: {d['mu2']} !!"
    assert np.allclose(d["std1"], 1.0, atol=0.1), f"mean: {d['std1']} !!"
    assert np.allclose(d["std2"], 0.5, atol=0.1), f"mean: {d['std2']} !!"


@pytest.mark.parametrize("kernel_type", ["rbf", "polynomial", "laplacian"])
def test_kernel_gradient_is_finite(kernel_type):
    x = torch.tensor([[0.0, 0.5], [1.0, -0.5]], dtype=torch.float32)
    source = torch.tensor([[0.0, 0.0], [0.2, 0.1]], dtype=torch.float32)
    target = torch.tensor([[1.0, 1.0], [1.2, 0.9]], dtype=torch.float32)
    kernel_params = (
        {"sigma": 1.0}
        if kernel_type in {"rbf", "laplacian"}
        else {"degree": 2, "offset": 1.0}
    )

    grad = compute_kernelized_steering_gradient(
        x=x,
        source=source,
        target=target,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
    )

    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()


def test_kernel_median_heuristic_sigma_is_positive_and_deterministic():
    source = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 2.0], [2.0, 2.0]], dtype=torch.float32)

    sigma_1 = compute_median_heuristic_sigma(source, target)
    sigma_2 = compute_median_heuristic_sigma(source, target)

    assert sigma_1 > 0
    assert sigma_1 == sigma_2


def test_kernel_hook_fit_forward_and_state_roundtrip():
    source = torch.zeros(4, 2)
    target = torch.ones(4, 2)
    labels = torch.cat([torch.ones(4), torch.zeros(4)]).to(torch.bool)
    responses = torch.cat([source, target], dim=0)

    hook = KernelSteeringHook(
        module_name="test",
        device="cpu",
        dtype=torch.float32,
        kernel_type="rbf",
        kernel_params=None,
        eta=0.5,
    )
    hook.fit(responses=responses, labels=labels)

    assert hook.source.shape == source.shape
    assert hook.target.shape == target.shape
    assert hook.kernel_params["sigma"] > 0

    zs = torch.zeros(2, 3, 2, dtype=torch.float32)
    zs_post = hook(None, None, zs)
    assert zs_post.shape == zs.shape
    assert zs_post.dtype == zs.dtype
    assert torch.isfinite(zs_post).all()

    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        state_path = f"{tempfolder}/kernel.statedict"
        hook.save_state_dict(state_path)
        loaded = KernelSteeringHook(module_name="test", device="cpu")
        loaded.from_state_path(state_path)

    assert torch.allclose(loaded.source, hook.source)
    assert torch.allclose(loaded.target, hook.target)
    assert loaded.kernel_params == hook.kernel_params


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": 0.0},
        {"eta": 0.0},
        {"step_count": 0},
    ],
)
def test_kernel_hook_identity_cases(kwargs):
    hook = KernelSteeringHook(
        module_name="test",
        device="cpu",
        kernel_params={"sigma": 1.0},
        **kwargs,
    )
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    responses = torch.tensor(
        [[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.2, 0.9]],
        dtype=torch.float32,
    )
    hook.fit(responses=responses, labels=labels)

    zs = torch.randn(2, 3, 2)
    assert torch.allclose(hook(None, None, zs), zs)


def test_kernel_hook_prompt_only_skips_decode_token():
    hook = KernelSteeringHook(
        module_name="test",
        device="cpu",
        kernel_params={"sigma": 1.0},
        prompt_only=True,
        eta=1.0,
    )
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    responses = torch.tensor(
        [[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.2, 0.9]],
        dtype=torch.float32,
    )
    hook.fit(responses=responses, labels=labels)

    decode_activation = torch.randn(2, 1, 2)
    assert torch.allclose(hook(None, None, decode_activation), decode_activation)


def test_kernel_hook_rejects_diffusion_activations():
    hook = KernelSteeringHook(
        module_name="test",
        device="cpu",
        kernel_params={"sigma": 1.0},
    )
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    responses = torch.randn(4, 2)
    hook.fit(responses=responses, labels=labels)

    with pytest.raises(NotImplementedError, match="text activations only"):
        hook(None, None, torch.randn(2, 2, 4, 4))

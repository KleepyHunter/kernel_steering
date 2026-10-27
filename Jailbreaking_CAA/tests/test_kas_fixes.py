from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("matplotlib")
pytest.importorskip("jaxtyping")

from pipeline import references
from pipeline.utils.hook_utils import (
    compute_median_heuristic_sigma_l1,
    get_kernelized_steering_hooks,
    get_kernelized_steering_input_pre_hook,
)


@pytest.mark.parametrize("kernel_type, kernel_params", [
    ("rbf", {"sigma": 3.0}),
    ("polynomial", {"degree": 2}),
    ("laplacian", {"sigma": 5.0}),
])
def test_kas_hook_accepts_float64_references(kernel_type, kernel_params):
    # The all-layer path (--use_mean_act True) stores references in float64.
    torch.manual_seed(0)
    source, target = torch.randn(7, 16, dtype=torch.float64), torch.randn(5, 16, dtype=torch.float64)
    activation = torch.randn(2, 4, 16).bfloat16()
    hook = get_kernelized_steering_input_pre_hook(
        source, target, kernel_type=kernel_type, kernel_params=kernel_params, eta=2.0, step_count=2
    )
    steered = hook(None, (activation,))[0]
    assert steered.dtype == torch.bfloat16
    assert steered.shape == activation.shape
    assert torch.isfinite(steered.float()).all()


def test_laplacian_hooks_pick_l1_median_sigma():
    # LLM-scale activations: L1 distances in the hundreds, so sigma=1 would zero every kernel value.
    torch.manual_seed(0)
    source, target = 3 * torch.randn(6, 512), 3 * torch.randn(6, 512) + 1
    combined = torch.cat([source, target])
    dists = torch.cdist(combined, combined, p=1)
    assert compute_median_heuristic_sigma_l1(source, target)["sigma"] == pytest.approx(dists[dists > 0].median().item())

    model_base = SimpleNamespace(model_block_modules=[None])
    (_, hook), = get_kernelized_steering_hooks(model_base, 0, source, target, kernel_type="laplacian", eta=2.0)[0]
    activation = 3 * torch.randn(1, 3, 512)
    assert not torch.equal(hook(None, (activation,))[0], activation)


def _patch_refusal_data(monkeypatch, pool, refusing):
    monkeypatch.setattr(references, "load_dataset_split", lambda **kwargs: list(pool))
    monkeypatch.setattr(
        references,
        "get_refusal_scores",
        lambda model, batch, *args, **kwargs: torch.tensor([1.0 if inst in refusing else -1.0 for inst in batch]),
    )


_MODEL_BASE = SimpleNamespace(model=None, tokenize_instructions_fn=None, refusal_toks=None)


def test_refill_tops_up_to_n_train(monkeypatch):
    pool = [f"p{i}" for i in range(100)]
    refusing = {f"p{i}" for i in range(0, 100, 3)}
    _patch_refusal_data(monkeypatch, pool, refusing)
    sampled = ["p0", "p1", "p2", "p3"]
    kept = [inst for inst in sampled if inst in refusing]
    refilled = references._refill_filtered(
        SimpleNamespace(n_train=4), _MODEL_BASE, "harmful", sampled, kept, lambda score: score > 0
    )
    assert len(refilled) == 4
    assert refilled[:2] == ["p0", "p3"]
    assert all(inst in refusing for inst in refilled)
    assert len(set(refilled)) == 4


def test_refill_keeps_unfiltered_sample(monkeypatch):
    _patch_refusal_data(monkeypatch, ["a", "b"], {"a", "b"})
    assert references._refill_filtered(
        SimpleNamespace(n_train=2), _MODEL_BASE, "harmful", ["a", "b"], ["a", "b"], lambda score: score > 0
    ) == ["a", "b"]


def test_refill_raises_when_pool_is_exhausted(monkeypatch):
    _patch_refusal_data(monkeypatch, ["a", "b", "c"], {"a"})
    with pytest.raises(ValueError, match="pass refusal filtering"):
        references._refill_filtered(
            SimpleNamespace(n_train=2), _MODEL_BASE, "harmful", ["a", "b"], ["a"], lambda score: score > 0
        )

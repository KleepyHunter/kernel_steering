import copy
from types import SimpleNamespace

import pytest
import torch

from pipeline.utils.chars import (
    CHARS_FORMAT,
    apply_chars,
    build_chars_steer_config,
    fit_chars_state,
    get_block_output_activations,
    get_chars_output_hook,
    solve_log_sinkhorn,
    validate_chars_state,
)


def _references():
    source = torch.tensor(
        [[0.0, 0.0], [0.2, 0.1], [4.0, 4.0], [4.2, 4.1]],
        dtype=torch.float32,
    )
    target = source + torch.tensor([1.5, -0.5])
    return source, target


def test_k1_is_exact_difference_in_means():
    source, target = _references()
    state = fit_chars_state(source, target, n_clusters=1)
    activations = torch.tensor([[[2.0, 3.0], [0.0, 0.0]]])
    actual = apply_chars(activations, state, alpha=1.0)
    expected = activations + (target.mean(0) - source.mean(0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_compact_equation_matches_direct_k_squared_formula():
    source, target = _references()
    state = fit_chars_state(source, target, n_clusters=2)
    activations = torch.tensor([[[0.1, 0.2], [3.8, 4.3]]])
    compact = apply_chars(activations, state, alpha=1.0)

    flat = activations.reshape(-1, 2)
    distances = torch.cdist(flat, state["source_centroids"]).square()
    sigma_squared = distances.median(dim=1, keepdim=True).values.clamp_min(1e-12)
    gates = torch.exp(-distances / (2 * sigma_squared))
    direct_shifts = []
    for x_index in range(flat.shape[0]):
        numerator = torch.zeros(2)
        denominator = torch.zeros(())
        for i in range(state["n_clusters"]):
            for j in range(state["n_clusters"]):
                weight = state["coupling"][i, j] * gates[x_index, i]
                numerator += weight * (
                    state["target_centroids"][j] - state["source_centroids"][i]
                )
                denominator += weight
        direct_shifts.append(numerator / denominator)
    direct = activations + torch.stack(direct_shifts).reshape_as(activations)
    torch.testing.assert_close(compact, direct, rtol=1e-5, atol=1e-6)


def test_sinkhorn_converges_to_requested_marginals():
    cost = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    source = torch.tensor([0.25, 0.75])
    target = torch.tensor([0.6, 0.4])
    coupling, _, error = solve_log_sinkhorn(cost, source, target)
    assert error <= 1e-6
    torch.testing.assert_close(coupling.sum(1), source, rtol=0, atol=1e-6)
    torch.testing.assert_close(coupling.sum(0), target, rtol=0, atol=1e-6)


def test_sinkhorn_allows_slow_but_convergent_transport():
    # Reduced from the Gemma-2 hallucination CAA fit that exposed the old
    # 500-iteration cap. This deterministic problem converges after 500.
    cost = torch.tensor(
        [
            [0.46122312545776367, 1.1135470867156982],
            [0.9999160170555115, 0.27815768122673035],
        ]
    )
    source = torch.tensor([0.7317073345184326, 0.2682926654815674])
    target = torch.tensor([0.7267441749572754, 0.2732557952404022])
    coupling, iterations, error = solve_log_sinkhorn(cost, source, target)
    assert 500 < iterations <= 1000
    assert error <= 1e-6
    torch.testing.assert_close(coupling.sum(1), source, rtol=0, atol=1e-6)
    torch.testing.assert_close(coupling.sum(0), target, rtol=0, atol=1e-6)


def test_orientation_reverses_the_transport_shift():
    source, target = _references()
    forward = fit_chars_state(source, target, n_clusters=1)
    reverse = fit_chars_state(target, source, n_clusters=1)
    activations = torch.zeros(1, 1, 2)
    forward_shift = apply_chars(activations, forward, alpha=1)
    reverse_shift = apply_chars(activations, reverse, alpha=1)
    torch.testing.assert_close(forward_shift, -reverse_shift, rtol=0, atol=0)


def test_state_and_fingerprint_are_reproducible():
    source, target = _references()
    first = fit_chars_state(source, target, n_clusters=2, metadata={"layer": 4})
    second = fit_chars_state(source, target, n_clusters=2, metadata={"layer": 4})
    assert first["state_fingerprint"] == second["state_fingerprint"]
    for key in ("source_centroids", "target_centroids", "coupling"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_state_fingerprint_rejects_mutation_and_wrong_metadata():
    source, target = _references()
    state = fit_chars_state(source, target, n_clusters=2, metadata={"layer": 4})
    mutated = copy.deepcopy(state)
    mutated["source_centroids"][0, 0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        validate_chars_state(mutated)
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_chars_state(state, expected_metadata={"layer": 5})
    changed_recipe = copy.deepcopy(state)
    changed_recipe["recipe"]["transport"]["regularization"] = 0.1
    with pytest.raises(ValueError, match="recipe"):
        validate_chars_state(changed_recipe)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_runtime_preserves_input_dtype_and_batch_shape(dtype):
    source, target = _references()
    state = fit_chars_state(source, target, n_clusters=2)
    activations = torch.zeros(3, 4, 2, dtype=dtype)
    result = apply_chars(activations, state, alpha=0.5)
    assert result.shape == activations.shape
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


def test_zero_distance_bandwidth_and_prompt_only_decode_are_safe():
    source, target = _references()
    state = fit_chars_state(source, target, n_clusters=2)
    activations = state["source_centroids"][0].reshape(1, 1, -1)
    assert torch.equal(
        apply_chars(activations, state, alpha=1, prompt_only=True),
        activations,
    )
    result = apply_chars(activations.repeat(1, 2, 1), state, alpha=1, prompt_only=True)
    assert torch.isfinite(result).all()


def test_large_finite_activations_remain_finite():
    source, target = _references()
    state = fit_chars_state(source, target, n_clusters=2)
    activations = torch.full((2, 3, 2), 1e15, dtype=torch.float32)
    result = apply_chars(activations, state, alpha=100.0)
    assert torch.isfinite(result).all()


def test_invalid_clusters_and_nonfinite_references_are_rejected():
    source, target = _references()
    with pytest.raises(ValueError, match="\\[1, 15\\]"):
        fit_chars_state(source, target, n_clusters=16)
    with pytest.raises(ValueError, match="distinct"):
        fit_chars_state(torch.zeros(4, 2), target, n_clusters=2)
    source[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        fit_chars_state(source, target, n_clusters=1)


def test_sinkhorn_nonconvergence_is_rejected():
    cost = torch.tensor([[0.0, 100.0], [100.0, 0.0]])
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_log_sinkhorn(
            cost,
            torch.tensor([0.9, 0.1]),
            torch.tensor([0.1, 0.9]),
            max_iter=1,
        )


def test_embedded_config_is_tensor_only_and_self_identifying():
    source, target = _references()
    state = fit_chars_state(source, target, n_clusters=2)
    config = build_chars_steer_config(
        state, layer=7, alpha=1.25, position=-1, prompt_only=True
    )
    item = config["layers.7"]
    assert item["action"] == "chars"
    assert item["state"]["format"] == CHARS_FORMAT
    assert item["state_fingerprint"] == state["state_fingerprint"]


def test_output_hook_preserves_tuple_tail():
    source, target = _references()
    state = fit_chars_state(source, target, n_clusters=1)
    hook = get_chars_output_hook(state, alpha=1.0)
    activations = torch.zeros(1, 2, 2)
    result = hook(None, None, (activations, "cache"))
    assert result[1] == "cache"
    torch.testing.assert_close(
        result[0], activations + (target.mean(0) - source.mean(0))
    )


def test_multi_layer_extraction_uses_one_batched_reference_pass():
    class Block(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment

        def forward(self, value):
            return value + self.increment

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(1), Block(2)])
            self.forward_calls = 0

        @property
        def device(self):
            return torch.device("cpu")

        def forward(self, input_ids, attention_mask):
            self.forward_calls += 1
            value = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 2)
            for block in self.blocks:
                value = block(value)
            return value

    model = Model()

    def tokenize(instructions, outputs=None):
        values = torch.tensor([[int(item)] for item in instructions])
        return SimpleNamespace(input_ids=values, attention_mask=torch.ones_like(values))

    model_base = SimpleNamespace(
        model=model,
        model_block_modules=model.blocks,
        tokenize_instructions_fn=tokenize,
    )
    activations = get_block_output_activations(
        model_base,
        ["1", "2", "3"],
        layers=(0, 1),
        position=-1,
        batch_size=2,
    )
    assert model.forward_calls == 2
    torch.testing.assert_close(
        activations[0], torch.tensor([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    )
    torch.testing.assert_close(
        activations[1], torch.tensor([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
    )

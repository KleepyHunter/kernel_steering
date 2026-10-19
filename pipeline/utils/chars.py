"""Core CHaRS equal-covariance activation transport.

This module deliberately contains only tensor state and deterministic numerical
recipes so that the same artifact can be used by the direct pipeline and
lm-evaluation-harness.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import torch
from sklearn.cluster import KMeans
from torch import Tensor
from threadpoolctl import threadpool_limits


CHARS_FORMAT = "chars_equal_cov_v2"
CHARS_MIN_CLUSTERS = 1
CHARS_MAX_CLUSTERS = 15
CHARS_KMEANS_SEED = 42
CHARS_KMEANS_N_INIT = 10
CHARS_KMEANS_MAX_ITER = 300
CHARS_KMEANS_TOL = 1e-4
CHARS_SINKHORN_REGULARIZATION = 0.05
CHARS_KMEANS_THREADS = 1
CHARS_SINKHORN_MAX_ITER = 1000
CHARS_SINKHORN_TOLERANCE = 1e-6
CHARS_BANDWIDTH_FLOOR = 1e-12


def chars_recipe() -> dict[str, Any]:
    """Return the immutable repository-owned CHaRS numerical recipe."""

    return {
        "name": CHARS_FORMAT,
        "kmeans": {
            "seed": CHARS_KMEANS_SEED,
            "algorithm": "lloyd",
            "n_init": CHARS_KMEANS_N_INIT,
            "max_iter": CHARS_KMEANS_MAX_ITER,
            "threads": CHARS_KMEANS_THREADS,
            "tol": CHARS_KMEANS_TOL,
        },
        "transport": {
            "cost": "squared_euclidean",
            "cost_normalization": "positive_median",
            "regularization": CHARS_SINKHORN_REGULARIZATION,
            "solver": "log_sinkhorn",
            "max_iter": CHARS_SINKHORN_MAX_ITER,
            "marginal_tolerance": CHARS_SINKHORN_TOLERANCE,
            "dtype": "float32",
        },
        "gating": {
            "kernel": "rbf",
            "sigma_squared": "per_token_median_source_centroid_distance_squared",
            "bandwidth_floor": CHARS_BANDWIDTH_FLOOR,
        },
        "local_map": "equal_covariance_translation",
        "hookpoint": "block_output",
    }


def _validate_clusters(n_clusters: int) -> int:
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, Integral):
        raise ValueError("CHaRS n_clusters must be an integer.")
    value = int(n_clusters)
    if value < CHARS_MIN_CLUSTERS or value > CHARS_MAX_CLUSTERS:
        raise ValueError(
            "CHaRS n_clusters must lie in "
            f"[{CHARS_MIN_CLUSTERS}, {CHARS_MAX_CLUSTERS}], got {value}."
        )
    return value


def _as_reference_matrix(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(
            f"CHaRS {name} activations must have shape [examples, d_model]."
        )
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"CHaRS {name} activations must be non-empty.")
    if not torch.isfinite(value).all():
        raise ValueError(f"CHaRS {name} activations must be finite.")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _tensor_bytes(value: Tensor) -> bytes:
    return (
        value.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    )


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("CHaRS metadata must contain only finite numbers.")
        return number
    raise TypeError(
        "CHaRS metadata must be weights-only compatible; "
        f"unsupported value {type(value).__name__}."
    )


def _fingerprint_mapping(payload: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256()

    def update(path: str, value: Any) -> None:
        if isinstance(value, Tensor):
            hasher.update(b"tensor\0")
            hasher.update(path.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(str(value.dtype).encode("ascii"))
            hasher.update(b"\0")
            hasher.update(json.dumps(list(value.shape)).encode("ascii"))
            hasher.update(b"\0")
            hasher.update(_tensor_bytes(value))
            return
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                update(f"{path}/{key}", value[key])
            return
        canonical = _canonical_json_value(value)
        hasher.update(b"value\0")
        hasher.update(path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )

    for key in sorted(payload, key=str):
        if key != "state_fingerprint":
            update(str(key), payload[key])
    return hasher.hexdigest()


def reference_fingerprint(source: Tensor, target: Tensor) -> str:
    """Fingerprint ordered source/target activation references."""

    source_matrix = _as_reference_matrix(source, name="source")
    target_matrix = _as_reference_matrix(target, name="target")
    return _fingerprint_mapping({"source": source_matrix, "target": target_matrix})


def get_block_output_activations(
    model_base: Any,
    instructions,
    *,
    outputs=None,
    layers,
    position: int,
    batch_size: int = 32,
) -> dict[int, Tensor]:
    """Collect one output position from multiple blocks in one reference pass."""

    normalized_layers = tuple(int(layer) for layer in layers)
    if not normalized_layers or len(set(normalized_layers)) != len(normalized_layers):
        raise ValueError("CHaRS extraction layers must be non-empty and unique.")
    if batch_size < 1:
        raise ValueError("CHaRS extraction batch_size must be positive.")
    if len(instructions) == 0:
        raise ValueError("CHaRS extraction requires at least one instruction.")
    if outputs is not None and len(outputs) != len(instructions):
        raise ValueError("CHaRS extraction outputs must match instructions.")
    n_model_layers = len(model_base.model_block_modules)
    if any(layer < 0 or layer >= n_model_layers for layer in normalized_layers):
        raise ValueError(
            f"CHaRS extraction layer is outside [0, {n_model_layers - 1}]."
        )

    collected: dict[int, list[Tensor]] = {layer: [] for layer in normalized_layers}
    handles = []

    def make_hook(layer: int):
        def hook_fn(module, input, output):
            activation = output[0] if isinstance(output, tuple) else output
            seq_len = activation.shape[1]
            index = position if position >= 0 else seq_len + position
            if index < 0 or index >= seq_len:
                raise ValueError(
                    f"CHaRS position {position} is invalid for sequence length {seq_len}."
                )
            collected[layer].append(
                activation[:, index, :].detach().to(device="cpu").clone()
            )

        return hook_fn

    for layer in normalized_layers:
        handles.append(
            model_base.model_block_modules[layer].register_forward_hook(
                make_hook(layer)
            )
        )
    try:
        # Hooks are attached to decoder blocks, so reference extraction does
        # not need the causal-LM head or its [batch, sequence, vocabulary]
        # logits. Hugging Face causal LMs expose the decoder as base_model;
        # lightweight test/custom models safely fall back to the original.
        forward_model = getattr(model_base.model, "base_model", model_base.model)
        forward_kwargs = {"use_cache": False} if forward_model is not model_base.model else {}
        for start in range(0, len(instructions), batch_size):
            batch_instructions = instructions[start : start + batch_size]
            if outputs is None:
                inputs = model_base.tokenize_instructions_fn(
                    instructions=batch_instructions
                )
            else:
                inputs = model_base.tokenize_instructions_fn(
                    instructions=batch_instructions,
                    outputs=outputs[start : start + batch_size],
                )
            with torch.no_grad():
                forward_model(
                    input_ids=inputs.input_ids.to(model_base.model.device),
                    attention_mask=inputs.attention_mask.to(model_base.model.device),
                    **forward_kwargs,
                )
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.cat(collected[layer], dim=0) for layer in normalized_layers}


def _fit_kmeans(samples: Tensor, n_clusters: int) -> tuple[Tensor, Tensor]:
    if samples.shape[0] < n_clusters:
        raise ValueError(
            "CHaRS requires at least n_clusters reference examples: "
            f"{samples.shape[0]} < {n_clusters}."
        )
    if torch.unique(samples, dim=0).shape[0] < n_clusters:
        raise ValueError(
            "CHaRS requires at least n_clusters distinct reference activations."
        )
    if n_clusters == 1:
        centroid = samples.mean(dim=0, keepdim=True)
        return centroid, torch.ones(1, dtype=torch.float32)

    estimator = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=CHARS_KMEANS_N_INIT,
        max_iter=CHARS_KMEANS_MAX_ITER,
        tol=CHARS_KMEANS_TOL,
        random_state=CHARS_KMEANS_SEED,
        algorithm="lloyd",
    )
    with threadpool_limits(limits=CHARS_KMEANS_THREADS):
        labels = estimator.fit_predict(samples.numpy())
    counts = torch.bincount(torch.from_numpy(labels), minlength=n_clusters).to(
        dtype=torch.float32
    )
    if int((counts > 0).sum().item()) != n_clusters:
        raise ValueError(
            "CHaRS k-means produced fewer than n_clusters fitted clusters."
        )
    centroids = torch.from_numpy(estimator.cluster_centers_).to(dtype=torch.float32)
    if not torch.isfinite(centroids).all():
        raise FloatingPointError("CHaRS k-means produced non-finite centroids.")
    return centroids.contiguous(), (counts / counts.sum()).contiguous()


def _normalize_cost(cost: Tensor) -> tuple[Tensor, Tensor]:
    positive = cost[cost > 0]
    scale = (
        positive.median()
        if positive.numel() > 0
        else torch.ones((), dtype=torch.float32)
    )
    if not torch.isfinite(scale) or float(scale.item()) <= 0.0:
        raise FloatingPointError("CHaRS transport cost normalization is invalid.")
    return cost / scale, scale


def solve_log_sinkhorn(
    cost: Tensor,
    source_masses: Tensor,
    target_masses: Tensor,
    *,
    regularization: float = CHARS_SINKHORN_REGULARIZATION,
    max_iter: int = CHARS_SINKHORN_MAX_ITER,
    tolerance: float = CHARS_SINKHORN_TOLERANCE,
) -> tuple[Tensor, int, float]:
    """Solve entropy-regularized discrete OT in the log domain."""

    cost = cost.detach().to(device="cpu", dtype=torch.float32).contiguous()
    source_masses = (
        source_masses.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    target_masses = (
        target_masses.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    if cost.ndim != 2 or cost.shape != (
        source_masses.numel(),
        target_masses.numel(),
    ):
        raise ValueError("CHaRS Sinkhorn inputs have incompatible shapes.")
    if (
        not torch.isfinite(cost).all()
        or not torch.isfinite(source_masses).all()
        or not torch.isfinite(target_masses).all()
    ):
        raise ValueError("CHaRS Sinkhorn inputs must be finite.")
    if torch.any(source_masses <= 0) or torch.any(target_masses <= 0):
        raise ValueError("CHaRS Sinkhorn marginals must be strictly positive.")
    if abs(float(source_masses.sum().item()) - 1.0) > tolerance:
        raise ValueError("CHaRS source masses must sum to one.")
    if abs(float(target_masses.sum().item()) - 1.0) > tolerance:
        raise ValueError("CHaRS target masses must sum to one.")
    if not math.isfinite(float(regularization)) or float(regularization) <= 0:
        raise ValueError("CHaRS Sinkhorn regularization must be positive.")
    if isinstance(max_iter, bool) or int(max_iter) < 1:
        raise ValueError("CHaRS Sinkhorn max_iter must be positive.")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise ValueError("CHaRS Sinkhorn tolerance must be positive.")

    log_kernel = -cost / float(regularization)
    log_source = source_masses.log()
    log_target = target_masses.log()
    log_u = torch.zeros_like(source_masses)
    log_v = torch.zeros_like(target_masses)
    coupling = torch.empty_like(cost)
    marginal_error = math.inf

    for iteration in range(1, int(max_iter) + 1):
        log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_target - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        coupling = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
        row_error = torch.max(torch.abs(coupling.sum(dim=1) - source_masses))
        column_error = torch.max(torch.abs(coupling.sum(dim=0) - target_masses))
        marginal_error = float(torch.maximum(row_error, column_error).item())
        if not math.isfinite(marginal_error) or not torch.isfinite(coupling).all():
            raise FloatingPointError("CHaRS Sinkhorn produced a non-finite state.")
        if marginal_error <= float(tolerance):
            return coupling.contiguous(), iteration, marginal_error

    raise RuntimeError(
        "CHaRS Sinkhorn did not converge within "
        f"{max_iter} iterations (marginal error {marginal_error:.3e})."
    )


def fit_chars_state(
    source: Tensor,
    target: Tensor,
    *,
    n_clusters: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit independently clustered source-to-target CHaRS transport."""

    n_clusters = _validate_clusters(n_clusters)
    source_matrix = _as_reference_matrix(source, name="source")
    target_matrix = _as_reference_matrix(target, name="target")
    if source_matrix.shape[1] != target_matrix.shape[1]:
        raise ValueError(
            "CHaRS source and target hidden dimensions must match: "
            f"{source_matrix.shape[1]} != {target_matrix.shape[1]}."
        )

    source_centroids, source_masses = _fit_kmeans(source_matrix, n_clusters)
    target_centroids, target_masses = _fit_kmeans(target_matrix, n_clusters)
    raw_cost = torch.cdist(source_centroids, target_centroids, p=2).square()
    normalized_cost, cost_scale = _normalize_cost(raw_cost)
    coupling, sinkhorn_iterations, marginal_error = solve_log_sinkhorn(
        normalized_cost,
        source_masses,
        target_masses,
    )

    row_masses = coupling.sum(dim=1)
    row_weighted_shifts = (
        coupling @ target_centroids - row_masses.unsqueeze(1) * source_centroids
    )
    state: dict[str, Any] = {
        "format": CHARS_FORMAT,
        "recipe": chars_recipe(),
        "metadata": _canonical_json_value(dict(metadata or {})),
        "n_clusters": n_clusters,
        "d_model": int(source_matrix.shape[1]),
        "source_centroids": source_centroids,
        "target_centroids": target_centroids,
        "source_masses": source_masses,
        "target_masses": target_masses,
        "coupling": coupling,
        "row_masses": row_masses.contiguous(),
        "row_weighted_shifts": row_weighted_shifts.contiguous(),
        "cost_scale": cost_scale.reshape(1).to(dtype=torch.float32),
        "sinkhorn_iterations": sinkhorn_iterations,
        "sinkhorn_marginal_error": marginal_error,
        "reference_fingerprint": reference_fingerprint(source_matrix, target_matrix),
    }
    state["state_fingerprint"] = _fingerprint_mapping(state)
    validate_chars_state(state)
    return state


def validate_chars_state(
    state: Mapping[str, Any],
    *,
    expected_n_clusters: int | None = None,
    expected_fingerprint: str | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Validate recipe, tensors, marginals, metadata, and canonical identity."""

    if not isinstance(state, Mapping):
        raise ValueError("CHaRS state must be a mapping.")
    if state.get("format") != CHARS_FORMAT:
        raise ValueError(f"Unsupported CHaRS state format: {state.get('format')!r}.")
    if _canonical_json_value(state.get("recipe")) != chars_recipe():
        raise ValueError(f"CHaRS state recipe does not match {CHARS_FORMAT}.")
    n_clusters = _validate_clusters(state.get("n_clusters"))
    if expected_n_clusters is not None and n_clusters != _validate_clusters(
        expected_n_clusters
    ):
        raise ValueError(
            "CHaRS state n_clusters mismatch: "
            f"{n_clusters} != {int(expected_n_clusters)}."
        )
    d_model = state.get("d_model")
    if isinstance(d_model, bool) or not isinstance(d_model, Integral) or d_model < 1:
        raise ValueError("CHaRS state d_model must be a positive integer.")

    shapes = {
        "source_centroids": (n_clusters, int(d_model)),
        "target_centroids": (n_clusters, int(d_model)),
        "source_masses": (n_clusters,),
        "target_masses": (n_clusters,),
        "coupling": (n_clusters, n_clusters),
        "row_masses": (n_clusters,),
        "row_weighted_shifts": (n_clusters, int(d_model)),
        "cost_scale": (1,),
    }
    for name, shape in shapes.items():
        value = state.get(name)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"CHaRS state tensor {name!r} must have shape {shape}.")
        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError(f"CHaRS state tensor {name!r} must be CPU float32.")
        if not torch.isfinite(value).all():
            raise ValueError(f"CHaRS state tensor {name!r} must be finite.")

    fingerprint = _fingerprint_mapping(state)
    stored = state.get("state_fingerprint")
    if not isinstance(stored, str) or stored != fingerprint:
        raise ValueError("CHaRS canonical state fingerprint is missing or invalid.")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError(
            "CHaRS state fingerprint mismatch: "
            f"{fingerprint} != {expected_fingerprint}."
        )

    source_masses = state["source_masses"]
    target_masses = state["target_masses"]
    coupling = state["coupling"]
    row_masses = state["row_masses"]
    if (
        torch.any(source_masses <= 0)
        or torch.any(target_masses <= 0)
        or torch.any(coupling < 0)
        or torch.any(row_masses <= 0)
        or float(state["cost_scale"].item()) <= 0.0
    ):
        raise ValueError("CHaRS state cluster masses must be positive.")
    sinkhorn_iterations = state.get("sinkhorn_iterations")
    sinkhorn_error = state.get("sinkhorn_marginal_error")
    if (
        isinstance(sinkhorn_iterations, bool)
        or not isinstance(sinkhorn_iterations, Integral)
        or not 1 <= int(sinkhorn_iterations) <= CHARS_SINKHORN_MAX_ITER
        or not isinstance(sinkhorn_error, Real)
        or not math.isfinite(float(sinkhorn_error))
        or float(sinkhorn_error) > CHARS_SINKHORN_TOLERANCE
    ):
        raise ValueError("CHaRS state records an invalid Sinkhorn convergence result.")
    tolerance = CHARS_SINKHORN_TOLERANCE
    if (
        torch.max(torch.abs(coupling.sum(dim=1) - source_masses)).item() > tolerance
        or torch.max(torch.abs(coupling.sum(dim=0) - target_masses)).item() > tolerance
        or torch.max(torch.abs(row_masses - coupling.sum(dim=1))).item() > tolerance
    ):
        raise ValueError("CHaRS state coupling does not satisfy its marginals.")
    expected_compact = (
        coupling @ state["target_centroids"]
        - row_masses.unsqueeze(1) * state["source_centroids"]
    )
    if not torch.allclose(
        state["row_weighted_shifts"],
        expected_compact,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("CHaRS compact row-weighted shifts are inconsistent.")

    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("CHaRS state metadata must be a mapping.")
    canonical_metadata = _canonical_json_value(metadata)
    if expected_metadata is not None:
        expected = _canonical_json_value(expected_metadata)
        for key, expected_value in expected.items():
            if canonical_metadata.get(key) != expected_value:
                raise ValueError(
                    f"CHaRS state metadata mismatch for {key!r}: "
                    f"{canonical_metadata.get(key)!r} != {expected_value!r}."
                )

    return fingerprint


def apply_chars(
    activations: Tensor,
    state: Mapping[str, Any],
    *,
    alpha: float,
    prompt_only: bool = False,
    _state_is_validated: bool = False,
) -> Tensor:
    """Apply Equation 12 in O(Kd) using compact row-weighted shifts."""

    if not _state_is_validated:
        validate_chars_state(state)
    if not isinstance(activations, Tensor) or activations.ndim != 3:
        raise ValueError("CHaRS expects activations with shape [batch, seq, d_model].")
    if activations.shape[-1] != state["d_model"]:
        raise ValueError(
            "CHaRS activation/state hidden dimensions do not match: "
            f"{activations.shape[-1]} != {state['d_model']}."
        )
    if not torch.isfinite(activations).all():
        raise ValueError("CHaRS input activations must be finite.")
    if not math.isfinite(float(alpha)):
        raise ValueError("CHaRS alpha must be finite.")
    if prompt_only and activations.shape[1] == 1:
        return activations
    if float(alpha) == 0.0:
        return activations

    original_dtype = activations.dtype
    shape = activations.shape
    flattened = activations.to(dtype=torch.float32).reshape(-1, shape[-1])
    source_centroids = state["source_centroids"].to(activations.device)
    row_masses = state["row_masses"].to(activations.device)
    row_weighted_shifts = state["row_weighted_shifts"].to(activations.device)

    if state["n_clusters"] == 1:
        # Algebraically Equation 12, without a gate/divide round trip:
        # K=1 therefore remains exactly difference-in-means.
        shift = (
            (state["target_centroids"][0] - state["source_centroids"][0])
            .to(activations.device)
            .expand_as(flattened)
        )
    else:
        squared_distances = torch.cdist(flattened, source_centroids, p=2).square()
        sigma_squared = squared_distances.median(dim=1, keepdim=True).values
        sigma_squared = sigma_squared.clamp_min(CHARS_BANDWIDTH_FLOOR)
        gates = torch.exp(-squared_distances / (2.0 * sigma_squared))
        denominator = (gates @ row_masses).clamp_min(torch.finfo(torch.float32).tiny)
        shift = (gates @ row_weighted_shifts) / denominator.unsqueeze(1)
    result = flattened + float(alpha) * shift
    if not torch.isfinite(result).all():
        raise FloatingPointError("CHaRS produced non-finite activations.")
    return result.reshape(shape).to(dtype=original_dtype)


def get_chars_output_hook(
    state: Mapping[str, Any],
    *,
    alpha: float,
    prompt_only: bool = False,
):
    validate_chars_state(state)
    runtime_states: dict[torch.device, dict[str, Any]] = {}

    def hook_fn(module, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        if activation.device not in runtime_states:
            runtime_state = dict(state)
            for name in (
                "source_centroids",
                "target_centroids",
                "row_masses",
                "row_weighted_shifts",
            ):
                runtime_state[name] = state[name].to(activation.device)
            runtime_states[activation.device] = runtime_state
        steered = apply_chars(
            activation,
            runtime_states[activation.device],
            alpha=alpha,
            prompt_only=prompt_only,
            _state_is_validated=True,
        )
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    return hook_fn


def build_chars_steer_config(
    state: Mapping[str, Any],
    *,
    layer: int,
    alpha: float,
    position: int,
    prompt_only: bool,
) -> dict[str, dict[str, Any]]:
    """Build a weights-only lm-evaluation-harness CHaRS artifact."""

    fingerprint = validate_chars_state(state)
    if isinstance(layer, bool) or not isinstance(layer, Integral) or int(layer) < 0:
        raise ValueError("CHaRS layer must be a non-negative integer.")
    if isinstance(position, bool) or not isinstance(position, Integral):
        raise ValueError("CHaRS position must be an integer.")
    if not math.isfinite(float(alpha)):
        raise ValueError("CHaRS alpha must be finite.")
    return {
        f"layers.{int(layer)}": {
            "action": "chars",
            "state": dict(state),
            "state_fingerprint": fingerprint,
            "n_clusters": int(state["n_clusters"]),
            "recipe": chars_recipe(),
            "alpha": float(alpha),
            "position": int(position),
            "hookpoint": "block_output",
            "prompt_only": bool(prompt_only),
        }
    }


def get_selected_block_output_activations(
    model_base: Any,
    instructions,
    *,
    outputs=None,
    layer: int,
    position: int,
    batch_size: int = 32,
) -> Tensor:
    """Collect block-output activations at one token position."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if layer < 0 or layer >= len(model_base.model_block_modules):
        raise ValueError(
            f"Layer {layer} is outside [0, {len(model_base.model_block_modules) - 1}]."
        )
    if outputs is not None and len(outputs) != len(instructions):
        raise ValueError(
            "When provided, outputs must match instructions for block-output "
            f"extraction: {len(outputs)} != {len(instructions)}."
        )
    if len(instructions) == 0:
        raise ValueError("Block-output extraction requires at least one instruction.")

    collected = []

    def hook_fn(module, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        seq_len = activation.shape[1]
        normalized_position = position if position >= 0 else seq_len + position
        if normalized_position < 0 or normalized_position >= seq_len:
            raise ValueError(
                f"Position {position} is invalid for sequence length {seq_len}."
            )
        collected.append(activation[:, normalized_position, :].detach().clone())

    handle = model_base.model_block_modules[layer].register_forward_hook(hook_fn)
    try:
        for start in range(0, len(instructions), batch_size):
            batch_instructions = instructions[start : start + batch_size]
            if outputs is None:
                inputs = model_base.tokenize_instructions_fn(
                    instructions=batch_instructions
                )
            else:
                inputs = model_base.tokenize_instructions_fn(
                    instructions=batch_instructions,
                    outputs=outputs[start : start + batch_size],
                )
            with torch.no_grad():
                model_base.model(
                    input_ids=inputs.input_ids.to(model_base.model.device),
                    attention_mask=inputs.attention_mask.to(model_base.model.device),
                )
    finally:
        handle.remove()

    return torch.cat(collected, dim=0)

"""Sparse Activation Steering (SAS): SAE-space steering and its direction extraction.

Self-contained: nothing outside this module and pipeline/sas_validation.py depends
on SAS, so removing both (plus their dispatch branches) removes the method.
"""

import os
from typing import Any, Dict, List, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks

try:
    from sae_lens import SAE
except ImportError:  # pragma: no cover - optional runtime dependency
    SAE = None


_SAE_CACHE: Dict[Tuple[str, str, str, str], Any] = {}


def resolve_gemma_sae_release(model_path: str, override_release: str | None = None) -> str:
    if override_release is not None:
        return override_release

    model_path_lower = model_path.lower()
    if "gemma-2-2b" in model_path_lower or "gemma2-2b" in model_path_lower:
        return "gemma-scope-2b-pt-res-canonical"
    if "gemma-2-9b" in model_path_lower or "gemma2-9b" in model_path_lower:
        return "gemma-scope-9b-pt-res-canonical"

    raise ValueError(
        "Could not infer Gemma Scope SAE release from model path. "
        "Set --sae_release explicitly for this model."
    )


def load_sae(
    release: str,
    sae_id: str,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
):
    if SAE is None:
        raise ImportError(
            "sae-lens is required for steer_method='sas'. "
            "Install it with: pip install sae-lens"
        )

    cache_key = (release, sae_id, str(device), str(dtype))
    if cache_key in _SAE_CACHE:
        return _SAE_CACHE[cache_key]

    loaded = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=str(device),
        dtype=dtype,
    )
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    sae.eval()
    sae.requires_grad_(False)
    _SAE_CACHE[cache_key] = sae
    return sae


def load_sae_for_layer(
    release: str,
    layer: int,
    width: str,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
):
    sae_id = f"layer_{layer}/width_{width}/canonical"
    sae = load_sae(release=release, sae_id=sae_id, device=device, dtype=dtype)
    return sae, sae_id


def _apply_sae_activation(z: Tensor, sae: Any) -> Tensor:
    activation_fn = getattr(sae, "activation_fn", None)
    if callable(activation_fn):
        return activation_fn(z)

    cfg = getattr(sae, "cfg", None)
    activation_name = getattr(cfg, "activation_fn_str", None)
    if activation_name in ("relu", "ReLU"):
        return torch.relu(z)
    return z


def get_sas_actadd_output_hook(
    vector: Float[Tensor, "d_sae"],
    coeff: Float[Tensor, ""],
    sae,
    use_delta_correction: bool = True,
    apply_post_activation: bool = True,
    prompt_only: bool = False,
):
    def hook_fn(module, input, output):
        nonlocal vector

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation = output

        _, seq_len, _ = activation.shape

        if prompt_only and seq_len == 1:
            if isinstance(output, tuple):
                return output
            return activation

        activation_fp32 = activation.to(torch.float32)
        z = sae.encode(activation_fp32)

        if use_delta_correction:
            delta = activation_fp32 - sae.decode(z)
        else:
            delta = None

        steering_vector = vector.to(device=z.device, dtype=z.dtype)
        z_steered = z + coeff * steering_vector

        if apply_post_activation:
            z_steered = _apply_sae_activation(z_steered, sae)

        steered_activation = sae.decode(z_steered)
        if delta is not None:
            steered_activation = steered_activation + delta

        steered_activation = steered_activation.to(activation)

        if isinstance(output, tuple):
            return (steered_activation, *output[1:])
        return steered_activation

    return hook_fn


def _normalize_positions(positions, seq_len: int) -> List[int]:
    if positions is None:
        positions = [-1]
    if isinstance(positions, int):
        positions = [positions]

    normalized_positions = []
    for pos in positions:
        idx = pos if pos >= 0 else seq_len + pos
        if idx < 0 or idx >= seq_len:
            continue
        normalized_positions.append(idx)

    if not normalized_positions:
        normalized_positions = [seq_len - 1]

    return normalized_positions


def _collect_sae_stats(
    model,
    instructions,
    outputs,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    layer: int,
    sae,
    positions,
    batch_size: int = 32,
):
    if outputs is not None and len(outputs) != len(instructions):
        raise ValueError(
            "When provided, SAS outputs must be the same length as instructions. "
            f"Got instructions={len(instructions)}, outputs={len(outputs)}."
        )

    sum_nonzero = None
    count_nonzero = None
    n_tokens = 0

    def capture_hook(module, input, output):
        nonlocal sum_nonzero, count_nonzero, n_tokens

        activation = output[0] if isinstance(output, tuple) else output
        activation_fp32 = activation.to(torch.float32)
        _, seq_len, _ = activation_fp32.shape
        pos_idx = _normalize_positions(positions, seq_len)

        selected = activation_fp32[:, pos_idx, :]
        z = sae.encode(selected)
        z_flat = z.reshape(-1, z.shape[-1])
        nonzero_mask = z_flat != 0

        if sum_nonzero is None:
            d_sae = z_flat.shape[-1]
            sum_nonzero = torch.zeros(d_sae, dtype=torch.float32, device=z_flat.device)
            count_nonzero = torch.zeros(d_sae, dtype=torch.long, device=z_flat.device)

        sum_nonzero += torch.where(nonzero_mask, z_flat, torch.zeros_like(z_flat)).sum(dim=0)
        count_nonzero += nonzero_mask.sum(dim=0)
        n_tokens += z_flat.shape[0]

    fwd_hooks = [(block_modules[layer], capture_hook)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i:i + batch_size]
        if outputs is None:
            inputs = tokenize_instructions_fn(instructions=batch_instructions)
        else:
            batch_outputs = outputs[i:i + batch_size]
            inputs = tokenize_instructions_fn(instructions=batch_instructions, outputs=batch_outputs)
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    if sum_nonzero is None or count_nonzero is None:
        raise RuntimeError("No SAE activations were collected while generating SAS direction.")

    return {
        "sum_nonzero": sum_nonzero,
        "count_nonzero": count_nonzero,
        "n_tokens": n_tokens,
    }


def _build_sparse_mean(stats, tau: float) -> Tensor:
    sum_nonzero = stats["sum_nonzero"]
    count_nonzero = stats["count_nonzero"]
    n_tokens = max(1, int(stats["n_tokens"]))

    active_frequency = count_nonzero.to(torch.float32) / float(n_tokens)
    valid = active_frequency >= tau
    has_nonzero = count_nonzero > 0

    sparse_mean = torch.zeros_like(sum_nonzero)
    masked = valid & has_nonzero
    sparse_mean[masked] = sum_nonzero[masked] / count_nonzero[masked].to(sum_nonzero.dtype)
    return sparse_mean


def generate_sas_direction(
    model_base,
    positive_instructions,
    negative_instructions,
    positive_outputs,
    negative_outputs,
    layer: int,
    sae_release: str,
    sae_width: str,
    feature_freq_threshold: float,
    positions=None,
    batch_size: int = 32,
):
    sae, _ = load_sae_for_layer(
        release=sae_release,
        layer=layer,
        width=sae_width,
        device=model_base.model.device,
        dtype=torch.float32,
    )

    positive_stats = _collect_sae_stats(
        model=model_base.model,
        instructions=positive_instructions,
        outputs=positive_outputs,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        layer=layer,
        sae=sae,
        positions=positions,
        batch_size=batch_size,
    )
    negative_stats = _collect_sae_stats(
        model=model_base.model,
        instructions=negative_instructions,
        outputs=negative_outputs,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        layer=layer,
        sae=sae,
        positions=positions,
        batch_size=batch_size,
    )

    mu_pos = _build_sparse_mean(positive_stats, tau=feature_freq_threshold)
    mu_neg = _build_sparse_mean(negative_stats, tau=feature_freq_threshold)

    shared_active = (mu_pos != 0) & (mu_neg != 0)
    mu_pos = mu_pos.clone()
    mu_neg = mu_neg.clone()
    mu_pos[shared_active] = 0
    mu_neg[shared_active] = 0

    return mu_pos - mu_neg


def generate_and_save_sas_direction(
    cfg,
    model_base,
    target_train,
    source_train,
    target_outputs=None,
    source_outputs=None,
    positions=None,
    layer=None,
):
    if layer is None:
        raise ValueError("SAS mode requires a target layer via 'layers'.")
    if cfg.sae_release is None:
        raise ValueError("SAS mode requires resolved SAE release.")

    artifact_dir = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(artifact_dir, exist_ok=True)

    direction = generate_sas_direction(
        model_base=model_base,
        positive_instructions=target_train,
        negative_instructions=source_train,
        positive_outputs=target_outputs,
        negative_outputs=source_outputs,
        layer=layer,
        sae_release=cfg.sae_release,
        sae_width=cfg.sae_width,
        feature_freq_threshold=cfg.sas_feature_freq_threshold,
        positions=positions,
    )
    torch.save(direction, os.path.join(artifact_dir, "mean_diffs.pt"))
    return direction


def build_sas_steer_config(cfg, *, layer: int, direction, prompt_only: bool) -> dict:
    sae_id = f"layer_{layer}/width_{cfg.sae_width}/canonical"
    return {
        f"layers.{layer}": {
            "steering_vector": direction,
            "steering_coefficient": cfg.sas_coefficient,
            "prompt_only": prompt_only,
            "action": "sas_sparse",
            "release": cfg.sae_release,
            "sae_id": sae_id,
            "layer": layer,
            "tau": cfg.sas_feature_freq_threshold,
            "use_delta_correction": cfg.sas_use_delta_correction,
            "apply_post_activation": cfg.sas_apply_post_activation,
        },
    }

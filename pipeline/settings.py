"""Method-dependent defaults, argument checks, and the run `Config`."""

import math
import os

from pipeline.config import Config
from pipeline.validation_selection import CHARS_VALID_CLUSTERS


def resolve_evaluation_protocol(steer_method: str, evaluation_protocol: str | None) -> str:
    if evaluation_protocol is None:
        return "caa" if steer_method in {"sas", "curveball"} else "legacy"
    return evaluation_protocol


def resolve_sas_extraction_source(steer_method: str, sas_extraction_source: str | None) -> str:
    if steer_method == "sas":
        if sas_extraction_source is None:
            return "caa"
        return sas_extraction_source
    return "legacy"


def resolve_curveball_extraction_source(steer_method: str, curveball_extraction_source: str | None) -> str:
    if steer_method == "curveball":
        if curveball_extraction_source is None:
            return "caa"
        return curveball_extraction_source
    return "legacy"


def resolve_chars_extraction_source(
    steer_method: str,
    chars_extraction_source: str | None,
    evaluation_protocol: str,
) -> str:
    if steer_method != "chars":
        return "legacy"
    if chars_extraction_source is not None:
        return chars_extraction_source
    return "caa" if evaluation_protocol == "caa" else "legacy"


def uses_caa_pair_extraction(
    kernel_extraction_source: str,
    steer_method: str = "kernel",
    curveball_extraction_source: str = "legacy",
    chars_extraction_source: str = "legacy",
    sas_extraction_source: str = "legacy",
) -> bool:
    if steer_method == "sas":
        return sas_extraction_source == "caa"
    if steer_method == "chars":
        return chars_extraction_source == "caa"
    if steer_method == "curveball":
        return curveball_extraction_source == "caa"
    return kernel_extraction_source == "caa"


def build_config(
    *,
    model_path,
    artifact_root,
    kernel_type,
    kernel_params,
    eta,
    lambda_reg,
    pos,
    layer,
    step_count,
    use_mean_act,
    prompt_only,
    seed,
    n_train,
    steer_method,
    kernel_extraction_source,
    sae_release,
    sae_width,
    sas_feature_freq_threshold,
    sas_extraction_source,
    sas_coefficient,
    sas_use_delta_correction,
    sas_apply_post_activation,
    curveball_extraction_source,
    curveball_k,
    curveball_kernel,
    curveball_degree,
    curveball_gamma,
    curveball_coef0,
    curveball_alpha,
    curveball_fit_inverse_alpha,
    chars_extraction_source,
    chars_clusters,
    chars_alpha,
    chars_state_path,
    caa_dataset_root,
    caa_behavior,
    evaluation_protocol,
    caa_eval_types,
    caa_eval_system_prompt,
    caa_eval_open_ended_score,
    caa_open_ended_judge_model,
    experiment_phase,
    selection_manifest,
    evaluation_interventions,
    split_seed,
    caa_validation_size,
    caa_eval_split,
    jailbreak_eval_split,
    jailbreak_utility_size,
    ce_loss_batch_size,
) -> Config:
    """Resolve method defaults, check the arguments, and build the run Config."""
    sas_extraction_source = resolve_sas_extraction_source(
        steer_method=steer_method,
        sas_extraction_source=sas_extraction_source,
    )
    curveball_extraction_source = resolve_curveball_extraction_source(
        steer_method=steer_method,
        curveball_extraction_source=curveball_extraction_source,
    )
    evaluation_protocol = resolve_evaluation_protocol(
        steer_method=steer_method,
        evaluation_protocol=evaluation_protocol,
    )
    chars_extraction_source = resolve_chars_extraction_source(
        steer_method=steer_method,
        chars_extraction_source=chars_extraction_source,
        evaluation_protocol=evaluation_protocol,
    )

    if steer_method not in {"kernel", "curveball", "chars", "sas"}:
        raise ValueError(f"Unsupported steer_method: {steer_method}")

    if pos is None:
        pos = -2 if uses_caa_pair_extraction(
            kernel_extraction_source=kernel_extraction_source,
            steer_method=steer_method,
            curveball_extraction_source=curveball_extraction_source,
            chars_extraction_source=chars_extraction_source,
            sas_extraction_source=sas_extraction_source,
        ) else -1

    if steer_method == "sas" and use_mean_act:
        raise ValueError("SAS mode does not support Mean-AcT extraction.")
    if steer_method == "curveball" and use_mean_act:
        raise ValueError("Curveball mode does not support Mean-AcT extraction.")
    if steer_method == "chars" and use_mean_act:
        raise ValueError("CHaRS mode does not support Mean-AcT extraction.")

    if use_mean_act and (uses_caa_pair_extraction(
        kernel_extraction_source=kernel_extraction_source,
        steer_method=steer_method,
        curveball_extraction_source=curveball_extraction_source,
        chars_extraction_source=chars_extraction_source,
        sas_extraction_source=sas_extraction_source,
    ) or evaluation_protocol in {"caa", "both"}):
        raise ValueError("CAA is unsupported for Mean-AcT in this branch.")

    if isinstance(chars_clusters, bool) or int(chars_clusters) not in CHARS_VALID_CLUSTERS:
        raise ValueError("chars_clusters must be an integer in [1, 15].")
    if not math.isfinite(float(chars_alpha)):
        raise ValueError("chars_alpha must be finite.")

    if caa_eval_open_ended_score and caa_open_ended_judge_model is None:
        raise ValueError("--caa_open_ended_judge_model is required when --caa_eval_open_ended_score=true.")

    if n_train < 1:
        raise ValueError("--n_train must be >= 1")

    print(f"kernel params: {kernel_params}")

    model_alias = os.path.basename(os.path.normpath(model_path))
    cfg = Config(
        model_alias=model_alias,
        model_path=model_path,
        artifact_root=artifact_root,
        seed=seed,
        n_train=n_train,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        eta=eta,
        lambda_reg=lambda_reg,
        step_count=step_count,
        pos=pos,
        layer=layer,
        use_mean_act=use_mean_act,
        prompt_only=prompt_only,
        steer_method=steer_method,
        kernel_extraction_source=kernel_extraction_source,
        sae_release=sae_release,
        sae_width=sae_width,
        sas_feature_freq_threshold=sas_feature_freq_threshold,
        sas_extraction_source=sas_extraction_source,
        sas_coefficient=sas_coefficient,
        sas_use_delta_correction=sas_use_delta_correction,
        sas_apply_post_activation=sas_apply_post_activation,
        curveball_extraction_source=curveball_extraction_source,
        curveball_k=curveball_k,
        curveball_kernel=curveball_kernel,
        curveball_degree=curveball_degree,
        curveball_gamma=curveball_gamma,
        curveball_coef0=curveball_coef0,
        curveball_alpha=curveball_alpha,
        curveball_fit_inverse_alpha=curveball_fit_inverse_alpha,
        chars_extraction_source=chars_extraction_source,
        chars_clusters=chars_clusters,
        chars_alpha=chars_alpha,
        chars_state_path=chars_state_path,
        caa_dataset_root=caa_dataset_root,
        caa_behavior=caa_behavior,
        evaluation_protocol=evaluation_protocol,
        caa_eval_types=tuple(caa_eval_types),
        caa_eval_system_prompt=(None if caa_eval_system_prompt == "none" else caa_eval_system_prompt),
        caa_eval_open_ended_score=caa_eval_open_ended_score,
        caa_open_ended_judge_model=caa_open_ended_judge_model,
        experiment_phase=experiment_phase,
        selection_manifest=selection_manifest,
        evaluation_interventions=(
            None if evaluation_interventions is None else tuple(evaluation_interventions)
        ),
        split_seed=split_seed,
        caa_validation_size=caa_validation_size,
        caa_eval_split=caa_eval_split,
        jailbreak_eval_split=jailbreak_eval_split,
        jailbreak_utility_size=jailbreak_utility_size,
        ce_loss_batch_size=ce_loss_batch_size,
    )
    return cfg

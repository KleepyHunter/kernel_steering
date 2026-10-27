"""Checks for `--experiment_phase select|test` runs.

A validation (select) or manifest-locked test run must use the registered layer and
strength grids and each method's fixed recipe; locked tests must also match the
selection manifest.
"""

import math

from pipeline import sas_validation
from pipeline.validation_selection import (
    CAA_LAYERS,
    CAA_MODEL,
    CAA_STRENGTHS,
    CHARS_RECIPES,
    CURVEBALL_CAA_RECIPE,
    INTERVENTION_TO_METHOD,
    JAILBREAK_LAYERS,
    JAILBREAK_STRENGTHS,
    application_mode,
    assert_chars_recipe_configuration,
    assert_curveball_caa_recipe_configuration,
    assert_locked_configuration,
    chars_validation_clusters,
    load_manifest,
    method_strength,
)


def _registered_jailbreak_layers(model_alias):
    for known_model, layers in JAILBREAK_LAYERS.items():
        if known_model.casefold() == model_alias.casefold():
            return layers
    raise ValueError(
        f"No validation layer grid is registered for model {model_alias!r}."
    )


def _in_float_grid(value, grid):
    return any(math.isclose(float(value), item, rel_tol=0.0, abs_tol=1e-12) for item in grid)


def curveball_caa_fixed_parameters(cfg):
    return {
        "kernel": str(cfg.curveball_kernel),
        "degree": int(cfg.curveball_degree),
        "gamma": None if cfg.curveball_gamma is None else float(cfg.curveball_gamma),
        "coef0": float(cfg.curveball_coef0),
        "kpca_components": int(cfg.curveball_k),
        "inverse_transform_alpha": float(cfg.curveball_fit_inverse_alpha),
        "extraction_source": str(cfg.curveball_extraction_source),
        "position": int(cfg.pos),
    }


def chars_fixed_parameters(cfg):
    experiment = "caa" if cfg.chars_extraction_source == "caa" else "jailbreak"
    return dict(CHARS_RECIPES[experiment])


def _assert_fixed_chars_recipe(cfg, experiment):
    expected = dict(CHARS_RECIPES[experiment])
    actual = chars_fixed_parameters(cfg)
    if actual != expected:
        raise ValueError(
            f"CHaRS {experiment} validation requires {expected}; requested={actual}."
        )


def _assert_fixed_curveball_caa_recipe(cfg):
    expected = dict(CURVEBALL_CAA_RECIPE)
    actual = curveball_caa_fixed_parameters(cfg)
    if actual != expected:
        raise ValueError(
            "CAA Curveball validation requires the fixed recipe "
            f"{expected}; requested={actual}."
        )


def validate_phase_configuration(cfg):
    if cfg.experiment_phase == "standard":
        return None
    if cfg.evaluation_protocol == "both":
        raise ValueError("select/test phases operate on one experiment at a time.")
    if cfg.split_seed != 42:
        raise ValueError("The validation protocol fixes the split seed to 42.")
    if cfg.use_mean_act:
        raise ValueError("Validation selection requires a single candidate layer.")
    if cfg.layer is None:
        raise ValueError("select/test phases require --layer.")
    if cfg.ce_loss_batch_size <= 0:
        raise ValueError("ce_loss_batch_size must be positive.")

    if cfg.steer_method == "kernel":
        if cfg.step_count != 1 or not math.isclose(cfg.lambda_reg, 0.0, abs_tol=1e-12):
            raise ValueError("KAS validation fixes step_count=1 and lambda_reg=0.")
        if cfg.kernel_type != "rbf" or cfg.kernel_params is not None:
            raise ValueError(
                "KAS selection requires RBF with no explicit kernel_params; "
                "bandwidth is computed from reference activations."
            )

    if cfg.evaluation_protocol == "caa":
        experiment = "caa"
        if cfg.steer_method not in {"kernel", "curveball", "chars", "sas"}:
            raise ValueError(
                "CAA selection supports ActAdd, KAS, Curveball, CHaRS, and SAS only."
            )
        if cfg.model_alias.casefold() != CAA_MODEL.casefold():
            raise ValueError(f"CAA validation selection is fixed to {CAA_MODEL}.")
        if cfg.pos != -2:
            raise ValueError("CAA validation extraction is fixed to the answer-token position.")
        if cfg.steer_method == "kernel":
            if cfg.kernel_extraction_source != "caa":
                raise ValueError("CAA ActAdd/KAS require CAA paired reference extraction.")
            if cfg.layer not in CAA_LAYERS or not _in_float_grid(cfg.eta, CAA_STRENGTHS):
                raise ValueError("CAA candidate is outside the registered layer/strength grid.")
        elif cfg.steer_method == "sas":
            if cfg.layer not in sas_validation.SAS_LAYERS:
                raise ValueError("CAA SAS candidate is outside the registered layer grid.")
            sas_validation.assert_fixed_sas_recipe(cfg, experiment)
        elif cfg.steer_method == "chars":
            if cfg.layer not in CAA_LAYERS or not _in_float_grid(
                cfg.chars_alpha, CAA_STRENGTHS
            ):
                raise ValueError("CAA CHaRS candidate is outside the layer/alpha grid.")
            locked_k = chars_validation_clusters(cfg.model_alias)[0]
            if cfg.chars_clusters != locked_k:
                raise ValueError(
                    f"CAA CHaRS validation fixes K={locked_k} for {cfg.model_alias}."
                )
            _assert_fixed_chars_recipe(cfg, experiment)
        else:
            if cfg.layer not in CAA_LAYERS or not _in_float_grid(
                cfg.curveball_alpha, CAA_STRENGTHS
            ):
                raise ValueError(
                    "CAA Curveball candidate is outside the registered layer/alpha grid."
                )
            _assert_fixed_curveball_caa_recipe(cfg)
        required_split = "validation" if cfg.experiment_phase == "select" else "test"
        if cfg.caa_eval_split != required_split:
            raise ValueError(
                f"{cfg.experiment_phase} phase requires caa_eval_split={required_split}."
            )
        if cfg.experiment_phase == "select" and tuple(cfg.caa_eval_types) != ("ab",):
            raise ValueError("CAA selection must use only the AB validation metric.")
    elif cfg.evaluation_protocol == "legacy":
        experiment = "jailbreak"
        layers = _registered_jailbreak_layers(cfg.model_alias)
        if cfg.steer_method not in {"kernel", "curveball", "chars", "sas"}:
            raise ValueError("Jailbreak validation supports ActAdd, KAS, Curveball, CHaRS, and SAS only.")
        if cfg.steer_method == "kernel":
            if cfg.kernel_extraction_source != "legacy":
                raise ValueError("Jailbreak ActAdd/KAS require the 128/128 legacy references.")
            if cfg.layer not in layers or not _in_float_grid(cfg.eta, JAILBREAK_STRENGTHS):
                raise ValueError("Jailbreak candidate is outside the model-specific grid.")
        elif cfg.steer_method == "curveball":
            if cfg.layer not in layers or not _in_float_grid(
                cfg.curveball_alpha, JAILBREAK_STRENGTHS
            ):
                raise ValueError("Jailbreak candidate is outside the model-specific grid.")
            if cfg.curveball_extraction_source != "legacy":
                raise ValueError(
                    "Jailbreak Curveball must use the 128/128 legacy reference split."
                )
            if (
                cfg.curveball_kernel != "poly"
                or cfg.curveball_degree != 3
                or not math.isclose(cfg.curveball_coef0, 1.0, abs_tol=1e-12)
                or cfg.curveball_k != 20
                or not math.isclose(
                    cfg.curveball_fit_inverse_alpha, 1.0, abs_tol=1e-12
                )
            ):
                raise ValueError("Curveball validation parameters do not match the fixed plan.")
        elif cfg.steer_method == "chars":
            if cfg.layer not in layers or not _in_float_grid(
                cfg.chars_alpha, JAILBREAK_STRENGTHS
            ):
                raise ValueError("Jailbreak CHaRS candidate is outside the layer/alpha grid.")
            locked_k = chars_validation_clusters(cfg.model_alias)[0]
            if cfg.chars_clusters != locked_k:
                raise ValueError(
                    f"Jailbreak CHaRS validation fixes K={locked_k} for {cfg.model_alias}."
                )
            _assert_fixed_chars_recipe(cfg, experiment)
        elif cfg.steer_method == "sas":
            if cfg.model_alias.casefold() != sas_validation.SAS_MODEL.casefold():
                raise ValueError(f"SAS validation supports only {sas_validation.SAS_MODEL}.")
            if cfg.layer not in sas_validation.SAS_LAYERS:
                raise ValueError("Jailbreak SAS candidate is outside the registered layer grid.")
            sas_validation.assert_fixed_sas_recipe(cfg, experiment)
        if cfg.n_train != 128 or cfg.jailbreak_utility_size != 256:
            raise ValueError(
                "Jailbreak selection fixes 128 harmful/harmless references and 256 utility examples."
            )
        required_split = "validation" if cfg.experiment_phase == "select" else "test"
        if cfg.jailbreak_eval_split != required_split:
            raise ValueError(
                f"{cfg.experiment_phase} phase requires jailbreak_eval_split={required_split}."
            )
    else:
        raise ValueError("select/test phases require evaluation_protocol=caa or legacy.")

    if cfg.experiment_phase == "test":
        if cfg.selection_manifest is None:
            raise ValueError("test phase requires --selection_manifest.")
        if not cfg.evaluation_interventions:
            raise ValueError(
                "test phase requires --evaluation_interventions with one locked method."
            )
        requested = [label for label in cfg.evaluation_interventions if label != "baseline"]
        if len(requested) > 1:
            raise ValueError("Run each validation-selected method in a separate test invocation.")
        manifest = load_manifest(cfg.selection_manifest)
        for label in requested:
            assert_locked_configuration(
                manifest,
                experiment=experiment,
                model=cfg.model_alias,
                method=INTERVENTION_TO_METHOD[label],
                mode=application_mode(cfg.prompt_only),
                layer=cfg.layer,
                strength=method_strength(
                    label,
                    eta=cfg.eta,
                    curveball_alpha=cfg.curveball_alpha,
                    chars_alpha=cfg.chars_alpha,
                    sas_coefficient=cfg.sas_coefficient,
                ),
                n_clusters=(cfg.chars_clusters if label == "chars" else None),
                behavior=cfg.caa_behavior if experiment == "caa" else None,
                prompt_condition=(
                    cfg.caa_eval_system_prompt or "none"
                    if experiment == "caa"
                    else None
                ),
            )
            if label == "sas":
                sas_validation.assert_sas_recipe_configuration(
                    manifest,
                    experiment=experiment,
                    model=cfg.model_alias,
                    coefficient=cfg.sas_coefficient,
                    sae_release=cfg.sae_release,
                    sae_width=cfg.sae_width,
                    feature_freq_threshold=cfg.sas_feature_freq_threshold,
                    extraction_source=cfg.sas_extraction_source,
                    position=cfg.pos,
                    use_delta_correction=cfg.sas_use_delta_correction,
                    apply_post_activation=cfg.sas_apply_post_activation,
                )
            if label == "curveball" and experiment == "caa":
                assert_curveball_caa_recipe_configuration(
                    manifest,
                    model=cfg.model_alias,
                    kernel=cfg.curveball_kernel,
                    degree=cfg.curveball_degree,
                    gamma=cfg.curveball_gamma,
                    coef0=cfg.curveball_coef0,
                    kpca_components=cfg.curveball_k,
                    inverse_transform_alpha=cfg.curveball_fit_inverse_alpha,
                    extraction_source=cfg.curveball_extraction_source,
                    position=cfg.pos,
                )
            if label == "chars":
                assert_chars_recipe_configuration(
                    manifest,
                    experiment=experiment,
                    extraction_source=cfg.chars_extraction_source,
                    model=cfg.model_alias,
                    position=cfg.pos,
                    n_clusters=cfg.chars_clusters,
                )
        return manifest
    return None

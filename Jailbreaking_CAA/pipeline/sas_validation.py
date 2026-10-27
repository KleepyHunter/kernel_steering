"""SAS validation protocol: fixed recipes, strength grids, and locked-manifest checks.

Kept apart from pipeline.validation_selection so the shared selection code never
requires SAS. validation_selection imports this module lazily, only for SAS scopes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from pipeline.validation_selection import CAA_MODEL, CAA_STRENGTHS, JAILBREAK_STRENGTHS

SAS_MODEL = CAA_MODEL
SAS_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAS_LAYERS: Tuple[int, ...] = (22, 23)
SAS_RECIPES: Mapping[str, Mapping[str, Any]] = {
    "caa": {
        "sae_release": SAS_SAE_RELEASE,
        "sae_width": "131k",
        "feature_freq_threshold": 0.7,
        "extraction_source": "caa",
        "position": -2,
        "use_delta_correction": True,
        "apply_post_activation": True,
    },
    "jailbreak": {
        "sae_release": SAS_SAE_RELEASE,
        "sae_width": "16k",
        "feature_freq_threshold": 0.7,
        "extraction_source": "legacy",
        "position": -1,
        "use_delta_correction": True,
        "apply_post_activation": True,
    },
}
SAS_COEFFICIENTS: Mapping[str, Tuple[float, ...]] = {
    "caa": CAA_STRENGTHS,
    "jailbreak": tuple(-strength for strength in JAILBREAK_STRENGTHS),
}


def sas_recipe(experiment: str) -> Dict[str, Any]:
    try:
        return dict(SAS_RECIPES[experiment])
    except KeyError as exc:
        raise ValueError(f"SAS is not registered for experiment={experiment!r}.") from exc


def sas_coefficients(experiment: str) -> Tuple[float, ...]:
    try:
        return tuple(float(value) for value in SAS_COEFFICIENTS[experiment])
    except KeyError as exc:
        raise ValueError(f"SAS is not registered for experiment={experiment!r}.") from exc


def validate_sas_candidate_recipe(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
) -> None:
    expected = sas_recipe(experiment)
    for record in records:
        if str(record.get("model", "")).casefold() != SAS_MODEL.casefold():
            raise ValueError(
                f"SAS validation supports only {SAS_MODEL}; found {record.get('model')!r}."
            )
        if record.get("strength_parameter") != "sas_coefficient":
            raise ValueError("SAS candidates must declare strength_parameter=sas_coefficient.")
        actual = record.get("fixed_parameters", {}).get("sas")
        if not isinstance(actual, Mapping):
            raise ValueError("SAS candidates must include fixed_parameters.sas.")
        fixed_recipe = {key: actual.get(key) for key in expected}
        if fixed_recipe != expected:
            raise ValueError(
                f"SAS candidate recipe mismatch for experiment={experiment}: "
                f"expected={expected}; found={fixed_recipe}."
            )
        coefficient = float(record["strength"])
        if not any(
            math.isclose(coefficient, allowed, rel_tol=0.0, abs_tol=1e-12)
            for allowed in sas_coefficients(experiment)
        ):
            raise ValueError(
                f"SAS coefficient {coefficient} is outside the {experiment} grid."
            )
        if "coefficient" in actual and not math.isclose(
            float(actual["coefficient"]), coefficient, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "Legacy SAS candidate coefficient does not match candidate strength."
            )


def assert_sas_recipe_configuration(
    manifest: Mapping[str, Any],
    *,
    experiment: str,
    model: str,
    coefficient: float,
    sae_release: Optional[str],
    sae_width: str,
    feature_freq_threshold: float,
    extraction_source: str,
    position: int,
    use_delta_correction: bool,
    apply_post_activation: bool,
) -> Mapping[str, Any]:
    if manifest.get("experiment") != experiment:
        raise ValueError(
            f"Manifest is for experiment={manifest.get('experiment')}, not {experiment}."
        )
    if model.casefold() != SAS_MODEL.casefold():
        raise ValueError(f"SAS validation supports only {SAS_MODEL}; found {model!r}.")

    expected = sas_recipe(experiment)
    manifest_recipe = manifest.get("grid", {}).get("sas")
    if not isinstance(manifest_recipe, Mapping):
        raise ValueError("Manifest does not contain a locked SAS recipe.")
    locked_recipe = {key: manifest_recipe.get(key) for key in expected}
    if locked_recipe != expected:
        raise ValueError(
            f"Manifest SAS recipe mismatch: expected={expected}; found={locked_recipe}."
        )

    declared_coefficients = manifest_recipe.get("coefficients")
    if declared_coefficients is None and "coefficient" in manifest_recipe:
        declared_coefficients = [manifest_recipe["coefficient"]]
    if not isinstance(declared_coefficients, Sequence) or isinstance(
        declared_coefficients, (str, bytes)
    ):
        raise ValueError("Manifest does not contain a locked SAS coefficient grid.")
    if not any(
        math.isclose(
            float(coefficient), float(allowed), rel_tol=0.0, abs_tol=1e-12
        )
        for allowed in declared_coefficients
    ):
        raise ValueError(
            f"SAS coefficient {coefficient} is not present in the locked manifest grid."
        )

    actual = {
        "sae_release": sae_release or SAS_SAE_RELEASE,
        "sae_width": str(sae_width),
        "feature_freq_threshold": float(feature_freq_threshold),
        "extraction_source": str(extraction_source),
        "position": int(position),
        "use_delta_correction": bool(use_delta_correction),
        "apply_post_activation": bool(apply_post_activation),
    }
    if actual != expected:
        raise ValueError(
            f"SAS configuration does not match the validation-locked recipe: "
            f"expected={expected}; requested={actual}."
        )
    return manifest_recipe


def assert_fixed_sas_recipe(cfg, experiment):
    expected = sas_recipe(experiment)
    actual = {
        "sae_release": cfg.sae_release or expected["sae_release"],
        "sae_width": str(cfg.sae_width),
        "feature_freq_threshold": float(cfg.sas_feature_freq_threshold),
        "extraction_source": str(cfg.sas_extraction_source),
        "position": int(cfg.pos),
        "use_delta_correction": bool(cfg.sas_use_delta_correction),
        "apply_post_activation": bool(cfg.sas_apply_post_activation),
    }
    if actual != expected:
        raise ValueError(
            f"SAS {experiment} validation requires the fixed recipe "
            f"{expected}; requested={actual}."
        )
    if not _in_grid(cfg.sas_coefficient, SAS_COEFFICIENTS[experiment]):
        raise ValueError(
            f"SAS {experiment} coefficient {cfg.sas_coefficient} is outside "
            f"the registered comparison grid {SAS_COEFFICIENTS[experiment]}."
        )


def sas_fixed_parameters(cfg):
    return {
        "sae_release": cfg.sae_release,
        "sae_width": str(cfg.sae_width),
        "feature_freq_threshold": float(cfg.sas_feature_freq_threshold),
        "extraction_source": str(cfg.sas_extraction_source),
        "position": int(cfg.pos),
        "use_delta_correction": bool(cfg.sas_use_delta_correction),
        "apply_post_activation": bool(cfg.sas_apply_post_activation),
    }


def sas_job_args(experiment: str, coefficient: float) -> list[str]:
    recipe = SAS_RECIPES[experiment]
    locked_coefficient = float(coefficient)
    if not any(
        math.isclose(locked_coefficient, allowed, rel_tol=0.0, abs_tol=1e-12)
        for allowed in SAS_COEFFICIENTS[experiment]
    ):
        raise ValueError(
            f"SAS coefficient {coefficient} is outside the {experiment} "
            f"comparison grid {SAS_COEFFICIENTS[experiment]}."
        )
    return [
        "--steer_method", "sas",
        "--pos", str(recipe["position"]),
        "--sae_release", str(recipe["sae_release"]),
        "--sae_width", str(recipe["sae_width"]),
        "--sas_feature_freq_threshold", str(recipe["feature_freq_threshold"]),
        "--sas_extraction_source", str(recipe["extraction_source"]),
        "--sas_coefficient", str(locked_coefficient),
        "--sas_use_delta_correction", str(recipe["use_delta_correction"]).lower(),
        "--sas_apply_post_activation", str(recipe["apply_post_activation"]).lower(),
    ]


def _in_grid(value, grid) -> bool:
    return any(
        math.isclose(float(value), float(item), rel_tol=0.0, abs_tol=1e-12)
        for item in grid
    )


def sas_manifest_grid(experiment: str) -> Dict[str, Any]:
    return {
        **sas_recipe(experiment),
        "models": [SAS_MODEL],
        "layers": list(SAS_LAYERS),
        "coefficients": list(sas_coefficients(experiment)),
        "strength_parameter": "sas_coefficient",
    }


def sas_show_grid(experiment: str) -> Dict[str, Any]:
    return {
        **sas_recipe(experiment),
        "coefficients": list(sas_coefficients(experiment)),
        "strength_parameter": "sas_coefficient",
    }


def validate_sas_steer_config(config, *, layer: int, manifest, model: str) -> None:
    recipe = SAS_RECIPES["jailbreak"]
    assert_sas_recipe_configuration(
        manifest,
        experiment="jailbreak",
        model=model,
        coefficient=config["steering_coefficient"],
        sae_release=config.get("release"),
        sae_width=recipe["sae_width"],
        feature_freq_threshold=config.get("tau"),
        extraction_source=recipe["extraction_source"],
        position=recipe["position"],
        use_delta_correction=config.get("use_delta_correction", False),
        apply_post_activation=config.get("apply_post_activation", False),
    )
    expected_sae_id = f"layer_{layer}/width_{recipe['sae_width']}/canonical"
    if config.get("sae_id") != expected_sae_id:
        raise ValueError(
            f"Serialized SAS sae_id must be {expected_sae_id}; "
            f"found {config.get('sae_id')}."
        )
    if int(config.get("layer", -1)) != int(layer):
        raise ValueError("Serialized SAS layer does not match the manifest.")

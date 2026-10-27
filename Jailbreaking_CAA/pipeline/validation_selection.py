"""Validation-only hyperparameter selection and locked-test utilities.

This module intentionally has no torch/transformers dependency.  It provides the
experiment specification, deterministic split helpers, candidate aggregation,
immutable selection manifests, and paired bootstrap confidence intervals used by
the CAA-style and jailbreak rebuttal experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)

CAA_BEHAVIORS: Tuple[str, ...] = (
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
    "survival-instinct",
    "sycophancy",
)
CAA_LAYERS: Tuple[int, ...] = (22, 23)
CAA_STRENGTHS: Tuple[float, ...] = (
    0.0,
    0.01,
    0.1,
    0.25,
    1.0,
    2.5,
    4.5,
    5.0,
    10.0,
    17.5,
    20.0,
    50.0,
    100.0,
    200.0,
    250.0,
    275.0,
    300.0,
    400.0,
    435.0,
    750.0,
)

JAILBREAK_STRENGTHS: Tuple[float, ...] = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    40.0,
    50.0,
    100.0,
    150.0,
    200.0,
)
JAILBREAK_LAYERS: Mapping[str, Tuple[int, ...]] = {
    "gemma-2-9b-it": (22, 23),
    "Llama-3.1-8B-Instruct": (12, 13),
    "Qwen2.5-7B-Instruct": (17, 18),
}

CAA_MODEL = "gemma-2-9b-it"

CURVEBALL_CAA_RECIPE: Mapping[str, Any] = {
    "kernel": "poly",
    "degree": 3,
    "gamma": None,
    "coef0": 1.0,
    "kpca_components": 20,
    "inverse_transform_alpha": 1.0,
    "extraction_source": "caa",
    "position": -2,
}

CHARS_VALID_CLUSTERS: Tuple[int, ...] = tuple(range(1, 16))

# The paper does not define one universal cluster count. For activation
# addition it fixes K=10 for Gemma2-9B-Instruct (Appendix E.5), while the
# Qwen2.5-7B-Instruct OT-coupling experiment reports K=5 (Table 5). It reports
# Llama3.1-8B-Instruct results without disclosing its selected K, so that model
# retains the repository's deterministic K=5 default rather than tuning K.
CHARS_VALIDATION_CLUSTERS: Mapping[str, int] = {
    "gemma-2-9b-it": 10,
    "Llama-3.1-8B-Instruct": 5,
    "Qwen2.5-7B-Instruct": 5,
}
CHARS_CLUSTER_PROVENANCE: Mapping[str, str] = {
    "gemma-2-9b-it": "paper_appendix_e5_actadd",
    "Llama-3.1-8B-Instruct": "repository_default_paper_value_unreported",
    "Qwen2.5-7B-Instruct": "paper_table_5_ot_coupling",
}
_CHARS_COMMON_RECIPE: Mapping[str, Any] = {
    "name": "chars_equal_cov_v2",
    "hookpoint": "block_output",
    "kmeans": {
        "seed": 42,
        "algorithm": "lloyd",
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4,
        "threads": 1,
    },
    "transport": {
        "cost": "squared_euclidean",
        "cost_normalization": "positive_median",
        "regularization": 0.05,
        "solver": "log_sinkhorn",
        "max_iter": 1000,
        "marginal_tolerance": 1e-6,
        "dtype": "float32",
    },
    "gating": {
        "kernel": "rbf",
        "sigma_squared": "per_token_median_source_centroid_distance_squared",
        "bandwidth_floor": 1e-12,
    },
    "local_map": "equal_covariance_translation",
}
CHARS_RECIPES: Mapping[str, Mapping[str, Any]] = {
    "caa": {
        **_CHARS_COMMON_RECIPE,
        "extraction_source": "caa",
        "position": -2,
        "orientation": "behavior_not_matching_to_behavior_matching",
    },
    "jailbreak": {
        **_CHARS_COMMON_RECIPE,
        "extraction_source": "legacy",
        "position": -1,
        "orientation": "harmful_to_harmless",
    },
}


def chars_validation_clusters(model_alias: str) -> Tuple[int, ...]:
    """Return the single validation-locked K for a registered model."""

    for known, n_clusters in CHARS_VALIDATION_CLUSTERS.items():
        if known.casefold() == str(model_alias).casefold():
            return (int(n_clusters),)
    raise ValueError(
        f"No validation-locked CHaRS K for model {model_alias!r}; "
        f"expected one of {list(CHARS_VALIDATION_CLUSTERS)}."
    )


def chars_cluster_provenance(model_alias: str) -> str:
    for known, provenance in CHARS_CLUSTER_PROVENANCE.items():
        if known.casefold() == str(model_alias).casefold():
            return str(provenance)
    raise ValueError(
        f"No CHaRS K provenance for model {model_alias!r}; "
        f"expected one of {list(CHARS_CLUSTER_PROVENANCE)}."
    )

INTERVENTION_TO_METHOD: Mapping[str, str] = {
    "baseline": "baseline",
    "actadd": "actadd",
    "kernel": "kas",
    "curveball": "curveball",
    "chars": "chars",
    "sas": "sas",
}
METHOD_TO_INTERVENTION: Mapping[str, str] = {
    value: key for key, value in INTERVENTION_TO_METHOD.items()
}


def canonical_question(question: str) -> str:
    """Normalize a question for exact duplicate/overlap checks."""

    return " ".join(str(question).split()).casefold()


def stable_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint_json(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def fingerprint_records(
    records: Sequence[Mapping[str, Any]],
    *,
    fields: Optional[Sequence[str]] = None,
) -> str:
    """Fingerprint records independently of their input ordering."""

    if fields is None:
        normalized = [dict(record) for record in records]
    else:
        normalized = [
            {field: record.get(field) for field in fields}
            for record in records
        ]
    normalized.sort(key=stable_json_dumps)
    return fingerprint_json(normalized)


def _derived_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def partition_caa_generation_records(
    generation_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    *,
    behavior: str,
    validation_size: int = 50,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Create disjoint CAA reference and validation partitions.

    Exact duplicate generation questions are collapsed, and any generation
    question appearing in the published test set is excluded before sampling.
    Sampling is deterministic per behavior and independent of JSON file order.
    """

    if validation_size <= 0:
        raise ValueError("validation_size must be positive.")

    test_questions = {
        canonical_question(record["question"])
        for record in test_records
    }

    unique_generation: Dict[str, Dict[str, Any]] = {}
    duplicate_count = 0
    test_overlap_count = 0
    for record in generation_records:
        if "question" not in record:
            raise ValueError("Every CAA generation record must contain 'question'.")
        question_key = canonical_question(record["question"])
        if question_key in test_questions:
            test_overlap_count += 1
            continue
        if question_key in unique_generation:
            duplicate_count += 1
            continue
        unique_generation[question_key] = dict(record)

    if len(unique_generation) <= validation_size:
        raise ValueError(
            f"Behavior '{behavior}' has {len(unique_generation)} eligible generation "
            f"questions, insufficient for validation_size={validation_size} plus references."
        )

    ordered_keys = sorted(unique_generation)
    rng = random.Random(_derived_seed(seed, behavior))
    validation_keys = set(rng.sample(ordered_keys, validation_size))

    reference = [
        unique_generation[key]
        for key in ordered_keys
        if key not in validation_keys
    ]
    validation = [
        unique_generation[key]
        for key in ordered_keys
        if key in validation_keys
    ]

    reference_questions = {
        canonical_question(record["question"]) for record in reference
    }
    validation_questions = {
        canonical_question(record["question"]) for record in validation
    }
    if reference_questions & validation_questions:
        raise AssertionError("CAA reference and validation partitions overlap.")
    if reference_questions & test_questions:
        raise AssertionError("CAA reference and test partitions overlap.")
    if validation_questions & test_questions:
        raise AssertionError("CAA validation and test partitions overlap.")

    metadata = {
        "behavior": behavior,
        "seed": seed,
        "validation_size": validation_size,
        "n_generation_input": len(generation_records),
        "n_reference": len(reference),
        "n_validation": len(validation),
        "n_test": len(test_records),
        "duplicates_removed": duplicate_count,
        "test_overlaps_removed": test_overlap_count,
        "fingerprints": {
            "reference": fingerprint_records(reference),
            "validation": fingerprint_records(validation),
            "test": fingerprint_records(list(test_records)),
        },
    }
    return reference, validation, metadata


def _sas():
    # Lazy: pipeline.sas_validation imports this module's grids, and SAS is only
    # needed when SAS candidates or manifests are actually present.
    from pipeline import sas_validation

    return sas_validation


def application_mode(prompt_only: bool) -> str:
    return "prompt_only" if prompt_only else "all_tokens"


def method_strength(
    intervention_label: str,
    *,
    eta: float,
    curveball_alpha: float,
    chars_alpha: float = 1.0,
    sas_coefficient: float = -1.0,
) -> Optional[float]:
    if intervention_label == "baseline":
        return None
    if intervention_label == "sas":
        return float(sas_coefficient)
    if intervention_label == "chars":
        return float(chars_alpha)
    if intervention_label == "curveball":
        return float(curveball_alpha)
    return float(eta)


def _candidate_scope(
    record: Mapping[str, Any],
    *,
    experiment: str,
) -> Tuple[Any, ...]:
    if experiment == "caa":
        return (
            record["model"],
            record["behavior"],
            record["prompt_condition"],
            record["application_mode"],
            record["method"],
        )
    if experiment == "jailbreak":
        return (
            record["model"],
            record["application_mode"],
            record["method"],
        )
    raise ValueError(f"Unknown experiment: {experiment}")


def _scope_dict(scope: Tuple[Any, ...], *, experiment: str) -> Dict[str, Any]:
    if experiment == "caa":
        model, behavior, prompt_condition, mode, method = scope
        return {
            "model": model,
            "behavior": behavior,
            "prompt_condition": prompt_condition,
            "application_mode": mode,
            "method": method,
        }
    model, mode, method = scope
    return {
        "model": model,
        "application_mode": mode,
        "method": method,
    }


def _expected_pairs(
    *,
    layers: Sequence[int],
    strengths: Sequence[float],
) -> set[Tuple[int, float]]:
    return {
        (int(layer), float(strength))
        for layer in layers
        for strength in strengths
    }


def _actual_pairs(records: Sequence[Mapping[str, Any]]) -> set[Tuple[int, float]]:
    return {
        (int(record["layer"]), float(record["strength"]))
        for record in records
    }


def _validate_complete_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
    strengths: Sequence[float],
    scope: Mapping[str, Any],
) -> None:
    expected = _expected_pairs(layers=layers, strengths=strengths)
    actual = _actual_pairs(records)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if len(actual) != len(records):
        raise ValueError(f"Duplicate layer/strength candidates for {dict(scope)}.")
    if missing or unexpected:
        raise ValueError(
            "Incomplete or unexpected candidate grid for "
            f"{dict(scope)}. Missing={missing}; unexpected={unexpected}."
        )


def _validate_complete_chars_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
    strengths: Sequence[float],
    clusters: Sequence[int],
    scope: Mapping[str, Any],
) -> None:
    expected = {
        (int(layer), float(strength), int(n_clusters))
        for layer in layers
        for strength in strengths
        for n_clusters in clusters
    }
    actual = {
        (
            int(record["layer"]),
            float(record["strength"]),
            int(record["n_clusters"]),
        )
        for record in records
    }
    if len(actual) != len(records):
        raise ValueError(
            f"Duplicate layer/strength/K CHaRS candidates for {dict(scope)}."
        )
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "Incomplete or unexpected CHaRS candidate grid for "
            f"{dict(scope)}. Missing={missing}; unexpected={unexpected}."
        )


def _sorted_candidate_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda record: stable_json_dumps(
            {
                "experiment": record.get("experiment"),
                "model": record.get("model"),
                "behavior": record.get("behavior"),
                "prompt_condition": record.get("prompt_condition"),
                "application_mode": record.get("application_mode"),
                "method": record.get("method"),
                "layer": record.get("layer"),
                "strength": record.get("strength"),
                "n_clusters": record.get("n_clusters"),
            }
        ),
    )


def _common_split_fingerprints(records: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    fingerprints = {
        stable_json_dumps(record.get("split_fingerprints", {}))
        for record in records
    }
    if len(fingerprints) != 1:
        raise ValueError("Candidate records disagree on split fingerprints.")
    return json.loads(next(iter(fingerprints)))


def _validate_curveball_caa_candidate_recipe(
    records: Sequence[Mapping[str, Any]],
) -> None:
    expected = dict(CURVEBALL_CAA_RECIPE)
    for record in records:
        if str(record.get("model", "")).casefold() != CAA_MODEL.casefold():
            raise ValueError(
                f"CAA Curveball validation supports only {CAA_MODEL}; "
                f"found {record.get('model')!r}."
            )
        if record.get("strength_parameter") != "curveball_alpha":
            raise ValueError(
                "CAA Curveball candidates must declare "
                "strength_parameter=curveball_alpha."
            )
        actual = record.get("fixed_parameters", {}).get("curveball")
        if actual != expected:
            raise ValueError(
                "CAA Curveball candidate recipe mismatch: "
                f"expected={expected}; found={actual}."
            )


def _chars_recipe(experiment: str) -> Dict[str, Any]:
    try:
        return dict(CHARS_RECIPES[experiment])
    except KeyError as exc:
        raise ValueError(
            f"CHaRS is not registered for experiment={experiment!r}."
        ) from exc


def _validate_chars_candidate_recipe(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
) -> None:
    expected = _chars_recipe(experiment)
    fingerprints_by_state: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for record in records:
        if record.get("strength_parameter") != "chars_alpha":
            raise ValueError(
                "CHaRS candidates must declare strength_parameter=chars_alpha."
            )
        n_clusters = record.get("n_clusters")
        if isinstance(n_clusters, bool) or int(n_clusters) not in CHARS_VALID_CLUSTERS:
            raise ValueError(f"Invalid CHaRS candidate K={n_clusters!r}.")
        locked_clusters = chars_validation_clusters(str(record.get("model", "")))
        if int(n_clusters) not in locked_clusters:
            raise ValueError(
                f"CHaRS candidate K={n_clusters} does not match the "
                f"validation-locked model value {locked_clusters[0]}."
            )
        actual = record.get("fixed_parameters", {}).get("chars")
        if actual != expected:
            raise ValueError(
                f"CHaRS {experiment} recipe mismatch: "
                f"expected={expected}; found={actual}."
            )
        fingerprint = record.get("state_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("CHaRS candidates require a canonical SHA-256 state fingerprint.")
        fingerprints_by_state[
            (int(record["layer"]), int(n_clusters))
        ].add(fingerprint)
    for state_key, fingerprints in fingerprints_by_state.items():
        if len(fingerprints) != 1:
            raise ValueError(
                "CHaRS candidates did not reuse one state across strengths for "
                f"layer/K={state_key}: {sorted(fingerprints)}."
            )


def _validate_chars_cross_scope_reuse(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
) -> None:
    grouped: Dict[Tuple[Any, ...], set[str]] = defaultdict(set)
    for record in records:
        if record.get("method") != "chars":
            continue
        if experiment == "caa":
            key = (
                record["model"],
                record["behavior"],
                int(record["layer"]),
                int(record["n_clusters"]),
            )
        else:
            key = (
                record["model"],
                int(record["layer"]),
                int(record["n_clusters"]),
            )
        grouped[key].add(str(record.get("state_fingerprint")))
    for state_key, fingerprints in grouped.items():
        if len(fingerprints) != 1:
            reuse_scope = "prompt conditions" if experiment == "caa" else "application modes"
            raise ValueError(
                f"CHaRS state was not reused across {reuse_scope} for "
                f"{state_key}: {sorted(fingerprints)}."
            )


def _validate_required_method_scopes(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
    require_chars: bool = False,
    require_sas: bool = False,
) -> None:
    grouped: Dict[Tuple[Any, ...], set[str]] = defaultdict(set)
    for record in records:
        if experiment == "caa":
            scope = (
                record["model"],
                record["behavior"],
                record["prompt_condition"],
                record["application_mode"],
            )
        else:
            scope = (record["model"], record["application_mode"])
        grouped[scope].add(record["method"])

    for scope, methods in grouped.items():
        if methods == {"chars"}:
            # The dedicated CHaRS campaign is intentionally method-filtered,
            # but its complete layer/strength/K grid is still mandatory.
            expected = {"chars"}
        elif methods == {"sas"}:
            # Same for the dedicated SAS campaign.
            expected = {"sas"}
        else:
            expected = {"actadd", "kas", "curveball"}
            if (
                require_sas
                and str(scope[0]).casefold() == _sas().SAS_MODEL.casefold()
            ):
                expected.add("sas")
            if require_chars:
                expected.add("chars")
        missing = sorted(expected - methods)
        unexpected = sorted(methods - expected)
        if missing or unexpected:
            raise ValueError(
                f"Incomplete or unexpected method coverage for {experiment} scope "
                f"{scope}. Missing={missing}; unexpected={unexpected}."
            )


def select_caa_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    require_complete_grid: bool = True,
) -> Dict[str, Any]:
    """Select CAA layer/strength using validation target probability only."""

    steered_records = [
        dict(record)
        for record in records
        if record.get("experiment") == "caa"
        and record.get("method") in {"actadd", "kas", "curveball", "chars", "sas"}
        and record.get("split") == "validation"
    ]
    if not steered_records:
        raise ValueError("No CAA validation candidate records were provided.")
    has_chars = any(record["method"] == "chars" for record in steered_records)
    has_sas = any(record["method"] == "sas" for record in steered_records)
    if has_chars:
        _validate_chars_cross_scope_reuse(steered_records, experiment="caa")
    if require_complete_grid:
        _validate_required_method_scopes(
            steered_records,
            experiment="caa",
            require_chars=has_chars,
            require_sas=has_sas,
        )

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for record in steered_records:
        metric = float(record["validation_target_probability"])
        if not math.isfinite(metric):
            raise ValueError(f"Non-finite CAA validation metric: {record}")
        grouped[_candidate_scope(record, experiment="caa")].append(record)

    selections: List[Dict[str, Any]] = []
    for scope, candidates in sorted(grouped.items(), key=lambda item: item[0]):
        scope_fields = _scope_dict(scope, experiment="caa")
        is_curveball = scope_fields["method"] == "curveball"
        is_chars = scope_fields["method"] == "chars"
        is_sas = scope_fields["method"] == "sas"
        if is_sas:
            _sas().validate_sas_candidate_recipe(candidates, experiment="caa")
        if is_curveball:
            _validate_curveball_caa_candidate_recipe(candidates)
        if is_chars:
            _validate_chars_candidate_recipe(candidates, experiment="caa")
        if require_complete_grid:
            if is_chars:
                _validate_complete_chars_grid(
                    candidates,
                    layers=CAA_LAYERS,
                    strengths=CAA_STRENGTHS,
                    clusters=chars_validation_clusters(scope_fields["model"]),
                    scope=scope_fields,
                )
            else:
                _validate_complete_grid(
                    candidates,
                    layers=_sas().SAS_LAYERS if is_sas else CAA_LAYERS,
                    strengths=(
                        _sas().sas_coefficients("caa") if is_sas else CAA_STRENGTHS
                    ),
                    scope=scope_fields,
                )

        selected = min(
            candidates,
            key=lambda record: (
                -float(record["validation_target_probability"]),
                float(record["strength"]),
                int(record.get("n_clusters", 0)),
                int(record["layer"]),
            ),
        )
        selections.append(
            {
                **scope_fields,
                "split_fingerprints": _common_split_fingerprints(candidates),
                **(
                    {
                        "strength_parameter": {
                            "curveball": "curveball_alpha",
                            "chars": "chars_alpha",
                            "sas": "sas_coefficient",
                        }[scope_fields["method"]]
                    }
                    if scope_fields["method"] in {"curveball", "chars", "sas"}
                    else {}
                ),
                "selected": {
                    "layer": int(selected["layer"]),
                    "strength": float(selected["strength"]),
                    **(
                        {"n_clusters": int(selected["n_clusters"])}
                        if is_chars else {}
                    ),
                },
                **(
                    {"state_fingerprint": selected["state_fingerprint"]}
                    if is_chars else {}
                ),
                "validation": {
                    "average_target_probability": float(
                        selected["validation_target_probability"]
                    ),
                    "n_examples": int(selected["n_validation_examples"]),
                },
                "tie_break": [
                    "higher_average_target_probability",
                    "lower_strength",
                    "lower_K" if is_chars else "lower_layer",
                    *( ["lower_layer"] if is_chars else [] ),
                ],
                "candidate_count": len(candidates),
            }
        )

    return {
        "schema_version": 3 if has_chars else 1,
        "experiment": "caa",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "selection_rule": "maximize_average_target_probability",
        "grid": {
            "layers": list(CAA_LAYERS),
            "strengths": list(CAA_STRENGTHS),
            "step_count": 1,
            "kernel": "rbf",
            "kernel_bandwidth": "reference_median_heuristic",
            "lambda_reg": 0.0,
            "curveball": {
                **dict(CURVEBALL_CAA_RECIPE),
                "models": [CAA_MODEL],
                "layers": list(CAA_LAYERS),
                "alphas": list(CAA_STRENGTHS),
                "strength_parameter": "curveball_alpha",
            },
            **(
                {
                    "chars": {
                        **_chars_recipe("caa"),
                        "layers": list(CAA_LAYERS),
                        "alphas": list(CAA_STRENGTHS),
                        "clusters": list(chars_validation_clusters(CAA_MODEL)),
                        "clusters_by_model": {
                            CAA_MODEL: list(chars_validation_clusters(CAA_MODEL))
                        },
                        "cluster_provenance_by_model": {
                            CAA_MODEL: chars_cluster_provenance(CAA_MODEL)
                        },
                        "strength_parameter": "chars_alpha",
                    }
                }
                if has_chars else {}
            ),
            **({"sas": _sas().sas_manifest_grid("caa")} if has_sas else {}),
        },
        "split_fingerprints": {
            "|".join(
                [
                    selection["model"],
                    selection["behavior"],
                    selection["prompt_condition"],
                    selection["application_mode"],
                ]
            ): selection["split_fingerprints"]
            for selection in selections
        },
        "selections": selections,
        "validation_candidates": _sorted_candidate_records(steered_records),
    }


def _model_layer_grid(model_alias: str) -> Tuple[int, ...]:
    if model_alias in JAILBREAK_LAYERS:
        return JAILBREAK_LAYERS[model_alias]
    lowered = model_alias.casefold()
    for known_model, layers in JAILBREAK_LAYERS.items():
        if known_model.casefold() == lowered:
            return layers
    raise ValueError(
        f"No jailbreak layer grid registered for '{model_alias}'. "
        f"Known models: {list(JAILBREAK_LAYERS)}"
    )


def select_jailbreak_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    utility_cap_ratio: float = 1.05,
    require_complete_grid: bool = True,
) -> Dict[str, Any]:
    """Select jailbreak configurations under the benign-perplexity cap."""

    if utility_cap_ratio < 1.0:
        raise ValueError("utility_cap_ratio must be at least 1.0.")

    validation_records = [
        dict(record)
        for record in records
        if record.get("experiment") == "jailbreak"
        and record.get("split") == "validation"
    ]
    if not validation_records:
        raise ValueError("No jailbreak validation candidate records were provided.")

    baseline_ppl: Dict[Tuple[str, str], float] = {}
    for record in validation_records:
        if record.get("method") != "baseline":
            continue
        key = (record["model"], record["application_mode"])
        ppl = float(record["validation_perplexity"])
        previous = baseline_ppl.get(key)
        if previous is not None and not math.isclose(
            previous, ppl, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"Inconsistent baseline perplexity for model/mode {key}: "
                f"{previous} versus {ppl}."
            )
        baseline_ppl[key] = ppl

    steered_records = [
        record
        for record in validation_records
        if record.get("method") in {"actadd", "kas", "curveball", "chars", "sas"}
    ]
    has_chars = any(record["method"] == "chars" for record in steered_records)
    has_sas = any(record["method"] == "sas" for record in steered_records)
    if has_chars:
        _validate_chars_cross_scope_reuse(steered_records, experiment="jailbreak")
    if require_complete_grid:
        _validate_required_method_scopes(
            steered_records,
            experiment="jailbreak",
            require_chars=has_chars,
            require_sas=has_sas,
        )
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for record in steered_records:
        asr = float(record["validation_asr"])
        ppl = float(record["validation_perplexity"])
        if not math.isfinite(asr) or not math.isfinite(ppl):
            raise ValueError(f"Non-finite jailbreak validation metric: {record}")
        grouped[_candidate_scope(record, experiment="jailbreak")].append(record)

    selections: List[Dict[str, Any]] = []
    for scope, candidates in sorted(grouped.items(), key=lambda item: item[0]):
        scope_fields = _scope_dict(scope, experiment="jailbreak")
        baseline_key = (
            scope_fields["model"],
            scope_fields["application_mode"],
        )
        if baseline_key not in baseline_ppl:
            raise ValueError(f"Missing baseline validation perplexity for {baseline_key}.")
        baseline = baseline_ppl[baseline_key]
        cap = utility_cap_ratio * baseline

        is_chars = scope_fields["method"] == "chars"
        is_sas = scope_fields["method"] == "sas"
        if is_sas:
            _sas().validate_sas_candidate_recipe(candidates, experiment="jailbreak")
        if is_chars:
            _validate_chars_candidate_recipe(candidates, experiment="jailbreak")
        if require_complete_grid:
            if is_chars:
                _validate_complete_chars_grid(
                    candidates,
                    layers=_model_layer_grid(scope_fields["model"]),
                    strengths=JAILBREAK_STRENGTHS,
                    clusters=chars_validation_clusters(scope_fields["model"]),
                    scope=scope_fields,
                )
            else:
                _validate_complete_grid(
                    candidates,
                    layers=(
                        _sas().SAS_LAYERS
                        if is_sas
                        else _model_layer_grid(scope_fields["model"])
                    ),
                    strengths=(
                        _sas().sas_coefficients("jailbreak")
                        if is_sas
                        else JAILBREAK_STRENGTHS
                    ),
                    scope=scope_fields,
                )

        admissible = [
            record
            for record in candidates
            if float(record["validation_perplexity"]) <= cap
        ]
        if not admissible:
            selections.append(
                {
                    **scope_fields,
                    **(
                        {"strength_parameter": "chars_alpha"}
                        if is_chars
                        else ({"strength_parameter": "sas_coefficient"} if is_sas else {})
                    ),
                    "selected": None,
                    "reason": "no_admissible_configuration",
                    "validation": {
                        "baseline_perplexity": baseline,
                        "perplexity_cap": cap,
                        "utility_cap_ratio": utility_cap_ratio,
                    },
                    "candidate_count": len(candidates),
                    "admissible_candidate_count": 0,
                }
            )
            continue

        selected = min(
            admissible,
            key=lambda record: (
                -float(record["validation_asr"]),
                float(record["validation_perplexity"]),
                # SAS coefficients are negative, so smaller means smaller magnitude.
                abs(float(record["strength"])) if is_sas else float(record["strength"]),
                int(record.get("n_clusters", 0)),
                int(record["layer"]),
            ),
        )
        selections.append(
            {
                **scope_fields,
                **(
                    {"strength_parameter": "chars_alpha"}
                    if is_chars
                    else ({"strength_parameter": "sas_coefficient"} if is_sas else {})
                ),
                "selected": {
                    "layer": int(selected["layer"]),
                    "strength": float(selected["strength"]),
                    **(
                        {"n_clusters": int(selected["n_clusters"])}
                        if is_chars else {}
                    ),
                },
                **(
                    {"state_fingerprint": selected["state_fingerprint"]}
                    if is_chars else {}
                ),
                "validation": {
                    "asr": float(selected["validation_asr"]),
                    "perplexity": float(selected["validation_perplexity"]),
                    "baseline_perplexity": baseline,
                    "perplexity_cap": cap,
                    "utility_cap_ratio": utility_cap_ratio,
                    "n_harmful_examples": int(selected["n_harmful_validation_examples"]),
                    "n_harmless_examples": int(selected["n_harmless_validation_examples"]),
                },
                "tie_break": [
                    "higher_asr",
                    "lower_perplexity",
                    "lower_strength_magnitude" if is_sas else "lower_strength",
                    *( ["lower_K"] if is_chars else [] ),
                    "lower_layer",
                ],
                "candidate_count": len(candidates),
                "admissible_candidate_count": len(admissible),
            }
        )

    split_records = steered_records or validation_records
    return {
        "schema_version": 3 if has_chars else 1,
        "experiment": "jailbreak",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "selection_rule": "maximize_asr_subject_to_perplexity_cap",
        "utility_cap_ratio": utility_cap_ratio,
        "grid": {
            "layers_by_model": {
                model: list(layers)
                for model, layers in JAILBREAK_LAYERS.items()
            },
            "strengths": list(JAILBREAK_STRENGTHS),
            "kas": {
                "kernel": "rbf",
                "kernel_bandwidth": "reference_median_heuristic",
                "lambda_reg": 0.0,
                "step_count": 1,
            },
            "curveball": {
                "kernel": "polynomial",
                "degree": 3,
                "bias": 1.0,
                "kpca_components": 20,
            },
            **(
                {
                    "chars": {
                        **_chars_recipe("jailbreak"),
                        "layers_by_model": {
                            model: list(layers)
                            for model, layers in JAILBREAK_LAYERS.items()
                        },
                        "alphas": list(JAILBREAK_STRENGTHS),
                        "clusters": sorted(set(CHARS_VALIDATION_CLUSTERS.values())),
                        "clusters_by_model": {
                            model: list(chars_validation_clusters(model))
                            for model in JAILBREAK_LAYERS
                        },
                        "cluster_provenance_by_model": {
                            model: chars_cluster_provenance(model)
                            for model in JAILBREAK_LAYERS
                        },
                        "strength_parameter": "chars_alpha",
                    }
                }
                if has_chars else {}
            ),
            **({"sas": _sas().sas_manifest_grid("jailbreak")} if has_sas else {}),
        },
        "split_fingerprints": _common_split_fingerprints(split_records),
        "baselines": [
            {
                "model": model,
                "application_mode": mode,
                "validation_perplexity": ppl,
            }
            for (model, mode), ppl in sorted(baseline_ppl.items())
        ],
        "selections": selections,
        "validation_candidates": _sorted_candidate_records(validation_records),
    }


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    return fingerprint_json(dict(manifest))


def write_manifest(
    manifest: Mapping[str, Any],
    output_path: os.PathLike[str] | str,
    *,
    overwrite: bool = False,
) -> str:
    """Write a manifest and companion digest.

    By default creation is exclusive: an existing manifest cannot be silently
    replaced after test results have been inspected.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    payload = json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(payload)

    digest = _manifest_digest(manifest)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    with digest_path.open(mode, encoding="utf-8") as handle:
        handle.write(f"{digest}  {path.name}\n")
    return digest


def load_manifest(
    manifest_path: os.PathLike[str] | str,
    *,
    verify_digest: bool = True,
) -> Dict[str, Any]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported manifest schema version: {manifest.get('schema_version')}"
        )

    if verify_digest:
        digest_path = path.with_suffix(path.suffix + ".sha256")
        if not digest_path.exists():
            raise ValueError(f"Manifest digest file is missing: {digest_path}")
        expected = digest_path.read_text(encoding="utf-8").split()[0]
        actual = _manifest_digest(manifest)
        if not hmac.compare_digest(expected, actual):
            raise ValueError(
                f"Manifest digest mismatch for {path}; the locked selection changed."
            )
    return manifest


def find_selection(
    manifest: Mapping[str, Any],
    *,
    model: str,
    method: str,
    mode: str,
    behavior: Optional[str] = None,
    prompt_condition: Optional[str] = None,
) -> Mapping[str, Any]:
    matches = []
    for selection in manifest.get("selections", []):
        if selection.get("model") != model:
            continue
        if selection.get("method") != method:
            continue
        if selection.get("application_mode") != mode:
            continue
        if manifest.get("experiment") == "caa":
            if selection.get("behavior") != behavior:
                continue
            if selection.get("prompt_condition") != prompt_condition:
                continue
        matches.append(selection)
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one locked selection, found "
            f"{len(matches)} for model={model}, method={method}, mode={mode}, "
            f"behavior={behavior}, prompt_condition={prompt_condition}."
        )
    return matches[0]


def assert_locked_configuration(
    manifest: Mapping[str, Any],
    *,
    experiment: str,
    model: str,
    method: str,
    mode: str,
    layer: int,
    strength: float,
    behavior: Optional[str] = None,
    prompt_condition: Optional[str] = None,
    n_clusters: Optional[int] = None,
    state_fingerprint: Optional[str] = None,
) -> Mapping[str, Any]:
    if manifest.get("experiment") != experiment:
        raise ValueError(
            f"Manifest is for experiment={manifest.get('experiment')}, "
            f"not {experiment}."
        )
    selection = find_selection(
        manifest,
        model=model,
        method=method,
        mode=mode,
        behavior=behavior,
        prompt_condition=prompt_condition,
    )
    locked = selection.get("selected")
    if locked is None:
        raise ValueError(
            f"No admissible locked configuration exists for method={method}."
        )
    if int(locked["layer"]) != int(layer) or not math.isclose(
        float(locked["strength"]),
        float(strength),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Test configuration does not match the validation-locked selection: "
            f"requested layer={layer}, strength={strength}; "
            f"locked layer={locked['layer']}, strength={locked['strength']}."
        )
    if method == "chars":
        if n_clusters is None:
            raise ValueError("Locked CHaRS configuration requires n_clusters.")
        locked_n_clusters = locked.get("n_clusters")
        if int(locked_n_clusters or -1) != int(n_clusters):
            raise ValueError(
                "Test CHaRS K does not match the validation-locked selection: "
                f"requested K={n_clusters}; locked K={locked_n_clusters}."
            )
        if state_fingerprint is not None and selection.get(
            "state_fingerprint"
        ) != state_fingerprint:
            raise ValueError(
                "Test CHaRS state fingerprint does not match the "
                "validation-locked selection."
            )
    return selection


def assert_chars_recipe_configuration(
    manifest: Mapping[str, Any],
    *,
    experiment: str,
    model: str,
    extraction_source: str,
    position: int,
    n_clusters: int,
) -> Mapping[str, Any]:
    manifest_experiment = manifest.get("experiment")
    if manifest_experiment != experiment:
        raise ValueError(
            f"Manifest is for experiment={manifest_experiment}, not {experiment}."
        )
    expected = _chars_recipe(experiment)
    manifest_recipe = manifest.get("grid", {}).get("chars")
    if not isinstance(manifest_recipe, Mapping):
        raise ValueError("Manifest does not contain a locked CHaRS recipe.")
    locked_recipe = {key: manifest_recipe.get(key) for key in expected}
    if locked_recipe != expected:
        raise ValueError(
            f"Manifest CHaRS recipe mismatch: expected={expected}; "
            f"found={locked_recipe}."
        )
    clusters_by_model = manifest_recipe.get("clusters_by_model")
    if isinstance(clusters_by_model, Mapping):
        locked_clusters = next(
            (
                values
                for known, values in clusters_by_model.items()
                if str(known).casefold() == str(model).casefold()
            ),
            (),
        )
    else:
        locked_clusters = manifest_recipe.get("clusters", ())
    if int(n_clusters) not in [int(value) for value in locked_clusters]:
        raise ValueError(
            f"CHaRS K={n_clusters} is not locked for model {model}."
        )
    if (
        str(extraction_source) != expected["extraction_source"]
        or int(position) != expected["position"]
    ):
        raise ValueError(
            "CHaRS configuration does not match the validation-locked recipe."
        )
    return manifest_recipe


def assert_curveball_caa_recipe_configuration(
    manifest: Mapping[str, Any],
    *,
    model: str,
    kernel: str,
    degree: int,
    gamma: Optional[float],
    coef0: float,
    kpca_components: int,
    inverse_transform_alpha: float,
    extraction_source: str,
    position: int,
) -> Mapping[str, Any]:
    if manifest.get("experiment") != "caa":
        raise ValueError(
            f"Manifest is for experiment={manifest.get('experiment')}, not caa."
        )
    if model.casefold() != CAA_MODEL.casefold():
        raise ValueError(
            f"CAA Curveball validation supports only {CAA_MODEL}; found {model!r}."
        )

    expected = dict(CURVEBALL_CAA_RECIPE)
    manifest_recipe = manifest.get("grid", {}).get("curveball")
    if not isinstance(manifest_recipe, Mapping):
        raise ValueError("Manifest does not contain a locked CAA Curveball recipe.")
    locked_recipe = {key: manifest_recipe.get(key) for key in expected}
    if locked_recipe != expected:
        raise ValueError(
            "Manifest CAA Curveball recipe mismatch: "
            f"expected={expected}; found={locked_recipe}."
        )

    actual = {
        "kernel": str(kernel),
        "degree": int(degree),
        "gamma": None if gamma is None else float(gamma),
        "coef0": float(coef0),
        "kpca_components": int(kpca_components),
        "inverse_transform_alpha": float(inverse_transform_alpha),
        "extraction_source": str(extraction_source),
        "position": int(position),
    }
    if actual != expected:
        raise ValueError(
            "Curveball configuration does not match the validation-locked CAA "
            f"recipe: expected={expected}; requested={actual}."
        )
    return manifest_recipe


def paired_bootstrap_difference(
    method_scores: Sequence[float],
    comparison_scores: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    if len(method_scores) != len(comparison_scores):
        raise ValueError("Paired score arrays must have equal length.")
    if not method_scores:
        raise ValueError("Paired bootstrap requires at least one example.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")

    differences = [
        float(method) - float(comparison)
        for method, comparison in zip(method_scores, comparison_scores)
    ]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("Bootstrap inputs must be finite.")

    rng = random.Random(seed)
    n = len(differences)
    resampled_means = []
    for _ in range(n_resamples):
        resampled_means.append(
            sum(differences[rng.randrange(n)] for _ in range(n)) / n
        )
    resampled_means.sort()

    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, int(math.floor(alpha * (n_resamples - 1))))
    upper_index = min(
        n_resamples - 1,
        int(math.ceil((1.0 - alpha) * (n_resamples - 1))),
    )
    return {
        "mean_difference": sum(differences) / n,
        "confidence": confidence,
        "ci_lower": resampled_means[lower_index],
        "ci_upper": resampled_means[upper_index],
        "n_examples": n,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def paired_scores_from_predictions(
    method_predictions: Sequence[Mapping[str, Any]],
    comparison_predictions: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    id_key: str = "prompt",
) -> Tuple[List[float], List[float]]:
    method_by_id = {
        str(prediction[id_key]): float(prediction[metric_key])
        for prediction in method_predictions
    }
    comparison_by_id = {
        str(prediction[id_key]): float(prediction[metric_key])
        for prediction in comparison_predictions
    }
    if set(method_by_id) != set(comparison_by_id):
        missing_method = sorted(set(comparison_by_id) - set(method_by_id))
        missing_comparison = sorted(set(method_by_id) - set(comparison_by_id))
        raise ValueError(
            "Prediction IDs are not paired. "
            f"Missing from method={missing_method}; "
            f"missing from comparison={missing_comparison}."
        )
    ordered_ids = sorted(method_by_id)
    return (
        [method_by_id[item_id] for item_id in ordered_ids],
        [comparison_by_id[item_id] for item_id in ordered_ids],
    )


def _prediction_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(record) for record in payload]
    if isinstance(payload, dict):
        for key in ("predictions", "completions"):
            if isinstance(payload.get(key), list):
                return [dict(record) for record in payload[key]]
    raise ValueError(
        "Prediction input must be a list or an object containing predictions/completions."
    )


def _load_candidate_records(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records.extend(dict(record) for record in payload)
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records.extend(dict(record) for record in payload["records"])
        else:
            raise ValueError(
                f"Candidate file must contain a list or {{'records': [...]}}: {path}"
            )
    return records


def _resolve_candidate_paths(
    explicit_paths: Sequence[str],
    candidate_root: Optional[str],
) -> List[Path]:
    paths = [Path(path) for path in explicit_paths]
    if candidate_root is not None:
        paths.extend(
            sorted(Path(candidate_root).rglob("selection_candidates.json"))
        )
    unique_paths = sorted(set(path.resolve() for path in paths))
    if not unique_paths:
        raise ValueError("No candidate files were found.")
    return unique_paths


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-only selection and locked-test utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("select-caa", "select-jailbreak"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--candidate-files",
            nargs="*",
            default=[],
            help="Explicit selection_candidates.json files.",
        )
        subparser.add_argument(
            "--candidate-root",
            default=None,
            help="Recursively scan this directory for selection_candidates.json.",
        )
        subparser.add_argument("--output", required=True)
        subparser.add_argument(
            "--allow-incomplete-grid",
            action="store_true",
            help="Development-only: do not require every planned layer/strength pair.",
        )
        subparser.add_argument(
            "--overwrite",
            action="store_true",
            help="Explicitly replace an existing manifest and digest.",
        )

    jailbreak_parser = subparsers.choices["select-jailbreak"]
    jailbreak_parser.add_argument("--utility-cap-ratio", type=float, default=1.05)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--manifest", required=True)
    check_parser.add_argument("--experiment", choices=("caa", "jailbreak"), required=True)
    check_parser.add_argument("--model", required=True)
    check_parser.add_argument(
        "--method", choices=("actadd", "kas", "curveball", "sas"), required=True
    )
    check_parser.add_argument(
        "--application-mode",
        choices=("prompt_only", "all_tokens"),
        required=True,
    )
    check_parser.add_argument("--layer", type=int, required=True)
    check_parser.add_argument("--strength", type=float, required=True)
    check_parser.add_argument("--behavior")
    check_parser.add_argument(
        "--prompt-condition", choices=("none", "pos", "neg")
    )

    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--method-predictions", required=True)
    bootstrap_parser.add_argument("--comparison-predictions", required=True)
    bootstrap_parser.add_argument("--metric-key", required=True)
    bootstrap_parser.add_argument("--id-key", default="prompt")
    bootstrap_parser.add_argument("--n-resamples", type=int, default=10_000)
    bootstrap_parser.add_argument("--seed", type=int, default=42)
    bootstrap_parser.add_argument("--output", required=True)

    grid_parser = subparsers.add_parser("show-grid")
    grid_parser.add_argument("--experiment", choices=("caa", "jailbreak"), required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command in {"select-caa", "select-jailbreak"}:
        paths = _resolve_candidate_paths(
            args.candidate_files,
            args.candidate_root,
        )
        records = _load_candidate_records(paths)
        require_complete = not args.allow_incomplete_grid
        if args.command == "select-caa":
            manifest = select_caa_candidates(
                records,
                require_complete_grid=require_complete,
            )
        else:
            manifest = select_jailbreak_candidates(
                records,
                utility_cap_ratio=args.utility_cap_ratio,
                require_complete_grid=require_complete,
            )
        manifest["candidate_files"] = [str(path) for path in paths]
        digest = write_manifest(
            manifest,
            args.output,
            overwrite=args.overwrite,
        )
        print(f"Wrote locked manifest: {args.output}")
        print(f"SHA256: {digest}")
        return

    if args.command == "check":
        manifest = load_manifest(args.manifest)
        selection = assert_locked_configuration(
            manifest,
            experiment=args.experiment,
            model=args.model,
            method=args.method,
            mode=args.application_mode,
            layer=args.layer,
            strength=args.strength,
            behavior=args.behavior,
            prompt_condition=args.prompt_condition,
        )
        print(json.dumps(selection, indent=2, sort_keys=True))
        return

    if args.command == "bootstrap":
        with open(args.method_predictions, "r", encoding="utf-8") as handle:
            method_predictions = json.load(handle)
        with open(args.comparison_predictions, "r", encoding="utf-8") as handle:
            comparison_predictions = json.load(handle)
        method_scores, comparison_scores = paired_scores_from_predictions(
            _prediction_records(method_predictions),
            _prediction_records(comparison_predictions),
            metric_key=args.metric_key,
            id_key=args.id_key,
        )
        result = paired_bootstrap_difference(
            method_scores,
            comparison_scores,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        _write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.experiment == "caa":
        grid = {
            "layers": list(CAA_LAYERS),
            "strengths": list(CAA_STRENGTHS),
            "configurations_per_method_scope": len(CAA_LAYERS)
            * len(CAA_STRENGTHS),
            "curveball": {
                **dict(CURVEBALL_CAA_RECIPE),
                "alphas": list(CAA_STRENGTHS),
                "strength_parameter": "curveball_alpha",
            },
            "sas": _sas().sas_show_grid("caa"),
        }
    else:
        grid = {
            "layers_by_model": {
                model: list(layers)
                for model, layers in JAILBREAK_LAYERS.items()
            },
            "strengths": list(JAILBREAK_STRENGTHS),
            "configurations_per_method_model_mode": 2
            * len(JAILBREAK_STRENGTHS),
            "sas": _sas().sas_show_grid("jailbreak"),
        }
    print(json.dumps(grid, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from types import SimpleNamespace

import pytest

from pipeline.validation_selection import (
    CAA_LAYERS,
    CAA_STRENGTHS,
    CHARS_VALIDATION_CLUSTERS,
    CHARS_RECIPES,
    assert_locked_configuration,
    select_caa_candidates,
    select_jailbreak_candidates,
)
from scripts.build_validation_jobs import (
    _caa_select_jobs,
    _chars_fit_caa_jobs,
    _chars_fit_jail_jobs,
    _jail_select_jobs,
)
from scripts.run_locked_tinybench import _validate_steer_config
from pipeline.utils.chars import build_chars_steer_config, fit_chars_state
import torch


def _chars_candidate(*, experiment, layer, strength, n_clusters, metric, fingerprint):
    common = {
        "experiment": experiment,
        "split": "validation",
        "model": "gemma-2-9b-it",
        "method": "chars",
        "strength_parameter": "chars_alpha",
        "application_mode": "all_tokens",
        "layer": layer,
        "strength": strength,
        "n_clusters": n_clusters,
        "state_fingerprint": fingerprint,
        "split_fingerprints": {"reference": "ref", "validation": "val"},
        "fixed_parameters": {"chars": dict(CHARS_RECIPES[experiment])},
    }
    if experiment == "caa":
        common.update(
            {
                "behavior": "sycophancy",
                "prompt_condition": "none",
                "validation_target_probability": metric,
                "n_validation_examples": 50,
            }
        )
    else:
        common.update(
            {
                "validation_asr": metric,
                "validation_perplexity": 10.0,
                "n_harmful_validation_examples": 39,
                "n_harmless_validation_examples": 256,
            }
        )
    return common


def test_caa_chars_tie_breaks_on_strength_then_layer_with_fixed_k():
    records = [
        _chars_candidate(
            experiment="caa",
            layer=22,
            strength=1.0,
            n_clusters=10,
            metric=0.8,
            fingerprint="a" * 64,
        ),
        _chars_candidate(
            experiment="caa",
            layer=23,
            strength=1.0,
            n_clusters=10,
            metric=0.8,
            fingerprint="b" * 64,
        ),
    ]
    manifest = select_caa_candidates(records, require_complete_grid=False)
    selection = manifest["selections"][0]
    assert manifest["schema_version"] == 3
    assert selection["selected"] == {
        "layer": 22,
        "strength": 1.0,
        "n_clusters": 10,
    }
    assert selection["state_fingerprint"] == "a" * 64

    invalid = dict(records[0], n_clusters=9)
    with pytest.raises(ValueError, match="validation-locked model value 10"):
        select_caa_candidates([invalid], require_complete_grid=False)


def test_complete_chars_grid_uses_one_fixed_k():
    records = []
    for layer in CAA_LAYERS:
        for strength in CAA_STRENGTHS:
            n_clusters = CHARS_VALIDATION_CLUSTERS["gemma-2-9b-it"]
            fingerprint = f"{layer:02x}{n_clusters:02x}".ljust(64, "0")
            records.append(
                _chars_candidate(
                    experiment="caa",
                    layer=layer,
                    strength=strength,
                    n_clusters=n_clusters,
                    metric=0.5,
                    fingerprint=fingerprint,
                )
            )
    manifest = select_caa_candidates(records, require_complete_grid=True)
    assert manifest["selections"][0]["candidate_count"] == 2 * 20
    assert manifest["grid"]["chars"]["clusters"] == [10]
    with pytest.raises(ValueError, match="CHaRS candidate grid"):
        select_caa_candidates(records[:-1], require_complete_grid=True)


def test_jailbreak_chars_obeys_cap_with_fixed_k():
    baseline = {
        "experiment": "jailbreak",
        "split": "validation",
        "model": "gemma-2-9b-it",
        "method": "baseline",
        "application_mode": "all_tokens",
        "validation_perplexity": 10.0,
    }
    records = [
        baseline,
        _chars_candidate(
            experiment="jailbreak",
            layer=22,
            strength=1.0,
            n_clusters=10,
            metric=0.7,
            fingerprint="c" * 64,
        ),
        _chars_candidate(
            experiment="jailbreak",
            layer=23,
            strength=1.0,
            n_clusters=10,
            metric=0.7,
            fingerprint="d" * 64,
        ),
    ]
    manifest = select_jailbreak_candidates(records, require_complete_grid=False)
    selection = manifest["selections"][0]
    assert selection["selected"] == {
        "layer": 22,
        "strength": 1.0,
        "n_clusters": 10,
    }
    assert selection["validation"]["perplexity_cap"] == 10.5


def test_locked_chars_rejects_k_and_state_mismatch():
    record = _chars_candidate(
        experiment="caa",
        layer=22,
        strength=1.0,
        n_clusters=10,
        metric=0.9,
        fingerprint="e" * 64,
    )
    manifest = select_caa_candidates([record], require_complete_grid=False)
    kwargs = {
        "experiment": "caa",
        "model": "gemma-2-9b-it",
        "method": "chars",
        "mode": "all_tokens",
        "layer": 22,
        "strength": 1.0,
        "behavior": "sycophancy",
        "prompt_condition": "none",
    }
    with pytest.raises(ValueError, match="CHaRS K"):
        assert_locked_configuration(manifest, **kwargs, n_clusters=9)
    with pytest.raises(ValueError, match="state fingerprint"):
        assert_locked_configuration(
            manifest,
            **kwargs,
            n_clusters=10,
            state_fingerprint="f" * 64,
        )


def test_chars_job_counts_and_cache_reuse():
    caa_args = SimpleNamespace(
        model_path="/models/gemma-2-9b-it",
        artifact_root="/artifacts/caa",
        chars_state_root="/states",
        prompt_condition="all",
        method="chars",
    )
    caa_jobs = _caa_select_jobs(caa_args)
    assert len(caa_jobs) == 6 * 3 * 2 * 20
    paths = {
        job["argv"][job["argv"].index("--chars_state_path") + 1]
        for job in caa_jobs
        if "coordinate-other-ais" in job["id"]
        and "-l22-" in job["id"]
        and job["id"].endswith("-k10")
    }
    assert paths == {
        "/states/caa/gemma-2-9b-it/coordinate-other-ais/layer-22/k-10/chars_state.pt"
    }

    jail_args = SimpleNamespace(
        model_path="/models/gemma-2-9b-it",
        artifact_root="/artifacts/jail",
        chars_state_root="/states",
        application_mode="all",
        method="chars",
    )
    jail_jobs = _jail_select_jobs(jail_args)
    assert len(jail_jobs) == 1 + 2 * 2 * 10
    state_paths = {
        job["argv"][job["argv"].index("--chars_state_path") + 1]
        for job in jail_jobs
        if "-chars-l22-" in job["id"] and job["id"].endswith("-k10")
    }
    assert state_paths == {
        "/states/jailbreak/gemma-2-9b-it/layer-22/k-10/chars_state.pt"
    }


def test_prerequisite_fit_job_counts():
    caa_jobs = _chars_fit_caa_jobs(
        SimpleNamespace(
            model_path="/models/gemma-2-9b-it",
            chars_state_root="/states",
        )
    )
    jail_jobs = _chars_fit_jail_jobs(
        SimpleNamespace(
            model_path="/models/Qwen2.5-7B-Instruct",
            chars_state_root="/states",
        )
    )
    assert len(caa_jobs) == 6
    assert len(jail_jobs) == 1
    assert jail_jobs[0]["argv"].count("--layers") == 1
    assert caa_jobs[0]["argv"][-1] == "10"
    assert jail_jobs[0]["argv"][-1] == "5"


def test_locked_tinybench_validates_embedded_chars_state(tmp_path):
    metadata = {
        "model": "gemma-2-9b-it",
        "layer": 22,
        "position": -1,
        "hookpoint": "block_output",
        "extraction_source": "legacy",
        "orientation": "harmful_to_harmless",
        "split_fingerprints": {"harmful_reference": "h", "harmless_reference": "s"},
    }
    source = torch.arange(20, dtype=torch.float32).reshape(10, 2) * 1e-3
    target = source + 1
    state = fit_chars_state(source, target, n_clusters=10, metadata=metadata)
    config = build_chars_steer_config(
        state, layer=22, alpha=1.0, position=-1, prompt_only=False
    )
    path = tmp_path / "steer_config_chars.pt"
    torch.save(config, path)
    manifest = {
        "experiment": "jailbreak",
        "grid": {
            "chars": {
                **dict(CHARS_RECIPES["jailbreak"]),
                "clusters": [10],
                "clusters_by_model": {"gemma-2-9b-it": [10]},
            }
        },
    }
    _validate_steer_config(
        path,
        "chars",
        22,
        1.0,
        False,
        manifest,
        "gemma-2-9b-it",
        n_clusters=10,
        state_fingerprint=state["state_fingerprint"],
    )
    with pytest.raises(ValueError, match="n_clusters mismatch|K"):
        _validate_steer_config(
            path,
            "chars",
            22,
            1.0,
            False,
            manifest,
            "gemma-2-9b-it",
            n_clusters=9,
            state_fingerprint=state["state_fingerprint"],
        )

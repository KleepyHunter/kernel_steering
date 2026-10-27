#!/usr/bin/env python3
"""Emit exhaustive validation-grid jobs and manifest-locked test jobs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.validation_selection import (
    CAA_BEHAVIORS,
    CAA_LAYERS,
    CAA_STRENGTHS,
    CHARS_RECIPES,
    CURVEBALL_CAA_RECIPE,
    CAA_MODEL,
    JAILBREAK_LAYERS,
    JAILBREAK_STRENGTHS,
    METHOD_TO_INTERVENTION,
    load_manifest,
    chars_validation_clusters,
)
from pipeline.sas_validation import SAS_COEFFICIENTS, SAS_MODEL, sas_job_args


def _base_command(model_path: str, artifact_root: str | None = None) -> list[str]:
    command = [
        sys.executable, "-m", "pipeline.run_pipeline", "--model_path", model_path
    ]
    if artifact_root is not None:
        command.extend(["--artifact_root", artifact_root])
    return command


def _fixed_args() -> list[str]:
    return [
        "--kernel_type", "rbf",
        "--lambda_reg", "0",
        "--step_count", "1",
        "--split_seed", "42",
    ]


def _prompt_flag(mode: str) -> list[str]:
    return ["--prompt_only"] if mode == "prompt_only" else []


def _curveball_caa_args(alpha: float) -> list[str]:
    recipe = CURVEBALL_CAA_RECIPE
    return [
        "--steer_method", "curveball",
        "--pos", str(recipe["position"]),
        "--curveball_extraction_source", str(recipe["extraction_source"]),
        "--curveball_kernel", str(recipe["kernel"]),
        "--curveball_degree", str(recipe["degree"]),
        "--curveball_coef0", str(recipe["coef0"]),
        "--curveball_k", str(recipe["kpca_components"]),
        "--curveball_fit_inverse_alpha", str(recipe["inverse_transform_alpha"]),
        "--curveball_alpha", str(float(alpha)),
    ]


def _chars_state_path(
    state_root: str,
    *,
    experiment: str,
    model_alias: str,
    layer: int,
    n_clusters: int,
    behavior: str | None = None,
) -> str:
    path = Path(state_root).expanduser().resolve() / experiment / model_alias
    if experiment == "caa":
        if behavior is None:
            raise ValueError("CAA CHaRS state paths require a behavior.")
        path /= behavior
    return str(path / f"layer-{layer}" / f"k-{n_clusters}" / "chars_state.pt")


def _chars_args(
    experiment: str,
    alpha: float,
    n_clusters: int,
    state_path: str | None = None,
) -> list[str]:
    recipe = CHARS_RECIPES[experiment]
    args = [
        "--steer_method", "chars",
        "--pos", str(recipe["position"]),
        "--chars_extraction_source", str(recipe["extraction_source"]),
        "--chars_alpha", str(float(alpha)),
        "--chars_clusters", str(int(n_clusters)),
    ]
    if state_path is not None:
        args.extend(["--chars_state_path", state_path])
    return args


def _emit(jobs: list[dict], output: str | None, output_format: str) -> None:
    handle = open(output, "w", encoding="utf-8") if output else sys.stdout
    try:
        if output_format == "shell":
            handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
            for job in jobs:
                handle.write(f"# {job['id']}\n")
                handle.write(shlex.join(job["argv"]) + "\n")
        else:
            for job in jobs:
                handle.write(json.dumps(job, sort_keys=True) + "\n")
    finally:
        if output:
            handle.close()
    print(f"Emitted {len(jobs)} jobs.", file=sys.stderr)


def _caa_select_jobs(args) -> list[dict]:
    model_alias = os.path.basename(os.path.normpath(args.model_path))
    args.chars_state_root = getattr(args, "chars_state_root", None) or os.path.join(
        args.artifact_root, "chars_states"
    )
    if model_alias.casefold() != CAA_MODEL.casefold():
        raise ValueError(f"CAA validation supports only {CAA_MODEL}.")
    method_filter = getattr(args, "method", "all")
    if method_filter not in {"all", "curveball", "chars", "sas"}:
        raise ValueError(f"Unsupported CAA validation method filter: {method_filter}.")
    jobs = []
    prompt_conditions = (
        ("none", "pos", "neg")
        if args.prompt_condition == "all"
        else (args.prompt_condition,)
    )
    for behavior in CAA_BEHAVIORS:
        for prompt_condition in prompt_conditions:
            for layer in CAA_LAYERS:
                for strength in CAA_STRENGTHS:
                    if method_filter == "sas":
                        continue
                    argv = _base_command(args.model_path, args.artifact_root) + _fixed_args() + [
                        "--experiment_phase", "select",
                        "--evaluation_protocol", "caa",
                        "--kernel_extraction_source", "caa",
                        "--caa_eval_split", "validation",
                        "--caa_eval_types", "ab",
                        "--caa_behavior", behavior,
                        "--caa_eval_system_prompt", prompt_condition,
                        "--caa_validation_size", "50",
                        "--layer", str(layer),
                        "--eta", str(strength),
                        "--evaluation_interventions", "actadd", "kernel",
                    ]
                    jobs.append({
                        "id": f"caa-select-{behavior}-{prompt_condition}-l{layer}-s{strength:g}",
                        "experiment": "caa",
                        "phase": "select",
                        "argv": argv,
                    })
                    for n_clusters in chars_validation_clusters(model_alias):
                        state_path = _chars_state_path(
                            args.chars_state_root,
                            experiment="caa",
                            model_alias=model_alias,
                            behavior=behavior,
                            layer=layer,
                            n_clusters=n_clusters,
                        )
                        jobs.append({
                            "id": (
                                f"caa-select-{behavior}-{prompt_condition}-"
                                f"chars-l{layer}-a{strength:g}-k{n_clusters}"
                            ),
                            "experiment": "caa",
                            "phase": "select",
                            "argv": (
                                _base_command(args.model_path, args.artifact_root)
                                + _fixed_args()
                                + [
                                    "--experiment_phase", "select",
                                    "--evaluation_protocol", "caa",
                                    "--caa_eval_split", "validation",
                                    "--caa_eval_types", "ab",
                                    "--caa_behavior", behavior,
                                    "--caa_eval_system_prompt", prompt_condition,
                                    "--caa_validation_size", "50",
                                    "--layer", str(layer),
                                ]
                                + _chars_args(
                                    "caa", strength, n_clusters, state_path
                                )
                                + ["--evaluation_interventions", "chars"]
                            ),
                        })
                    jobs.append({
                        "id": (
                            f"caa-select-{behavior}-{prompt_condition}-"
                            f"curveball-l{layer}-a{strength:g}"
                        ),
                        "experiment": "caa",
                        "phase": "select",
                        "argv": (
                            _base_command(args.model_path, args.artifact_root)
                            + _fixed_args()
                            + [
                                "--experiment_phase", "select",
                                "--evaluation_protocol", "caa",
                                "--caa_eval_split", "validation",
                                "--caa_eval_types", "ab",
                                "--caa_behavior", behavior,
                                "--caa_eval_system_prompt", prompt_condition,
                                "--caa_validation_size", "50",
                                "--layer", str(layer),
                            ]
                            + _curveball_caa_args(strength)
                            + ["--evaluation_interventions", "curveball"]
                        ),
                    })
                if method_filter != "sas":
                    continue
                for coefficient in SAS_COEFFICIENTS["caa"]:
                    jobs.append({
                        "id": (
                            f"caa-select-{behavior}-{prompt_condition}-"
                            f"sas-l{layer}-c{coefficient:g}"
                        ),
                        "experiment": "caa",
                        "phase": "select",
                        "argv": (
                            _base_command(args.model_path, args.artifact_root)
                            + _fixed_args()
                            + [
                                "--experiment_phase", "select",
                                "--evaluation_protocol", "caa",
                                "--caa_eval_split", "validation",
                                "--caa_eval_types", "ab",
                                "--caa_behavior", behavior,
                                "--caa_eval_system_prompt", prompt_condition,
                                "--caa_validation_size", "50",
                                "--layer", str(layer),
                            ]
                            + sas_job_args("caa", coefficient)
                            + ["--evaluation_interventions", "sas"]
                        ),
                    })
    if method_filter == "all":
        # CHaRS and SAS run as their own campaigns.
        jobs = [
            job for job in jobs
            if "-chars-" not in job["id"] and "-sas-" not in job["id"]
        ]
    else:
        jobs = [job for job in jobs if f"-{method_filter}-" in job["id"]]

    return jobs


def _model_layers(model_alias: str) -> tuple[int, ...]:
    for known, layers in JAILBREAK_LAYERS.items():
        if known.casefold() == model_alias.casefold():
            return layers
    raise ValueError(f"Unknown model alias {model_alias!r}; expected one of {list(JAILBREAK_LAYERS)}")


def _jail_select_jobs(args) -> list[dict]:
    model_alias = os.path.basename(os.path.normpath(args.model_path))
    args.chars_state_root = getattr(args, "chars_state_root", None) or os.path.join(
        args.artifact_root, "chars_states"
    )
    layers = _model_layers(model_alias)
    method_filter = getattr(args, "method", "all")
    if method_filter not in {"all", "chars", "sas"}:
        raise ValueError(
            f"Unsupported jailbreak validation method filter: {method_filter}."
        )
    if method_filter == "sas" and model_alias.casefold() != SAS_MODEL.casefold():
        raise ValueError(f"SAS validation supports only {SAS_MODEL}.")
    common = _base_command(args.model_path, args.artifact_root) + _fixed_args() + [
        "--experiment_phase", "select",
        "--evaluation_protocol", "legacy",
        "--jailbreak_eval_split", "validation",
    ]
    jobs = [{
        "id": f"jail-select-{model_alias}-baseline",
        "experiment": "jailbreak",
        "phase": "select",
        "argv": common + [
            "--steer_method", "kernel",
            "--kernel_extraction_source", "legacy",
            "--layer", str(layers[0]),
            "--eta", str(JAILBREAK_STRENGTHS[0]),
            "--evaluation_interventions", "baseline",
        ],
    }]
    modes = (
        ("prompt_only", "all_tokens")
        if args.application_mode == "all"
        else (args.application_mode,)
    )
    for mode in modes:
        for layer in layers:
            for strength in JAILBREAK_STRENGTHS:
                if method_filter == "sas":
                    continue
                jobs.append({
                    "id": f"jail-select-{model_alias}-{mode}-kernel-l{layer}-s{strength:g}",
                    "experiment": "jailbreak",
                    "phase": "select",
                    "argv": common + [
                        "--steer_method", "kernel",
                        "--kernel_extraction_source", "legacy",
                        "--layer", str(layer),
                        "--eta", str(strength),
                        "--evaluation_interventions", "actadd", "kernel",
                    ] + _prompt_flag(mode),
                })
                for n_clusters in chars_validation_clusters(model_alias):
                    state_path = _chars_state_path(
                        args.chars_state_root,
                        experiment="jailbreak",
                        model_alias=model_alias,
                        layer=layer,
                        n_clusters=n_clusters,
                    )
                    jobs.append({
                        "id": (
                            f"jail-select-{model_alias}-{mode}-chars-"
                            f"l{layer}-a{strength:g}-k{n_clusters}"
                        ),
                        "experiment": "jailbreak",
                        "phase": "select",
                        "argv": (
                            common
                            + ["--layer", str(layer)]
                            + _chars_args(
                                "jailbreak", strength, n_clusters, state_path
                            )
                            + ["--evaluation_interventions", "chars"]
                            + _prompt_flag(mode)
                        ),
                    })
                jobs.append({
                    "id": f"jail-select-{model_alias}-{mode}-curveball-l{layer}-s{strength:g}",
                    "experiment": "jailbreak",
                    "phase": "select",
                    "argv": common + [
                        "--steer_method", "curveball",
                        "--curveball_extraction_source", "legacy",
                        "--curveball_kernel", "poly",
                        "--curveball_degree", "3",
                        "--curveball_coef0", "1",
                        "--curveball_k", "20",
                        "--curveball_fit_inverse_alpha", "1",
                        "--curveball_alpha", str(strength),
                        "--layer", str(layer),
                        "--evaluation_interventions", "curveball",
                    ] + _prompt_flag(mode),
                })
            if method_filter != "sas":
                continue
            for coefficient in SAS_COEFFICIENTS["jailbreak"]:
                jobs.append({
                    "id": (
                        f"jail-select-{model_alias}-{mode}-"
                        f"sas-l{layer}-c{coefficient:g}"
                    ),
                    "experiment": "jailbreak",
                    "phase": "select",
                    "argv": (
                        common
                        + ["--layer", str(layer)]
                        + sas_job_args("jailbreak", coefficient)
                        + [
                            "--ce_loss_batch_size", "1",
                            "--evaluation_interventions", "sas",
                        ]
                        + _prompt_flag(mode)
                    ),
                })
    if method_filter == "all":
        # CHaRS and SAS run as their own campaigns.
        jobs = [
            job for job in jobs
            if "-chars-" not in job["id"] and "-sas-" not in job["id"]
        ]
    else:
        jobs = [
            job for job in jobs
            if "-baseline" in job["id"] or f"-{method_filter}-" in job["id"]
        ]
    return jobs


def _chars_fit_caa_jobs(args) -> list[dict]:
    model_alias = os.path.basename(os.path.normpath(args.model_path))
    if model_alias.casefold() != CAA_MODEL.casefold():
        raise ValueError(f"CAA validation supports only {CAA_MODEL}.")
    jobs = []
    for behavior in CAA_BEHAVIORS:
        jobs.append({
            "id": f"chars-fit-caa-{behavior}",
            "experiment": "caa",
            "phase": "fit",
            "argv": [
                sys.executable,
                str(REPO_ROOT / "scripts" / "fit_chars_states.py"),
                "--model-path", args.model_path,
                "--output-root", args.chars_state_root,
                "--experiment", "caa",
                "--caa-behavior", behavior,
                "--caa-validation-size", "50",
                "--split-seed", "42",
                "--layers", *[str(layer) for layer in CAA_LAYERS],
                "--clusters", *[
                    str(value) for value in chars_validation_clusters(model_alias)
                ],
            ],
        })
    return jobs


def _chars_fit_jail_jobs(args) -> list[dict]:
    model_alias = os.path.basename(os.path.normpath(args.model_path))
    layers = _model_layers(model_alias)
    return [{
        "id": f"chars-fit-jail-{model_alias}",
        "experiment": "jailbreak",
        "phase": "fit",
        "argv": [
            sys.executable,
            str(REPO_ROOT / "scripts" / "fit_chars_states.py"),
            "--model-path", args.model_path,
            "--output-root", args.chars_state_root,
            "--experiment", "jailbreak",
            "--split-seed", "42",
            "--n-train", "128",
            "--layers", *[str(layer) for layer in layers],
            "--clusters", *[
                str(value) for value in chars_validation_clusters(model_alias)
            ],
        ],
    }]


def _locked_test_jobs(args) -> list[dict]:
    manifest = load_manifest(args.manifest)
    model_alias = os.path.basename(os.path.normpath(args.model_path))
    selections = [
        item for item in manifest.get("selections", [])
        if item.get("model", "").casefold() == model_alias.casefold()
    ]
    if not selections:
        raise ValueError(f"Manifest has no selections for {model_alias}.")
    jobs = []
    if manifest["experiment"] == "caa":
        scopes = sorted({
            (item["behavior"], item["prompt_condition"], item["application_mode"])
            for item in selections
        })
        for behavior, prompt_condition, mode in scopes:
            jobs.append({
                "id": f"caa-test-{behavior}-{prompt_condition}-{mode}-baseline",
                "experiment": "caa",
                "phase": "test",
                "argv": _base_command(args.model_path, args.artifact_root) + _fixed_args() + [
                    "--experiment_phase", "test",
                    "--selection_manifest", args.manifest,
                    "--evaluation_protocol", "caa",
                    "--kernel_extraction_source", "caa",
                    "--caa_eval_split", "test",
                    "--caa_eval_types", "ab",
                    "--caa_behavior", behavior,
                    "--caa_eval_system_prompt", prompt_condition,
                    "--layer", str(CAA_LAYERS[0]),
                    "--eta", "0",
                    "--evaluation_interventions", "baseline",
                ] + _prompt_flag(mode),
            })
        for item in selections:
            if item.get("selected") is None:
                continue
            selected = item["selected"]
            method = item["method"]
            intervention = METHOD_TO_INTERVENTION[method]
            if method == "sas":
                method_args = sas_job_args("caa", selected["strength"])
            elif method == "curveball":
                method_args = _curveball_caa_args(selected["strength"])
            elif method == "chars":
                state_path = None
                if args.chars_state_root is not None:
                    state_path = _chars_state_path(
                        args.chars_state_root,
                        experiment="caa",
                        model_alias=model_alias,
                        behavior=item["behavior"],
                        layer=selected["layer"],
                        n_clusters=selected["n_clusters"],
                    )
                method_args = _chars_args(
                    "caa",
                    selected["strength"],
                    selected["n_clusters"],
                    state_path,
                )
            else:
                method_args = [
                    "--steer_method", "kernel",
                    "--kernel_extraction_source", "caa",
                    "--eta", str(selected["strength"]),
                ]
            jobs.append({
                "id": f"caa-test-{item['behavior']}-{item['prompt_condition']}-{item['application_mode']}-{method}",
                "experiment": "caa",
                "phase": "test",
                "argv": (
                    _base_command(args.model_path, args.artifact_root)
                    + _fixed_args()
                    + [
                        "--experiment_phase", "test",
                        "--selection_manifest", args.manifest,
                        "--evaluation_protocol", "caa",
                        "--caa_eval_split", "test",
                        "--caa_eval_types", "ab",
                        "--caa_behavior", item["behavior"],
                        "--caa_eval_system_prompt", item["prompt_condition"],
                        "--layer", str(selected["layer"]),
                    ]
                    + method_args
                    + ["--evaluation_interventions", intervention]
                    + _prompt_flag(item["application_mode"])
                ),
            })
    else:
        layers = _model_layers(model_alias)
        jobs.append({
            "id": f"jail-test-{model_alias}-baseline",
            "experiment": "jailbreak",
            "phase": "test",
            "argv": _base_command(args.model_path, args.artifact_root) + _fixed_args() + [
                "--experiment_phase", "test",
                "--selection_manifest", args.manifest,
                "--evaluation_protocol", "legacy",
                "--jailbreak_eval_split", "test",
                "--steer_method", "kernel",
                "--kernel_extraction_source", "legacy",
                "--layer", str(layers[0]),
                "--eta", str(JAILBREAK_STRENGTHS[0]),
                "--evaluation_interventions", "baseline",
            ],
        })
        for item in selections:
            if item.get("selected") is None:
                continue
            selected = item["selected"]
            method = item["method"]
            intervention = METHOD_TO_INTERVENTION[method]
            if method == "curveball":
                method_args = [
                    "--steer_method", "curveball",
                    "--curveball_extraction_source", "legacy",
                    "--curveball_alpha", str(selected["strength"]),
                ]
            elif method == "sas":
                method_args = sas_job_args("jailbreak", selected["strength"])
            elif method == "chars":
                state_path = None
                if args.chars_state_root is not None:
                    state_path = _chars_state_path(
                        args.chars_state_root,
                        experiment="jailbreak",
                        model_alias=model_alias,
                        layer=selected["layer"],
                        n_clusters=selected["n_clusters"],
                    )
                method_args = _chars_args(
                    "jailbreak",
                    selected["strength"],
                    selected["n_clusters"],
                    state_path,
                )
            else:
                method_args = [
                    "--steer_method", "kernel",
                    "--kernel_extraction_source", "legacy",
                    "--eta", str(selected["strength"]),
                ]
            jobs.append({
                "id": f"jail-test-{model_alias}-{item['application_mode']}-{method}",
                "experiment": "jailbreak",
                "phase": "test",
                "argv": (
                    _base_command(args.model_path, args.artifact_root)
                    + _fixed_args()
                    + [
                        "--experiment_phase", "test",
                        "--selection_manifest", args.manifest,
                        "--evaluation_protocol", "legacy",
                        "--jailbreak_eval_split", "test",
                        "--layer", str(selected["layer"]),
                    ]
                    + method_args
                    + ["--evaluation_interventions", intervention]
                    + _prompt_flag(item["application_mode"])
                ),
            })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--format", choices=("shell", "jsonl"), default="shell")
    subparsers = parser.add_subparsers(dest="command", required=True)
    caa = subparsers.add_parser("caa-select")
    caa.add_argument("--model-path", required=True)
    caa.add_argument("--artifact-root", required=True)
    caa.add_argument("--method", choices=("all", "curveball", "chars", "sas"), default="all")
    caa.add_argument("--chars-state-root")
    caa.add_argument(
        "--prompt-condition",
        choices=("all", "none", "pos", "neg"),
        default="all",
    )
    jail = subparsers.add_parser("jail-select")
    jail.add_argument("--model-path", required=True)
    jail.add_argument("--artifact-root", required=True)
    jail.add_argument("--method", choices=("all", "chars", "sas"), default="all")
    jail.add_argument("--chars-state-root")
    jail.add_argument(
        "--application-mode",
        choices=("all", "prompt_only", "all_tokens"),
        default="all",
    )
    test = subparsers.add_parser("locked-test")
    test.add_argument("--model-path", required=True)
    test.add_argument("--artifact-root", required=True)
    test.add_argument("--manifest", required=True)
    test.add_argument("--chars-state-root")
    fit_caa = subparsers.add_parser("chars-fit-caa")
    fit_caa.add_argument("--model-path", required=True)
    fit_caa.add_argument("--chars-state-root", required=True)
    fit_jail = subparsers.add_parser("chars-fit-jail")
    fit_jail.add_argument("--model-path", required=True)
    fit_jail.add_argument("--chars-state-root", required=True)
    args = parser.parse_args()
    builders = {
        "caa-select": _caa_select_jobs,
        "jail-select": _jail_select_jobs,
        "locked-test": _locked_test_jobs,
        "chars-fit-caa": _chars_fit_caa_jobs,
        "chars-fit-jail": _chars_fit_jail_jobs,
    }
    _emit(builders[args.command](args), args.output, args.format)


if __name__ == "__main__":
    main()

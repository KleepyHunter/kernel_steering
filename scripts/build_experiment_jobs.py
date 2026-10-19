#!/usr/bin/env python3
"""Emit standard-phase job files: jailbreak test runs and the KAS ablations.

Each job is one `python -m pipeline.run_pipeline` call in standard phase, which
evaluates JAILBREAKBENCH ASR (step 3b) and Pile / Alpaca perplexity (step 5).
Run the emitted JSONL with `scripts/validation_workflow.sh run-local` or
`scripts/slurm_validation_array.sh`, as for validation jobs.

Subcommands:
  standard-test   validation-selected settings from jailbreak manifests
  single-pair     KAS (1 and 10 steps) and ActAdd fitted on one prompt pair
  n-train         KAS and ActAdd for a growing number of reference pairs
  kernel-choice   KAS with the Laplacian, polynomial, and RBF kernels
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.sas_validation import sas_job_args  # noqa: E402
from pipeline.validation_selection import METHOD_TO_INTERVENTION, load_manifest  # noqa: E402
from scripts.build_validation_jobs import (  # noqa: E402
    _base_command,
    _chars_args,
    _emit,
    _fixed_args,
    _prompt_flag,
)

APPLICATION_MODES = ("prompt_only", "all_tokens")

# Layer and KAS step sizes per model for the single-pair ablation:
# (layer, eta for 1 gradient step, eta for 10 gradient steps).
SINGLE_PAIR_SETTINGS = {
    "gemma-2-9b-it": (22, 200.0, 40.0),
    "Llama-3.1-8B-Instruct": (12, 5.0, 1.0),
    "Qwen2.5-7B-Instruct": (18, 40.0, 8.0),
}


def _model_alias(model_path: str) -> str:
    return os.path.basename(os.path.normpath(model_path))


def _kas_args(*, layer: int, eta: float, step_count: int, seed: int, lambda_reg: float) -> list[str]:
    return [
        "--steer_method", "kernel",
        "--kernel_extraction_source", "legacy",
        "--evaluation_protocol", "legacy",
        "--pos", "-1",
        "--layer", str(layer),
        "--eta", str(float(eta)),
        "--step_count", str(step_count),
        "--lambda_reg", str(float(lambda_reg)),
        "--seed", str(seed),
    ]


def _standard_test_method_args(method: str, selected: dict) -> list[str]:
    strength = selected["strength"]
    if method in {"actadd", "kas"}:
        return [
            "--steer_method", "kernel",
            "--kernel_extraction_source", "legacy",
            "--eta", str(strength),
        ]
    if method == "curveball":
        return [
            "--steer_method", "curveball",
            "--curveball_extraction_source", "legacy",
            "--curveball_kernel", "poly",
            "--curveball_degree", "3",
            "--curveball_coef0", "1",
            "--curveball_k", "20",
            "--curveball_fit_inverse_alpha", "1",
            "--curveball_alpha", str(strength),
        ]
    if method == "chars":
        return _chars_args("jailbreak", strength, selected["n_clusters"])
    if method == "sas":
        return sas_job_args("jailbreak", strength)
    raise ValueError(f"Unsupported jailbreak method in manifest: {method!r}.")


def _standard_test_jobs(args) -> list[dict]:
    jobs = []
    manifests = []
    for manifest_path in args.manifest:
        manifest = load_manifest(manifest_path)
        if manifest.get("experiment") != "jailbreak":
            raise ValueError(f"{manifest_path} is not a jailbreak manifest.")
        manifests.append(manifest)
    for model_path in args.model_path:
        jobs.extend(_standard_test_jobs_for_model(model_path, manifests, args.artifact_root))
    ids = [job["id"] for job in jobs]
    duplicates = sorted({job_id for job_id in ids if ids.count(job_id) > 1})
    if duplicates:
        raise ValueError(f"Manifests select the same method/mode more than once: {duplicates}.")
    if not jobs:
        raise ValueError("The manifests have no selected configuration for the given models.")
    return jobs


def _standard_test_jobs_for_model(model_path, manifests, artifact_root) -> list[dict]:
    model_alias = _model_alias(model_path)
    jobs = []
    for manifest in manifests:
        for item in manifest.get("selections", []):
            if item.get("model", "").casefold() != model_alias.casefold():
                continue
            if item.get("selected") is None:
                continue
            method = item["method"]
            selected = item["selected"]
            mode = item["application_mode"]
            # One artifact root per method: the standard-phase path does not
            # include the intervention, so ActAdd and KAS selected at the same
            # layer and strength would otherwise share (and race on) a run dir.
            jobs.append({
                "id": f"jail-standard-{model_alias}-{mode}-{method}",
                "experiment": "jailbreak",
                "phase": "standard",
                "argv": (
                    _base_command(model_path, os.path.join(artifact_root, method))
                    + _fixed_args()
                    + ["--evaluation_protocol", "legacy", "--layer", str(selected["layer"])]
                    + _standard_test_method_args(method, selected)
                    + ["--evaluation_interventions", METHOD_TO_INTERVENTION[method]]
                    + _prompt_flag(mode)
                ),
            })
    return jobs


def _single_pair_jobs(args) -> list[dict]:
    jobs = []
    for model_path in args.model_path:
        model_alias = _model_alias(model_path)
        if model_alias not in SINGLE_PAIR_SETTINGS:
            raise ValueError(
                f"No single-pair settings for {model_alias}; expected one of {list(SINGLE_PAIR_SETTINGS)}."
            )
        layer, eta_one_step, eta_ten_steps = SINGLE_PAIR_SETTINGS[model_alias]
        for seed in args.seeds:
            for step_count, eta in ((1, eta_one_step), (10, eta_ten_steps)):
                jobs.append({
                    "id": f"single-pair-{model_alias}-seed{seed}-steps{step_count}-eta{eta:g}",
                    "experiment": "single_pair",
                    "phase": "standard",
                    "argv": (
                        _base_command(model_path, args.artifact_root)
                        + _kas_args(layer=layer, eta=eta, step_count=step_count, seed=seed, lambda_reg=args.lambda_reg)
                        + ["--kernel_type", "rbf", "--n_train", "1"]
                    ),
                })
    return jobs


def _n_train_jobs(args) -> list[dict]:
    model_alias = _model_alias(args.model_path)
    jobs = []
    for n_train in args.n_train:
        for mode in APPLICATION_MODES:
            jobs.append({
                "id": f"n-train-{model_alias}-n{n_train}-{mode}",
                "experiment": "n_train",
                "phase": "standard",
                "argv": (
                    _base_command(args.model_path, args.artifact_root)
                    + _kas_args(layer=args.layer, eta=args.eta, step_count=1, seed=args.seed, lambda_reg=args.lambda_reg)
                    + ["--kernel_type", "rbf", "--n_train", str(n_train)]
                    + _prompt_flag(mode)
                ),
            })
    return jobs


def _kernel_choice_jobs(args) -> list[dict]:
    model_alias = _model_alias(args.model_path)
    jobs = []
    for kernel in args.kernels:
        for mode in APPLICATION_MODES:
            jobs.append({
                "id": f"kernel-choice-{model_alias}-{kernel}-{mode}",
                "experiment": "kernel_choice",
                "phase": "standard",
                "argv": (
                    _base_command(args.model_path, args.artifact_root)
                    + _kas_args(layer=args.layer, eta=args.eta, step_count=1, seed=args.seed, lambda_reg=args.lambda_reg)
                    + ["--kernel_type", kernel]
                    + _prompt_flag(mode)
                ),
            })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output")
    parser.add_argument("--format", choices=("shell", "jsonl"), default="shell")
    subparsers = parser.add_subparsers(dest="command", required=True)

    test = subparsers.add_parser("standard-test")
    test.add_argument("--model-path", nargs="+", required=True)
    test.add_argument("--artifact-root", required=True)
    test.add_argument("--manifest", nargs="+", required=True, help="One or more jailbreak manifests.")

    single = subparsers.add_parser("single-pair")
    single.add_argument("--model-path", nargs="+", required=True)
    single.add_argument("--artifact-root", required=True)
    single.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="One reference pair per seed.")
    single.add_argument("--lambda-reg", type=float, default=0.1)

    n_train = subparsers.add_parser("n-train")
    n_train.add_argument("--model-path", required=True)
    n_train.add_argument("--artifact-root", required=True)
    n_train.add_argument("--n-train", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    n_train.add_argument("--layer", type=int, default=22)
    n_train.add_argument("--eta", type=float, default=150.0)
    n_train.add_argument("--seed", type=int, default=1)
    n_train.add_argument("--lambda-reg", type=float, default=0.1)

    kernel = subparsers.add_parser("kernel-choice")
    kernel.add_argument("--model-path", required=True)
    kernel.add_argument("--artifact-root", required=True)
    kernel.add_argument("--kernels", nargs="+", default=["laplacian", "polynomial", "rbf"])
    kernel.add_argument("--layer", type=int, default=13)
    kernel.add_argument("--eta", type=float, default=5.0)
    kernel.add_argument("--seed", type=int, default=1)
    kernel.add_argument("--lambda-reg", type=float, default=0.1)

    args = parser.parse_args()
    builders = {
        "standard-test": _standard_test_jobs,
        "single-pair": _single_pair_jobs,
        "n-train": _n_train_jobs,
        "kernel-choice": _kernel_choice_jobs,
    }
    _emit(builders[args.command](args), args.output, args.format)


if __name__ == "__main__":
    main()

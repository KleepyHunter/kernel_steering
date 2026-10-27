#!/usr/bin/env python3
"""Run TinyBenchmarks only after validating a locked selection/config file."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.validation_selection import (
    CHARS_RECIPES,
    assert_chars_recipe_configuration,
    assert_locked_configuration,
    load_manifest,
)
from pipeline.sas_validation import validate_sas_steer_config
from pipeline.utils.chars import validate_chars_state

DEFAULT_TASKS = "tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande"


def _validate_steer_config(
    path, method, layer, strength, prompt_only, manifest, model,
    n_clusters=None, state_fingerprint=None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to inspect the serialized steer config") from exc
    payload = torch.load(path, map_location="cpu", weights_only=True)
    key = f"layers.{layer}"
    if set(payload) != {key}:
        raise ValueError(f"Steer config must contain only {key}; found {sorted(payload)}")
    config = payload[key]
    expected_action = {
        "actadd": "add",
        "kas": "kernel",
        "curveball": "curveball_kpca",
        "chars": "chars",
        "sas": "sas_sparse",
    }[method]
    if config.get("action") != expected_action:
        raise ValueError(f"Expected action={expected_action}, found {config.get('action')}")
    strength_key = {
        "kas": "eta",
        "chars": "alpha",
    }.get(method, "steering_coefficient")
    if not math.isclose(float(config[strength_key]), float(strength), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Serialized steering strength does not match the manifest.")
    if bool(config.get("prompt_only", False)) != prompt_only:
        raise ValueError("Serialized prompt_only mode does not match the manifest.")
    if method == "sas":
        validate_sas_steer_config(config, layer=layer, manifest=manifest, model=model)
    if method == "chars":
        if n_clusters is None or state_fingerprint is None:
            raise ValueError("Locked CHaRS TinyBench requires K and state fingerprint.")
        state = config.get("state")
        if not isinstance(state, dict):
            raise ValueError("Serialized CHaRS config is missing its embedded state.")
        fingerprint = validate_chars_state(
            state,
            expected_n_clusters=n_clusters,
            expected_fingerprint=state_fingerprint,
            expected_metadata={
                "model": model,
                "layer": int(layer),
                "position": -1,
                "hookpoint": "block_output",
                "extraction_source": "legacy",
                "orientation": "harmful_to_harmless",
            },
        )
        if config.get("state_fingerprint") != fingerprint:
            raise ValueError("Serialized CHaRS config fingerprint does not match state.")
        if int(config.get("n_clusters", -1)) != int(n_clusters):
            raise ValueError("Serialized CHaRS K does not match the manifest.")
        if config.get("recipe") != state.get("recipe"):
            raise ValueError("Serialized CHaRS config recipe does not match state.")
        recipe = CHARS_RECIPES["jailbreak"]
        if config.get("hookpoint") != recipe["hookpoint"]:
            raise ValueError("Serialized CHaRS hookpoint is invalid.")
        if int(config.get("position", 0)) != recipe["position"]:
            raise ValueError("Serialized CHaRS position is invalid.")
        assert_chars_recipe_configuration(
            manifest,
            experiment="jailbreak",
            model=model,
            extraction_source=recipe["extraction_source"],
            position=recipe["position"],
            n_clusters=n_clusters,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--method",
        choices=("baseline", "actadd", "kas", "curveball", "chars", "sas"),
        required=True,
    )
    parser.add_argument("--application-mode", choices=("prompt_only", "all_tokens"), required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--steer-path")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--harness-root", default=str(REPO_ROOT / "lm-evaluation-harness"))
    parser.add_argument("--lm-eval", default="lm_eval")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if manifest.get("experiment") != "jailbreak":
        raise ValueError("TinyBench locked utility evaluation requires a jailbreak manifest.")
    model_alias = os.path.basename(os.path.normpath(args.model_path))
    prompt_only = args.application_mode == "prompt_only"
    selected = None
    if args.method != "baseline":
        if args.layer is None or args.strength is None or args.steer_path is None:
            raise ValueError("Steered TinyBench runs require layer, strength, and steer-path.")
        selected = assert_locked_configuration(
            manifest,
            experiment="jailbreak",
            model=model_alias,
            method=args.method,
            mode=args.application_mode,
            layer=args.layer,
            strength=args.strength,
            n_clusters=args.n_clusters,
        )
        _validate_steer_config(
            args.steer_path,
            args.method,
            args.layer,
            args.strength,
            prompt_only,
            manifest,
            model_alias,
            n_clusters=args.n_clusters,
            state_fingerprint=selected.get("state_fingerprint"),
        )
        command = [
            args.lm_eval,
            "--model", "steered",
            "--model_args", f"pretrained={args.model_path},steer_path={args.steer_path}",
        ]
    else:
        if args.steer_path is not None:
            raise ValueError("Baseline TinyBench must not receive a steer config.")
        command = [args.lm_eval, "--model", "hf", "--model_args", f"pretrained={args.model_path}"]

    command += [
        "--tasks", args.tasks,
        "--device", args.device,
        "--batch_size", args.batch_size,
        "--output_path", args.output_path,
    ]
    metadata = {
        "experiment": "jailbreak",
        "phase": "test",
        "model": model_alias,
        "method": args.method,
        "application_mode": args.application_mode,
        "validation_selected": None if selected is None else selected["selected"],
        "state_fingerprint": (
            None if selected is None else selected.get("state_fingerprint")
        ),
        "selection_manifest": str(Path(args.manifest).resolve()),
        "steer_path": None if args.steer_path is None else str(Path(args.steer_path).resolve()),
        "tasks": args.tasks.split(","),
        "command": command,
    }
    metadata_path = Path(args.output_path + ".selection.json")
    if not args.dry_run:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        subprocess.run(command, cwd=args.harness_root, check=True)
    print(shlex.join(command))


if __name__ == "__main__":
    main()

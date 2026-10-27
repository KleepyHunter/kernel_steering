#!/usr/bin/env python3
"""Fit validation-locked CHaRS states from one shared activation extraction."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.load_dataset import (
    load_caa_reference_validation,
    load_dataset_split,
)
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.chars import (
    fit_chars_state,
    get_block_output_activations,
    validate_chars_state,
)
from pipeline.validation_selection import (
    CHARS_VALID_CLUSTERS,
    chars_validation_clusters,
    fingerprint_records,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment", choices=("caa", "jailbreak"), required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--caa-behavior")
    parser.add_argument("--caa-dataset-root", default="caa")
    parser.add_argument("--caa-validation-size", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _references(args):
    if args.experiment == "caa":
        if args.caa_behavior is None:
            raise ValueError("CAA CHaRS fitting requires --caa-behavior.")
        pairs, _, split_metadata = load_caa_reference_validation(
            behavior=args.caa_behavior,
            dataset_root=args.caa_dataset_root,
            validation_size=args.caa_validation_size,
            seed=args.split_seed,
        )
        instructions = [item["instruction"] for item in pairs]
        source_outputs = [item["negative_output"] for item in pairs]
        target_outputs = [item["positive_output"] for item in pairs]
        return {
            "source_instructions": instructions,
            "source_outputs": source_outputs,
            "target_instructions": instructions,
            "target_outputs": target_outputs,
            "position": -2,
            "extraction_source": "caa",
            "orientation": "behavior_not_matching_to_behavior_matching",
            "split_fingerprints": {
                "caa_reference": split_metadata["fingerprints"]["reference"]
            },
            "scope": args.caa_behavior,
        }

    if args.n_train != 128 or args.split_seed != 42:
        raise ValueError(
            "Validation-locked jailbreak CHaRS fitting requires 128 references and seed 42."
        )
    rng = random.Random(args.split_seed)
    harmful = rng.sample(
        load_dataset_split(harmtype="harmful", split="train", instructions_only=True),
        args.n_train,
    )
    harmless = rng.sample(
        load_dataset_split(harmtype="harmless", split="train", instructions_only=True),
        args.n_train,
    )
    return {
        "source_instructions": harmful,
        "source_outputs": None,
        "target_instructions": harmless,
        "target_outputs": None,
        "position": -1,
        "extraction_source": "legacy",
        "orientation": "harmful_to_harmless",
        "split_fingerprints": {
            "harmful_reference": fingerprint_records(
                [{"instruction": item} for item in harmful]
            ),
            "harmless_reference": fingerprint_records(
                [{"instruction": item} for item in harmless]
            ),
        },
        "scope": None,
    }


def _state_root(args, model_alias, scope):
    root = Path(args.output_root).expanduser().resolve()
    if args.experiment == "caa":
        return root / "caa" / model_alias / str(scope)
    return root / "jailbreak" / model_alias


def main():
    args = _parse_args()
    if args.split_seed != 42:
        raise ValueError("Validation-locked CHaRS fitting fixes split seed 42.")
    layers = tuple(dict.fromkeys(args.layers))
    model_alias = os.path.basename(os.path.normpath(args.model_path))
    expected_clusters = chars_validation_clusters(model_alias)
    clusters = (
        expected_clusters
        if args.clusters is None
        else tuple(dict.fromkeys(args.clusters))
    )
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("CHaRS fit layers must be non-negative and non-empty.")
    if not clusters or any(value not in CHARS_VALID_CLUSTERS for value in clusters):
        raise ValueError("CHaRS fit clusters must lie in [1, 15].")
    if clusters != expected_clusters:
        raise ValueError(
            f"Validation-locked CHaRS fitting for {model_alias} requires "
            f"K={expected_clusters[0]}; received {list(clusters)}."
        )

    references = _references(args)
    model_base = construct_model_base(args.model_path)
    source_by_layer = get_block_output_activations(
        model_base,
        references["source_instructions"],
        outputs=references["source_outputs"],
        layers=layers,
        position=references["position"],
        batch_size=args.batch_size,
    )
    target_by_layer = get_block_output_activations(
        model_base,
        references["target_instructions"],
        outputs=references["target_outputs"],
        layers=layers,
        position=references["position"],
        batch_size=args.batch_size,
    )

    root = _state_root(args, model_alias, references["scope"])
    entries = []
    for layer in layers:
        metadata = {
            "model": model_alias,
            "layer": int(layer),
            "position": int(references["position"]),
            "hookpoint": "block_output",
            "extraction_source": references["extraction_source"],
            "orientation": references["orientation"],
            "split_fingerprints": references["split_fingerprints"],
        }
        for n_clusters in clusters:
            state = fit_chars_state(
                source_by_layer[layer],
                target_by_layer[layer],
                n_clusters=n_clusters,
                metadata=metadata,
            )
            path = root / f"layer-{layer}" / f"k-{n_clusters}" / "chars_state.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = torch.load(path, map_location="cpu", weights_only=True)
                validate_chars_state(
                    existing,
                    expected_n_clusters=n_clusters,
                    expected_fingerprint=state["state_fingerprint"],
                    expected_metadata=metadata,
                )
            else:
                torch.save(state, path)
            entries.append(
                {
                    "layer": layer,
                    "n_clusters": n_clusters,
                    "state_fingerprint": state["state_fingerprint"],
                    "path": str(path),
                }
            )

    bundle = {
        "format": "chars_state_bundle_v1",
        "experiment": args.experiment,
        "model": model_alias,
        "scope": references["scope"],
        "position": references["position"],
        "split_fingerprints": references["split_fingerprints"],
        "states": entries,
    }
    bundle_path = root / "bundle.json"
    serialized = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    if bundle_path.exists() and bundle_path.read_text() != serialized:
        raise ValueError(f"Existing CHaRS bundle manifest differs: {bundle_path}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(serialized)
    print(bundle_path)


if __name__ == "__main__":
    main()

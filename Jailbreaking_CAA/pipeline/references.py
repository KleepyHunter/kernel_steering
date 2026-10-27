"""Reference prompts each steering method is fitted on.

Jailbreak runs use harmful/harmless instruction splits; CAA runs use paired
behavior-matching / behavior-not-matching answers to the same questions.
"""

import random
from dataclasses import dataclass
from typing import Optional

from dataset.load_dataset import (
    load_caa_pairs,
    load_caa_reference_validation,
    load_dataset_split,
)
from pipeline.settings import uses_caa_pair_extraction
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.validation_selection import fingerprint_records


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (target_train, source_train, target_val, source_val)
    """
    random.seed(cfg.seed if cfg.n_train == 1 else 42)
    target_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    source_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    target_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    source_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    return target_train, source_train, target_val, source_val


def maybe_log_single_train_prompts(cfg, target_train, source_train):
    """Log source/target train prompts when n_train is configured as 1."""
    if cfg.n_train != 1:
        return

    def _render_prompt(dataset, label):
        if len(dataset) == 0:
            print(f"[n_train=1] {label} prompt: <EMPTY AFTER FILTERING>")
            return
        if len(dataset) > 1:
            print(f"[n_train=1] {label} prompts count={len(dataset)}; showing first prompt.")
        print(f"[n_train=1] {label} prompt: {dataset[0]}")

    print("[n_train=1] Logging source/target training prompts.")
    _render_prompt(target_train, "target")
    _render_prompt(source_train, "source")


def _refill_filtered(cfg, model_base, harmtype, sampled, kept, keep):
    """Top `kept` up to cfg.n_train with unsampled train prompts that pass `keep`.

    Replacements come from the rest of the train split in a fixed seed-42
    order, drawn from a private RNG so the global sampling state is untouched.
    """
    if len(kept) >= cfg.n_train:
        return kept
    kept = list(kept)
    sampled_set = set(sampled)
    candidates = [inst for inst in load_dataset_split(harmtype=harmtype, split='train', instructions_only=True)
                  if inst not in sampled_set]
    random.Random(42).shuffle(candidates)
    n_filtered = cfg.n_train - len(kept)
    while len(kept) < cfg.n_train and candidates:
        batch, candidates = candidates[:32], candidates[32:]
        scores = get_refusal_scores(model_base.model, batch, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        for inst, score in zip(batch, scores.tolist()):
            if keep(score) and len(kept) < cfg.n_train:
                kept.append(inst)
    if len(kept) < cfg.n_train:
        raise ValueError(
            f"Only {len(kept)} {harmtype} train prompts pass refusal filtering; --n_train {cfg.n_train} needs more."
        )
    print(f"[n_train={cfg.n_train}] Replaced {n_filtered} filtered {harmtype} train prompts.")
    return kept


def filter_data(cfg, model_base, target_train, source_train, target_val, source_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (target_train, source_train, target_val, source_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train and cfg.n_train == 1:
        print("[n_train=1] Skipping train filtering.")
    elif cfg.filter_train:
        target_train_scores = get_refusal_scores(model_base.model, target_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        source_train_scores = get_refusal_scores(model_base.model, source_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        target_kept = filter_examples(target_train, target_train_scores, 0, lambda x, y: x > y)
        source_kept = filter_examples(source_train, source_train_scores, 0, lambda x, y: x < y)
        # Default runs (n_train=128) keep whatever survives filtering. Other sizes
        # (the reference-count ablation) refill to exactly n_train pairs.
        if cfg.n_train != 128:
            target_kept = _refill_filtered(cfg, model_base, 'harmful', target_train, target_kept, lambda score: score > 0)
            source_kept = _refill_filtered(cfg, model_base, 'harmless', source_train, source_kept, lambda score: score < 0)
        target_train, source_train = target_kept, source_kept

    return target_train, source_train, target_val, source_val


@dataclass
class References:
    """Fitting data for one run. `target_*` is the behavior steered toward."""

    target_train: list
    source_train: list
    target_outputs: Optional[list]
    source_outputs: Optional[list]
    caa_validation_dataset: Optional[list]
    caa_split_metadata: Optional[dict]
    split_fingerprints: dict


def load_references(cfg, model_base) -> References:
    target_outputs, source_outputs = None, None
    caa_validation_dataset = None
    caa_split_metadata = None
    reference_split_fingerprints = {}

    if uses_caa_pair_extraction(
            kernel_extraction_source=cfg.kernel_extraction_source,
            steer_method=cfg.steer_method,
            curveball_extraction_source=cfg.curveball_extraction_source,
            chars_extraction_source=cfg.chars_extraction_source,
            sas_extraction_source=cfg.sas_extraction_source,
        ):
        if cfg.experiment_phase in {"select", "test"}:
            caa_pairs, caa_validation_dataset, caa_split_metadata = (
                load_caa_reference_validation(
                    behavior=cfg.caa_behavior,
                    dataset_root=cfg.caa_dataset_root,
                    validation_size=cfg.caa_validation_size,
                    seed=cfg.split_seed,
                )
            )
        else:
            caa_pairs = load_caa_pairs(
                behavior=cfg.caa_behavior,
                split="generate",
                dataset_root=cfg.caa_dataset_root,
            )
        if len(caa_pairs) == 0:
            raise ValueError("CAA dataset split is empty. Cannot extract a contrastive direction.")

        instructions = [x["instruction"] for x in caa_pairs]
        target_outputs = [x["positive_output"] for x in caa_pairs]
        source_outputs = [x["negative_output"] for x in caa_pairs]

        if not (len(instructions) == len(target_outputs) == len(source_outputs)):
            raise ValueError(
                "CAA extraction requires equal-length instruction/positive/negative lists. "
                f"Got lengths: instructions={len(instructions)}, "
                f"positive={len(target_outputs)}, negative={len(source_outputs)}"
            )

        target_train = instructions
        source_train = instructions
        reference_split_fingerprints = {
            "caa_reference": (
                caa_split_metadata["fingerprints"]["reference"]
                if caa_split_metadata is not None
                else fingerprint_records(
                    [
                        {
                            "instruction": instruction,
                            "positive_output": positive,
                            "negative_output": negative,
                        }
                        for instruction, positive, negative in zip(
                            instructions, target_outputs, source_outputs
                        )
                    ]
                )
            )
        }
        maybe_log_single_train_prompts(cfg, target_train, source_train)
    else:
        # Load and sample datasets
        target_train, source_train, target_val, source_val = load_and_sample_datasets(cfg)

        # Validation-selection references are exactly the seed-42 samples.
        if cfg.experiment_phase == "standard":
            target_train, source_train, target_val, source_val = filter_data(
                cfg, model_base, target_train, source_train, target_val, source_val
            )
        maybe_log_single_train_prompts(cfg, target_train, source_train)
        reference_split_fingerprints = {
            "harmful_reference": fingerprint_records(
                [{"instruction": item} for item in target_train]
            ),
            "harmless_reference": fingerprint_records(
                [{"instruction": item} for item in source_train]
            ),
        }

    return References(
        target_train=target_train,
        source_train=source_train,
        target_outputs=target_outputs,
        source_outputs=source_outputs,
        caa_validation_dataset=caa_validation_dataset,
        caa_split_metadata=caa_split_metadata,
        split_fingerprints=reference_split_fingerprints,
    )

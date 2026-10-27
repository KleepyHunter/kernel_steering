"""Evaluation stages of a run.

- Standard jailbreak runs: JAILBREAKBENCH ASR.
- Validation / locked-test jailbreak runs: ASR on the 39 validation (or JAILBREAKBENCH
  test) prompts and, for selection, perplexity on the 256-example utility split.
- CAA runs: behavior-matching answer probability, optionally open-ended scoring.
- Every run (step 5): CE loss / perplexity on Pile and Alpaca.
"""

import gc
import json
import os

import torch

from dataset.load_dataset import load_caa_test_dataset, load_dataset, load_dataset_split
from pipeline import sas_validation
from pipeline.phase_validation import chars_fixed_parameters, curveball_caa_fixed_parameters
from pipeline.submodules.evaluate_caa import (
    evaluate_choice_dataset,
    evaluate_open_ended_dataset,
    get_caa_system_prompt,
)
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import (
    evaluate_alpaca_validation,
    evaluate_loss,
    load_alpaca_validation_examples,
)
from pipeline.validation_selection import (
    INTERVENTION_TO_METHOD,
    application_mode,
    assert_locked_configuration,
    canonical_question,
    fingerprint_records,
    method_strength,
)


def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

    del completions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)
    return evaluation


def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate CE loss / perplexity on Pile and Alpaca."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, dataset_labels=["pile", "alpaca"])

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)


def run_loss_evaluations(cfg, model_base, interventions):
    """Step 5 of every run: CE loss / perplexity for each evaluated intervention."""
    for intervention_label, (fwd_pre_hooks, fwd_hooks) in interventions.items():
        evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label)


def run_validation_jailbreak_evaluations(
    cfg,
    model_base,
    interventions,
    target_train,
    source_train,
    locked_manifest,
):
    harmful_validation = load_dataset_split(harmtype="harmful", split="val")
    harmless_validation = load_dataset_split(harmtype="harmless", split="val")
    utility_examples, utility_metadata = load_alpaca_validation_examples(
        harmless_validation,
        n_examples=cfg.jailbreak_utility_size,
        seed=cfg.split_seed,
    )
    jailbreak_test = load_dataset("jailbreakbench")
    split_fingerprints = {
        "harmful_reference": fingerprint_records(
            [{"instruction": item} for item in target_train]
        ),
        "harmless_reference": fingerprint_records(
            [{"instruction": item} for item in source_train]
        ),
        "harmful_validation": fingerprint_records(harmful_validation),
        "harmless_utility_validation": utility_metadata["split_fingerprint"],
        "jailbreakbench_test": fingerprint_records(jailbreak_test),
    }
    if len(harmful_validation) != 39:
        raise ValueError(
            f"Expected 39 harmful validation prompts, found {len(harmful_validation)}."
        )
    harmful_reference_questions = {canonical_question(item) for item in target_train}
    harmful_validation_questions = {
        canonical_question(item["instruction"]) for item in harmful_validation
    }
    jailbreak_test_questions = {
        canonical_question(item["instruction"]) for item in jailbreak_test
    }
    if (
        harmful_reference_questions & harmful_validation_questions
        or harmful_reference_questions & jailbreak_test_questions
        or harmful_validation_questions & jailbreak_test_questions
    ):
        raise ValueError(
            "Harmful reference, ASR validation, and JAILBREAKBENCH splits overlap."
        )
    harmless_reference_questions = {canonical_question(item) for item in source_train}
    utility_questions = {
        canonical_question(item["instruction"]) for item in utility_examples
    }
    if harmless_reference_questions & utility_questions:
        raise ValueError("Harmless reference and utility-validation splits overlap.")
    if cfg.experiment_phase == "test" and (
        locked_manifest.get("split_fingerprints", {}) != split_fingerprints
    ):
        raise ValueError(
            "Jailbreak split fingerprints do not match the validation-locked manifest."
        )

    if cfg.jailbreak_eval_split == "validation":
        evaluation_jobs = [("harmful_validation", harmful_validation)]
    else:
        evaluation_jobs = [("jailbreakbench", jailbreak_test)]

    evaluations = {}
    for dataset_name, dataset in evaluation_jobs:
        for intervention_label, (fwd_pre_hooks, fwd_hooks) in interventions.items():
            generate_and_save_completions_for_dataset(
                cfg,
                model_base,
                fwd_pre_hooks,
                fwd_hooks,
                intervention_label,
                dataset_name,
                dataset=dataset,
            )
            evaluation = evaluate_completions_and_save_results_for_dataset(
                cfg,
                intervention_label,
                dataset_name,
                eval_methodologies=cfg.jailbreak_eval_methodologies,
            )
            if cfg.experiment_phase == "test" and intervention_label != "baseline":
                locked = assert_locked_configuration(
                    locked_manifest,
                    experiment="jailbreak",
                    model=cfg.model_alias,
                    method=INTERVENTION_TO_METHOD[intervention_label],
                    mode=application_mode(cfg.prompt_only),
                    layer=cfg.layer,
                    strength=method_strength(
                        intervention_label,
                        eta=cfg.eta,
                        curveball_alpha=cfg.curveball_alpha,
                        chars_alpha=cfg.chars_alpha,
                        sas_coefficient=cfg.sas_coefficient,
                    ),
                    n_clusters=(
                        cfg.chars_clusters if intervention_label == "chars" else None
                    ),
                    state_fingerprint=(
                        getattr(cfg, "chars_state_fingerprint", None)
                        if intervention_label == "chars" else None
                    ),
                )
                evaluation["validation_selected"] = locked["selected"]
                evaluation_path = os.path.join(
                    cfg.artifact_path(),
                    "completions",
                    f"{dataset_name}_{intervention_label}_evaluations.json",
                )
                with open(evaluation_path, "w") as handle:
                    json.dump(evaluation, handle, indent=4)
            evaluations[(dataset_name, intervention_label)] = evaluation

    if cfg.experiment_phase != "select":
        return

    candidate_records = []
    utility_dir = os.path.join(cfg.artifact_path(), "validation_utility")
    os.makedirs(utility_dir, exist_ok=True)
    for intervention_label, (fwd_pre_hooks, fwd_hooks) in interventions.items():
        utility = evaluate_alpaca_validation(
            model_base,
            utility_examples,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            batch_size=cfg.ce_loss_batch_size,
            prompt_only=cfg.prompt_only,
        )
        with open(os.path.join(utility_dir, f"{intervention_label}.json"), "w") as handle:
            json.dump(utility, handle, indent=4)
        evaluation = evaluations[("harmful_validation", intervention_label)]
        modes = ("prompt_only", "all_tokens") if intervention_label == "baseline" else (application_mode(cfg.prompt_only),)
        for mode in modes:
            candidate_records.append(
                {
                    "experiment": "jailbreak",
                    "split": "validation",
                    "model": cfg.model_alias,
                    "method": INTERVENTION_TO_METHOD[intervention_label],
                    **(
                        {
                            "strength_parameter": {
                                "chars": "chars_alpha",
                                "sas": "sas_coefficient",
                            }[intervention_label]
                        }
                        if intervention_label in {"chars", "sas"}
                        else {}
                    ),
                    "application_mode": mode,
                    "layer": None if intervention_label == "baseline" else cfg.layer,
                    "strength": method_strength(
                        intervention_label,
                        eta=cfg.eta,
                        curveball_alpha=cfg.curveball_alpha,
                        chars_alpha=cfg.chars_alpha,
                        sas_coefficient=cfg.sas_coefficient,
                    ),
                    **(
                        {
                            "n_clusters": cfg.chars_clusters,
                            "state_fingerprint": cfg.chars_state_fingerprint,
                        }
                        if intervention_label == "chars" else {}
                    ),
                    "validation_asr": evaluation["llamaguard3_success_rate"],
                    "validation_perplexity": utility["perplexity"],
                    "n_harmful_validation_examples": len(harmful_validation),
                    "n_harmless_validation_examples": len(utility_examples),
                    "split_fingerprints": split_fingerprints,
                    "fixed_parameters": {
                        "kas": {
                            "kernel": "rbf",
                            "bandwidth": "reference_median_heuristic",
                            "lambda_reg": 0.0,
                            "gradient_steps": 1,
                        },
                        "curveball": {
                            "kernel": "polynomial",
                            "degree": 3,
                            "bias": 1.0,
                            "kpca_components": 20,
                            "inverse_transform_alpha": cfg.curveball_fit_inverse_alpha,
                        },
                        **(
                            {"chars": chars_fixed_parameters(cfg)}
                            if intervention_label == "chars"
                            else {}
                        ),
                        **(
                            {"sas": sas_validation.sas_fixed_parameters(cfg)}
                            if intervention_label == "sas"
                            else {}
                        ),
                    },
                }
            )
    with open(os.path.join(cfg.artifact_path(), "selection_candidates.json"), "w") as handle:
        json.dump(candidate_records, handle, indent=4)


def run_standard_jailbreak_evaluations(cfg, model_base, interventions, skip_baseline):
    """Step 3 of a standard (non-validation) jailbreak run: JAILBREAKBENCH completions + ASR."""
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    # 3a. Generate and save completions on target evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        if not skip_baseline:
            # generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
            pass
        for intervention_label, (fwd_pre_hooks, fwd_hooks) in interventions.items():
            generate_and_save_completions_for_dataset(
                cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name
            )

    # 3b. Evaluate completions and save results on target evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        if not skip_baseline:
            # evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
            pass
        for intervention_label in interventions.keys():
            evaluate_completions_and_save_results_for_dataset(
                cfg,
                intervention_label,
                dataset_name,
                eval_methodologies=cfg.jailbreak_eval_methodologies,
            )


def run_caa_evaluations(
    cfg,
    model_base,
    interventions,
    validation_dataset=None,
    split_metadata=None,
    locked_manifest=None,
):
    system_prompt = get_caa_system_prompt(cfg.caa_behavior, cfg.caa_eval_system_prompt)
    eval_dir = os.path.join(cfg.artifact_path(), "caa_eval")
    os.makedirs(eval_dir, exist_ok=True)
    prompt_condition = cfg.caa_eval_system_prompt or "none"

    if cfg.experiment_phase == "test":
        requested = [
            label for label in interventions if label != "baseline"
        ]
        if requested:
            locked = assert_locked_configuration(
                locked_manifest,
                experiment="caa",
                model=cfg.model_alias,
                method=INTERVENTION_TO_METHOD[requested[0]],
                mode=application_mode(cfg.prompt_only),
                layer=cfg.layer,
                strength=method_strength(
                    requested[0],
                    eta=cfg.eta,
                    curveball_alpha=cfg.curveball_alpha,
                    chars_alpha=cfg.chars_alpha,
                    sas_coefficient=cfg.sas_coefficient,
                ),
                n_clusters=(cfg.chars_clusters if requested[0] == "chars" else None),
                state_fingerprint=(
                    getattr(cfg, "chars_state_fingerprint", None)
                    if requested[0] == "chars" else None
                ),
                behavior=cfg.caa_behavior,
                prompt_condition=prompt_condition,
            )
            expected = locked.get("split_fingerprints", {})
        else:
            scope_matches = [
                item for item in locked_manifest.get("selections", [])
                if item.get("model") == cfg.model_alias
                and item.get("behavior") == cfg.caa_behavior
                and item.get("prompt_condition") == prompt_condition
                and item.get("application_mode") == application_mode(cfg.prompt_only)
            ]
            if not scope_matches:
                raise ValueError("Manifest has no matching CAA scope for the baseline test.")
            fingerprints = {
                json.dumps(item.get("split_fingerprints", {}), sort_keys=True)
                for item in scope_matches
            }
            if len(fingerprints) != 1:
                raise ValueError("Manifest methods disagree on CAA split fingerprints.")
            expected = json.loads(next(iter(fingerprints)))
        actual = (split_metadata or {}).get("fingerprints", {})
        if expected != actual:
            raise ValueError(
                "CAA split fingerprints do not match the validation-locked manifest."
            )

    summary = {
        "metadata": {
            "experiment_phase": cfg.experiment_phase,
            "evaluation_split": cfg.caa_eval_split,
            "steer_method": cfg.steer_method,
            "behavior": cfg.caa_behavior,
            "layer": cfg.layer,
            "pos": cfg.pos,
            "eta": cfg.eta,
            "lambda_reg": cfg.lambda_reg,
            "step_count": cfg.step_count,
            "chars_alpha": cfg.chars_alpha,
            "chars_clusters": cfg.chars_clusters,
            "chars_extraction_source": cfg.chars_extraction_source,
            "chars_state_fingerprint": getattr(cfg, "chars_state_fingerprint", None),
            "system_prompt": prompt_condition,
            "types": list(cfg.caa_eval_types),
            "split_metadata": split_metadata,
            "selection_manifest": cfg.selection_manifest,
        },
        "results": {},
    }
    candidate_records = []

    for eval_type in cfg.caa_eval_types:
        summary["results"][eval_type] = {}
        if cfg.caa_eval_split == "validation":
            if eval_type != "ab" or validation_dataset is None:
                raise ValueError(
                    "CAA validation selection supports only the held-out AB split."
                )
            dataset = validation_dataset
        else:
            dataset = load_caa_test_dataset(
                eval_type=eval_type,
                dataset_root=cfg.caa_dataset_root,
                behavior=cfg.caa_behavior if eval_type in {"ab", "open_ended"} else None,
            )

        for intervention_label, (fwd_pre_hooks, fwd_hooks) in interventions.items():
            if eval_type == "open_ended":
                result = evaluate_open_ended_dataset(
                    model_base=model_base,
                    dataset=dataset,
                    max_new_tokens=cfg.max_new_tokens,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                    system_prompt=system_prompt,
                    score_outputs=cfg.caa_eval_open_ended_score,
                    behavior=cfg.caa_behavior,
                    judge_model=cfg.caa_open_ended_judge_model,
                )
            else:
                target_key = "answer_matching_behavior" if eval_type == "ab" else "correct"
                result = evaluate_choice_dataset(
                    model_base=model_base,
                    dataset=dataset,
                    target_key=target_key,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                    system_prompt=system_prompt,
                )

            if cfg.experiment_phase == "test" and intervention_label != "baseline":
                locked = assert_locked_configuration(
                    locked_manifest,
                    experiment="caa",
                    model=cfg.model_alias,
                    method=INTERVENTION_TO_METHOD[intervention_label],
                    mode=application_mode(cfg.prompt_only),
                    layer=cfg.layer,
                    strength=method_strength(
                        intervention_label,
                        eta=cfg.eta,
                        curveball_alpha=cfg.curveball_alpha,
                        chars_alpha=cfg.chars_alpha,
                        sas_coefficient=cfg.sas_coefficient,
                    ),
                    n_clusters=(
                        cfg.chars_clusters if intervention_label == "chars" else None
                    ),
                    state_fingerprint=(
                        getattr(cfg, "chars_state_fingerprint", None)
                        if intervention_label == "chars" else None
                    ),
                    behavior=cfg.caa_behavior,
                    prompt_condition=prompt_condition,
                )
                result["summary"]["validation_selected"] = locked["selected"]

            raw_path = os.path.join(eval_dir, f"raw_{eval_type}_{intervention_label}.json")
            result_path = os.path.join(eval_dir, f"summary_{eval_type}_{intervention_label}.json")
            with open(raw_path, "w") as handle:
                json.dump(result["predictions"], handle, indent=4)
            with open(result_path, "w") as handle:
                json.dump(result["summary"], handle, indent=4)
            summary["results"][eval_type][intervention_label] = result["summary"]

            if cfg.experiment_phase == "select" and eval_type == "ab":
                candidate_records.append(
                    {
                        "experiment": "caa",
                        "split": "validation",
                        "model": cfg.model_alias,
                        "method": INTERVENTION_TO_METHOD[intervention_label],
                        **(
                            {
                                "strength_parameter": {
                                    "curveball": "curveball_alpha",
                                    "chars": "chars_alpha",
                                    "sas": "sas_coefficient",
                                }[intervention_label]
                            }
                            if intervention_label in {"curveball", "chars", "sas"}
                            else {}
                        ),
                        "behavior": cfg.caa_behavior,
                        "prompt_condition": prompt_condition,
                        "application_mode": application_mode(cfg.prompt_only),
                        "layer": None if intervention_label == "baseline" else cfg.layer,
                        "strength": method_strength(
                            intervention_label,
                            eta=cfg.eta,
                            curveball_alpha=cfg.curveball_alpha,
                            chars_alpha=cfg.chars_alpha,
                            sas_coefficient=cfg.sas_coefficient,
                        ),
                        **(
                            {
                                "n_clusters": cfg.chars_clusters,
                                "state_fingerprint": cfg.chars_state_fingerprint,
                            }
                            if intervention_label == "chars" else {}
                        ),
                        "validation_target_probability": result["summary"]["average_target_probability"],
                        "n_validation_examples": result["summary"]["n_examples"],
                        "split_fingerprints": split_metadata["fingerprints"],
                        "fixed_parameters": (
                            {"sas": sas_validation.sas_fixed_parameters(cfg)}
                            if intervention_label == "sas"
                            else {"curveball": curveball_caa_fixed_parameters(cfg)}
                            if intervention_label == "curveball"
                            else (
                                {"chars": chars_fixed_parameters(cfg)}
                                if intervention_label == "chars"
                                else {
                                    "kernel": "rbf",
                                    "kernel_bandwidth": "reference_median_heuristic",
                                    "lambda_reg": 0.0,
                                    "activation_position": "answer_token",
                                    "gradient_steps": 1,
                                }
                            )
                        ),
                    }
                )

    with open(os.path.join(eval_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=4)
    if candidate_records:
        with open(os.path.join(eval_dir, "selection_candidates.json"), "w") as handle:
            json.dump(candidate_records, handle, indent=4)

    for eval_type, eval_results in summary["results"].items():
        for intervention, result_summary in eval_results.items():
            probability = result_summary.get("average_target_probability")
            if probability is not None:
                print(
                    f"CAA_SUMMARY {cfg.caa_eval_split} {eval_type} "
                    f"{intervention} target_probability={probability:.6f}"
                )

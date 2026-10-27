"""Entry point: `python -m pipeline.run_pipeline --model_path MODEL [options]`.

One invocation fits one steering method on reference prompts and evaluates it.
Every run goes through the same stages:

1. settings / phase_validation: resolve defaults, check arguments, build the Config
2. references: load the prompts (and CAA answer pairs) to fit on
3. interventions: collect reference activations, fit the method's direction or
   state, export its steer config, build forward hooks
4. evaluation: jailbreak (standard, validation, or locked test) and/or CAA,
   depending on --evaluation_protocol and --experiment_phase
5. evaluation: CE loss / perplexity on Pile and Alpaca for every evaluated intervention
"""

import torch

from pipeline.cli import parse_arguments
from pipeline.evaluation import (
    run_caa_evaluations,
    run_loss_evaluations,
    run_standard_jailbreak_evaluations,
    run_validation_jailbreak_evaluations,
)
from pipeline.interventions import build_interventions
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.phase_validation import validate_phase_configuration
from pipeline.references import load_references
from pipeline.settings import build_config
from pipeline.utils.sas import resolve_gemma_sae_release


def run_pipeline(
    model_path,
    artifact_root=None,
    kernel_type='rbf',
    kernel_params=None,
    eta=0.1,
    lambda_reg=0.01,
    pos=None,
    layer=None,
    step_count=1,
    use_mean_act=False,
    prompt_only=False,
    skip_baseline=False,
    seed=42,
    n_train=128,
    steer_method="kernel",
    kernel_extraction_source="legacy",
    sae_release=None,
    sae_width="16k",
    sas_feature_freq_threshold=0.01,
    sas_extraction_source=None,
    sas_coefficient=-1.0,
    sas_use_delta_correction=True,
    sas_apply_post_activation=True,
    curveball_extraction_source=None,
    curveball_k=20,
    curveball_kernel="poly",
    curveball_degree=3,
    curveball_gamma=None,
    curveball_coef0=1.0,
    curveball_alpha=1.0,
    curveball_fit_inverse_alpha=1.0,
    chars_extraction_source=None,
    chars_clusters=5,
    chars_alpha=1.0,
    chars_state_path=None,
    caa_dataset_root="caa",
    caa_behavior="refusal",
    evaluation_protocol=None,
    caa_eval_types=("ab", "truthful_qa", "mmlu"),
    caa_eval_system_prompt="none",
    caa_eval_open_ended_score=False,
    caa_open_ended_judge_model=None,
    experiment_phase="standard",
    selection_manifest=None,
    evaluation_interventions=None,
    split_seed=42,
    caa_validation_size=50,
    caa_eval_split="test",
    jailbreak_eval_split="test",
    jailbreak_utility_size=256,
    ce_loss_batch_size=2,
):
    """Run the full pipeline."""
    # Every keyword argument except skip_baseline configures the run.
    options = dict(locals())
    skip_baseline = options.pop("skip_baseline")
    cfg = build_config(**options)
    locked_manifest = validate_phase_configuration(cfg)

    model_base = construct_model_base(cfg.model_path)

    # Get the dimensionality of the model's intermediate activations
    d_model = model_base.model.config.hidden_size

    print(f"Model hidden size (d_model): {d_model}")

    if cfg.steer_method == "sas":
        cfg.sae_release = resolve_gemma_sae_release(cfg.model_path, cfg.sae_release)

    # Only the all-layer KAS path and the steer-config export read this
    # fallback; single-layer KAS hooks read cfg.kernel_params directly.
    kernel_params = cfg.kernel_params
    # currently only have RBF kernel setups
    if cfg.kernel_type == 'rbf' and kernel_params is None:
        kernel_params = {'sigma': 1}  # heuristic choice

    refs = load_references(cfg, model_base)
    interventions = build_interventions(cfg, model_base, refs, locked_manifest, kernel_params)

    if (
        cfg.evaluation_protocol == "legacy"
        and cfg.experiment_phase in {"select", "test"}
    ):
        run_validation_jailbreak_evaluations(
            cfg=cfg,
            model_base=model_base,
            interventions=interventions.legacy,
            target_train=refs.target_train,
            source_train=refs.source_train,
            locked_manifest=locked_manifest,
        )
    elif cfg.evaluation_protocol in {"legacy", "both"}:
        run_standard_jailbreak_evaluations(
            cfg, model_base, interventions.legacy, skip_baseline
        )

    if cfg.evaluation_protocol in {"caa", "both"}:
        run_caa_evaluations(
            cfg=cfg,
            model_base=model_base,
            interventions=interventions.caa,
            validation_dataset=refs.caa_validation_dataset,
            split_metadata=refs.caa_split_metadata,
            locked_manifest=locked_manifest,
        )

    # 5. CE loss / perplexity of every intervention evaluated above.
    loss_interventions = {}
    if cfg.evaluation_protocol in {"legacy", "both"}:
        loss_interventions.update(interventions.legacy)
    if cfg.evaluation_protocol in {"caa", "both"}:
        loss_interventions.update(interventions.caa)
    run_loss_evaluations(cfg, model_base, loss_interventions)


if __name__ == "__main__":
    args = parse_arguments()
    torch.manual_seed(args.seed)
    run_pipeline(**vars(args))

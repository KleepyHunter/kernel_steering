"""Fit each steering method and build its forward hooks ("interventions").

An intervention is a `(fwd_pre_hooks, fwd_hooks)` pair keyed by label: `baseline`,
`actadd`, `kernel` (KAS), `curveball`, `chars`, or `sas`. Each method also exports a
steer config for lm-eval (TinyBenchmarks) when a run is not a validation run.
"""

import os
import pickle
from dataclasses import dataclass

import torch

from pipeline.settings import uses_caa_pair_extraction
from pipeline.submodules.generate_directions import (
    generate_activations_mean_act,
    generate_directions,
)
from pipeline.utils.chars import (
    build_chars_steer_config,
    fit_chars_state,
    get_chars_output_hook,
    get_selected_block_output_activations,
    validate_chars_state,
)
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_curveball_kpca_output_hook,
    get_kernelized_steering_hooks,
    get_kernelized_steering_input_pre_hook,
)
from pipeline.utils.sas import (
    build_sas_steer_config,
    generate_and_save_sas_direction,
    get_sas_actadd_output_hook,
    load_sae_for_layer,
)
from pipeline.utils.utils import compute_median_heuristic_sigma
from pipeline.validation_selection import application_mode, assert_locked_configuration


@dataclass
class Interventions:
    legacy: dict  # jailbreak evaluation, label -> (fwd_pre_hooks, fwd_hooks)
    caa: dict  # CAA evaluation, label -> (fwd_pre_hooks, fwd_hooks)


def generate_and_save_candidate_directions(
    cfg,
    model_base,
    target_train,
    source_train,
    target_outputs=None,
    source_outputs=None,
    use_output_hook: bool = False,
):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    target_activation, source_activation = generate_directions(
        model_base,
        target_train,
        source_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"),
        method="kernel",
        target_outputs=target_outputs,
        source_outputs=source_outputs,
        use_output_hook=use_output_hook,
    )

    return target_activation, source_activation


def generate_and_save_curveball_state(
    cfg,
    model_base,
    target_train,
    source_train,
    target_outputs=None,
    source_outputs=None,
    positions=None,
    layer=None,
):
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    curveball_state = generate_directions(
        model_base,
        target_train,
        source_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"),
        method="kernel",
        steer_method="curveball",
        target_outputs=target_outputs,
        source_outputs=source_outputs,
        use_output_hook=cfg.curveball_extraction_source == "caa",
        positions=positions,
        layers=layer,
        curveball_k=cfg.curveball_k,
        curveball_kernel=cfg.curveball_kernel,
        curveball_degree=cfg.curveball_degree,
        curveball_gamma=cfg.curveball_gamma,
        curveball_coef0=cfg.curveball_coef0,
        curveball_fit_inverse_alpha=cfg.curveball_fit_inverse_alpha,
    )

    with open(os.path.join(cfg.artifact_path(), 'generate_directions/curveball_state.pkl'), "wb") as f:
        pickle.dump(curveball_state, f)
    return curveball_state


def generate_and_save_chars_state(
    cfg,
    model_base,
    target_train,
    source_train,
    *,
    target_outputs=None,
    source_outputs=None,
    reference_split_fingerprints=None,
):
    state_dir = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(state_dir, exist_ok=True)

    if cfg.chars_extraction_source == "caa":
        # CAA: behavior-not-matching (negative) -> behavior-matching (positive).
        transport_source_instructions = source_train
        transport_source_outputs = source_outputs
        transport_target_instructions = target_train
        transport_target_outputs = target_outputs
        orientation = "behavior_not_matching_to_behavior_matching"
    else:
        # Legacy jailbreak: harmful reference -> harmless reference.
        transport_source_instructions = target_train
        transport_source_outputs = target_outputs
        transport_target_instructions = source_train
        transport_target_outputs = source_outputs
        orientation = "harmful_to_harmless"

    expected_metadata = {
        "model": cfg.model_alias,
        "layer": int(cfg.layer),
        "position": int(cfg.pos),
        "hookpoint": "block_output",
        "extraction_source": cfg.chars_extraction_source,
        "orientation": orientation,
        "split_fingerprints": dict(reference_split_fingerprints or {}),
    }
    # Every run collects its own activations and fits the state.
    source_activations = get_selected_block_output_activations(
        model_base,
        transport_source_instructions,
        outputs=transport_source_outputs,
        layer=cfg.layer,
        position=cfg.pos,
        batch_size=32,
    )
    target_activations = get_selected_block_output_activations(
        model_base,
        transport_target_instructions,
        outputs=transport_target_outputs,
        layer=cfg.layer,
        position=cfg.pos,
        batch_size=32,
    )
    state = fit_chars_state(
        source_activations,
        target_activations,
        n_clusters=cfg.chars_clusters,
        metadata=expected_metadata,
    )
    if cfg.chars_state_path is not None:
        # A pre-fitted state (scripts/fit_chars_states.py) must equal this refit.
        cached = torch.load(
            cfg.chars_state_path,
            map_location="cpu",
            weights_only=True,
        )
        validate_chars_state(
            cached,
            expected_n_clusters=cfg.chars_clusters,
            expected_fingerprint=state["state_fingerprint"],
            expected_metadata=expected_metadata,
        )

    fingerprint = validate_chars_state(
        state,
        expected_n_clusters=cfg.chars_clusters,
        expected_metadata=expected_metadata,
    )
    state_path = os.path.join(state_dir, "chars_state.pt")
    torch.save(state, state_path)
    verified = torch.load(state_path, map_location="cpu", weights_only=True)
    validate_chars_state(
        verified,
        expected_n_clusters=cfg.chars_clusters,
        expected_fingerprint=fingerprint,
        expected_metadata=expected_metadata,
    )
    return verified


def generate_steer_config(cfg, num_layers, source, target, kernel_type, kernel_params, eta, lambda_reg, step_count, layer, direction, use_mean_act, prompt_only, chars_state=None):
    has_locked_intervention = any(
        label != "baseline" for label in (cfg.evaluation_interventions or ())
    )
    if cfg.experiment_phase == "select" or (
        cfg.experiment_phase == "test" and not has_locked_intervention
    ):
        print("Skipping lm-eval steer config export without a locked intervention.")
        return False

    if cfg.steer_method == "chars":
        if chars_state is None:
            raise ValueError("CHaRS export requires a verified state.")
        steer_config_chars = build_chars_steer_config(
            chars_state,
            layer=layer,
            alpha=cfg.chars_alpha,
            position=cfg.pos,
            prompt_only=prompt_only,
        )
        steer_dir = os.path.join(cfg.artifact_path(), "steer_config")
        os.makedirs(steer_dir, exist_ok=True)
        torch.save(
            steer_config_chars,
            os.path.join(steer_dir, "steer_config_chars.pt"),
        )
        return True

    if cfg.steer_method == "sas":
        steer_dir = os.path.join(cfg.artifact_path(), "steer_config")
        os.makedirs(steer_dir, exist_ok=True)
        torch.save(
            build_sas_steer_config(cfg, layer=layer, direction=direction, prompt_only=prompt_only),
            os.path.join(steer_dir, "steer_config_sas.pt"),
        )
        return True
    if cfg.steer_method == "curveball":
        steer_config_curveball = {
            f"layers.{layer}": {
                "steering_coefficient": cfg.curveball_alpha,
                "prompt_only": prompt_only,
                "action": "curveball_kpca",
                "state_path": os.path.join(cfg.artifact_path(), "generate_directions", "curveball_state.pkl"),
                "curveball_k": cfg.curveball_k,
                "curveball_kernel": cfg.curveball_kernel,
                "curveball_degree": cfg.curveball_degree,
                "curveball_gamma": cfg.curveball_gamma,
                "curveball_coef0": cfg.curveball_coef0,
                "curveball_fit_inverse_alpha": cfg.curveball_fit_inverse_alpha,
            },
        }
        if not os.path.exists(os.path.join(cfg.artifact_path(), 'steer_config')):
            os.makedirs(os.path.join(cfg.artifact_path(), 'steer_config'))
        torch.save(steer_config_curveball, os.path.join(cfg.artifact_path(), 'steer_config/steer_config_curveball.pt'))
        return True

    if not use_mean_act:
        # actadd config
        steer_config_dim = {
            f"layers.{layer}": {
                "steering_vector": direction,
                "steering_coefficient": eta,
                "action": "add",
                "prompt_only": prompt_only,
            },
        }
        # kernel config
        kernel_params = compute_median_heuristic_sigma(source, target)
        steer_config_kernel = {
            f"layers.{layer}": {
                "source": source,
                "target": target,
                "kernel_type": kernel_type,
                "kernel_params": kernel_params,
                "eta": eta,
                "lambda_reg": lambda_reg,
                "step_count": step_count,
                "prompt_only": prompt_only,
                "action": "kernel"
            },
        }
    else:
        steer_config_dim = dict()
        steer_config_kernel = dict()
        for l in range(num_layers):
            # kernel config
            kernel_params = compute_median_heuristic_sigma(source[l], target[l])
            steer_config_kernel[f"layers.{l}"] = {
                "source": source[l],
                "target": target[l],
                "kernel_type": kernel_type,
                "kernel_params": kernel_params,
                "eta": eta,
                "lambda_reg": lambda_reg,
                "step_count": step_count,
                "prompt_only": prompt_only,
                "action": "kernel"
            }

    if not os.path.exists(os.path.join(cfg.artifact_path(), 'steer_config')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'steer_config'))
    torch.save(steer_config_dim, os.path.join(cfg.artifact_path(), 'steer_config/steer_config_dim.pt'))
    torch.save(steer_config_kernel, os.path.join(cfg.artifact_path(), 'steer_config/steer_config_kernel.pt'))
    return True


def filter_interventions(cfg, interventions):
    if cfg.evaluation_interventions is None:
        return interventions
    requested = set(cfg.evaluation_interventions)
    missing = requested - set(interventions)
    if missing:
        raise ValueError(f"Requested interventions are unavailable in this run: {sorted(missing)}")
    return {label: hooks for label, hooks in interventions.items() if label in requested}


def _chars_interventions(cfg, model_base, refs, locked_manifest, kernel_params):
    if cfg.layer is None:
        raise ValueError("CHaRS mode requires --layer.")
    chars_state = generate_and_save_chars_state(
        cfg,
        model_base,
        refs.target_train,
        refs.source_train,
        target_outputs=refs.target_outputs,
        source_outputs=refs.source_outputs,
        reference_split_fingerprints=refs.split_fingerprints,
    )
    cfg.chars_state_fingerprint = validate_chars_state(chars_state)
    if cfg.experiment_phase == "test" and "chars" in (
        cfg.evaluation_interventions or ()
    ):
        assert_locked_configuration(
            locked_manifest,
            experiment=(
                "caa" if cfg.evaluation_protocol == "caa" else "jailbreak"
            ),
            model=cfg.model_alias,
            method="chars",
            mode=application_mode(cfg.prompt_only),
            layer=cfg.layer,
            strength=cfg.chars_alpha,
            n_clusters=cfg.chars_clusters,
            state_fingerprint=cfg.chars_state_fingerprint,
            behavior=(
                cfg.caa_behavior if cfg.evaluation_protocol == "caa" else None
            ),
            prompt_condition=(
                (cfg.caa_eval_system_prompt or "none")
                if cfg.evaluation_protocol == "caa" else None
            ),
        )
    if generate_steer_config(
        cfg,
        model_base.model.config.num_hidden_layers,
        None,
        None,
        cfg.kernel_type,
        kernel_params,
        cfg.eta,
        cfg.lambda_reg,
        cfg.step_count,
        cfg.layer,
        None,
        False,
        cfg.prompt_only,
        chars_state=chars_state,
    ):
        print("CHaRS steer config saved.")

    chars_fwd_pre_hooks = []
    chars_fwd_hooks = [(
        model_base.model_block_modules[cfg.layer],
        get_chars_output_hook(
            chars_state,
            alpha=cfg.chars_alpha,
            prompt_only=cfg.prompt_only,
        ),
    )]
    return {"chars": (chars_fwd_pre_hooks, chars_fwd_hooks)}


def _sas_interventions(cfg, model_base, refs, locked_manifest, kernel_params):
    if cfg.layer is None:
        raise ValueError("SAS mode requires --layer.")

    sas_direction = generate_and_save_sas_direction(
        cfg,
        model_base,
        refs.target_train,
        refs.source_train,
        target_outputs=refs.target_outputs,
        source_outputs=refs.source_outputs,
        positions=cfg.pos,
        layer=cfg.layer,
    )
    if generate_steer_config(
        cfg,
        model_base.model.config.num_hidden_layers,
        None,
        None,
        cfg.kernel_type,
        kernel_params,
        cfg.eta,
        cfg.lambda_reg,
        cfg.step_count,
        cfg.layer,
        sas_direction,
        False,
        cfg.prompt_only,
    ):
        print("SAS steer config saved.")

    sae, _ = load_sae_for_layer(
        release=cfg.sae_release,
        layer=cfg.layer,
        width=cfg.sae_width,
        device=model_base.model.device,
        dtype=torch.float32,
    )
    sas_fwd_pre_hooks, sas_fwd_hooks = [], [(
        model_base.model_block_modules[cfg.layer],
        get_sas_actadd_output_hook(
            vector=sas_direction,
            coeff=cfg.sas_coefficient,
            sae=sae,
            use_delta_correction=cfg.sas_use_delta_correction,
            apply_post_activation=cfg.sas_apply_post_activation,
            prompt_only=cfg.prompt_only,
        ),
    )]
    return {"sas": (sas_fwd_pre_hooks, sas_fwd_hooks)}


def _curveball_interventions(cfg, model_base, refs, locked_manifest, kernel_params):
    if cfg.layer is None:
        raise ValueError("Curveball mode requires --layer.")

    curveball_state = generate_and_save_curveball_state(
        cfg,
        model_base,
        refs.target_train,
        refs.source_train,
        target_outputs=refs.target_outputs,
        source_outputs=refs.source_outputs,
        positions=cfg.pos,
        layer=cfg.layer,
    )
    if generate_steer_config(
        cfg,
        model_base.model.config.num_hidden_layers,
        None,
        None,
        cfg.kernel_type,
        kernel_params,
        cfg.eta,
        cfg.lambda_reg,
        cfg.step_count,
        cfg.layer,
        None,
        False,
        cfg.prompt_only,
    ):
        print("Curveball steer config saved.")

    curveball_fwd_pre_hooks, curveball_fwd_hooks = [], [(
        model_base.model_block_modules[cfg.layer],
        get_curveball_kpca_output_hook(
            kpca=curveball_state["kpca"],
            z_hat=curveball_state["z_hat"],
            alpha=cfg.curveball_alpha,
            prompt_only=cfg.prompt_only,
        ),
    )]
    return {"curveball": (curveball_fwd_pre_hooks, curveball_fwd_hooks)}


def _kas_and_actadd_interventions(cfg, model_base, refs, locked_manifest, kernel_params):
    # 1. Generate candidate target/source directions
    target_directions, source_directions = generate_and_save_candidate_directions(
        cfg,
        model_base,
        refs.target_train,
        refs.source_train,
        target_outputs=refs.target_outputs,
        source_outputs=refs.source_outputs,
        use_output_hook=uses_caa_pair_extraction(
            kernel_extraction_source=cfg.kernel_extraction_source,
            steer_method=cfg.steer_method,
            curveball_extraction_source=cfg.curveball_extraction_source,
            chars_extraction_source=cfg.chars_extraction_source,
        ),
    )

    # 2. Select the most effective direction
    # get keys of the target_directions dict
    print(f"Available positions and layers for directions: {list(target_directions.keys())}")
    target_activations, source_activations = target_directions[(cfg.pos, cfg.layer)], source_directions[(cfg.pos, cfg.layer)]  # [n_samples, d_model]

    # take the mean direction across samples
    raw_direction = (target_activations.mean(dim=0) - source_activations.mean(dim=0))  # [d_model]

    # save the steer config for tinyBenchmark eval
    if generate_steer_config(cfg, model_base.model.config.num_hidden_layers, source_activations, target_activations, cfg.kernel_type, kernel_params, cfg.eta, cfg.lambda_reg, cfg.step_count, cfg.layer, raw_direction, cfg.use_mean_act, cfg.prompt_only):
        print("Steer config saved for tinyBenchmark eval.")

    # experiment: source -> target
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[cfg.layer], get_activation_addition_input_pre_hook(vector=raw_direction, coeff=cfg.eta, prompt_only=cfg.prompt_only))], []
    kernel_fwd_pre_hooks, kernel_fwd_hooks = get_kernelized_steering_hooks(model_base, cfg.layer, source_activations, target_activations,
                                                                           kernel_type=cfg.kernel_type, kernel_params=cfg.kernel_params,
                                                                           eta=cfg.eta, lambda_reg=cfg.lambda_reg, step_count=cfg.step_count,
                                                                           prompt_only=cfg.prompt_only)
    return {
        "actadd": (actadd_fwd_pre_hooks, actadd_fwd_hooks),
        "kernel": (kernel_fwd_pre_hooks, kernel_fwd_hooks),
    }


def _kas_all_layer_interventions(cfg, model_base, refs, locked_manifest, kernel_params):
    # 1. Generate activations of all layers, having applied sequential steering
    source_activations, target_activations = generate_activations_mean_act(
        model_base=model_base,
        source_instructions=refs.source_train,
        target_instructions=refs.target_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"),
        kernel_type=cfg.kernel_type,
        kernel_params=kernel_params,
        eta=cfg.eta,
        lambda_reg=cfg.lambda_reg,
        step_count=cfg.step_count
    )
    # Print the shape of the activations for error checking
    print("Shape of source activations:", source_activations.shape)
    print("Shape of target activations:", target_activations.shape)

    # save the steer config for tinyBenchmark eval
    if generate_steer_config(cfg, model_base.model.config.num_hidden_layers, source_activations, target_activations, cfg.kernel_type, kernel_params, cfg.eta, cfg.lambda_reg, cfg.step_count, cfg.layer, None, cfg.use_mean_act, cfg.prompt_only):
        print("Steer config saved for tinyBenchmark eval.")

    # 2. Create hooks for the experiment
    kernel_fwd_pre_hooks = []
    num_layers = model_base.model.config.num_hidden_layers
    for layer in range(num_layers):
        kernel_fwd_pre_hooks.append(
            (model_base.model_block_modules[layer],
             get_kernelized_steering_input_pre_hook(
                 source=source_activations[layer],
                 target=target_activations[layer],
                 kernel_type=cfg.kernel_type,
                 kernel_params=kernel_params,
                 eta=cfg.eta,
                 lambda_reg=cfg.lambda_reg,
                 step_count=cfg.step_count,
                 prompt_only=cfg.prompt_only
             ))
        )

    kernel_fwd_hooks = []
    return {"kernel": (kernel_fwd_pre_hooks, kernel_fwd_hooks)}


def build_interventions(cfg, model_base, refs, locked_manifest, kernel_params) -> Interventions:
    """Fit the configured method and return its interventions, filtered by --evaluation_interventions."""
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    legacy_interventions = {}
    if cfg.experiment_phase in {"select", "test"}:
        legacy_interventions["baseline"] = (
            baseline_fwd_pre_hooks, baseline_fwd_hooks
        )
    caa_interventions = {"baseline": (baseline_fwd_pre_hooks, baseline_fwd_hooks)}

    if cfg.steer_method == "chars":
        methods = _chars_interventions(cfg, model_base, refs, locked_manifest, kernel_params)
    elif cfg.steer_method == "sas":
        methods = _sas_interventions(cfg, model_base, refs, locked_manifest, kernel_params)
    elif cfg.steer_method == "curveball":
        methods = _curveball_interventions(cfg, model_base, refs, locked_manifest, kernel_params)
    elif not cfg.use_mean_act:
        methods = _kas_and_actadd_interventions(cfg, model_base, refs, locked_manifest, kernel_params)
    else:
        methods = _kas_all_layer_interventions(cfg, model_base, refs, locked_manifest, kernel_params)
    legacy_interventions.update(methods)
    caa_interventions.update(methods)

    return Interventions(
        legacy=filter_interventions(cfg, legacy_interventions),
        caa=filter_interventions(cfg, caa_interventions),
    )

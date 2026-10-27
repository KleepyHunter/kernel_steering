#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
PIPELINE_CONDA_ENV="refusal_direction"
GPU="${GPU:-<GPU_ID>}"
if [[ "${GPU}" == *"<"* ]]; then
    echo "Set GPU to the GPU id to use (it is the placeholder ${GPU})." >&2
    exit 2
fi
LM_EVAL_CONDA_ENV="${LM_EVAL_CONDA_ENV:-lm_eval}"
RUN_LM_EVAL="${RUN_LM_EVAL:-true}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande}"
LM_EVAL_DEVICE="${LM_EVAL_DEVICE:-cuda:0}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"

activate_env() {
    local env_name="$1"
    conda deactivate || true
    conda activate "${env_name}"
}

activate_env "${PIPELINE_CONDA_ENV}"

# -----------------------------------------------------------------------------
# Sweep config (edit here)
# -----------------------------------------------------------------------------
OUTDIR="${OUTDIR:-outputs_sas_wrapper_legacy}"
mkdir -p "${OUTDIR}"

SEEDS=(0)
PROMPT_ONLY_FLAGS=("--prompt_only" "")
MODEL_PATHS=("google/gemma-2-9b-it")
LAYERS=(22)
SAE_WIDTHS=("16k")
TAUS=(0.7)

STEER_METHOD="sas"
SAS_EXTRACTION_SOURCE="legacy"      # force legacy target/source splits
SAS_COEFFICIENT="${SAS_COEFFICIENT:--1.0}"
SAS_USE_DELTA_CORRECTION="${SAS_USE_DELTA_CORRECTION:-true}"
SAS_APPLY_POST_ACTIVATION="${SAS_APPLY_POST_ACTIVATION:-true}"
SAE_RELEASE_OVERRIDE="${SAE_RELEASE_OVERRIDE:-}"
EVALUATION_PROTOCOL="legacy"
POS="-1"                            # legacy default; override if needed
CAA_DATASET_ROOT="caa"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
get_model_name() {
    local model_path="$1"
    echo "${model_path##*/}"
}

is_prompt_only() {
    local prompt_flag="$1"
    if [[ "$prompt_flag" == "--prompt_only" ]]; then
        echo "True"
    else
        echo "False"
    fi
}

resolve_sas_steer_config_path() {
    local model_path="$1"
    local seed="$2"
    local layer="$3"
    local prompt_only="$4"
    local pos_override="$5"
    local sae_width="$6"
    local tau="$7"
    local sas_coefficient="$8"

    python3 - "$model_path" "$seed" "$layer" "$prompt_only" "$pos_override" \
        "$EVALUATION_PROTOCOL" "$SAS_EXTRACTION_SOURCE" "$sae_width" "$tau" "$sas_coefficient" \
        "$SAS_USE_DELTA_CORRECTION" "$SAS_APPLY_POST_ACTIVATION" "$SAE_RELEASE_OVERRIDE" \
        "$CAA_DATASET_ROOT" <<'PY'
import os
import sys

from pipeline.config import Config

(
    model_path,
    seed,
    layer,
    prompt_only_raw,
    pos_override_raw,
    evaluation_protocol,
    sas_extraction_source,
    sae_width,
    tau,
    sas_coefficient,
    sas_use_delta_correction_raw,
    sas_apply_post_activation_raw,
    sae_release_override,
    caa_dataset_root,
) = sys.argv[1:]


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y", "t"}


def resolve_gemma_sae_release(model_path_value, override_release=None):
    if override_release:
        return override_release

    model_path_lower = model_path_value.lower()
    if "gemma-2-2b" in model_path_lower or "gemma2-2b" in model_path_lower:
        return "gemma-scope-2b-pt-res-canonical"
    if "gemma-2-9b" in model_path_lower or "gemma2-9b" in model_path_lower:
        return "gemma-scope-9b-pt-res-canonical"

    raise ValueError(
        "Could not infer Gemma Scope SAE release from model path. "
        "Set SAE_RELEASE_OVERRIDE explicitly for this model."
    )


prompt_only = str_to_bool(prompt_only_raw)
sas_use_delta_correction = str_to_bool(sas_use_delta_correction_raw)
sas_apply_post_activation = str_to_bool(sas_apply_post_activation_raw)

if pos_override_raw:
    pos = int(pos_override_raw)
else:
    pos = -2 if sas_extraction_source == "caa" else -1

sae_release = resolve_gemma_sae_release(model_path, sae_release_override or None)

cfg = Config(
    model_alias=os.path.basename(model_path),
    model_path=model_path,
    seed=int(seed),
    kernel_type="rbf",
    kernel_params=None,
    eta=1.0,
    lambda_reg=0.1,
    pos=pos,
    layer=int(layer),
    use_mean_act=False,
    step_count=1,
    prompt_only=prompt_only,
    kernel_extraction_source="legacy",
    steer_method="sas",
    sae_release=sae_release,
    sae_width=sae_width,
    sas_feature_freq_threshold=float(tau),
    sas_extraction_source=sas_extraction_source,
    sas_coefficient=float(sas_coefficient),
    sas_use_delta_correction=sas_use_delta_correction,
    sas_apply_post_activation=sas_apply_post_activation,
    caa_dataset_root=caa_dataset_root,
    caa_behavior="refusal",
    evaluation_protocol=evaluation_protocol,
)

print(os.path.join(cfg.artifact_path(), "steer_config", "steer_config_sas.pt"))
PY
}

# -----------------------------------------------------------------------------
# Main sweep loop
# -----------------------------------------------------------------------------
for sd in "${SEEDS[@]}"; do
for md in "${MODEL_PATHS[@]}"; do
for po in "${PROMPT_ONLY_FLAGS[@]}"; do
for width in "${SAE_WIDTHS[@]}"; do
for tau in "${TAUS[@]}"; do
for layer in "${LAYERS[@]}"; do

name="seed${sd}_model_$(get_model_name "$md")_layer${layer}_tau${tau}_width${width}_legacy_prompt_only-$(is_prompt_only "$po")"
logdir="${OUTDIR}/model_$(get_model_name "$md")/layer-${layer}/tau-${tau}/sae_width-${width}/prompt_only-$(is_prompt_only "$po")"
mkdir -p "${logdir}"
logfile="${logdir}/seed_${sd}.log"

echo "Running: ${name}"
activate_env "${PIPELINE_CONDA_ENV}"

cmd=(
    python3 -m pipeline.run_pipeline
    --model_path "$md"
    --steer_method "$STEER_METHOD"
    --layer "$layer"
    --seed "$sd"
    --pos "$POS"
    --sae_width "$width"
    --sas_feature_freq_threshold "$tau"
    --sas_extraction_source "$SAS_EXTRACTION_SOURCE"
    --sas_coefficient "$SAS_COEFFICIENT"
    --sas_use_delta_correction "$SAS_USE_DELTA_CORRECTION"
    --sas_apply_post_activation "$SAS_APPLY_POST_ACTIVATION"
    --evaluation_protocol "$EVALUATION_PROTOCOL"
)

if [[ "$po" == "--prompt_only" ]]; then
    cmd+=(--prompt_only)
fi

if [[ -n "$SAE_RELEASE_OVERRIDE" ]]; then
    cmd+=(--sae_release "$SAE_RELEASE_OVERRIDE")
fi

CUDA_VISIBLE_DEVICES="${GPU}" OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
"${cmd[@]}" > "${logfile}" 2>&1

if [[ "${RUN_LM_EVAL}" == "true" ]]; then
    steer_path="$(
        resolve_sas_steer_config_path \
            "$md" "$sd" "$layer" "$(is_prompt_only "$po")" "$POS" "$width" "$tau" "$SAS_COEFFICIENT"
    )"

    if [[ ! -f "${steer_path}" ]]; then
        echo "Missing SAS steer config: ${steer_path}" | tee -a "${logfile}"
        exit 1
    fi

    activate_env "${LM_EVAL_CONDA_ENV}"
    CUDA_VISIBLE_DEVICES="${GPU}" lm_eval \
        --model steered \
        --model_args "pretrained=$md" "steer_path=${steer_path}" \
        --tasks "${LM_EVAL_TASKS}" \
        --device "${LM_EVAL_DEVICE}" \
        --batch_size "${LM_EVAL_BATCH_SIZE}" >> "${logfile}" 2>&1
    activate_env "${PIPELINE_CONDA_ENV}"
fi

echo "Finished: ${name}"
echo "-------------------------------"

done
done
done
done
done
done

echo "All SAS legacy jobs completed."

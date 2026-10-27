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
LM_EVAL_CONDA_ENV="${LM_EVAL_CONDA_ENV:-steering-lm-eval}"
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
OUTDIR="${OUTDIR:-outputs_kernel_wrapper_legacy}"
mkdir -p "${OUTDIR}"

SEEDS=(0)
PROMPT_ONLY_FLAGS=("" "--prompt_only")
# PROMPT_ONLY_FLAGS=("")
MODEL_PATHS=("google/gemma-2-9b-it")
LAYERS=(21)
ETAS=(40.0)

STEER_METHOD="kernel"
KERNEL_TYPE="polynomial"
KERNEL_PARAMS_JSON='{"degree": 2}'             # Example: {"sigma": 48}
LAMBDA_REG="0.0"
STEP_COUNT="10"
KERNEL_EXTRACTION_SOURCE="legacy"
EVALUATION_PROTOCOL="legacy"
POS="-1"

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

resolve_kernel_steer_config_path() {
    local model_path="$1"
    local seed="$2"
    local layer="$3"
    local prompt_only="$4"
    local pos_override="$5"
    local eta="$6"

    python3 - "$model_path" "$seed" "$layer" "$prompt_only" "$pos_override" \
        "$eta" "$LAMBDA_REG" "$STEP_COUNT" "$KERNEL_TYPE" "$KERNEL_PARAMS_JSON" \
        "$KERNEL_EXTRACTION_SOURCE" "$EVALUATION_PROTOCOL" <<'PY'
import ast
import json
import os
import sys

from pipeline.config import Config

(
    model_path,
    seed,
    layer,
    prompt_only_raw,
    pos_override_raw,
    eta,
    lambda_reg,
    step_count,
    kernel_type,
    kernel_params_raw,
    kernel_extraction_source,
    evaluation_protocol,
) = sys.argv[1:]


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y", "t"}


def parse_kernel_params(raw: str):
    text = (raw or "").strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise ValueError("KERNEL_PARAMS_JSON must parse to a dict.")
    return parsed


prompt_only = str_to_bool(prompt_only_raw)
if pos_override_raw:
    pos = int(pos_override_raw)
else:
    pos = -2 if kernel_extraction_source == "caa" else -1

cfg = Config(
    model_alias=os.path.basename(model_path),
    model_path=model_path,
    seed=int(seed),
    kernel_type=kernel_type,
    kernel_params=parse_kernel_params(kernel_params_raw),
    eta=float(eta),
    lambda_reg=float(lambda_reg),
    pos=pos,
    layer=int(layer),
    use_mean_act=False,
    step_count=int(step_count),
    prompt_only=prompt_only,
    steer_method="kernel",
    kernel_extraction_source=kernel_extraction_source,
    evaluation_protocol=evaluation_protocol,
)

print(os.path.join(cfg.artifact_path(), "steer_config", "steer_config_kernel.pt"))
PY
}

# -----------------------------------------------------------------------------
# Main sweep loop
# -----------------------------------------------------------------------------
for sd in "${SEEDS[@]}"; do
for md in "${MODEL_PATHS[@]}"; do
for po in "${PROMPT_ONLY_FLAGS[@]}"; do
for layer in "${LAYERS[@]}"; do
for et in "${ETAS[@]}"; do

name="seed${sd}_model_$(get_model_name "$md")_layer${layer}_eta${et}_step_count-${STEP_COUNT}_legacy_prompt_only-$(is_prompt_only "$po")"
logdir="${OUTDIR}/model_$(get_model_name "$md")/kernel_type-${KERNEL_TYPE}/layer-${layer}/eta-${et}/step_count-${STEP_COUNT}/prompt_only-$(is_prompt_only "$po")"
mkdir -p "${logdir}"
logfile="${logdir}/seed_${sd}.log"

echo "Running: ${name}"
activate_env "${PIPELINE_CONDA_ENV}"

cmd=(
    python3 -m pipeline.run_pipeline
    --model_path "$md"
    --steer_method "$STEER_METHOD"
    --kernel_type "$KERNEL_TYPE"
    --eta "$et"
    --lambda_reg "$LAMBDA_REG"
    --seed "$sd"
    --layer "$layer"
    --pos "$POS"
    --step_count "$STEP_COUNT"
    --kernel_extraction_source "$KERNEL_EXTRACTION_SOURCE"
    --evaluation_protocol "$EVALUATION_PROTOCOL"
)

if [[ "$po" == "--prompt_only" ]]; then
    cmd+=(--prompt_only)
fi

if [[ -n "$KERNEL_PARAMS_JSON" ]]; then
    cmd+=(--kernel_params "$KERNEL_PARAMS_JSON")
fi

CUDA_VISIBLE_DEVICES="${GPU}" OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
"${cmd[@]}" > "${logfile}" 2>&1

if [[ "${RUN_LM_EVAL}" == "true" ]]; then
    steer_path="$(
        resolve_kernel_steer_config_path \
            "$md" "$sd" "$layer" "$(is_prompt_only "$po")" "$POS" "$et"
    )"

    if [[ ! -f "${steer_path}" ]]; then
        echo "Missing kernel steer config: ${steer_path}" | tee -a "${logfile}"
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

echo "All kernel legacy jobs completed."

#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
PIPELINE_CONDA_ENV="${PIPELINE_CONDA_ENV:-refusal_direction}"
GPU="${GPU:-<GPU_ID>}"
if [[ "${GPU}" == *"<"* ]]; then
    echo "Set GPU to the GPU id to use (it is the placeholder ${GPU})." >&2
    exit 2
fi
LM_EVAL_CONDA_ENV="${LM_EVAL_CONDA_ENV:-steering-lm-eval}"
RUN_LM_EVAL="${RUN_LM_EVAL:-false}"
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
OUTDIR="${OUTDIR:-outputs_kernel_wrapper}"
mkdir -p "${OUTDIR}"

SEEDS=(0)
PROMPT_ONLY_FLAGS=("")
MODEL_PATHS=("Qwen/Qwen2.5-7B-Instruct")
LAYERS=(18)
ETAS=(8.0)

STEER_METHOD="kernel"
KERNEL_TYPE="rbf"
KERNEL_PARAMS_JSON=""             # Example: {"sigma": 48}
LAMBDA_REG="0.0"
STEP_COUNT="10"
KERNEL_EXTRACTION_SOURCE="caa"    # legacy | caa
EVALUATION_PROTOCOL="caa"         # legacy | caa | both
POS=""                            # Empty -> pipeline default (-2 for CAA extraction)
SKIP_BASELINE="false"

CAA_DATASET_ROOT="caa"
CAA_BEHAVIORS=("refusal" "hallucination" "sycophancy" "myopic-reward" "survival-instinct" "coordinate-other-ais" "corrigible-neutral-HHH")
CAA_EVAL_TYPES=("ab")
CAA_EVAL_SYSTEM_PROMPTS=("none")
CAA_EVAL_OPEN_ENDED_SCORE="false"
CAA_OPEN_ENDED_JUDGE_MODEL=""

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
    local behavior="$7"
    local system_prompt="$8"

    python3 - "$model_path" "$seed" "$layer" "$prompt_only" "$pos_override" \
        "$eta" "$LAMBDA_REG" "$STEP_COUNT" "$KERNEL_TYPE" "$KERNEL_PARAMS_JSON" \
        "$KERNEL_EXTRACTION_SOURCE" "$EVALUATION_PROTOCOL" "$CAA_DATASET_ROOT" \
        "$behavior" "$system_prompt" "$CAA_EVAL_OPEN_ENDED_SCORE" <<'PY'
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
    caa_dataset_root,
    caa_behavior,
    caa_system_prompt_raw,
    caa_eval_open_ended_score_raw,
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
caa_eval_open_ended_score = str_to_bool(caa_eval_open_ended_score_raw)

if pos_override_raw:
    pos = int(pos_override_raw)
else:
    pos = -2 if kernel_extraction_source == "caa" else -1

caa_system_prompt = None if caa_system_prompt_raw in {"", "none"} else caa_system_prompt_raw

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
    caa_dataset_root=caa_dataset_root,
    caa_behavior=caa_behavior,
    evaluation_protocol=evaluation_protocol,
    caa_eval_system_prompt=caa_system_prompt,
    caa_eval_open_ended_score=caa_eval_open_ended_score,
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
for cb in "${CAA_BEHAVIORS[@]}"; do
for sys_prompt in "${CAA_EVAL_SYSTEM_PROMPTS[@]}"; do

name="seed${sd}_model_$(get_model_name "$md")_layer${layer}_eta-${et}_step_count-${STEP_COUNT}_caa_behavior-${cb}_sysprompt-${sys_prompt}_prompt_only-$(is_prompt_only "$po")"
logdir="${OUTDIR}/model_$(get_model_name "$md")/layer-${layer}/eta-${et}/step_count-${STEP_COUNT}/caa_behavior-${cb}/sys_prompt-${sys_prompt}/prompt_only-$(is_prompt_only "$po")"
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
    --step_count "$STEP_COUNT"
    --kernel_extraction_source "$KERNEL_EXTRACTION_SOURCE"
    --evaluation_protocol "$EVALUATION_PROTOCOL"
    --caa_dataset_root "$CAA_DATASET_ROOT"
    --caa_behavior "$cb"
    --caa_eval_system_prompt "$sys_prompt"
    --caa_eval_open_ended_score "$CAA_EVAL_OPEN_ENDED_SCORE"
)

if [[ -n "$POS" ]]; then
    cmd+=(--pos "$POS")
fi

if [[ "$po" == "--prompt_only" ]]; then
    cmd+=(--prompt_only)
fi

if [[ "${SKIP_BASELINE,,}" == "true" ]]; then
    cmd+=(--skip_baseline)
fi

if [[ -n "$KERNEL_PARAMS_JSON" ]]; then
    cmd+=(--kernel_params "$KERNEL_PARAMS_JSON")
fi

if [[ -n "$CAA_OPEN_ENDED_JUDGE_MODEL" ]]; then
    cmd+=(--caa_open_ended_judge_model "$CAA_OPEN_ENDED_JUDGE_MODEL")
fi

if [[ "${#CAA_EVAL_TYPES[@]}" -gt 0 ]]; then
    cmd+=(--caa_eval_types "${CAA_EVAL_TYPES[@]}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
"${cmd[@]}" > "${logfile}" 2>&1

if [[ "${RUN_LM_EVAL}" == "true" ]]; then
    steer_path="$(
        resolve_kernel_steer_config_path \
            "$md" "$sd" "$layer" "$(is_prompt_only "$po")" "$POS" "$et" "$cb" "$sys_prompt"
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
done
done

echo "All kernel CAA jobs completed."

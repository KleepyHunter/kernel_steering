#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
PIPELINE_CONDA_ENV="refusal_direction"
LM_EVAL_CONDA_ENV="steering-lm-eval"
LOGIN_HF="false"
HF_TOKEN="${HF_TOKEN:-}"
GPU="${GPU:-<GPU_ID>}"
if [[ "${GPU}" == *"<"* ]]; then
    echo "Set GPU to the GPU id to use (it is the placeholder ${GPU})." >&2
    exit 2
fi

activate_env() {
    local env_name="$1"
    conda deactivate || true
    conda activate "${env_name}"
}

activate_env "${PIPELINE_CONDA_ENV}"

if [[ "${LOGIN_HF}" == "true" ]]; then
    if [[ -z "${HF_TOKEN}" ]]; then
        echo "LOGIN_HF=true but HF_TOKEN is empty."
        exit 1
    fi
    huggingface-cli login --token "${HF_TOKEN}"
fi

# -----------------------------------------------------------------------------
# Sweep config (edit here)
# -----------------------------------------------------------------------------
OUTDIR="outputs_llama3.1_8b_instruct"
mkdir -p "${OUTDIR}"

ETAS=(-2.0 -1.0 1.0 2.0)
SEEDS=(0)
# Keep this style to match your current scripts.
PROMPT_ONLY_FLAGS=("--prompt_only" "")

MODEL_PATHS=("meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it")

# Core pipeline params
KERNEL_TYPE="rbf"
KERNEL_PARAMS_JSON=None  # Example: {"sigma": 1.5}. Empty -> None
LAMBDA_REG="0"
POS="-2"                # Empty string -> None (auto default inside pipeline)
STEP_COUNT="1"
USE_MEAN_ACT="false"
SKIP_BASELINE="false"
RUN_LM_EVAL="true"
# Where pipeline.run_pipeline writes runs (its default artifact root): <PATH_TO_REPO>/pipeline/runs
RUNS_DIR="${RUNS_DIR:-<PATH_TO_REPO>/pipeline/runs}"

# CAA params
KERNEL_EXTRACTION_SOURCE="caa"   # legacy | caa
CAA_DATASET_ROOT="caa"
CAA_BEHAVIORS=("refusal" "hallucination" "sycophancy" "myopic-reward" "survival-instinct" "coordinate-other-ais" "corrigible-neutral-HHH")
# CAA_BEHAVIORS=("survival-instinct")
EVALUATION_PROTOCOL="caa"        # legacy | caa | both
CAA_EVAL_TYPES=("ab")  # ab open_ended truthful_qa mmlu
CAA_EVAL_SYSTEM_PROMPT=("none" "pos" "neg")      # none | pos | neg
CAA_EVAL_OPEN_ENDED_SCORE="false"
CAA_OPEN_ENDED_JUDGE_MODEL="gpt-4o-mini"       # required only when open-ended score is true

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
get_layer_number() {
    local model_path="$1"
    if [[ "$model_path" == *"meta-llama/Llama-3.1-8B-Instruct"* ]]; then
        echo "13 12"
    elif [[ "$model_path" == *"Qwen/Qwen2.5-7B-Instruct"* ]]; then
        echo "18 17"
    else
        # Safe default fallback; edit as needed.
        echo "24"
    fi
}

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

format_kernel_params_for_path() {
    local params_raw="$1"
    if [[ -z "$params_raw" ]]; then
        echo "None"
        return
    fi

    python3 - "$params_raw" <<'PY'
import ast
import json
import sys

raw = sys.argv[1]
text = raw.strip()

if text == "" or text.lower() == "none":
    print("None")
    raise SystemExit(0)

try:
    parsed = json.loads(text)
except json.JSONDecodeError:
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        raise SystemExit(
            f"Invalid KERNEL_PARAMS_JSON: {raw}. Expected JSON dict like '{{\"sigma\": 48}}'."
        )

if not isinstance(parsed, dict):
    raise SystemExit(
        f"Invalid KERNEL_PARAMS_JSON: {raw}. Expected a dict-like value."
    )

print(repr(parsed))
PY
}

# -----------------------------------------------------------------------------
# Main sweep loop
# -----------------------------------------------------------------------------
for sd in "${SEEDS[@]}"; do
for md in "${MODEL_PATHS[@]}"; do
for po in "${PROMPT_ONLY_FLAGS[@]}"; do
for et in "${ETAS[@]}"; do
for cb in "${CAA_BEHAVIORS[@]}"; do
for layer in $(get_layer_number "$md"); do

name="seed${sd}_model_$(get_model_name "$md")_eta${et}_layer${layer}_caa_behavior-${cb}_prompt_only-$(is_prompt_only "$po")"
logdir="${OUTDIR}/model_$(get_model_name "$md")/eta-${et}/layer-${layer}/caa_behavior-${cb}/use_mean_act-${USE_MEAN_ACT}/prompt_only-$(is_prompt_only "$po")"
mkdir -p "${logdir}"
logfile="${logdir}/seed_${sd}.log"

echo "Running: ${name}"

activate_env "${PIPELINE_CONDA_ENV}"

cmd=(
    python3 -m pipeline.run_pipeline
    --model_path "$md"
    --kernel_type "$KERNEL_TYPE"
    --eta "$et"
    --lambda_reg "$LAMBDA_REG"
    --seed "$sd"
    --layer "$layer"
    --step_count "$STEP_COUNT"
    --kernel_extraction_source "$KERNEL_EXTRACTION_SOURCE"
    --caa_dataset_root "$CAA_DATASET_ROOT"
    --caa_behavior "$cb"
    --evaluation_protocol "$EVALUATION_PROTOCOL"
    --caa_eval_system_prompt "$CAA_EVAL_SYSTEM_PROMPT"
    --caa_eval_open_ended_score "$CAA_EVAL_OPEN_ENDED_SCORE"
)

if [[ -n "$POS" ]]; then
    cmd+=(--pos "$POS")
fi

# run_pipeline.py uses a bool parser for this arg; pass only when true.
if [[ "${USE_MEAN_ACT,,}" == "true" ]]; then
    cmd+=(--use_mean_act True)
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

# Keep your lm-eval flow available, but disabled by default.
if [[ "${RUN_LM_EVAL}" == "true" ]]; then
    activate_env "${LM_EVAL_CONDA_ENV}"
    kernel_params_path_component="$(format_kernel_params_for_path "$KERNEL_PARAMS_JSON")"
    steer_base_path="${RUNS_DIR}/seed-$sd/use_mean_act-False/kernel_type-rbf/kernel_params-$kernel_params_path_component/eta-$et/lambda_reg-$LAMBDA_REG/pos-$POS/layer-$layer/step_count-$STEP_COUNT/prompt_only-$(is_prompt_only "$po")/kernel_extraction_source-$KERNEL_EXTRACTION_SOURCE/caa_dataset_root-$CAA_DATASET_ROOT/caa_behavior-$cb/evaluation_protocol-$EVALUATION_PROTOCOL/$(get_model_name "$md")/steer_config"

    if [[ "$sd" -eq 0 ]]; then
        CUDA_VISIBLE_DEVICES="${GPU}" lm_eval --model steered --model_args "pretrained=$md" "steer_path=$steer_base_path/steer_config_dim.pt" --tasks tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande --device=cuda:0 --batch_size auto >> "${logfile}" 2>&1
        CUDA_VISIBLE_DEVICES="${GPU}" lm_eval --model steered --model_args "pretrained=$md" "steer_path=$steer_base_path/steer_config_kernel.pt" --tasks tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande --device=cuda:0 --batch_size auto >> "${logfile}" 2>&1
    else
        CUDA_VISIBLE_DEVICES="${GPU}" lm_eval --model steered --model_args "pretrained=$md" "steer_path=$steer_base_path/steer_config_kernel.pt" --tasks tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande --device=cuda:0 --batch_size auto >> "${logfile}" 2>&1
    fi
fi

echo "Finished: ${name}"
echo "-------------------------------"

done
done
done
done
done
done

echo "All jobs completed."

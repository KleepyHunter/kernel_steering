#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
PIPELINE_CONDA_ENV="refusal_direction"
LM_EVAL_CONDA_ENV="${LM_EVAL_CONDA_ENV:-steering-lm-eval}"
GPU="${GPU:-<GPU_ID>}"
if [[ "${GPU}" == *"<"* ]]; then
    echo "Set GPU to the GPU id to use (it is the placeholder ${GPU})." >&2
    exit 2
fi
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
OUTDIR="${OUTDIR:-outputs_curveball_wrapper}"
mkdir -p "${OUTDIR}"

SEEDS=(0)
# PROMPT_ONLY_FLAGS=("--prompt_only" "")
PROMPT_ONLY_FLAGS=("")
MODEL_PATHS=("Qwen/Qwen2.5-32B-Instruct")
CURVEBALL_KS=(20)
CURVEBALL_ALPHAS=(1.0 5.0 10.0 20.0 50.0 100.0 150.0 -1.0 -5.0 -10.0 -20.0 -50.0 -100.0 -150.0)
CURVEBALL_KERNEL="poly"
CURVEBALL_DEGREE="3"
CURVEBALL_GAMMA=""
CURVEBALL_COEF0="1.0"
CURVEBALL_FIT_INVERSE_ALPHA="1.0"

STEER_METHOD="curveball"
CURVEBALL_EXTRACTION_SOURCE="caa"     # caa | legacy
EVALUATION_PROTOCOL="caa"             # legacy | caa | both
CAA_DATASET_ROOT="caa"
CAA_BEHAVIORS=("hallucination" "refusal" "sycophancy" "myopic-reward" "survival-instinct" "coordinate-other-ais" "corrigible-neutral-HHH")
CAA_EVAL_TYPES=("ab")
CAA_EVAL_SYSTEM_PROMPTS=("none")
CAA_EVAL_OPEN_ENDED_SCORE="false"
CAA_OPEN_ENDED_JUDGE_MODEL=""

POS=""

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
get_layer_number() {
    local model_path="$1"
    if [[ "$model_path" == *"google/gemma-2-9b-it"* ]]; then
        echo "22"
    elif [[ "$model_path" == *"meta-llama/Llama-3.1-8B-Instruct"* ]]; then
        echo "12"
    elif [[ "$model_path" == *"Qwen/Qwen2.5-7B-Instruct"* ]]; then
        echo "18"
    elif [[ "$model_path" == *"Qwen/Qwen2.5-32B-Instruct"* ]]; then
        echo "46"
    else
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

resolve_curveball_steer_config_path() {
    local model_path="$1"
    local seed="$2"
    local layer="$3"
    local prompt_only="$4"
    local pos_override="$5"
    local behavior="$6"
    local system_prompt="$7"
    local curveball_k="$8"
    local curveball_alpha="$9"

    python3 - "$model_path" "$seed" "$layer" "$prompt_only" "$pos_override" \
        "$EVALUATION_PROTOCOL" "$CURVEBALL_EXTRACTION_SOURCE" "$CAA_DATASET_ROOT" "$behavior" \
        "$system_prompt" "$CAA_EVAL_OPEN_ENDED_SCORE" "$curveball_k" "$CURVEBALL_KERNEL" \
        "$CURVEBALL_DEGREE" "$CURVEBALL_GAMMA" "$CURVEBALL_COEF0" "$curveball_alpha" \
        "$CURVEBALL_FIT_INVERSE_ALPHA" <<'PY'
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
    curveball_extraction_source,
    caa_dataset_root,
    caa_behavior,
    caa_system_prompt_raw,
    caa_eval_open_ended_score_raw,
    curveball_k,
    curveball_kernel,
    curveball_degree,
    curveball_gamma_raw,
    curveball_coef0,
    curveball_alpha,
    curveball_fit_inverse_alpha,
) = sys.argv[1:]


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y", "t"}


prompt_only = str_to_bool(prompt_only_raw)
caa_eval_open_ended_score = str_to_bool(caa_eval_open_ended_score_raw)
curveball_gamma = None if curveball_gamma_raw in {"", "none", "None"} else float(curveball_gamma_raw)

if pos_override_raw:
    pos = int(pos_override_raw)
else:
    pos = -2 if curveball_extraction_source == "caa" else -1

caa_system_prompt = None if caa_system_prompt_raw in {"", "none"} else caa_system_prompt_raw

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
    steer_method="curveball",
    kernel_extraction_source="legacy",
    curveball_extraction_source=curveball_extraction_source,
    curveball_k=int(curveball_k),
    curveball_kernel=curveball_kernel,
    curveball_degree=int(curveball_degree),
    curveball_gamma=curveball_gamma,
    curveball_coef0=float(curveball_coef0),
    curveball_alpha=float(curveball_alpha),
    curveball_fit_inverse_alpha=float(curveball_fit_inverse_alpha),
    caa_dataset_root=caa_dataset_root,
    caa_behavior=caa_behavior,
    evaluation_protocol=evaluation_protocol,
    caa_eval_system_prompt=caa_system_prompt,
    caa_eval_open_ended_score=caa_eval_open_ended_score,
)

print(os.path.join(cfg.artifact_path(), "steer_config", "steer_config_curveball.pt"))
PY
}

# -----------------------------------------------------------------------------
# Main sweep loop
# -----------------------------------------------------------------------------
for sd in "${SEEDS[@]}"; do
for md in "${MODEL_PATHS[@]}"; do
for po in "${PROMPT_ONLY_FLAGS[@]}"; do
for cbk in "${CURVEBALL_KS[@]}"; do
for cba in "${CURVEBALL_ALPHAS[@]}"; do
for cb in "${CAA_BEHAVIORS[@]}"; do
for sys_prompt in "${CAA_EVAL_SYSTEM_PROMPTS[@]}"; do
for layer in $(get_layer_number "$md"); do

name="seed${sd}_model_$(get_model_name "$md")_layer${layer}_curveball_k-${cbk}_curveball_alpha-${cba}_caa_behavior-${cb}_sysprompt-${sys_prompt}_prompt_only-$(is_prompt_only "$po")"
logdir="${OUTDIR}/model_$(get_model_name "$md")/layer-${layer}/curveball_k-${cbk}/curveball_alpha-${cba}/caa_behavior-${cb}/sys_prompt-${sys_prompt}/prompt_only-$(is_prompt_only "$po")"
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
    --curveball_extraction_source "$CURVEBALL_EXTRACTION_SOURCE"
    --curveball_k "$cbk"
    --curveball_kernel "$CURVEBALL_KERNEL"
    --curveball_degree "$CURVEBALL_DEGREE"
    --curveball_coef0 "$CURVEBALL_COEF0"
    --curveball_alpha "$cba"
    --curveball_fit_inverse_alpha "$CURVEBALL_FIT_INVERSE_ALPHA"
    --evaluation_protocol "$EVALUATION_PROTOCOL"
    --caa_dataset_root "$CAA_DATASET_ROOT"
    --caa_behavior "$cb"
    --caa_eval_system_prompt "$sys_prompt"
    --caa_eval_open_ended_score "$CAA_EVAL_OPEN_ENDED_SCORE"
)

if [[ -n "$CURVEBALL_GAMMA" ]]; then
    cmd+=(--curveball_gamma "$CURVEBALL_GAMMA")
fi

if [[ -n "$POS" ]]; then
    cmd+=(--pos "$POS")
fi

if [[ "$po" == "--prompt_only" ]]; then
    cmd+=(--prompt_only)
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
        resolve_curveball_steer_config_path \
            "$md" "$sd" "$layer" "$(is_prompt_only "$po")" "$POS" "$cb" "$sys_prompt" "$cbk" "$cba"
    )"

    if [[ ! -f "${steer_path}" ]]; then
        echo "Missing curveball steer config: ${steer_path}" | tee -a "${logfile}"
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
done

echo "All curveball jobs completed."

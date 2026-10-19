#!/usr/bin/env bash
set -euo pipefail

# Standalone TinyBenchmarks run with the lm-evaluation-harness copy in this repository.
# Run it in the `refusal_direction` environment (environment.yml), which includes lm_eval.
#
# Usage:
#   scripts/run_tinybench.sh MODEL_PATH OUTPUT_PATH [STEER_PATH]
#
#   Without STEER_PATH the unsteered model is scored (lm_eval --model hf). With STEER_PATH,
#   a steer config written by a pipeline run (<run>/steer_config/steer_config_*.pt), the
#   steered model is scored (lm_eval --model steered).
#   For manifest-locked utility tests use scripts/run_locked_tinybench.py instead.
#
# Environment overrides:
#   TASKS=tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande
#   CUDA_VISIBLE_DEVICES=<GPU_ID>   (required)
#   DEVICE=cuda:0  BATCH_SIZE=auto
#   LM_EVAL=lm_eval   (an lm_eval executable installed from ./lm-evaluation-harness)

if [[ $# -lt 2 || $# -gt 3 ]]; then
  sed -n '7,19p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARNESS_ROOT="${REPO_ROOT}/lm-evaluation-harness"

MODEL_PATH="$1"
OUTPUT_PATH="$(realpath -m "$2")"
STEER_PATH="${3:-}"
TASKS="${TASKS:-tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
LM_EVAL="${LM_EVAL:-lm_eval}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
export CUDA_VISIBLE_DEVICES
if [[ "${CUDA_VISIBLE_DEVICES}" == *"<"* ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to the GPU id(s) to use (it is the placeholder ${CUDA_VISIBLE_DEVICES})." >&2
  exit 2
fi

if ! command -v "${LM_EVAL}" >/dev/null 2>&1; then
  echo "${LM_EVAL} not found; activate the refusal_direction environment (environment.yml) or set LM_EVAL." >&2
  exit 1
fi
# The steering actions live in this repository's copy of the harness; refuse any other install.
LM_EVAL_PYTHON="$(dirname "$(command -v "${LM_EVAL}")")/python"
if [[ -x "${LM_EVAL_PYTHON}" ]]; then
  installed="$("${LM_EVAL_PYTHON}" -c 'import lm_eval, os; print(os.path.realpath(os.path.dirname(os.path.dirname(lm_eval.__file__))))')"
  if [[ "${installed}" != "$(realpath "${HARNESS_ROOT}")" ]]; then
    echo "${LM_EVAL} uses lm_eval from ${installed}, not ${HARNESS_ROOT}." >&2
    echo "Install the copy: pip install -e ${HARNESS_ROOT}" >&2
    exit 1
  fi
fi

if [[ -n "${STEER_PATH}" ]]; then
  STEER_PATH="$(realpath "${STEER_PATH}")"
  [[ -f "${STEER_PATH}" ]] || { echo "Missing steer config: ${STEER_PATH}" >&2; exit 1; }
  model_args=(--model steered --model_args "pretrained=${MODEL_PATH},steer_path=${STEER_PATH}")
else
  model_args=(--model hf --model_args "pretrained=${MODEL_PATH}")
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
cd "${HARNESS_ROOT}"
set -x
"${LM_EVAL}" "${model_args[@]}" \
  --tasks "${TASKS}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_PATH}"

#!/usr/bin/env bash
set -euo pipefail

# Run the validation-locked CHaRS jailbreak workflow for every registered model.
#
# Usage:
#   scripts/jailbreak/validate_chars_all_models.sh [prepare|run|select|all]
#
# Sequential on the currently visible GPU(s) by default. To run one model per
# GPU concurrently, set CHARS_MODEL_GPUS to three comma-separated GPU IDs:
#   CHARS_MODEL_GPUS=<GPU_ID_1>,<GPU_ID_2>,<GPU_ID_3> scripts/jailbreak/validate_chars_all_models.sh all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ACTION="${1:-all}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/chars_jailbreak_v2}"
APPLICATION_MODE="${JAILBREAK_APPLICATION_MODE:-all}"
UTILITY_CAP_RATIO="${UTILITY_CAP_RATIO:-1.05}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_GPUS="${CHARS_MODEL_GPUS:-}"

MODELS=(
  "google/gemma-2-9b-it"
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

case "${ACTION}" in
  prepare|run|select|all) ;;
  *) echo "Usage: $0 [prepare|run|select|all]" >&2; exit 2 ;;
esac
case "${APPLICATION_MODE}" in
  all|prompt_only|all_tokens) ;;
  *) echo "Invalid JAILBREAK_APPLICATION_MODE." >&2; exit 2 ;;
esac

run_model() {
  local model_path="$1"
  echo "CHaRS jailbreak ${ACTION}: ${model_path}"
  MODEL_PATH="${model_path}" \
  CAMPAIGN="${CAMPAIGN}" \
  JAILBREAK_APPLICATION_MODE="${APPLICATION_MODE}" \
  UTILITY_CAP_RATIO="${UTILITY_CAP_RATIO}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    "${SCRIPT_DIR}/validate_chars.sh" "${ACTION}"
}

if [[ -z "${MODEL_GPUS}" ]]; then
  for model_path in "${MODELS[@]}"; do
    run_model "${model_path}"
  done
  exit 0
fi

IFS=',' read -r -a gpu_ids <<< "${MODEL_GPUS}"
if [[ "${#gpu_ids[@]}" -ne "${#MODELS[@]}" ]]; then
  echo "CHARS_MODEL_GPUS must contain exactly ${#MODELS[@]} comma-separated GPU IDs." >&2
  exit 2
fi

pids=()
for index in "${!MODELS[@]}"; do
  if [[ -z "${gpu_ids[$index]}" ]]; then
    echo "CHARS_MODEL_GPUS contains an empty GPU ID." >&2
    exit 2
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu_ids[$index]}"
    run_model "${MODELS[$index]}"
  ) &
  pids+=("$!")
done

failures=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "CHaRS jailbreak failed for ${MODELS[$index]}." >&2
    failures=1
  fi
done
exit "${failures}"

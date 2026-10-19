#!/usr/bin/env bash
set -euo pipefail

# Jailbreak validation for ActAdd, KAS, and Curveball on Gemma, Llama, and Qwen:
# prepare and run the validation grid for each model, then lock the selections in
# ${CAMPAIGN}/manifests/jailbreak_validation.json (used by scripts/jailbreak/test.sh).
# Runs end to end; restarts skip completed jobs. Needs the LlamaGuard server
# (scripts/serve_llamaguard.sh).
#
# Run from the existing PyTorch/Transformers environment.
# Optional overrides:
#   GEMMA_MODEL=google/gemma-2-9b-it
#   LLAMA_MODEL=meta-llama/Llama-3.1-8B-Instruct
#   QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
#   CAMPAIGN=/path/to/output
#   CUDA_VISIBLE_DEVICES=<GPU_ID>
#   LLAMAGUARD_VLLM_SERVER=http://localhost:8000/v1
#   JAILBREAK_APPLICATION_MODE=all  # all, prompt_only, or all_tokens

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-2-9b-it}"
LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/rebuttal}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
LLAMAGUARD_VLLM_SERVER="${LLAMAGUARD_VLLM_SERVER:-http://0.0.0.0:8000/v1}"
JAILBREAK_APPLICATION_MODE="${JAILBREAK_APPLICATION_MODE:-all}"
export CUDA_VISIBLE_DEVICES OPENAI_API_KEY LLAMAGUARD_VLLM_SERVER

case "${JAILBREAK_APPLICATION_MODE}" in
  all|prompt_only|all_tokens) ;;
  *)
    echo "Invalid JAILBREAK_APPLICATION_MODE: ${JAILBREAK_APPLICATION_MODE}. Expected all, prompt_only, or all_tokens." >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"

run_model_grid() {
  local model_path="$1"
  local model_alias="${model_path##*/}"
  local job_suffix=""
  if [[ "${JAILBREAK_APPLICATION_MODE}" != "all" ]]; then
    job_suffix="_${JAILBREAK_APPLICATION_MODE}"
  fi
  local job_file="${CAMPAIGN}/jobs/jail_${model_alias}${job_suffix}.jsonl"
  local status_dir="${job_file}.status"

  echo "Preparing jailbreaking validation jobs for ${model_alias}..."
  scripts/validation_workflow.sh prepare-jail "${model_path}" "${CAMPAIGN}" "${JAILBREAK_APPLICATION_MODE}"

  echo "Running ${model_alias} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "Progress markers: ${status_dir}"
  scripts/validation_workflow.sh run-local     "${job_file}" 0 "$(($(wc -l < "${job_file}") - 1))" "${status_dir}"
}

run_model_grid "${GEMMA_MODEL}"
run_model_grid "${LLAMA_MODEL}"
run_model_grid "${QWEN_MODEL}"

if [[ "${JAILBREAK_APPLICATION_MODE}" == "all" ]]; then
  MANIFEST="${CAMPAIGN}/manifests/jailbreak_validation.json"
else
  MANIFEST="${CAMPAIGN}/manifests/jailbreak_validation_${JAILBREAK_APPLICATION_MODE}.json"
fi
if [[ -f "${MANIFEST}" ]]; then
  echo "Manifest already exists; verifying it instead of overwriting it."
  python3 -c 'import sys; from pipeline.validation_selection import load_manifest; load_manifest(sys.argv[1]); print("Manifest verified:", sys.argv[1])' "${MANIFEST}"
else
  echo "Selecting jailbreaking configurations under the 1.05x utility cap..."
  scripts/validation_workflow.sh select-jail "${CAMPAIGN}" "${JAILBREAK_APPLICATION_MODE}"
fi

echo "Jailbreaking validation complete for application mode: ${JAILBREAK_APPLICATION_MODE}"
echo "Manifest: ${MANIFEST}"

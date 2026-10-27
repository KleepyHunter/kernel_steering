#!/usr/bin/env bash
set -euo pipefail

# Jailbreak test runs at the validation-selected settings (main jailbreak table).
#
# One standard-phase run per selected method and application mode. Each reports
# JAILBREAKBENCH ASR (step 3b) and Pile / Alpaca perplexity (step 5), and
# exports its steer config for TinyBenchmarks. Run the validation scripts first:
# scripts/jailbreak/validate.sh (ActAdd, KAS, Curveball) and
# scripts/jailbreak/validate_chars_all_models.sh (CHaRS).
#
# Usage:
#   scripts/jailbreak/test.sh [prepare|run|all]
#
# Environment overrides:
#   MODELS="google/gemma-2-9b-it meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
#   MANIFESTS="..."  Space-separated jailbreak manifests. Default: the ActAdd/KAS/
#                    Curveball manifest from validation_runs/rebuttal plus every CHaRS
#                    manifest under validation_runs/chars_jailbreak*/. Add the SAS
#                    manifest here to include SAS.
#   CAMPAIGN=/path/to/output (default: validation_runs/jailbreak_test)
#   CUDA_VISIBLE_DEVICES=<GPU_ID>
#   LLAMAGUARD_VLLM_SERVER=http://0.0.0.0:8000/v1  (see scripts/serve_llamaguard.sh)
#   PYTHON_BIN=python3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${1:-all}"
MODELS="${MODELS:-google/gemma-2-9b-it meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/jailbreak_test}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
LLAMAGUARD_VLLM_SERVER="${LLAMAGUARD_VLLM_SERVER:-http://0.0.0.0:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES LLAMAGUARD_VLLM_SERVER OPENAI_API_KEY PYTHON_BIN

case "${ACTION}" in
  prepare|run|all) ;;
  *)
    echo "Usage: $0 [prepare|run|all]" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"

default_manifests() {
  local main="${REPO_ROOT}/validation_runs/rebuttal/manifests/jailbreak_validation.json"
  [[ -f "${main}" ]] && echo "${main}"
  local chars
  for chars in "${REPO_ROOT}"/validation_runs/chars_jailbreak*/manifests/chars_jail_*.json; do
    [[ -f "${chars}" ]] && echo "${chars}"
  done
  return 0
}

MANIFESTS="${MANIFESTS:-$(default_manifests | tr '\n' ' ')}"
JOB_FILE="${CAMPAIGN}/jobs/jailbreak_test.jsonl"
STATUS_DIR="${JOB_FILE}.status"

prepare_jobs() {
  read -r -a manifests <<< "${MANIFESTS}"
  if [[ "${#manifests[@]}" -eq 0 ]]; then
    echo "No jailbreak manifests found; run the validation scripts or set MANIFESTS." >&2
    exit 1
  fi
  local manifest
  for manifest in "${manifests[@]}"; do
    "${PYTHON_BIN}" -c 'import sys; from pipeline.validation_selection import load_manifest; load_manifest(sys.argv[1]); print("Manifest verified:", sys.argv[1])' "${manifest}"
  done

  read -r -a models <<< "${MODELS}"
  mkdir -p "$(dirname "${JOB_FILE}")" "${CAMPAIGN}/artifacts"
  "${PYTHON_BIN}" scripts/build_experiment_jobs.py \
    --format jsonl \
    --output "${JOB_FILE}" \
    standard-test \
    --model-path "${models[@]}" \
    --artifact-root "${CAMPAIGN}/artifacts" \
    --manifest "${manifests[@]}"

  local count
  count="$(wc -l < "${JOB_FILE}")"
  echo "Jailbreak test jobs: ${JOB_FILE} (${count})"
  echo "Slurm array: sbatch --array=0-$((count - 1)) --export=ALL,JOB_FILE=${JOB_FILE},STATUS_DIR=${STATUS_DIR},OPENAI_API_KEY=${OPENAI_API_KEY},LLAMAGUARD_VLLM_SERVER=${LLAMAGUARD_VLLM_SERVER} scripts/slurm_validation_array.sh"
}

run_jobs() {
  if [[ ! -f "${JOB_FILE}" ]]; then
    echo "Missing ${JOB_FILE}; run $0 prepare first." >&2
    exit 1
  fi
  scripts/validation_workflow.sh run-local "${JOB_FILE}" 0 "$(($(wc -l < "${JOB_FILE}") - 1))" "${STATUS_DIR}"
  echo "Jailbreak test runs complete: ${CAMPAIGN}/artifacts/<method>"
}

case "${ACTION}" in
  prepare) prepare_jobs ;;
  run) run_jobs ;;
  all)
    prepare_jobs
    run_jobs
    ;;
esac

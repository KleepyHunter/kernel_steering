#!/usr/bin/env bash
set -euo pipefail

# Dedicated Gemma 2 9B jailbreak SAS validation sweep.
# The unsteered baseline is included because the selector requires its perplexity.
#
# Usage:
#   scripts/jailbreak/validate_sas.sh [prepare|run|select|all]
#
# Environment overrides:
#   GEMMA_MODEL=google/gemma-2-9b-it
#   CAMPAIGN=/path/to/output
#   CUDA_VISIBLE_DEVICES=<GPU_ID>
#   JAILBREAK_APPLICATION_MODE=all  # all, prompt_only, or all_tokens
#   LLAMAGUARD_VLLM_SERVER=http://localhost:8000/v1
#   UTILITY_CAP_RATIO=1.05
#   PYTHON_BIN=python3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${1:-all}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-2-9b-it}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/sas}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
JAILBREAK_APPLICATION_MODE="${JAILBREAK_APPLICATION_MODE:-all}"
LLAMAGUARD_VLLM_SERVER="${LLAMAGUARD_VLLM_SERVER:-http://0.0.0.0:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
UTILITY_CAP_RATIO="${UTILITY_CAP_RATIO:-1.05}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES LLAMAGUARD_VLLM_SERVER OPENAI_API_KEY

case "${ACTION}" in
  prepare|run|select|all) ;;
  *)
    echo "Usage: $0 [prepare|run|select|all]" >&2
    exit 2
    ;;
esac

case "${JAILBREAK_APPLICATION_MODE}" in
  all|prompt_only|all_tokens) ;;
  *)
    echo "Invalid JAILBREAK_APPLICATION_MODE: ${JAILBREAK_APPLICATION_MODE}." >&2
    exit 2
    ;;
esac

suffix=""
expected_jobs=41
if [[ "${JAILBREAK_APPLICATION_MODE}" != "all" ]]; then
  suffix="_${JAILBREAK_APPLICATION_MODE}"
  expected_jobs=21
fi

JOB_FILE="${CAMPAIGN}/jobs/jail_sas_strength_grid_select${suffix}.jsonl"
STATUS_DIR="${JOB_FILE}.status"
ARTIFACT_ROOT="${CAMPAIGN}/artifacts/jail_sas_strength_grid_select${suffix}"
MANIFEST="${CAMPAIGN}/manifests/jailbreak_sas_strength_grid_validation${suffix}.json"

cd "${REPO_ROOT}"

prepare_jobs() {
  mkdir -p "$(dirname "${JOB_FILE}")" "${ARTIFACT_ROOT}"
  local command=(
    "${PYTHON_BIN}" scripts/build_validation_jobs.py
    --format jsonl
    --output "${JOB_FILE}"
    jail-select
    --model-path "${GEMMA_MODEL}"
    --artifact-root "${ARTIFACT_ROOT}"
    --application-mode "${JAILBREAK_APPLICATION_MODE}"
    --method sas
  )
  "${command[@]}"

  local count
  count="$(wc -l < "${JOB_FILE}")"
  if [[ "${count}" -ne "${expected_jobs}" ]]; then
    echo "Expected ${expected_jobs} jailbreak baseline/SAS jobs, found ${count}." >&2
    exit 1
  fi
  echo "Jailbreak SAS jobs: ${JOB_FILE} (${count}; includes baseline)"
  echo "Local run: $0 run"
  echo "Slurm array: sbatch --array=0-$((count - 1)) --export=ALL,JOB_FILE=${JOB_FILE},STATUS_DIR=${STATUS_DIR} scripts/slurm_validation_array.sh"
}

run_jobs() {
  if [[ ! -f "${JOB_FILE}" ]]; then
    echo "Missing ${JOB_FILE}; run $0 prepare first." >&2
    exit 1
  fi
  local count
  count="$(wc -l < "${JOB_FILE}")"
  local command=(
    scripts/validation_workflow.sh run-local
    "${JOB_FILE}" 0 "$((count - 1))" "${STATUS_DIR}"
  )
  "${command[@]}"
}

select_manifest() {
  if [[ ! -f "${JOB_FILE}" ]]; then
    echo "Missing ${JOB_FILE}; run $0 prepare first." >&2
    exit 1
  fi
  local count completed
  count="$(wc -l < "${JOB_FILE}")"
  completed=0
  if [[ -d "${STATUS_DIR}" ]]; then
    completed="$(find "${STATUS_DIR}" -maxdepth 1 -type f -name "*.done.json" | wc -l)"
  fi
  if [[ "${completed}" -ne "${count}" ]]; then
    echo "Cannot select: ${completed}/${count} jailbreak baseline/SAS jobs are complete." >&2
    exit 1
  fi

  mkdir -p "$(dirname "${MANIFEST}")"
  if [[ -f "${MANIFEST}" ]]; then
    "${PYTHON_BIN}" -c 'import sys; from pipeline.validation_selection import load_manifest; load_manifest(sys.argv[1]); print("Manifest verified:", sys.argv[1])' "${MANIFEST}"
  else
    local selector=(
      "${PYTHON_BIN}" -m pipeline.validation_selection select-jailbreak
      --candidate-root "${ARTIFACT_ROOT}"
      --utility-cap-ratio "${UTILITY_CAP_RATIO}"
      --allow-incomplete-grid
      --output "${MANIFEST}"
    )
    "${selector[@]}"
  fi
  echo "Jailbreak SAS manifest: ${MANIFEST}"
}

case "${ACTION}" in
  prepare)
    prepare_jobs
    ;;
  run)
    run_jobs
    ;;
  select)
    select_manifest
    ;;
  all)
    prepare_jobs
    run_jobs
    select_manifest
    ;;
esac

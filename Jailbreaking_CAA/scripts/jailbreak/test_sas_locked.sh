#!/usr/bin/env bash
set -euo pipefail

# Run manifest-locked JAILBREAKBENCH tests for the selected Gemma 2 9B SAS
# configurations. The plan contains one shared baseline plus the selected
# prompt-only and all-token interventions.
#
# Usage:
#   scripts/jailbreak/test_sas_locked.sh [prepare|run|all]
#
# Environment overrides:
#   GEMMA_MODEL=google/gemma-2-9b-it
#   CAMPAIGN=/path/to/test/output
#   CUDA_VISIBLE_DEVICES=<GPU_ID>
#   SAS_MANIFEST=/path/to/jailbreak_sas_manifest.json
#   LLAMAGUARD_VLLM_SERVER=http://localhost:8000/v1
#   OPENAI_API_KEY=dummy
#   PYTHON_BIN=python3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${1:-all}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-2-9b-it}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/sas_jailbreak_locked_test}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
SAS_MANIFEST="${SAS_MANIFEST:-${REPO_ROOT}/validation_runs/sas_jailbreak/manifests/jailbreak_sas_strength_grid_validation.json}"
LLAMAGUARD_VLLM_SERVER="${LLAMAGUARD_VLLM_SERVER:-http://localhost:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES LLAMAGUARD_VLLM_SERVER OPENAI_API_KEY

case "${ACTION}" in
  prepare|run|all) ;;
  *)
    echo "Usage: $0 [prepare|run|all]" >&2
    exit 2
    ;;
esac

TAG="jailbreak_sas"
JOB_FILE="${CAMPAIGN}/jobs/test_${TAG}.jsonl"
STATUS_DIR="${JOB_FILE}.status"
EXPECTED_JOBS=3

cd "${REPO_ROOT}"

prepare_jobs() {
  if [[ ! -f "${SAS_MANIFEST}" ]]; then
    echo "Missing SAS manifest: ${SAS_MANIFEST}" >&2
    exit 1
  fi

  scripts/validation_workflow.sh prepare-tests \
    "${GEMMA_MODEL}" "${SAS_MANIFEST}" "${CAMPAIGN}" "${TAG}"

  local count
  count="$(wc -l < "${JOB_FILE}")"
  if [[ "${count}" -ne "${EXPECTED_JOBS}" ]]; then
    echo "Expected ${EXPECTED_JOBS} SAS jailbreak test jobs, found ${count}." >&2
    exit 1
  fi

  echo "SAS jailbreak test jobs: ${JOB_FILE} (${count})"
  echo "Local run: $0 run"
  echo "Slurm array: sbatch --array=0-$((count - 1)) --export=ALL,JOB_FILE=${JOB_FILE},STATUS_DIR=${STATUS_DIR},OPENAI_API_KEY=${OPENAI_API_KEY},LLAMAGUARD_VLLM_SERVER=${LLAMAGUARD_VLLM_SERVER} scripts/slurm_validation_array.sh"
}

run_jobs() {
  if [[ ! -f "${JOB_FILE}" ]]; then
    echo "Missing ${JOB_FILE}; run $0 prepare first." >&2
    exit 1
  fi

  local count
  count="$(wc -l < "${JOB_FILE}")"
  scripts/validation_workflow.sh run-local \
    "${JOB_FILE}" 0 "$((count - 1))" "${STATUS_DIR}"
  echo "SAS jailbreak held-out test complete: ${STATUS_DIR}"
}

case "${ACTION}" in
  prepare)
    prepare_jobs
    ;;
  run)
    run_jobs
    ;;
  all)
    prepare_jobs
    run_jobs
    ;;
esac

echo "SAS jailbreak locked-test campaign: ${CAMPAIGN}"

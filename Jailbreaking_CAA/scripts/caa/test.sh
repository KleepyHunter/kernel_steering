#!/usr/bin/env bash
set -euo pipefail

# Manifest-locked held-out CAA tests for the unsteered baseline, ActAdd, KAS, and
# Curveball (CAA tables, all system-prompt conditions). Run
# scripts/caa/validate.sh first; it writes the manifest these tests are locked to.
# CHaRS and SAS tests: scripts/caa/test_chars_neg.sh, scripts/caa/test_curveball_sas.sh.
#
# Usage:
#   scripts/caa/test.sh [prepare|run|all]
#
# Environment overrides:
#   GEMMA_MODEL=google/gemma-2-9b-it
#   CAA_PROMPT_CONDITION=all  # all, none, pos, or neg (must match the validation run)
#   VALIDATION_CAMPAIGN=/path/used/by/scripts/caa/validate.sh (default: validation_runs/rebuttal)
#   MANIFEST=/path/to/caa_manifest.json (default: derived from the two settings above)
#   CAMPAIGN=/path/to/test/output (default: validation_runs/caa_test)
#   CUDA_VISIBLE_DEVICES=<GPU_ID>
#   PYTHON_BIN=python3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${1:-all}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-2-9b-it}"
CAA_PROMPT_CONDITION="${CAA_PROMPT_CONDITION:-all}"
VALIDATION_CAMPAIGN="${VALIDATION_CAMPAIGN:-${REPO_ROOT}/validation_runs/rebuttal}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/caa_test}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES PYTHON_BIN

case "${ACTION}" in
  prepare|run|all) ;;
  *)
    echo "Usage: $0 [prepare|run|all]" >&2
    exit 2
    ;;
esac

case "${CAA_PROMPT_CONDITION}" in
  all)
    suffix=""
    # 6 behaviors x 3 prompt conditions x (baseline + ActAdd + KAS + Curveball).
    expected_jobs=72
    ;;
  none|pos|neg)
    suffix="_${CAA_PROMPT_CONDITION}"
    expected_jobs=24
    ;;
  *)
    echo "Invalid CAA_PROMPT_CONDITION: ${CAA_PROMPT_CONDITION}. Expected all, none, pos, or neg." >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"

MANIFEST="${MANIFEST:-${VALIDATION_CAMPAIGN}/manifests/caa_validation${suffix}.json}"
TAG="caa${suffix}"
JOB_FILE="${CAMPAIGN}/jobs/test_${TAG}.jsonl"
STATUS_DIR="${JOB_FILE}.status"

prepare_jobs() {
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing CAA manifest: ${MANIFEST}; run scripts/caa/validate.sh first." >&2
    exit 1
  fi
  "${PYTHON_BIN}" -c 'import sys; from pipeline.validation_selection import load_manifest; load_manifest(sys.argv[1]); print("Manifest verified:", sys.argv[1])' "${MANIFEST}"

  scripts/validation_workflow.sh prepare-tests \
    "${GEMMA_MODEL}" "${MANIFEST}" "${CAMPAIGN}" "${TAG}"

  local count
  count="$(wc -l < "${JOB_FILE}")"
  if [[ "${count}" -ne "${expected_jobs}" ]]; then
    echo "Expected ${expected_jobs} CAA locked-test jobs, found ${count}." >&2
    exit 1
  fi
  echo "CAA held-out jobs: ${JOB_FILE} (${count})"
  echo "Slurm array: sbatch --array=0-$((count - 1)) --export=ALL,JOB_FILE=${JOB_FILE},STATUS_DIR=${STATUS_DIR} scripts/slurm_validation_array.sh"
}

run_jobs() {
  if [[ ! -f "${JOB_FILE}" ]]; then
    echo "Missing ${JOB_FILE}; run $0 prepare first." >&2
    exit 1
  fi
  scripts/validation_workflow.sh run-local "${JOB_FILE}" 0 "$(($(wc -l < "${JOB_FILE}") - 1))" "${STATUS_DIR}"
  echo "CAA held-out test complete: ${STATUS_DIR}"
}

case "${ACTION}" in
  prepare) prepare_jobs ;;
  run) run_jobs ;;
  all)
    prepare_jobs
    run_jobs
    ;;
esac

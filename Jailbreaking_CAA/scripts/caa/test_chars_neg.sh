#!/usr/bin/env bash
set -euo pipefail

# Run manifest-locked held-out CAA tests for the negative-prompt CHaRS
# selections, reusing the fingerprint-verified states fitted during validation.
# Usage: scripts/caa/test_chars_neg.sh [prepare|run|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${1:-all}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-2-9b-it}"
VALIDATION_CAMPAIGN="${CHARS_VALIDATION_CAMPAIGN:-${REPO_ROOT}/validation_runs/chars_caa_neg}"
MANIFEST="${CHARS_MANIFEST:-${VALIDATION_CAMPAIGN}/manifests/chars_caa_neg.json}"
STATE_ROOT="${CHARS_STATE_ROOT:-${VALIDATION_CAMPAIGN}/chars_states}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/chars_caa_locked_test_neg}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
JOBS="${CAMPAIGN}/jobs/test_caa_chars_neg.jsonl"
STATUS_DIR="${JOBS}.status"
ARTIFACT_ROOT="${CAMPAIGN}/artifacts/tests/caa_chars_neg"
EXPECTED_JOBS=12
export CUDA_VISIBLE_DEVICES PYTHON_BIN

case "${ACTION}" in
  prepare|run|all) ;;
  *)
    echo "Usage: $0 [prepare|run|all]" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"

prepare_jobs() {
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing CHaRS manifest: ${MANIFEST}" >&2
    echo "Run scripts/caa/validate_chars_neg.sh all first." >&2
    exit 1
  fi
  if [[ ! -d "${STATE_ROOT}" ]]; then
    echo "Missing CHaRS state root: ${STATE_ROOT}" >&2
    exit 1
  fi

  "${PYTHON_BIN}" -c \
    'import sys; from pipeline.validation_selection import load_manifest; load_manifest(sys.argv[1]); print("Manifest verified:", sys.argv[1])' \
    "${MANIFEST}"

  mkdir -p "$(dirname "${JOBS}")" "${ARTIFACT_ROOT}"
  "${PYTHON_BIN}" scripts/build_validation_jobs.py \
    --format jsonl \
    --output "${JOBS}" \
    locked-test \
    --model-path "${GEMMA_MODEL}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --manifest "${MANIFEST}" \
    --chars-state-root "${STATE_ROOT}"

  local count
  count="$(wc -l < "${JOBS}")"
  if [[ "${count}" -ne "${EXPECTED_JOBS}" ]]; then
    echo "Expected ${EXPECTED_JOBS} locked-test jobs, found ${count}." >&2
    exit 1
  fi
  echo "Prepared ${count} locked CHaRS CAA negative test jobs: ${JOBS}"
}

run_jobs() {
  if [[ ! -f "${JOBS}" ]]; then
    echo "Missing ${JOBS}; run $0 prepare first." >&2
    exit 1
  fi
  local count
  count="$(wc -l < "${JOBS}")"
  scripts/validation_workflow.sh run-local \
    "${JOBS}" 0 "$((count - 1))" "${STATUS_DIR}"
  echo "Held-out CHaRS CAA negative tests complete: ${STATUS_DIR}"
}

case "${ACTION}" in
  prepare) prepare_jobs ;;
  run) run_jobs ;;
  all)
    prepare_jobs
    run_jobs
    ;;
esac

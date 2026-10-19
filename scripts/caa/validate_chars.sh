#!/usr/bin/env bash
set -euo pipefail

# CHaRS CAA workflow: fit K=10 states, validate layer/alpha, lock manifest.
# Usage: scripts/caa/validate_chars.sh [prepare|run|select|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ACTION="${1:-all}"
MODEL_PATH="${GEMMA_MODEL:-google/gemma-2-9b-it}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/chars_caa}"
PROMPT_CONDITION="${CAA_PROMPT_CONDITION:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STATE_ROOT="${CAMPAIGN}/chars_states"
FIT_JOBS="${CAMPAIGN}/jobs/chars_fit_caa.jsonl"
GRID_JOBS="${CAMPAIGN}/jobs/chars_caa_${PROMPT_CONDITION}.jsonl"
FIT_STATUS="${FIT_JOBS}.status"
GRID_STATUS="${GRID_JOBS}.status"
ARTIFACT_ROOT="${CAMPAIGN}/artifacts/chars_caa_${PROMPT_CONDITION}"
MANIFEST="${CAMPAIGN}/manifests/chars_caa_${PROMPT_CONDITION}.json"

case "${ACTION}" in prepare|run|select|all) ;; *) echo "Usage: $0 [prepare|run|select|all]" >&2; exit 2 ;; esac
case "${PROMPT_CONDITION}" in all|none|pos|neg) ;; *) echo "Invalid CAA_PROMPT_CONDITION." >&2; exit 2 ;; esac
prompt_count=1
[[ "${PROMPT_CONDITION}" == "all" ]] && prompt_count=3
expected_grid=$((6 * prompt_count * 2 * 20))
cd "${REPO_ROOT}"

prepare() {
  mkdir -p "${CAMPAIGN}/jobs" "${ARTIFACT_ROOT}"
  "${PYTHON_BIN}" scripts/build_validation_jobs.py --format jsonl --output "${FIT_JOBS}" \
    chars-fit-caa --model-path "${MODEL_PATH}" --chars-state-root "${STATE_ROOT}"
  "${PYTHON_BIN}" scripts/build_validation_jobs.py --format jsonl --output "${GRID_JOBS}" \
    caa-select --model-path "${MODEL_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
    --method chars --chars-state-root "${STATE_ROOT}" \
    --prompt-condition "${PROMPT_CONDITION}"
  [[ "$(wc -l < "${FIT_JOBS}")" -eq 6 ]]
  [[ "$(wc -l < "${GRID_JOBS}")" -eq "${expected_grid}" ]]
  echo "Prepared 6 fit jobs and ${expected_grid} CHaRS validation jobs."
}

run() {
  scripts/validation_workflow.sh run-local "${FIT_JOBS}" 0 5 "${FIT_STATUS}"
  scripts/validation_workflow.sh run-local "${GRID_JOBS}" 0 "$((expected_grid - 1))" "${GRID_STATUS}"
}

select_manifest() {
  [[ "$(find "${FIT_STATUS}" -maxdepth 1 -name "*.done.json" | wc -l)" -eq 6 ]]
  [[ "$(find "${GRID_STATUS}" -maxdepth 1 -name "*.done.json" | wc -l)" -eq "${expected_grid}" ]]
  mkdir -p "$(dirname "${MANIFEST}")"
  "${PYTHON_BIN}" -m pipeline.validation_selection select-caa \
    --candidate-root "${ARTIFACT_ROOT}" --output "${MANIFEST}"
  echo "Locked CHaRS CAA manifest: ${MANIFEST}"
}

case "${ACTION}" in
  prepare) prepare ;;
  run) run ;;
  select) select_manifest ;;
  all) prepare; run; select_manifest ;;
esac

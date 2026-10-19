#!/usr/bin/env bash
set -euo pipefail

# CHaRS jailbreak workflow: fit fixed-K states, validate layer/alpha, lock manifest.
# Usage: scripts/jailbreak/validate_chars.sh [prepare|run|select|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ACTION="${1:-all}"
MODEL_PATH="${MODEL_PATH:-google/gemma-2-9b-it}"
MODEL_ALIAS="$(basename "${MODEL_PATH%/}")"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/chars_jailbreak}"
APPLICATION_MODE="${JAILBREAK_APPLICATION_MODE:-all}"
UTILITY_CAP_RATIO="${UTILITY_CAP_RATIO:-1.05}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STATE_ROOT="${CAMPAIGN}/chars_states"
FIT_JOBS="${CAMPAIGN}/jobs/chars_fit_${MODEL_ALIAS}.jsonl"
GRID_JOBS="${CAMPAIGN}/jobs/chars_jail_${MODEL_ALIAS}_${APPLICATION_MODE}.jsonl"
FIT_STATUS="${FIT_JOBS}.status"
GRID_STATUS="${GRID_JOBS}.status"
ARTIFACT_ROOT="${CAMPAIGN}/artifacts/chars_jail_${MODEL_ALIAS}_${APPLICATION_MODE}"
MANIFEST="${CAMPAIGN}/manifests/chars_jail_${MODEL_ALIAS}_${APPLICATION_MODE}.json"

case "${ACTION}" in prepare|run|select|all) ;; *) echo "Usage: $0 [prepare|run|select|all]" >&2; exit 2 ;; esac
case "${APPLICATION_MODE}" in all|prompt_only|all_tokens) ;; *) echo "Invalid JAILBREAK_APPLICATION_MODE." >&2; exit 2 ;; esac
mode_count=1
[[ "${APPLICATION_MODE}" == "all" ]] && mode_count=2
expected_grid=$((1 + mode_count * 2 * 10))
cd "${REPO_ROOT}"

prepare() {
  mkdir -p "${CAMPAIGN}/jobs" "${ARTIFACT_ROOT}"
  "${PYTHON_BIN}" scripts/build_validation_jobs.py --format jsonl --output "${FIT_JOBS}" \
    chars-fit-jail --model-path "${MODEL_PATH}" --chars-state-root "${STATE_ROOT}"
  "${PYTHON_BIN}" scripts/build_validation_jobs.py --format jsonl --output "${GRID_JOBS}" \
    jail-select --model-path "${MODEL_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
    --method chars --chars-state-root "${STATE_ROOT}" \
    --application-mode "${APPLICATION_MODE}"
  [[ "$(wc -l < "${FIT_JOBS}")" -eq 1 ]]
  [[ "$(wc -l < "${GRID_JOBS}")" -eq "${expected_grid}" ]]
  echo "Prepared 1 fit job and ${expected_grid} baseline/CHaRS validation jobs."
}

run() {
  scripts/validation_workflow.sh run-local "${FIT_JOBS}" 0 0 "${FIT_STATUS}"
  scripts/validation_workflow.sh run-local "${GRID_JOBS}" 0 "$((expected_grid - 1))" "${GRID_STATUS}"
}

select_manifest() {
  [[ "$(find "${FIT_STATUS}" -maxdepth 1 -name "*.done.json" | wc -l)" -eq 1 ]]
  [[ "$(find "${GRID_STATUS}" -maxdepth 1 -name "*.done.json" | wc -l)" -eq "${expected_grid}" ]]
  mkdir -p "$(dirname "${MANIFEST}")"
  "${PYTHON_BIN}" -m pipeline.validation_selection select-jailbreak \
    --candidate-root "${ARTIFACT_ROOT}" --utility-cap-ratio "${UTILITY_CAP_RATIO}" \
    --output "${MANIFEST}"
  echo "Locked CHaRS jailbreak manifest: ${MANIFEST}"
}

case "${ACTION}" in
  prepare) prepare ;;
  run) run ;;
  select) select_manifest ;;
  all) prepare; run; select_manifest ;;
esac

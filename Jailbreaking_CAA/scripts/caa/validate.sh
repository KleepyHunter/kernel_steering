#!/usr/bin/env bash
set -euo pipefail

# CAA validation for ActAdd, KAS, and Curveball on Gemma (6 behaviors): prepare and
# run the validation grid, then lock the selections in
# ${CAMPAIGN}/manifests/caa_validation[_<condition>].json (used by scripts/caa/test.sh).
# Runs end to end; restarts skip completed jobs.
#
# Run from the existing PyTorch/Transformers environment.
# Optional overrides:
#   GEMMA_MODEL=google/gemma-2-9b-it
#   CAMPAIGN=/path/to/output
#   CUDA_VISIBLE_DEVICES=<GPU_ID>
#   CAA_PROMPT_CONDITION=all  # all, none, pos, or neg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-2-9b-it}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/rebuttal}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
CAA_PROMPT_CONDITION="${CAA_PROMPT_CONDITION:-all}"
export CUDA_VISIBLE_DEVICES

cd "${REPO_ROOT}"

echo "Preparing CAA validation jobs..."
scripts/validation_workflow.sh prepare-caa \
  "${GEMMA_MODEL}" "${CAMPAIGN}" "${CAA_PROMPT_CONDITION}"

if [[ "${CAA_PROMPT_CONDITION}" == "all" ]]; then
  JOB_FILE="${CAMPAIGN}/jobs/caa_select.jsonl"
else
  JOB_FILE="${CAMPAIGN}/jobs/caa_select_${CAA_PROMPT_CONDITION}.jsonl"
fi
STATUS_DIR="${JOB_FILE}.status"

echo "Running the CAA validation grid on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Progress markers: ${STATUS_DIR}"
scripts/validation_workflow.sh run-local "${JOB_FILE}" 0 "$(($(wc -l < "${JOB_FILE}") - 1))" "${STATUS_DIR}"

if [[ "${CAA_PROMPT_CONDITION}" == "all" ]]; then
  MANIFEST="${CAMPAIGN}/manifests/caa_validation.json"
else
  MANIFEST="${CAMPAIGN}/manifests/caa_validation_${CAA_PROMPT_CONDITION}.json"
fi
if [[ -f "${MANIFEST}" ]]; then
  echo "Manifest already exists; verifying it instead of overwriting it."
  python3 -c 'import sys; from pipeline.validation_selection import load_manifest; load_manifest(sys.argv[1]); print("Manifest verified:", sys.argv[1])' "${MANIFEST}"
else
  echo "Selecting CAA configurations from validation metrics..."
  scripts/validation_workflow.sh select-caa "${CAMPAIGN}" "${CAA_PROMPT_CONDITION}"
fi

echo "CAA validation complete for prompt condition: ${CAA_PROMPT_CONDITION}"
echo "Manifest: ${MANIFEST}"

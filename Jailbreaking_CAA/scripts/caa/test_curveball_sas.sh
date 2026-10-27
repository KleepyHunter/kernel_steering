#!/usr/bin/env bash
set -euo pipefail

# Run manifest-locked held-out CAA tests for the selected Curveball and/or SAS
# configurations from the negative-prompt validation campaigns. Each method runs
# on its own by default; "both" merges them to share the baseline jobs.
#
# Usage:
#   scripts/caa/test_curveball_sas.sh [prepare|run|all] [curveball|sas|both]
#
# Environment overrides:
#   GEMMA_MODEL=google/gemma-2-9b-it
#   CAMPAIGN=/path/to/test/output
#   CUDA_VISIBLE_DEVICES=<GPU_ID>
#   SAS_MANIFEST=/path/to/caa_sas_manifest.json
#   CURVEBALL_MANIFEST=/path/to/caa_curveball_manifest.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${1:-all}"
METHOD="${2:-curveball}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-2-9b-it}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/caa_locked_test_neg}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
SAS_MANIFEST="${SAS_MANIFEST:-${REPO_ROOT}/validation_runs/sas_caa_neg/manifests/caa_sas_strength_grid_validation_neg.json}"
CURVEBALL_MANIFEST="${CURVEBALL_MANIFEST:-${REPO_ROOT}/validation_runs/curveball_caa_neg/manifests/caa_curveball_strength_grid_validation_neg.json}"
export CUDA_VISIBLE_DEVICES

case "${ACTION}" in
  prepare|run|all) ;;
  *)
    echo "Usage: $0 [prepare|run|all] [curveball|sas|both]" >&2
    exit 2
    ;;
esac

case "${METHOD}" in
  sas|curveball|both) ;;
  *)
    echo "Invalid method: ${METHOD}. Expected sas, curveball, or both." >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"

manifest_for() {
  case "$1" in
    sas) echo "${SAS_MANIFEST}" ;;
    curveball) echo "${CURVEBALL_MANIFEST}" ;;
  esac
}

tag_for() {
  echo "caa_$1_neg"
}

job_file_for() {
  echo "${CAMPAIGN}/jobs/test_$(tag_for "$1").jsonl"
}

combined_job_file() {
  echo "${CAMPAIGN}/jobs/test_caa_sas_curveball_neg.jsonl"
}

prepare_method() {
  local method="$1"
  local manifest tag job_file count
  manifest="$(manifest_for "${method}")"
  tag="$(tag_for "${method}")"
  job_file="$(job_file_for "${method}")"

  if [[ ! -f "${manifest}" ]]; then
    echo "Missing ${method} manifest: ${manifest}" >&2
    exit 1
  fi

  scripts/validation_workflow.sh prepare-tests \
    "${GEMMA_MODEL}" "${manifest}" "${CAMPAIGN}" "${tag}"

  count="$(wc -l < "${job_file}")"
  if [[ "${count}" -ne 12 ]]; then
    echo "Expected 12 ${method} locked-test jobs, found ${count}." >&2
    exit 1
  fi

  echo "${method} held-out jobs: ${job_file} (${count})"
  echo "Slurm array: sbatch --array=0-$((count - 1)) --export=ALL,JOB_FILE=${job_file},STATUS_DIR=${job_file}.status scripts/slurm_validation_array.sh"
}

merge_combined_jobs() {
  local sas_jobs curveball_jobs combined count
  sas_jobs="$(job_file_for sas)"
  curveball_jobs="$(job_file_for curveball)"
  combined="$(combined_job_file)"
  mkdir -p "$(dirname "${combined}")"

  cp "${sas_jobs}" "${combined}"
  grep -v -- '-baseline"' "${curveball_jobs}" >> "${combined}"

  count="$(wc -l < "${combined}")"
  if [[ "${count}" -ne 18 ]]; then
    echo "Expected 18 combined CAA test jobs, found ${count}." >&2
    exit 1
  fi
  echo "Combined held-out jobs: ${combined} (${count}; 6 shared baseline + 6 SAS + 6 Curveball)"
  echo "Slurm array: sbatch --array=0-$((count - 1)) --export=ALL,JOB_FILE=${combined},STATUS_DIR=${combined}.status scripts/slurm_validation_array.sh"
}

run_job_file() {
  local job_file="$1"
  local label="$2"
  local status_dir count
  status_dir="${job_file}.status"

  if [[ ! -f "${job_file}" ]]; then
    echo "Missing ${job_file}; run $0 prepare ${METHOD} first." >&2
    exit 1
  fi

  count="$(wc -l < "${job_file}")"
  scripts/validation_workflow.sh run-local \
    "${job_file}" 0 "$((count - 1))" "${status_dir}"
  echo "${label} held-out test complete: ${status_dir}"
}

if [[ "${METHOD}" == "both" ]]; then
  case "${ACTION}" in
    prepare)
      prepare_method sas
      prepare_method curveball
      merge_combined_jobs
      ;;
    run)
      run_job_file "$(combined_job_file)" "combined SAS/Curveball"
      ;;
    all)
      prepare_method sas
      prepare_method curveball
      merge_combined_jobs
      run_job_file "$(combined_job_file)" "combined SAS/Curveball"
      ;;
  esac
else
  case "${ACTION}" in
    prepare)
      prepare_method "${METHOD}"
      ;;
    run)
      run_job_file "$(job_file_for "${METHOD}")" "${METHOD}"
      ;;
    all)
      prepare_method "${METHOD}"
      run_job_file "$(job_file_for "${METHOD}")" "${METHOD}"
      ;;
  esac
fi

echo "CAA locked-test campaign: ${CAMPAIGN}"

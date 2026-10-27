#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  scripts/validation_workflow.sh prepare-caa MODEL_PATH CAMPAIGN_DIR [all|none|pos|neg]
  scripts/validation_workflow.sh prepare-jail MODEL_PATH CAMPAIGN_DIR [all|prompt_only|all_tokens]
  scripts/validation_workflow.sh run-index JOB_FILE ZERO_BASED_INDEX [STATUS_DIR]
  scripts/validation_workflow.sh run-local JOB_FILE [START_INDEX] [END_INDEX] [STATUS_DIR]
  scripts/validation_workflow.sh select-caa CAMPAIGN_DIR [all|none|pos|neg]
  scripts/validation_workflow.sh select-jail CAMPAIGN_DIR [all|prompt_only|all_tokens]
  scripts/validation_workflow.sh prepare-tests MODEL_PATH MANIFEST CAMPAIGN_DIR TAG

For Slurm, first prepare a job file, then use:
  N=$(wc -l < JOB_FILE)
  sbatch --array=0-$((N-1)) --export=ALL,JOB_FILE=JOB_FILE,STATUS_DIR=STATUS_DIR YOUR_SBATCH_SCRIPT

The sbatch script should execute:
  python3 scripts/run_validation_job.py --job-file "$JOB_FILE" --status-dir "$STATUS_DIR"
EOF
}

# The experiment scripts default CUDA_VISIBLE_DEVICES to the placeholder <GPU_ID>;
# running jobs needs a real GPU id. (Unset keeps all GPUs visible, as before.)
require_gpu() {
  if [[ "${CUDA_VISIBLE_DEVICES:-}" == *"<"* ]]; then
    echo "CUDA_VISIBLE_DEVICES is the placeholder ${CUDA_VISIBLE_DEVICES}; set it to the GPU id(s) to use." >&2
    exit 2
  fi
}

command="${1:-}"
if [[ -z "${command}" ]]; then
  usage
  exit 2
fi
shift
cd "${REPO_ROOT}"

case "${command}" in
  prepare-caa)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    model_path="$1"
    campaign_dir="$2"
    prompt_condition="${3:-all}"
    case "${prompt_condition}" in
      all|none|pos|neg) ;;
      *)
        echo "Invalid CAA prompt condition: ${prompt_condition}. Expected all, none, pos, or neg." >&2
        exit 2
        ;;
    esac
    if [[ "${prompt_condition}" == "all" ]]; then
      job_file="${campaign_dir}/jobs/caa_select.jsonl"
      artifact_root="${campaign_dir}/artifacts/caa_select"
    else
      job_file="${campaign_dir}/jobs/caa_select_${prompt_condition}.jsonl"
      artifact_root="${campaign_dir}/artifacts/caa_select_${prompt_condition}"
    fi
    mkdir -p "$(dirname "${job_file}")" "${artifact_root}"
    "${PYTHON_BIN}" scripts/build_validation_jobs.py       --format jsonl       --output "${job_file}"       caa-select       --model-path "${model_path}"       --artifact-root "${artifact_root}"       --prompt-condition "${prompt_condition}"
    echo "CAA jobs: ${job_file} ($(wc -l < "${job_file}"))"
    ;;

  prepare-jail)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    model_path="$1"
    campaign_dir="$2"
    application_mode="${3:-all}"
    case "${application_mode}" in
      all|prompt_only|all_tokens) ;;
      *)
        echo "Invalid jailbreak application mode: ${application_mode}. Expected all, prompt_only, or all_tokens." >&2
        exit 2
        ;;
    esac
    model_alias="$(basename "${model_path%/}")"
    if [[ "${application_mode}" == "all" ]]; then
      job_file="${campaign_dir}/jobs/jail_${model_alias}.jsonl"
      artifact_root="${campaign_dir}/artifacts/jail_select"
    else
      job_file="${campaign_dir}/jobs/jail_${model_alias}_${application_mode}.jsonl"
      artifact_root="${campaign_dir}/artifacts/jail_select_${application_mode}"
    fi
    mkdir -p "$(dirname "${job_file}")" "${artifact_root}"
    "${PYTHON_BIN}" scripts/build_validation_jobs.py       --format jsonl       --output "${job_file}"       jail-select       --model-path "${model_path}"       --artifact-root "${artifact_root}"       --application-mode "${application_mode}"
    echo "Jailbreaking jobs: ${job_file} ($(wc -l < "${job_file}"))"
    ;;

  run-index)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    require_gpu
    job_file="$1"
    index="$2"
    status_dir="${3:-${job_file}.status}"
    "${PYTHON_BIN}" scripts/run_validation_job.py       --job-file "${job_file}"       --index "${index}"       --status-dir "${status_dir}"
    ;;

  run-local)
    [[ $# -ge 1 && $# -le 4 ]] || { usage; exit 2; }
    require_gpu
    job_file="$1"
    count="$(wc -l < "${job_file}")"
    start="${2:-0}"
    end="${3:-$((count - 1))}"
    status_dir="${4:-${job_file}.status}"
    if (( start < 0 || end < start || end >= count )); then
      echo "Invalid range ${start}-${end}; job indices are 0-$((count - 1))." >&2
      exit 2
    fi
    for ((index=start; index<=end; index++)); do
      "${PYTHON_BIN}" scripts/run_validation_job.py         --job-file "${job_file}"         --index "${index}"         --status-dir "${status_dir}"
    done
    ;;

  select-caa)
    [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
    campaign_dir="$1"
    prompt_condition="${2:-all}"
    case "${prompt_condition}" in
      all|none|pos|neg) ;;
      *)
        echo "Invalid CAA prompt condition: ${prompt_condition}. Expected all, none, pos, or neg." >&2
        exit 2
        ;;
    esac
    if [[ "${prompt_condition}" == "all" ]]; then
      candidate_root="${campaign_dir}/artifacts/caa_select"
      manifest="${campaign_dir}/manifests/caa_validation.json"
    else
      candidate_root="${campaign_dir}/artifacts/caa_select_${prompt_condition}"
      manifest="${campaign_dir}/manifests/caa_validation_${prompt_condition}.json"
    fi
    mkdir -p "$(dirname "${manifest}")"
    "${PYTHON_BIN}" -m pipeline.validation_selection select-caa       --candidate-root "${candidate_root}"       --output "${manifest}"
    ;;

  select-jail)
    [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
    campaign_dir="$1"
    application_mode="${2:-all}"
    case "${application_mode}" in
      all|prompt_only|all_tokens) ;;
      *)
        echo "Invalid jailbreak application mode: ${application_mode}. Expected all, prompt_only, or all_tokens." >&2
        exit 2
        ;;
    esac
    if [[ "${application_mode}" == "all" ]]; then
      candidate_root="${campaign_dir}/artifacts/jail_select"
      manifest="${campaign_dir}/manifests/jailbreak_validation.json"
    else
      candidate_root="${campaign_dir}/artifacts/jail_select_${application_mode}"
      manifest="${campaign_dir}/manifests/jailbreak_validation_${application_mode}.json"
    fi
    mkdir -p "$(dirname "${manifest}")"
    "${PYTHON_BIN}" -m pipeline.validation_selection select-jailbreak       --candidate-root "${candidate_root}"       --utility-cap-ratio 1.05       --output "${manifest}"
    ;;

  prepare-tests)
    [[ $# -eq 4 ]] || { usage; exit 2; }
    model_path="$1"
    manifest="$2"
    campaign_dir="$3"
    tag="$4"
    job_file="${campaign_dir}/jobs/test_${tag}.jsonl"
    artifact_root="${campaign_dir}/artifacts/tests/${tag}"
    mkdir -p "$(dirname "${job_file}")" "${artifact_root}"
    "${PYTHON_BIN}" scripts/build_validation_jobs.py       --format jsonl       --output "${job_file}"       locked-test       --model-path "${model_path}"       --artifact-root "${artifact_root}"       --manifest "${manifest}"
    echo "Locked test jobs: ${job_file} ($(wc -l < "${job_file}"))"
    ;;

  *)
    usage
    exit 2
    ;;
esac

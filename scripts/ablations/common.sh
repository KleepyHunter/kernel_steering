# Shared by the ablation scripts in this directory; source it, do not run it.
#
# Expects: ACTION, CAMPAIGN, JOB_FILE, and a build_jobs function that writes
# ${JOB_FILE} via scripts/build_experiment_jobs.py.

ABLATIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ABLATIONS_DIR}/../.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-<GPU_ID>}"
LLAMAGUARD_VLLM_SERVER="${LLAMAGUARD_VLLM_SERVER:-http://0.0.0.0:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES LLAMAGUARD_VLLM_SERVER OPENAI_API_KEY PYTHON_BIN

check_action() {
  case "${ACTION}" in
    prepare|run|all) ;;
    *)
      echo "Usage: $0 [prepare|run|all]" >&2
      exit 2
      ;;
  esac
}

prepare_jobs() {
  mkdir -p "$(dirname "${JOB_FILE}")" "${CAMPAIGN}/artifacts"
  build_jobs
  local count
  count="$(wc -l < "${JOB_FILE}")"
  if [[ -n "${EXPECTED_JOBS:-}" && "${count}" -ne "${EXPECTED_JOBS}" ]]; then
    echo "Expected ${EXPECTED_JOBS} jobs, found ${count}." >&2
    exit 1
  fi
  echo "Jobs: ${JOB_FILE} (${count})"
  echo "Slurm array: sbatch --array=0-$((count - 1)) --export=ALL,JOB_FILE=${JOB_FILE},STATUS_DIR=${JOB_FILE}.status,OPENAI_API_KEY=${OPENAI_API_KEY},LLAMAGUARD_VLLM_SERVER=${LLAMAGUARD_VLLM_SERVER} scripts/slurm_validation_array.sh"
}

run_jobs() {
  if [[ ! -f "${JOB_FILE}" ]]; then
    echo "Missing ${JOB_FILE}; run $0 prepare first." >&2
    exit 1
  fi
  scripts/validation_workflow.sh run-local "${JOB_FILE}" 0 "$(($(wc -l < "${JOB_FILE}") - 1))" "${JOB_FILE}.status"
  echo "Runs complete: ${CAMPAIGN}/artifacts"
}

run_action() {
  cd "${REPO_ROOT}"
  case "${ACTION}" in
    prepare) prepare_jobs ;;
    run) run_jobs ;;
    all)
      prepare_jobs
      run_jobs
      ;;
  esac
}

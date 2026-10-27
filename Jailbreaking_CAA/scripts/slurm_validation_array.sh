#!/usr/bin/env bash
#SBATCH --job-name=kas-validation
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail

: "${JOB_FILE:?Submit with --export=ALL,JOB_FILE=/absolute/path/to/jobs.jsonl}"
STATUS_DIR="${STATUS_DIR:-${JOB_FILE}.status}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/run_validation_job.py   --job-file "${JOB_FILE}"   --status-dir "${STATUS_DIR}"

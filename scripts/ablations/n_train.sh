#!/usr/bin/env bash
set -euo pipefail

# Ablation: number of reference pairs (appendix table "Num. of references").
#
# Fits KAS (and ActAdd, evaluated in the same run) on n_train harmful/harmless
# reference pairs, for prompt-only and all-token steering. Reports JAILBREAKBENCH
# ASR and perplexity (standard phase). Runs with n_train != 128 are stored under
# an extra "n_train-<N>" directory level. For n_train other than 1 and 128,
# prompts dropped by refusal filtering are replaced from the rest of the train
# split, so exactly N pairs are used. n_train=1 is never filtered, and 128 (the
# default pipeline) keeps whatever passes the filter.
#
# Usage:
#   scripts/ablations/n_train.sh [prepare|run|all]
#
# Environment overrides:
#   MODEL_PATH=google/gemma-2-9b-it
#   N_TRAIN="1 2 4 8 16 32 64 128"
#   LAYER=22  ETA=150  SEED=1  LAMBDA_REG=0.1
#   CAMPAIGN=/path/to/output (default: validation_runs/ablation_n_train)
#   CUDA_VISIBLE_DEVICES=<GPU_ID>  LLAMAGUARD_VLLM_SERVER=http://0.0.0.0:8000/v1  PYTHON_BIN=python3

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ACTION="${1:-all}"
MODEL_PATH="${MODEL_PATH:-google/gemma-2-9b-it}"
N_TRAIN="${N_TRAIN:-1 2 4 8 16 32 64 128}"
LAYER="${LAYER:-22}"
ETA="${ETA:-150}"
SEED="${SEED:-1}"
LAMBDA_REG="${LAMBDA_REG:-0.1}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/ablation_n_train}"
JOB_FILE="${CAMPAIGN}/jobs/n_train.jsonl"

build_jobs() {
  read -r -a n_train <<< "${N_TRAIN}"
  EXPECTED_JOBS=$(( ${#n_train[@]} * 2 ))
  "${PYTHON_BIN}" scripts/build_experiment_jobs.py --format jsonl --output "${JOB_FILE}" \
    n-train \
    --model-path "${MODEL_PATH}" \
    --artifact-root "${CAMPAIGN}/artifacts" \
    --n-train "${n_train[@]}" \
    --layer "${LAYER}" --eta "${ETA}" --seed "${SEED}" --lambda-reg "${LAMBDA_REG}"
}

check_action
run_action

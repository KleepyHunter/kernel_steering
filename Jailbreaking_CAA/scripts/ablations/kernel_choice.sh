#!/usr/bin/env bash
set -euo pipefail

# Ablation: kernel choice (appendix table "KAS ... across kernels").
#
# Fits KAS with the Laplacian, polynomial, and RBF kernels (bandwidth / kernel
# parameters left to their defaults), for prompt-only and all-token steering.
# Reports JAILBREAKBENCH ASR and perplexity (standard phase).
#
# Usage:
#   scripts/ablations/kernel_choice.sh [prepare|run|all]
#
# Environment overrides:
#   MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
#   KERNELS="laplacian polynomial rbf"
#   LAYER=13  ETA=5  SEED=1  LAMBDA_REG=0.1
#   CAMPAIGN=/path/to/output (default: validation_runs/ablation_kernel_choice)
#   CUDA_VISIBLE_DEVICES=<GPU_ID>  LLAMAGUARD_VLLM_SERVER=http://0.0.0.0:8000/v1  PYTHON_BIN=python3

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ACTION="${1:-all}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
KERNELS="${KERNELS:-laplacian polynomial rbf}"
LAYER="${LAYER:-13}"
ETA="${ETA:-5}"
SEED="${SEED:-1}"
LAMBDA_REG="${LAMBDA_REG:-0.1}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/ablation_kernel_choice}"
JOB_FILE="${CAMPAIGN}/jobs/kernel_choice.jsonl"

build_jobs() {
  read -r -a kernels <<< "${KERNELS}"
  EXPECTED_JOBS=$(( ${#kernels[@]} * 2 ))
  "${PYTHON_BIN}" scripts/build_experiment_jobs.py --format jsonl --output "${JOB_FILE}" \
    kernel-choice \
    --model-path "${MODEL_PATH}" \
    --artifact-root "${CAMPAIGN}/artifacts" \
    --kernels "${kernels[@]}" \
    --layer "${LAYER}" --eta "${ETA}" --seed "${SEED}" --lambda-reg "${LAMBDA_REG}"
}

check_action
run_action

#!/usr/bin/env bash
set -euo pipefail

# Ablation: jailbreaking from a single contrastive pair.
#
# For each model and seed, one harmful/harmless reference pair is sampled
# (--n_train 1, the pair is chosen by --seed) and KAS is fitted with one
# gradient step and with ten. Every run also evaluates ActAdd at the same eta.
# Reports JAILBREAKBENCH ASR and Pile / Alpaca perplexity (standard phase).
# Layers and step sizes per model are in scripts/build_experiment_jobs.py
# (SINGLE_PAIR_SETTINGS): Gemma layer 22 (eta 200 / 40), Llama layer 12
# (eta 5 / 1), Qwen layer 18 (eta 40 / 8).
#
# Usage:
#   scripts/ablations/single_pair.sh [prepare|run|all]
#
# Environment overrides:
#   MODELS="google/gemma-2-9b-it meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
#   SEEDS="0 1 2 3 4 5 6 7 8 9"   one sampled pair per seed
#   LAMBDA_REG=0.1
#   CAMPAIGN=/path/to/output (default: validation_runs/ablation_single_pair)
#   CUDA_VISIBLE_DEVICES=<GPU_ID>  LLAMAGUARD_VLLM_SERVER=http://0.0.0.0:8000/v1  PYTHON_BIN=python3

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ACTION="${1:-all}"
MODELS="${MODELS:-google/gemma-2-9b-it meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
LAMBDA_REG="${LAMBDA_REG:-0.1}"
CAMPAIGN="${CAMPAIGN:-${REPO_ROOT}/validation_runs/ablation_single_pair}"
JOB_FILE="${CAMPAIGN}/jobs/single_pair.jsonl"

build_jobs() {
  read -r -a models <<< "${MODELS}"
  read -r -a seeds <<< "${SEEDS}"
  EXPECTED_JOBS=$(( ${#models[@]} * ${#seeds[@]} * 2 ))
  "${PYTHON_BIN}" scripts/build_experiment_jobs.py --format jsonl --output "${JOB_FILE}" \
    single-pair \
    --model-path "${models[@]}" \
    --artifact-root "${CAMPAIGN}/artifacts" \
    --seeds "${seeds[@]}" \
    --lambda-reg "${LAMBDA_REG}"
}

check_action
run_action

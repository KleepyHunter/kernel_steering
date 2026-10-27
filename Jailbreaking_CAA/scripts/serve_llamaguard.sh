#!/usr/bin/env bash
set -euo pipefail

# Serve the Llama-Guard-3-8B judge that scores jailbreak ASR, through vLLM's
# OpenAI-compatible API. Run it in the pipeline environment and leave it running;
# point the pipeline at it with LLAMAGUARD_VLLM_SERVER=http://<host>:<port>/v1.
#
# Usage:
#   scripts/serve_llamaguard.sh
#
# Environment overrides:
#   LLAMAGUARD_GPU=<GPU_ID>     GPU to serve on (required)
#   LLAMAGUARD_PORT=8000
#   LLAMAGUARD_GPU_MEMORY=0.4   fraction of that GPU's memory vLLM may use
#   LLAMAGUARD_MAX_MODEL_LEN    optional context limit (e.g. 8192). The default 131072 needs
#                               about 16 GB of KV cache, so set it when lowering the memory share.

LLAMAGUARD_GPU="${LLAMAGUARD_GPU:-<GPU_ID>}"
if [[ "${LLAMAGUARD_GPU}" == *"<"* ]]; then
  echo "Set LLAMAGUARD_GPU to the GPU id to serve on (it is the placeholder ${LLAMAGUARD_GPU})." >&2
  exit 2
fi

extra_args=()
if [[ -n "${LLAMAGUARD_MAX_MODEL_LEN:-}" ]]; then
  extra_args+=(--max-model-len "${LLAMAGUARD_MAX_MODEL_LEN}")
fi

CUDA_VISIBLE_DEVICES="${LLAMAGUARD_GPU}" exec vllm serve meta-llama/Llama-Guard-3-8B \
  --port "${LLAMAGUARD_PORT:-8000}" \
  --gpu-memory-utilization "${LLAMAGUARD_GPU_MEMORY:-0.4}" \
  --enforce_eager \
  "${extra_args[@]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"     # default: Image/data (already holds style_prompts.json)
export CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cache}"  # default: Image/cache
# export HF_HUB_CACHE=""  # optionally, models will be saved here
# export HF_TOKEN=""  # required for some specific models like Gemma-2

if [ -z "$1" ]; then
  echo "Usage: $0 <seed> [gpu_id]"
  exit 1
fi

seed=$1
GPU_ID=${2:-0}

python -m act.scripts.pipeline \
"task_params=toxicity" \
"seed=$seed" \
"results_dir=results_$seed" \
"responses.batch_size=20" \
"model=gemma-2-2b.yaml" \
"wandb.mode=disabled" \
"intervention_params.incremental=incr" \
"intervention_params=kernel" \
"intervention_params.hook_params.kernel_type=polynomial" \
"intervention_params.hook_params.kernel_params={degree:2,offset:1.0}" \
"intervention_params.hook_params.eta=1.0" \
"intervention_params.hook_params.step_count=1" \
"wandb.mode=disabled" \
"device=cuda:$GPU_ID" \
"model.dtype=float32"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"     # default: Image/data (already holds style_prompts.json)
export CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cache}"  # default: Image/cache
# export HF_HUB_CACHE=""  # optionally, models will be saved here
# export HF_TOKEN=""  # required for some specific models like Gemma-2

# python -m act.scripts.download_external_data
# if [ -z "$1" ]; then
#   echo "Usage: $0 <seed> [gpu_id]"
#   exit 1
# fi

seed=$1
GPU_ID=${2:-0}


# # seed=38

python -m act.scripts.pipeline \
"task_params=toxicity" \
"seed=$seed" \
"results_dir=results_$seed" \
"responses.batch_size=20" \
"model=gemma-2-2b.yaml" \
"intervention_params=mean_ot" \
"intervention_params.incremental=incr" \
"wandb.mode=disabled" \
"device=cuda:$GPU_ID"
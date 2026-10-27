#!/usr/bin/env bash

# Paths: override by exporting DATA_DIR / CACHE_DIR before running
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"     # default: Image/data (already holds style_prompts.json)
export CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cache}"  # default: Image/cache
# export HF_HUB_CACHE=""   # optionally, models will be saved here
# export HF_TOKEN="hf_..."  # set this only if needed for gated models

# --- Argument parsing ---
if [ -z "$1" ]; then
  echo "Usage: $0 <seed> [gpu_id] [style]"
  exit 1
fi

seed=$1
GPU_ID=${2:-0}       # default: GPU 0
style=${3:-cyberpunk}   # any key of data/style_prompts.json

model=FLUX.1-schnell.yaml
# model=FLUX.1-dev.yaml
# model=SDXL-Lightning.yaml

model_name="${model%.yaml}"   # strip the ".yaml" suffix
intervention_params=mean_ot

# --- Run pipeline for multiple styles ---
for sty in $style; do
  results_dir="results_${model_name}_${intervention_params}_${sty}_${seed}"
  sweep_idx=0
  for sweep in "0.0 1.0 11" "2.0 10.0 9"; do
    sweep_idx=$((sweep_idx + 1))
    read min_strength max_strength strength_steps <<< "$sweep"
    evaluation="['text-to-image-generation']"
    if [ "$sweep_idx" -eq 2 ]; then
      evaluation="['text-to-image-generation','clip_score']"
    fi

    python -m act.scripts.pipeline \
        --config-name text_to_image_generation.yaml \
        "task_params=coco_styles" \
        "model=${model}" \
        "seed=$seed" \
        "intervention_params=${intervention_params}" \
        "task_params.src_subsets=['none']" \
        "task_params.dst_subsets=[$sty]" \
        "results_dir=${results_dir}" \
        "task_params.prompt_subset=['none']" \
        "responses.batch_size=8" \
        "responses.max_batches=64" \
        "interventions.max_batches=null" \
        "intervention_params.incremental=incr" \
        "wandb.mode=disabled" \
        "evaluation=${evaluation}" \
        "text_to_image_generation.batch_size=4" \
        "text_to_image_generation.max_batches=15" \
        "text_to_image_generation.create_gif=true" \
        "text_to_image_generation.min_strength=${min_strength}" \
        "text_to_image_generation.max_strength=${max_strength}" \
        "text_to_image_generation.strength_steps=${strength_steps}" \
        "device=cuda:$GPU_ID"
  done
done

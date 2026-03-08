#!/bin/bash
set -euo pipefail

# File: sample.sh

# Updated checkpoint path and label path
experiment_dir="results_a100/gp42-training/059-SiT-XL-2-Linear-velocity-None"
ckpt="${experiment_dir}/checkpoints/0005000/model.safetensors"
if [ ! -f "$ckpt" ]; then
    echo "Warning: $ckpt not found, checking for ema.pt"
    ckpt="${experiment_dir}/checkpoints/0005000/ema.pt"
fi

label_path="${experiment_dir}/used_labels.json"
steps=250
cfg_scale=4.0
base_dir="samples_trained_only"
stoch_dir="${base_dir}/SiT-XL-2-model-cfg-${cfg_scale}-8-JUMP_FLOW-${steps}-stoch"
det_dir="${base_dir}/SiT-XL-2-model-cfg-${cfg_scale}-8-JUMP_FLOW-${steps}-det"

mkdir -p "$stoch_dir" "$det_dir"

run_mode() {
  local jump_flag="$1"
  local out_dir="$2"
  local mode_name="$3"
  local -a pids=()

  echo "Running 8 independent JUMP_FLOW sampling workers on GPUs 0-7..."
  GPUS=(0 1 2 3 4 5 6 7)
  for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}
    rank=$i
    world_size=${#GPUS[@]}
    local seed=$gpu
    local out_file="${out_dir}/gpu${gpu}_seed${seed}.png"
    # Use specific GPU and set matching rank/world_size
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python sample.py JUMP_FLOW \
      --model "SiT-XL/2" \
      --ckpt "$ckpt" \
      --vae "ema" \
      --label-path "$label_path" \
      --rank "$rank" \
      --world-size "$world_size" \
      --num-sampling-steps "$steps" \
      --cfg-scale "$cfg_scale" \
      --seed "$seed" \
      $jump_flag \
      --bregman-type cosh \
      --out-file "$out_file" &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [ "$failed" -ne 0 ]; then
    echo "${mode_name} sampling failed." >&2
    exit 1
  fi
}

run_mode "--stochastic-jump" "$stoch_dir" "stochastic"
run_mode "--no-stochastic-jump" "$det_dir" "deterministic"

echo "All sampling jobs finished."

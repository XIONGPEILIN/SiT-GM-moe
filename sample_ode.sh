#!/bin/bash
set -euo pipefail

# File: sample_ode.sh

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
base_dir="samples_ode"
out_dir="${base_dir}/SiT-XL-2-model-cfg-${cfg_scale}-ODE-${steps}"

mkdir -p "$out_dir"

echo "Running 8 independent ODE sampling workers on GPUs 0-7..."
GPUS=(0 1 2 3 4 5 6 7)
pids=()

for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}
  rank=$i
  world_size=${#GPUS[@]}
  seed=$gpu
  out_file="${out_dir}/gpu${gpu}_seed${seed}.png"
  
  # Use specific GPU and set matching rank/world_size
  # Mode is set to ODE, and we use dopri5 as the default solver
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python sample.py ODE \
    --model "SiT-XL/2" \
    --ckpt "$ckpt" \
    --vae "ema" \
    --label-path "$label_path" \
    --rank "$rank" \
    --world-size "$world_size" \
    --num-sampling-steps "$steps" \
    --cfg-scale "$cfg_scale" \
    --seed "$seed" \
    --sampling-method "dopri5" \
    --bregman-type cosh \
    --out-file "$out_file" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "ODE sampling failed." >&2
  exit 1
fi

echo "All ODE sampling jobs finished. Results are in ${out_dir}"

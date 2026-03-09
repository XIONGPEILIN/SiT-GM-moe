#!/bin/bash
set -euo pipefail

# File: sample_test.sh (Simplified for single image check)

# Explicitly use the requested checkpoint
ckpt="results_a100/gp42-training/059-SiT-XL-2-Linear-velocity-None/checkpoints/0005000/model.safetensors"

if [ ! -f "$ckpt" ]; then
    echo "Error: $ckpt not found!"
    exit 1
fi

label_path="results_a100/gp42-training/059-SiT-XL-2-Linear-velocity-None/used_labels.json"
steps=250
cfg_scale=4.0
seed=42

echo "Running JUMP_FLOW sampling (stochastic) on CUDA:1..."
echo "Using checkpoint: $ckpt"

CUDA_VISIBLE_DEVICES=1 .venv/bin/python sample.py JUMP_FLOW \
  --model "SiT-XL/2" \
  --ckpt "$ckpt" \
  --vae "ema" \
  --label-path "$label_path" \
  --rank 0 \
  --world-size 1 \
  --limit 1 \
  --num-sampling-steps "$steps" \
  --cfg-scale "$cfg_scale" \
  --seed "$seed" \
  --stochastic-jump \
  --bregman-type cosh \
  --out-file "./sample_jump.png"

echo "Sampling finished."
echo "Image saved to ./sample_jump.png"
echo "Jump probabilities saved to ./sample_jump.probs.pt"

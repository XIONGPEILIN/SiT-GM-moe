#!/bin/bash
set -e

echo "=== Quick Sampling Test for SiT-GM-moe ==="

# Using CPU or single GPU for a quick generation check
CUDA_VISIBLE_DEVICES=0 python3 sample.py MIXED \
  --model SiT-XL/2 \
  --image-size 256 \
  --num-classes 1000 \
  --path-type Linear \
  --prediction velocity \
  --num-bins 128 \
  --jump-range 3.0 \
  --time-schedule cubic \
  --cfg-scale 4.0 \
  --num-sampling-steps 50 \
  --global-seed 42 \
  --ckpt "none" # Use 'none' to skip loading completely

echo "=== Test Completed Successfully! ==="

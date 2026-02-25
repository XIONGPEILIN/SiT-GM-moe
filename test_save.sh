#!/bin/bash
set -e

echo "=== Quick Save Weight Test for SiT-GM-moe ==="

# We will run the actual train.py but override args to save very quickly
# --ckpt-every 2 means it will save a checkpoint on step 2
# --sample-every 10 means it won't trigger the long image generation before saving
# --global-batch-size 32 to use minimal resources
# --epochs 1 just to run briefly

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    train.py \
    --model "SiT-XL/2" \
    --feature-path "/home/yanai-lab/xiong-p/SiT-GM-moe/imagenet_feature" \
    --results-dir "/home/yanai-lab/xiong-p/SiT-GM-moe/results_a100" \
    --global-batch-size 32 \
    --num-workers 2 \
    --num-bins 128 \
    --jump-range 3.0 \
    --sampler-type jump_flow \
    --time-schedule cubic \
    --epochs 1 \
    --log-every 1 \
    --ckpt-every 2 \
    --sample-every 100 \
    --cfg-scale 4.0 

echo "=== Save Weight Test Completed Successfully! ==="

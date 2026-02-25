#!/bin/bash
# Training script for SiT-GM-moe on A100 80GB x8
# Configured for maximum efficient throughput without gradient checkpointing.
#
# Usage:
#   bash train_a100_80gb.sh <FEATURE_PATH> [RESULTS_DIR] [CKPT_PATH]
#
# Args:
#   FEATURE_PATH : Path to pre-encoded VAE features (contains _features and _labels dirs)
#   RESULTS_DIR  : Output directory for checkpoints and logs (default: results_a100)
#   CKPT_PATH    : Optional path to a checkpoint to resume from

set -e

FEATURE_PATH="${1:-/home/yanai-lab/xiong-p/SiT-GM-moe/imagenet_feature}"
RESULTS_DIR="${2:-results_a100}"
CKPT_PATH="${3:-}"

# -------------------------------------------------------------------
# Hardware: 8x GPU (e.g. A100 80GB or RTX 6000 96GB)
# Config:   SiT-XL/2, num_bins=128, jump_range=3.0
# Batch:    128 per GPU (= 1024 global across 8 GPUs)
# Precision: FP32 (Full Precision)
# New Args:  Uses cubic time schedule and specific flow sampling args
# -------------------------------------------------------------------

NUM_GPUS=8
GLOBAL_BATCH=640  # 128 per GPU
MODEL="SiT-XL/2"
NUM_BINS=128
JUMP_RANGE=3.0
SAMPLER_TYPE="jump_flow"
TIME_SCHEDULE="cubic"
NUM_WORKERS=8

CKPT_ARG=""
if [ -n "$CKPT_PATH" ]; then
    CKPT_ARG="--ckpt $CKPT_PATH"
fi

echo "=========================================="
echo " SiT-GM-moe Training on 80GB/96GB x${NUM_GPUS} (FP32)"
echo " Feature Data: $FEATURE_PATH"
echo " Results: $RESULTS_DIR"
echo " Model:   $MODEL ($TIME_SCHEDULE schedule)"
echo " Global BS: $GLOBAL_BATCH  (${NUM_GPUS} GPUs × $((GLOBAL_BATCH / NUM_GPUS))/GPU)"
echo "=========================================="

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision no \
    train.py \
    --model "$MODEL" \
    --feature-path "$FEATURE_PATH" \
    --results-dir "$RESULTS_DIR" \
    --global-batch-size $GLOBAL_BATCH \
    --num-workers $NUM_WORKERS \
    --num-bins $NUM_BINS \
    --jump-range $JUMP_RANGE \
    --sampler-type $SAMPLER_TYPE \
    --time-schedule $TIME_SCHEDULE \
    --epochs 14000 \
    --log-every 10 \
    --ckpt-every 50000 \
    --sample-every 10000000000000000000000000000000000000000000000000000000000000000000000000000 \
    --cfg-scale 4.0 \
    --wandb \
    $CKPT_ARG

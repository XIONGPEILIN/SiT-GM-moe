#!/bin/bash
# Optimized training script for SiT-GM-moe on gp40 (A6000 48GB x8)
set -e
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
FEATURE_PATH="${1:-/home/yanai-lab/xiong-p/SiT-GM-moe/imagenet_feature}"
RESULTS_DIR="${2:-results_a100/gp40-training}"
CKPT_PATH="${3:-}"




# -------------------------------------------------------------------
# Batch Size Calculation:
# A6000 has 48GB. BATCH_PER_GPU=30.
# Global Batch is calculated without gradient accumulation steps.
# -------------------------------------------------------------------
NUM_GPUS=7
BATCH_PER_GPU=1
GLOBAL_BATCH=$((BATCH_PER_GPU * NUM_GPUS * 16))


MODEL="SiT-XL/2"
SAMPLER_TYPE="jump_flow"
NUM_WORKERS=7
DATASET_REPEAT=1
MAX_TRAIN_SAMPLES=64

CKPT_ARG=""
RESUME_ARG=""
if [ -n "$CKPT_PATH" ]; then
    if [ -d "$CKPT_PATH" ]; then
        RESUME_ARG="--resume $CKPT_PATH"
    else
        CKPT_ARG="--ckpt $CKPT_PATH"
    fi
fi

# Runtime temp/cache policy
WORKSPACE_DIR="/home/yanai-lab/xiong-p/SiT-GM-moe"
USER_TMP="${USER_TMP:-$WORKSPACE_DIR/.cache}"
mkdir -p "$USER_TMP"
export TMPDIR="$USER_TMP"
export TORCHINDUCTOR_CACHE_DIR="$USER_TMP/torchinductor"
export TRITON_CACHE_DIR="$USER_TMP/triton"
export XDG_CACHE_HOME="$USER_TMP/xdg_cache"

# -------------------------------------------------------------------
# NCCL tuning for gp40 (measured with 8-GPU accelerate all-reduce):
# - Best tested setting on this node was NCCL_P2P_LEVEL=SYS
# - Adding Tree,Ring was neutral to slightly positive
# - Larger buffers / extra NCCL threads did not improve throughput
# -------------------------------------------------------------------
export NCCL_P2P_LEVEL=SYS
export NCCL_ALGO=Tree,Ring
export NCCL_DEBUG=WARN

echo "=========================================="
echo " SiT-GM-moe Training on A6000 48GB x${NUM_GPUS}"
echo " Node:    gp40 (User Optimal P2P Config)"
echo " Global BS: $GLOBAL_BATCH ($((GLOBAL_BATCH/NUM_GPUS)) per GPU)"
echo " NCCL: ALGO=$NCCL_ALGO P2P_LEVEL=$NCCL_P2P_LEVEL"
echo " Sampling: DISABLED (via huge interval)"
echo "=========================================="

accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    train.py \
    --model "$MODEL" \
    --feature-path "$FEATURE_PATH" \
    --results-dir "$RESULTS_DIR" \
    --global-batch-size $GLOBAL_BATCH \
    --num-workers $NUM_WORKERS \
    --sampler-type $SAMPLER_TYPE \
    --epochs 1400000000 \
    --log-every 10 \
    --ckpt-every 5000 \
    --sample-every 1000000000000000000 \
    --cfg-scale 1 \
    --wandb \
    --gradient-checkpointing \
    --dataset-repeat $DATASET_REPEAT \
    $CKPT_ARG \
    $RESUME_ARG

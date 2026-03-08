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
RESULTS_DIR="${2:-results_a100/3090-jump-flow}"
CKPT_PATH="${3:-results_a100/3090-jump-flow/001-SiT-XL-2-Linear-velocity-None/checkpoints/0005000}"

# -------------------------------------------------------------------
# Hardware: 8x GPU (e.g. A100 80GB or RTX 6000 96GB)
# Config:   SiT-XL/2, num_bins=128, jump_range=3.0
# Batch:    128 per GPU (= 1024 global across 8 GPUs)
# Precision: FP32 (Full Precision)
# New Args:  Uses the current fixed linear CondOT setup
# -------------------------------------------------------------------

NUM_GPUS=8
BATCH_PER_GPU=15
GLOBAL_BATCH=$((BATCH_PER_GPU * NUM_GPUS*16))
MODEL="SiT-XL/2"
SAMPLER_TYPE="jump_flow"
# Keep workers conservative by default for stability on shared/NFS setups.
# You can override: NUM_WORKERS=2 bash stara6000.sh ...
NUM_WORKERS=8
MAX_TRAIN_SAMPLES=8
DATASET_REPEAT=100000

CKPT_ARG=""
RESUME_ARG=""
if [ -n "$CKPT_PATH" ]; then
    if [ -d "$CKPT_PATH" ]; then
        RESUME_ARG="--resume $CKPT_PATH"
    else
        CKPT_ARG="--ckpt $CKPT_PATH"
    fi
fi

export NCCL_MIN_NCHANNELS=4



# -------------------------------------------------------------------
# Runtime temp/cache policy
# -------------------------------------------------------------------
# Ensure persistent cache in workspace instead of transient /tmp
# to avoid recompilation and potential /tmp space issues.
WORKSPACE_DIR="/home/yanai-lab/xiong-p/SiT-GM-moe"
USER_TMP="${USER_TMP:-$WORKSPACE_DIR/.cache}"
mkdir -p "$USER_TMP"

export TMPDIR="$USER_TMP"
export TMP="$USER_TMP"
export TEMP="$USER_TMP"

# Torch Inductor Cache (set multiple variants for compatibility)
export TORCHINDUCTOR_CACHE_DIR="$USER_TMP/torchinductor"
export TORCH_INDUCTOR_CACHE_DIR="$USER_TMP/torchinductor"
export TORCH_INDUCTOR_CACHE_BASE_DIR="$USER_TMP/torchinductor"

# Triton Cache
export TRITON_CACHE_DIR="$USER_TMP/triton"
export XDG_CACHE_HOME="$USER_TMP/xdg_cache"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"


echo "=========================================="
echo " SiT-GM-moe Training on 24GB x${NUM_GPUS}"
echo " Feature Data: $FEATURE_PATH"
echo " Results: $RESULTS_DIR"
echo " Model:   $MODEL"
echo " Global BS: $GLOBAL_BATCH  (${NUM_GPUS} GPUs × $((GLOBAL_BATCH / NUM_GPUS))/GPU)"
echo " Sampler: $SAMPLER_TYPE"
echo " Max train samples: $MAX_TRAIN_SAMPLES"
echo " Dataset repeat: $DATASET_REPEAT"
echo " Resume arg: ${RESUME_ARG:-<none>}"
echo " Ckpt arg: ${CKPT_ARG:-<none>}"
echo " NCCL: ALGO=Auto (Default) MIN_NCHANNELS=$NCCL_MIN_NCHANNELS"
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
    --sample-every 999999999999999999999999 \
    --cfg-scale 4 \
    --wandb \
    --gradient-checkpointing \
    --gradient_accumulation_steps 1 \
    --max-train-samples $MAX_TRAIN_SAMPLES \
    --dataset-repeat $DATASET_REPEAT \
    $CKPT_ARG \
    $RESUME_ARG

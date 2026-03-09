#!/bin/bash
# Optimized training script for SiT-GM-moe on A100 80GB (Single GPU + DeepSpeed)
set -e
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
NUM_GPUS=1
FEATURE_PATH="/home/yanai-lab/xiong-p/SiT-GM-moe/imagenet_feature"
RESULTS_DIR="results_a100/a100-training-full"

# -------------------------------------------------------------------
# Batch Size Calculation:
# A100 80GB + DeepSpeed ZeRO + Gradient Checkpointing.
# -------------------------------------------------------------------
BATCH_PER_GPU=256
GLOBAL_BATCH=$BATCH_PER_GPU

MODEL="SiT-XL/2-org"
SAMPLER_TYPE="jump_flow"
NUM_WORKERS=8
DATASET_REPEAT=1

# MuonWithAuxAdam tuning knobs
MUON_LR="0.02"
MUON_MOMENTUM="0.95"
MUON_WD="0.01"

AUX_ADAM_LR="0.0001"

EMBED_LR="0.0001"
HEAD_LR="0.0001"

AUX_ADAM_BETA1="0.95"
AUX_ADAM_BETA2="0.99"
AUX_ADAM_EPS="1e-8"
AUX_ADAM_WD="0.01"
GRAD_CLIP_NORM="0"
EMA="true"
EMA_DECAY="0.999"
SAMPLE_USE_EMA="true"

# CKPT logic remains unchanged
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

echo "=========================================="
echo " SiT-GM-moe Training on A100 80GB (1-GPU)"
echo " Global BS: $GLOBAL_BATCH"
echo " Model:     $MODEL"
echo " Precision: FP32 (mixed_precision=no)"
echo "=========================================="

EMA_FLAG=""
if [ "$EMA" = "true" ]; then
    EMA_FLAG="--ema"
else
    EMA_FLAG="--no-ema"
fi

SAMPLE_USE_EMA_FLAG=""
if [ "$SAMPLE_USE_EMA" = "true" ]; then
    SAMPLE_USE_EMA_FLAG="--sample-use-ema"
else
    SAMPLE_USE_EMA_FLAG="--no-sample-use-ema"
fi

accelerate launch --num_processes=$NUM_GPUS --mixed_precision=no \
    traina100.py \
    --model "$MODEL" \
    --feature-path "$FEATURE_PATH" \
    --results-dir "$RESULTS_DIR" \
    --global-batch-size $GLOBAL_BATCH \
    --num-workers $NUM_WORKERS \
    --sampler-type $SAMPLER_TYPE \
    --epochs 1400000000 \
    --log-every 5 \
    --ckpt-every 5000 \
    --sample-every 1000000000000000000 \
    --cfg-scale 1 \
    --gradient_accumulation_steps 2 \
    --dataset-repeat $DATASET_REPEAT \
    --compile-mode max-autotune \
    --muon-lr "$MUON_LR" \
    --muon-momentum "$MUON_MOMENTUM" \
    --muon-weight-decay "$MUON_WD" \
    --embed-lr "$EMBED_LR" \
    --head-lr "$HEAD_LR" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --aux-adam-lr "$AUX_ADAM_LR" \
    --aux-adam-beta1 "$AUX_ADAM_BETA1" \
    --aux-adam-beta2 "$AUX_ADAM_BETA2" \
    --aux-adam-eps "$AUX_ADAM_EPS" \
    --aux-adam-weight-decay "$AUX_ADAM_WD" \
    $EMA_FLAG \
    --ema-decay "$EMA_DECAY" \
    $SAMPLE_USE_EMA_FLAG \
    --bregman-type mse \
    $CKPT_ARG \
    --no-ema-resume \
    $RESUME_ARG \
    --wandb \
    --compile \
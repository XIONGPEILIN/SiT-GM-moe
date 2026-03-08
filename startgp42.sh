#!/bin/bash
# Optimized training script for SiT-GM-moe on gp42 (7x RTX PRO 6000 Blackwell 96GB)
set -e
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4
FEATURE_PATH="${1:-/home/yanai-lab/xiong-p/SiT-GM-moe/imagenet_feature}"
RESULTS_DIR="${2:-results_a100/gp42-training}"
CKPT_PATH="${3:-results_a100/gp42-training/059-SiT-XL-2-Linear-velocity-None/checkpoints/0005000/ema.pt}"

# -------------------------------------------------------------------
# Batch Size Calculation:
# Blackwell has 96GB. Adjust BATCH_PER_GPU as needed.
# Global Batch is calculated without gradient accumulation steps.
# -------------------------------------------------------------------

BATCH_PER_GPU=50
GLOBAL_BATCH=$((BATCH_PER_GPU * NUM_GPUS * 16))

MODEL="SiT-XL/2"
SAMPLER_TYPE="jump_flow"
NUM_WORKERS=4
# DATASET_REPEAT=10000
# MAX_TRAIN_SAMPLES=64

# MuonWithAuxAdam tuning knobs (Optimized via online research)
MUON_LR="0.02"
MUON_MOMENTUM="0.95"
MUON_WD="0.01"
AUX_ADAM_LR="0.0003"
AUX_ADAM_BETA1="0.9"
AUX_ADAM_BETA2="0.95"
AUX_ADAM_EPS="1e-8"
AUX_ADAM_WD="0.01"
EMA="true"
EMA_DECAY="0.999"
SAMPLE_USE_EMA="true"
USE_DEEPSPEED="true"
DEEPSPEED_CONFIG_FILE="ds_config_zero1.json"

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
# NCCL tuning for gp42 (Blackwell Optimized for GPUs 4,5,6,7):
# - Ring algorithm with P2P LEVEL 5: Best for 4-GPU PCIe topology
# - Symmetric Memory & CE Threshold: Reduces SM overhead & small message latency
# - Buffsize 4MB: Maximizes PCIe Gen4/Gen5 pipeline throughput
# - Performance: ~74ms for 1GB tensor (Solid PCIe bottleneck)
# -------------------------------------------------------------------
export NCCL_ALGO="Ring"
export NCCL_P2P_LEVEL=5
export NCCL_SYMMETRIC_MEMORY=1
export NCCL_CE_THRESHOLD=1
export NCCL_BUFFSIZE=4194304
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

echo "=========================================="
echo " SiT-GM-moe Training on Blackwell 96GB x${NUM_GPUS}"
echo " Node:    gp42 (Optimal Blackwell NCCL Config)"
echo " Global BS: $GLOBAL_BATCH ($((GLOBAL_BATCH/NUM_GPUS)) per GPU)"
echo " NCCL: ALGO=$NCCL_ALGO, P2P=5, SYMM=1, CE=1"
echo " Loss weights: flow=0.5, Jump=0.5 (Markov Superposition)"
echo " Max Train Samples: $MAX_TRAIN_SAMPLES"
echo " Sampling: DISABLED (via huge interval)"
echo " Muon: lr=$MUON_LR, momentum=$MUON_MOMENTUM, wd=$MUON_WD"
echo " Aux-Adam: lr=$AUX_ADAM_LR, betas=($AUX_ADAM_BETA1,$AUX_ADAM_BETA2), eps=$AUX_ADAM_EPS, wd=$AUX_ADAM_WD"
echo " EMA: enabled=$EMA, decay=$EMA_DECAY, sample_use_ema=$SAMPLE_USE_EMA"
echo " DeepSpeed: enabled=$USE_DEEPSPEED, config=$DEEPSPEED_CONFIG_FILE"
echo "=========================================="

DEEPSPEED_ARGS=""
if [ "$USE_DEEPSPEED" = "true" ]; then
    DEEPSPEED_ARGS="--use_deepspeed --deepspeed_config_file $DEEPSPEED_CONFIG_FILE"
fi

EMA_FLAG=""
if [ "$EMA" = "true" ]; then
    EMA_FLAG="--ema"
elif [ "$EMA" = "false" ]; then
    EMA_FLAG="--no-ema"
else
    echo "Invalid EMA value: $EMA (must be true or false)"
    exit 1
fi

SAMPLE_USE_EMA_FLAG=""
if [ "$SAMPLE_USE_EMA" = "true" ]; then
    SAMPLE_USE_EMA_FLAG="--sample-use-ema"
elif [ "$SAMPLE_USE_EMA" = "false" ]; then
    SAMPLE_USE_EMA_FLAG="--no-sample-use-ema"
else
    echo "Invalid SAMPLE_USE_EMA value: $SAMPLE_USE_EMA (must be true or false)"
    exit 1
fi

accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 $DEEPSPEED_ARGS \
    train.py \
    --model "$MODEL" \
    --feature-path "$FEATURE_PATH" \
    --results-dir "$RESULTS_DIR" \
    --global-batch-size $GLOBAL_BATCH \
    --num-workers $NUM_WORKERS \
    --sampler-type $SAMPLER_TYPE \
    --max-train-samples $MAX_TRAIN_SAMPLES \
    --epochs 1400000000 \
    --log-every 1 \
    --ckpt-every 5000 \
    --sample-every 1000000000000000000 \
    --cfg-scale 1 \
    --wandb \
    --gradient_accumulation_steps 1 \
    --dataset-repeat $DATASET_REPEAT \
    --compile \
    --compile-mode max-autotune \
    --muon-lr "$MUON_LR" \
    --muon-momentum "$MUON_MOMENTUM" \
    --muon-weight-decay "$MUON_WD" \
    --aux-adam-lr "$AUX_ADAM_LR" \
    --aux-adam-beta1 "$AUX_ADAM_BETA1" \
    --aux-adam-beta2 "$AUX_ADAM_BETA2" \
    --aux-adam-eps "$AUX_ADAM_EPS" \
    --aux-adam-weight-decay "$AUX_ADAM_WD" \
    --gradient-checkpointing \
    $EMA_FLAG \
    --ema-decay "$EMA_DECAY" \
    $SAMPLE_USE_EMA_FLAG \
    --bregman-type cosh \
    $CKPT_ARG \
    --no-ema-resume \
    $RESUME_ARG

#!/bin/bash
# Evaluation script for SiT-GM-moe on A100 80GB x8
# Configured for generating FID samples without interrupting training.
#
# Usage:
#   bash eval_a100_80gb.sh <CKPT_PATH> [SAMPLE_DIR]
#
# Args:
#   CKPT_PATH  : Path to the checkpoint (.pt) to evaluate
#   SAMPLE_DIR : Output directory for generated images (default: results_a100/samples)

set -e

if [ -z "$1" ]; then
    echo "Error: Checkpoint path is required."
    echo "Usage: bash eval_a100_80gb.sh <CKPT_PATH> [SAMPLE_DIR]"
    exit 1
fi

CKPT_PATH="$1"
SAMPLE_DIR="${2:-results_a100/samples}"

# -------------------------------------------------------------------
# Config matching training structure
# -------------------------------------------------------------------

NUM_GPUS=8
MODEL="SiT-XL/2"
NUM_BINS=128
JUMP_RANGE=3.0
TIME_SCHEDULE="cubic"

# Adjust batch size based on your available memory during evaluation.
# Since we chunk the VAE, 16 per GPU is usually safe for A100 80GB
PER_PROC_BATCH_SIZE=8 
NUM_SAMPLES=50000

echo "=========================================="
echo " SiT-GM-moe Evaluation on 80GB x${NUM_GPUS} (FP32)"
echo " Checkpoint: $CKPT_PATH"
echo " Sample Dir: $SAMPLE_DIR"
echo " Model:      $MODEL ($TIME_SCHEDULE schedule)"
echo " Total generation: $NUM_SAMPLES images."
echo "=========================================="

/home/yanai-lab/xiong-p/SiT-GM-moe/.venv/bin/accelerate launch \
    --num_processes $NUM_GPUS \
    sample_ddp.py MIXED \
    --model "$MODEL" \
    --ckpt "$CKPT_PATH" \
    --sample-dir "$SAMPLE_DIR" \
    --image-size 256 \
    --num-classes 1000 \
    --path-type Linear \
    --prediction velocity \
    --num-bins $NUM_BINS \
    --jump-range $JUMP_RANGE \
    --time-schedule $TIME_SCHEDULE \
    --num-sampling-steps 50 \
    --cfg-scale 4.0 \
    --per-proc-batch-size $PER_PROC_BATCH_SIZE \
    --num-fid-samples $NUM_SAMPLES

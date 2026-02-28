#!/bin/bash
# Sampling script for SiT-GM-moe on A100 80GB x8
# Generates 50,000 samples for FID evaluation using MIXED (Flow+Jump) sampler.
#
# Usage:
#   bash sample_a100_80gb.sh <CKPT_PATH> [SAMPLE_DIR] [MODE]
#
# Args:
#   CKPT_PATH   : Path to a trained SiT checkpoint (.pt file)
#   SAMPLE_DIR  : Directory to save samples (default: samples_a100)
#   MODE        : Sampling mode: ODE, SDE, or MIXED (default: MIXED)

set -e

CKPT_PATH="${1:?Error: CKPT_PATH is required. Usage: $0 <CKPT_PATH> [SAMPLE_DIR] [MODE]}"
SAMPLE_DIR="${2:-samples_a6000}"
MODE="${3:-MIXED}"

# -------------------------------------------------------------------
# Hardware: 8x A100 80GB
# Config:   SiT-XL/2, num_bins=128, jump_range=3.0
# Batch:    128 per GPU (sampling has no grad, so can be very large)
# Precision: FP32
# -------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
PER_PROC_BS=32
NUM_FID_SAMPLES=50000
NUM_STEPS=1000
CFG_SCALE=1
MODEL="SiT-XL/2"
NUM_BINS=128
JUMP_RANGE=3.0

echo "=========================================="
echo " SiT-GM-moe Sampling on A100 80GB x${NUM_GPUS} (FP32)"
echo " Checkpoint: $CKPT_PATH"
echo " Output dir: $SAMPLE_DIR"
echo " Mode:       $MODE"
echo " Total FID samples: $NUM_FID_SAMPLES"
echo " Per-GPU batch: $PER_PROC_BS"
echo "=========================================="

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision no \
    sample_ddp.py "$MODE" \
    --model "$MODEL" \
    --ckpt "$CKPT_PATH" \
    --sample-dir "$SAMPLE_DIR" \
    --per-proc-batch-size $PER_PROC_BS \
    --num-fid-samples $NUM_FID_SAMPLES \
    --num-sampling-steps $NUM_STEPS \
    --cfg-scale $CFG_SCALE \
    --num-bins $NUM_BINS \
    --jump-range $JUMP_RANGE \
    --tf32

echo ""
echo "Done! Samples written to: $SAMPLE_DIR"
echo "To compute FID, run:"
echo "  python evaluator.py <ref_stats.npz> ${SAMPLE_DIR}/*.npz"

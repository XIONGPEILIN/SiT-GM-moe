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
GLOBAL_BATCH=640  
MODEL="SiT-XL/2"
NUM_BINS=1024
JUMP_RANGE=4.0
SAMPLER_TYPE="jump"
TIME_SCHEDULE="linear"
NUM_WORKERS=3

CKPT_ARG=""
if [ -n "$CKPT_PATH" ]; then
    CKPT_ARG="--ckpt $CKPT_PATH"
fi

# -------------------------------------------------------------------
# NCCL 优化：针对跨 NUMA 双路服务器 (2x EPYC + 8x A6000)
# -------------------------------------------------------------------

# 1. Tree 算法：相比默认 Ring，Tree 在非均匀拓扑下更高效
#    Ring 的环形路径必须经过最慢的跨 NUMA 链路两次
#    Tree 可以减少跨 NUMA 通信次数（先 NUMA 内聚合，再跨 NUMA 合并）
export NCCL_ALGO=Tree,Ring

# 2. 强制启用 NVLink P2P 传输，即使跨 NUMA 也尝试 P2P
#    默认情况下 NCCL 可能因为跨 NUMA 而回退到 SHM(共享内存)拷贝
export NCCL_P2P_LEVEL=5
export NCCL_P2P_DISABLE=0

# 3. 共享内存缓冲区加大：跨 NUMA 回退到 SHM 时，加大缓冲区减少碎片传输
export NCCL_SHM_DISABLE=0
export NCCL_BUFFSIZE=16777216        # 16MB (默认 4MB)，减少传输次数
export NCCL_NTHREADS=512             # NCCL 内部线程数（默认 256），加速数据搬运

# 4. 优化 NUMA 亲和性：让每个 GPU 进程绑定到最近的 CPU 核心
#    避免 GPU 0-3 的进程跑到 NUMA 1 的 CPU 上导致额外的内存跨域访问
export NCCL_SOCKET_NTHREADS=4        # Socket 通信线程数
export NCCL_NSOCKS_PERTHREAD=4       # 每线程 socket 数

# 5. 调试信息（首次运行看通信路径是否正确，确认后可改为 WARN）
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

echo "=========================================="
echo " SiT-GM-moe Training on 48GB x${NUM_GPUS} (FP32)"
echo " Feature Data: $FEATURE_PATH"
echo " Results: $RESULTS_DIR"
echo " Model:   $MODEL ($TIME_SCHEDULE schedule)"
echo " Global BS: $GLOBAL_BATCH  (${NUM_GPUS} GPUs × $((GLOBAL_BATCH / NUM_GPUS))/GPU)"
echo " NCCL: ALGO=$NCCL_ALGO P2P_LEVEL=$NCCL_P2P_LEVEL BUFFSIZE=$NCCL_BUFFSIZE"
echo "=========================================="

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
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


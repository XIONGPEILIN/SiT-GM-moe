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
RESULTS_DIR="${2:-results_a100/a6000-jump-flow}"
CKPT_PATH="${3:-}"

# -------------------------------------------------------------------
# Hardware: 8x GPU (e.g. A100 80GB or RTX 6000 96GB)
# Config:   SiT-XL/2, num_bins=128, jump_range=3.0
# Batch:    128 per GPU (= 1024 global across 8 GPUs)
# Precision: FP32 (Full Precision)
# New Args:  Uses the current fixed linear CondOT setup
# -------------------------------------------------------------------

NUM_GPUS=8
GLOBAL_BATCH=768
MODEL="SiT-XL/2"
SAMPLER_TYPE="jump_flow"
# Keep workers conservative by default for stability on shared/NFS setups.
# You can override: NUM_WORKERS=2 bash stara6000.sh ...
NUM_WORKERS=8
MAX_TRAIN_SAMPLES=64
DATASET_REPEAT=10000

CKPT_ARG=""
RESUME_ARG=""
if [ -n "$CKPT_PATH" ]; then
    if [ -d "$CKPT_PATH" ]; then
        RESUME_ARG="--resume $CKPT_PATH"
    else
        CKPT_ARG="--ckpt $CKPT_PATH"
    fi
fi

# -------------------------------------------------------------------
# NCCL 优化：针对跨 NUMA 双路服务器 (2x EPYC + 8x A6000)
# -------------------------------------------------------------------

# 1. Tree 算法：相比默认 Ring，Tree 在非均匀拓扑下更高效
#    Ring 的环形路径必须经过最慢的跨 NUMA 链路两次
#    Tree 可以减少跨 NUMA 通信次数（先 NUMA 内聚合，再跨 NUMA 合并）
export NCCL_ALGO=Tree,Ring

# 2. P2P 传输策略
#    使用 "PHB" 级别：允许同 CPU 内的所有通信(含 NVLink 和 PCIe) 走 P2P 极速通道，
#    仅仅把跨 CPU 的 SYS 连接通信安全回退到共享内存，防死锁。
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL      # 回退到 NVL：实测 PHB 级别也会死锁，只有纯 NVLink 是安全的
export NCCL_BLOCKING_WAIT=0    # 非阻塞等待，减少死锁风险

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
# [新增] 限制 CUDA 并行流，减少 Context Switch 开销
export CUDA_DEVICE_MAX_CONNECTIONS=1
# [新增] 强制 NCCL 开辟更多并行通道来加速共享内存的拷贝
export NCCL_MIN_NCHANNELS=4
# -------------------------------------------------------------------
# OS / PyTorch 级优化
# -------------------------------------------------------------------

# 6. 控制 CPU 线程数，防止 DataLoader 跨 NUMA 争抢
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 7. PyTorch CUDA 内存分配器优化
#    expandable_segments 减少显存碎片，避免 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo " SiT-GM-moe Training on 48GB x${NUM_GPUS}"
echo " Feature Data: $FEATURE_PATH"
echo " Results: $RESULTS_DIR"
echo " Model:   $MODEL"
echo " Global BS: $GLOBAL_BATCH  (${NUM_GPUS} GPUs × $((GLOBAL_BATCH / NUM_GPUS))/GPU)"
echo " Sampler: $SAMPLER_TYPE"
echo " Max train samples: $MAX_TRAIN_SAMPLES"
echo " Dataset repeat: $DATASET_REPEAT"
echo " Resume arg: ${RESUME_ARG:-<none>}"
echo " Ckpt arg: ${CKPT_ARG:-<none>}"
echo " NCCL: ALGO=$NCCL_ALGO P2P_LEVEL=$NCCL_P2P_LEVEL BUFFSIZE=$NCCL_BUFFSIZE"
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
    --sample-every 5000 \
    --cfg-scale 4 \
    --wandb \
    --max-train-samples $MAX_TRAIN_SAMPLES \
    --dataset-repeat $DATASET_REPEAT \
    $CKPT_ARG \
    $RESUME_ARG

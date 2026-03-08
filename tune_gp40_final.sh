#!/bin/bash

# Ensure we use the correct virtual environment
export PATH="/home/yanai-lab/xiong-p/SiT-GM-moe/.venv/bin:$PATH"
export PYTHONPATH="/home/yanai-lab/xiong-p/SiT-GM-moe"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="32000"

TORCHRUN="/home/yanai-lab/xiong-p/SiT-GM-moe/.venv/bin/torchrun"
SCRIPT="test_allreduce.py"

echo "Final Comprehensive Tuning on gp40..."

declare -a configs=(
    "DEFAULT"
    "NCCL_SHM_DISABLE=1"
    "NCCL_P2P_DISABLE=1"
    "NCCL_P2P_LEVEL=5"
    "NCCL_ALGO=Tree NCCL_MIN_NCHANNELS=8"
    "NCCL_IB_DISABLE=1"
    # User's theory set
    "NCCL_ALGO=Tree,Ring NCCL_P2P_LEVEL=5 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_BUFFSIZE=16777216 NCCL_NTHREADS=512"
    # User's theory set but with SHM_DISABLE=1 (based on previous find)
    "NCCL_ALGO=Tree,Ring NCCL_P2P_LEVEL=5 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=1 NCCL_BUFFSIZE=16777216 NCCL_NTHREADS=512"
    # Hybrid: Disable P2P and IB, increase channels
    "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=16"
)

for config in "${configs[@]}"; do
    echo "Testing: $config"
    MASTER_PORT=$((MASTER_PORT + 1))
    if [ "$config" == "DEFAULT" ]; then
        output=$($TORCHRUN --master_port=$MASTER_PORT --nproc_per_node=8 $SCRIPT 2>/dev/null | grep RESULT_BW | cut -d':' -f2)
    else
        output=$(eval "export $config; $TORCHRUN --master_port=$MASTER_PORT --nproc_per_node=8 $SCRIPT 2>/dev/null" | grep RESULT_BW | cut -d':' -f2)
    fi
    echo " -> $output GB/s"
done

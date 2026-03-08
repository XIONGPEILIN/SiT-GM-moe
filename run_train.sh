#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --model SiT-XL/2 \
    --data-path /sys/fs/cgroup/memory \
    --global-batch-size 256 \
    --epochs 1000 \
    --log-every 100 \
    --ckpt-every 10000 \
    --num-workers 8

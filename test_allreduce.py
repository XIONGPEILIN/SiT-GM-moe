import os
import time
import torch
import torch.distributed as dist
import sys

def test_allreduce(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Message size (128 MB)
    size_mb = 128
    tensor_size = size_mb * 1024 * 1024 // 4 # float32
    
    # Warmup
    data = torch.randn(tensor_size, device=device)
    dist.barrier()
    for _ in range(5):
        dist.all_reduce(data)
    torch.cuda.synchronize()

    # Collective All-Reduce Bandwidth
    dist.barrier()
    iters = 20
    data = torch.randn(tensor_size, device=device)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(data)
    torch.cuda.synchronize()
    end = time.time()
    
    duration = end - start
    # All-reduce algorithmic bandwidth: 2 * (N-1)/N * Size / Time
    bw = (2 * (world_size - 1) / world_size) * (size_mb * iters) / (duration * 1024)
    
    if rank == 0:
        print(f"RESULT_BW:{bw:.2f}")

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    test_allreduce(rank, world_size)
    dist.destroy_process_group()

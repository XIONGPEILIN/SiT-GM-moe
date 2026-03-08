#!/usr/bin/env python3
"""
NCCL All-Reduce Benchmark
Measures the time to all-reduce a 24GB bf16 tensor across GPUs.
Reports bandwidth in GB/s and latency in ms.
"""
import os
import sys
import time
from datetime import timedelta
import torch
import torch.distributed as dist


def main():
    # Init distributed with timeout
    try:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
    except Exception as e:
        print(f"[ERROR] Failed to init process group: {e}")
        sys.exit(1)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 1 GB in bf16 = 1 * 1024^3 / 2 = 536,870,912 elements
    target_bytes = 1 * (1024 ** 3)  # 1 GB
    elem_size = 2  # bf16
    num_elements = target_bytes // elem_size

    if rank == 0:
        print(f"[NCCL Bench] world_size={world_size}, dtype=bf16")
        print(
            f"[NCCL Bench] Tensor: {num_elements:,} elements = {target_bytes / 1e9:.2f} GB")
        # Print NCCL env vars
        for key in sorted(os.environ):
            if key.startswith("NCCL"):
                print(f"  {key}={os.environ[key]}")
        sys.stdout.flush()

    # Allocate tensor
    tensor = torch.randn(num_elements, dtype=torch.bfloat16, device=device)

    # Warmup (3 iterations)
    for _ in range(3):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark (5 iterations)
    n_iters = 5
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for i in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    if rank == 0:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        # Algo bandwidth = 2*(n-1)/n * data_size / time  (for all-reduce ring)
        # Bus bandwidth = data_size * 2 * (n-1) / n / time
        algo_bw = target_bytes / min_time / 1e9  # GB/s (simple)
        bus_bw = target_bytes * 2 * \
            (world_size - 1) / world_size / min_time / 1e9

        print(f"\n{'='*60}")
        print(f"[RESULT] Iterations: {n_iters}")
        print(f"[RESULT] Times (s):  {[f'{t:.4f}' for t in times]}")
        print(f"[RESULT] Min:  {min_time*1000:.1f} ms")
        print(f"[RESULT] Avg:  {avg_time*1000:.1f} ms")
        print(f"[RESULT] Max:  {max_time*1000:.1f} ms")
        print(f"[RESULT] Algo BW (min): {algo_bw:.2f} GB/s")
        print(f"[RESULT] Bus  BW (min): {bus_bw:.2f} GB/s")
        print(f"{'='*60}")
        sys.stdout.flush()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

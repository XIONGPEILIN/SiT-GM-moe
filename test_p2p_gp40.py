import os
import time
import torch
import torch.distributed as dist
import sys

def test_p2p_matrix(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Message size (128 MB)
    size_mb = 128
    tensor_size = size_mb * 1024 * 1024 // 4 # float32
    
    # Warmup
    data = torch.randn(tensor_size, device=device)
    dist.barrier()
    dist.all_reduce(data)
    torch.cuda.synchronize()

    results = torch.zeros((world_size, world_size), device=device)

    for src in range(world_size):
        for dst in range(world_size):
            if src == dst:
                continue
            
            dist.barrier()
            iters = 5
            
            if rank == src:
                send_data = torch.randn(tensor_size, device=device)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(iters):
                    dist.send(send_data, dst)
                torch.cuda.synchronize()
                end = time.time()
                bandwidth = (size_mb * iters) / ((end - start) * 1024) # GB/s
                results[src, dst] = bandwidth
            elif rank == dst:
                recv_data = torch.empty(tensor_size, device=device)
                for _ in range(iters):
                    dist.recv(recv_data, src)
                torch.cuda.synchronize()
    
    # Gather results to rank 0
    dist.reduce(results, dst=0, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print("\nP2P Bandwidth Matrix (GB/s):")
        header = "Rank | " + " | ".join([f"{i:6}" for i in range(world_size)])
        print(header)
        print("-" * len(header))
        for i in range(world_size):
            row_str = f"{i:4} | " + " | ".join([f"{results[i, j]:6.2f}" if i != j else "   -  " for j in range(world_size)])
            print(row_str)

    # All-Reduce Bus Bandwidth
    dist.barrier()
    iters = 20
    data = torch.randn(tensor_size, device=device)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(data)
    torch.cuda.synchronize()
    end = time.time()
    bw = (2 * (world_size - 1) / world_size) * (size_mb * iters) / ((end - start) * 1024)
    if rank == 0:
        print(f"\nFinal All-Reduce Bus Bandwidth: {bw:.2f} GB/s")

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    test_p2p_matrix(rank, world_size)
    dist.destroy_process_group()

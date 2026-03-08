# File: test_p2p.py
import os
import torch
import torch.distributed as dist
from datetime import timedelta

def main():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    tensor = torch.zeros(1024, 1024, device=device) # 4MB
    
    print(f"Rank {rank} starting P2P...")
    if rank == 0:
        dist.send(tensor, dst=1)
        print("Rank 0 sent data.")
    elif rank == 1:
        dist.recv(tensor, src=0)
        print("Rank 1 received data.")
    
    dist.barrier()
    print(f"Rank {rank} finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

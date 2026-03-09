import torch
import time
from functools import partial
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(device)}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDNN Version: {torch.backends.cudnn.version()}")

# Parameters for benchmark
B, H, S, D = 16, 16, 2048, 128  # Adjust as needed
dtype = torch.bfloat16

print(f"\nBenchmarking Attention with shape (B={B}, H={H}, S={S}, D={D}) dtype={dtype}")

def benchmark(name, func, *args, **kwargs):
    # Warmup
    for _ in range(10):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / 100
    print(f"{name}: {avg_time_ms:.4f} ms")
    return avg_time_ms

# Data preparation
q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=False)
k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=False)
v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=False)

# 1. Standard SDPA (cuDNN / FlashAttn / Math)
def run_sdpa():
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

print("\n--- Running Benchmarks ---")
time_sdpa = benchmark("SDPA (Default - cuDNN/FlashAttn)", run_sdpa)

# 2. FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention
    
    # Define a no-op score mod (equivalent to standard attention)
    def noop_mod(score, b, h, q_idx, kv_idx):
        return score

    # 2a. FlexAttention (Default - likely Triton)
    def run_flex_default():
        return flex_attention(q, k, v, score_mod=noop_mod)
    
    # Compile first to be fair if we compare against compiled flash backend
    flex_default_compiled = torch.compile(run_flex_default)
    
    # Verify it runs first
    try:
        run_flex_default()
        time_flex_default = benchmark("FlexAttention (Default Backend)", run_flex_default)
    except Exception as e:
        print(f"FlexAttention (Default Backend) failed: {e}")

    # 2b. FlexAttention (Flash Backend)
    # The blog mentions: flex_attention(..., kernel_options={"BACKEND": "FLASH"})
    try:
        flex_flash_fn = torch.compile(
            partial(flex_attention, score_mod=noop_mod, kernel_options={"BACKEND": "FLASH"}),
            dynamic=False
        )
        
        # Warmup specific for compile
        try:
            flex_flash_fn(q, k, v)
            time_flex_flash = benchmark("FlexAttention (BACKEND='FLASH')", lambda: flex_flash_fn(q, k, v))
        except Exception as e:
             print(f"FlexAttention (BACKEND='FLASH') Execution failed: {e}")
             
    except Exception as e:
        print(f"FlexAttention (BACKEND='FLASH') setup failed: {e}")

except ImportError:
    print("FlexAttention not available in this PyTorch version.")
except Exception as e:
    print(f"Error testing FlexAttention: {e}")

print("\nBenchmark Complete.")

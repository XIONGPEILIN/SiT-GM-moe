import torch
import torch.nn as nn
from models import SiT_models

def estimate_vram():
    gpu_options = {"A100": 80, "RTX6000": 96}
    print("=== SiT-GM-moe Max Model Predictor ===")
    print(f"Detected GPUs: A100 (80GB) and RTX 6000 (96GB)")
    
    num_bins = 128
    in_channels = 4
    seq_len = 256 # (256/2)^2
    
    configs = [
        ("XL/2", 28, 1152),
        ("XXL/2 (Hypothetical)", 40, 2048),
        ("Giant/2 (Hypothetical)", 48, 3072),
        ("Colossus/2 (Hypothetical)", 64, 4096),
    ]

    for name, depth, hidden in configs:
        # Params calculation for Transformer
        # Approx: depth * (12 * hidden^2) + embed + head
        # Jump head adds: in_channels * (num_bins + 1) * hidden
        total_params = depth * (12 * (hidden**2)) 
        # Add head memory
        total_params += in_channels * (num_bins + 1) * hidden
        
        static_gb = total_params * (4 + 8 + 4) / (1024**3) # Param + Opt + Grad
        
        # Activations for BS=8
        bs = 8
        mem_per_layer = (seq_len * hidden * 10) * 4 / (1024**3)
        attn_matrix_mem = ( (hidden//64) * seq_len * seq_len) * 4 / (1024**3)
        total_act = bs * depth * (mem_per_layer + attn_matrix_mem)
        
        total_vram = static_gb + total_act + 2.0 # System buffer
        
        print(f"\nModel: {name}")
        print(f"  Config: Depth={depth}, Hidden={hidden}")
        print(f"  Params: {total_params/1e9:.2f} B")
        print(f"  Est. VRAM (BS={bs}): {total_vram:.2f} GB")
        
        if total_vram < 80:
            status = "Fits on A100 (80GB)"
        elif total_vram < 96:
            status = "Fits on RTX 6000 (96GB)"
        else:
            status = "TOO LARGE"
        print(f"  Status: {status}")

if __name__ == "__main__":
    estimate_vram()

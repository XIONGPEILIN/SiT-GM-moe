
import torch
import numpy as np

def check_kt_dist():
    # Simulate CondOT path: xt = t*z + (1-t)*eps
    # z (x1) ~ N(0, 1), eps (x0) ~ N(0, 1)
    N = 1000000
    z = torch.randn(N)
    eps = torch.randn(N)
    
    ts = [0.1, 0.5, 0.9, 0.99]
    for t in ts:
        xt = t * z + (1 - t) * eps
        # k_t(x|z) = x^2 - (t+1)*x*z - (1-t)^2 + t*z^2
        kt = xt**2 - (t + 1) * xt * z - (1 - t)**2 + t * z**2
        
        pos_fraction = (kt > 0).float().mean().item()
        avg_kt_pos = kt[kt > 0].mean().item() if (kt > 0).any() else 0
        
        # lambda_target = [kt]_+ / (1-t)^3
        lambda_t = torch.clamp(kt, min=0) / ((1-t)**3 + 1e-8)
        avg_lambda = lambda_t.mean().item()
        
        print(f"t={t:.2f}: P(kt>0)={pos_fraction:.4f}, avg(kt|kt>0)={avg_kt_pos:.4e}, avg(lambda)={avg_lambda:.4e}")

if __name__ == "__main__":
    check_kt_dist()

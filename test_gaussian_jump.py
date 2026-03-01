import torch
import torch as th
from models import SiT_models
from transport import create_transport, Sampler
import sys


def test_gaussian_jump():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    # 1. Initialize Model
    # We use a small input size (latents 8x8) and XL/2 to check parameters
    model = SiT_models["SiT-XL/2"](
        input_size=8,
        num_classes=1000
    ).to(device)
    model.eval()

    print(f"Model initialized. Jump channels: {model.jump_channels}")
    expected_jump_channels = model.in_channels * 3
    assert model.jump_channels == expected_jump_channels, f"Expected {expected_jump_channels}, got {model.jump_channels}"

    # 2. Test Forward with CFG
    img_size = 8
    C = model.in_channels
    z = torch.randn(2, C, img_size, img_size).to(
        device)  # Batch 2 for CFG split
    t = torch.tensor([0.5, 0.5]).to(device)
    y = torch.tensor([1, 1000]).to(device)  # One class, one null

    output = model.forward_with_cfg(z, t, y, cfg_scale=4.0)
    # Output should be (2, flow_C + 3*jump_C, H, W) -> (2, 4 + 12, 8, 8) = (2, 16, 8, 8)
    print(f"CFG Output shape: {output.shape}")
    assert output.shape == (2, model.out_channels, img_size, img_size)

    # 3. Test Training Loss
    transport = create_transport(path_type="Linear", prediction="velocity")
    x1 = torch.randn(4, C, img_size, img_size).to(device)
    model_kwargs = dict(y=torch.randint(0, 1000, (4,)).to(device))

    # We need to wrap model in a way that training_losses can call it
    # SiT-GM code expects the model to handle the forward.
    loss_dict = transport.training_losses(model, x1, model_kwargs)

    print("Loss Calculation:")
    for k, v in loss_dict.items():
        if v.numel() > 1:
            print(f"  {k} shape: {v.shape} (taking mean for print)")
            val = v.mean().item()
        else:
            val = v.item()
        print(f"  {k}: {val:.4f}")
        assert not th.isnan(v).any(), f"NaN detected in {k}"
        assert th.isfinite(v).all(), f"Infinite value detected in {k}"

    # 4. Test Sampling Logic (PF-ODE)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_jump_flow(num_steps=5)  # few steps for speed

    # Prepare inputs for sample_fn: (z, model_fn, **model_kwargs)
    zs = torch.randn(2, C, img_size, img_size).to(device)
    ys = torch.tensor([5, 5]).to(device)
    # Mocking CFG expansion inside test or just testing jump_flow
    model_fn = model.forward

    print("Testing Sampling Loop...")
    samples_trajectory = sample_fn(zs, model_fn, y=ys)
    final_samples = samples_trajectory[-1]

    print(f"Final sampled shape: {final_samples.shape}")
    assert final_samples.shape == zs.shape
    assert not th.isnan(final_samples).any(), "NaN in samples!"

    print("\n[SUCCESS] Gaussian Jump Implementation test passed!")


if __name__ == "__main__":
    try:
        test_gaussian_jump()
    except Exception as e:
        print(f"\n[FAILED] Test crashed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

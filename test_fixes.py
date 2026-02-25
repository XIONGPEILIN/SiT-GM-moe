import torch
from models import SiT_models
from transport import create_transport, Sampler

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Test model creation with custom params (validates #13)
    latent_size = 4  # small for fast test
    model = SiT_models["SiT-XL/2"](
        input_size=latent_size,
        num_classes=10,
        num_bins=8,
        jump_range=4.0
    ).to(device)
    print("Model created successfully.")
    
    # 2. Test transport creation preventing train_eps=0 override (validates #1)
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        train_eps=1e-5,
        sample_eps=1e-5
    )
    assert transport.train_eps > 0, "train_eps should not be 0!"
    print(f"Transport created with train_eps={transport.train_eps}")
    
    # 3. Test training_losses for NaNs or shape issues (validates #5, #7)
    batch_size = 2
    x1 = torch.randn(batch_size, 4, latent_size, latent_size).to(device)
    y = torch.randint(0, 10, (batch_size,)).to(device)
    model_kwargs = dict(y=y)
    
    loss_dict = transport.training_losses(model, x1, model_kwargs)
    print("loss_flow:", loss_dict["loss_flow"].item())
    print("loss_jump:", loss_dict["loss_jump"].item())
    print("loss (total):", loss_dict["loss"].item())
    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN!"
    
    # 4. Test CFG forward pass (validates #4)
    model.eval()
    cfg_scale = 4.0
    model_out = model.forward_with_cfg(
        torch.cat([x1, x1], dim=0),
        torch.rand(batch_size * 2).to(device),
        torch.cat([y, y], dim=0),
        cfg_scale
    )
    print("CFG output shape:", model_out.shape)
    
    # 5. Test jump flow sampler (validates #3, #8)
    transport_sampler = Sampler(transport)
    sample_fn = transport_sampler.sample_jump_flow(num_steps=5, jump_alpha=0.5)
    
    zs = torch.randn(batch_size, 4, latent_size, latent_size).to(device)
    samples = sample_fn(zs, model, **model_kwargs)
    print("Sampler generated", len(samples), "steps.")
    print("Final sample shape:", samples[-1].shape)
    
    print("All tests passed!")

if __name__ == "__main__":
    main()

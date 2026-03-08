# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# File: sample.py

"""
Sample new images from a pre-trained SiT.
"""
import torch
import json
import os
from collections import OrderedDict
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
from time import time
from pathlib import Path


def load_pretrained_compatible(model, state_dict):
    """
    Load checkpoints robustly across legacy/new SiT head layouts.
    - Legacy pretrain may have a single `final_layer.*`.
    - Current model uses `final_layer_flow.*` and `final_layer_jump.*`.
    """
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Strip common wrappers.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k[len("module."):]: v for k, v in state_dict.items()
        }
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {
            k[len("_orig_mod."):]: v for k, v in state_dict.items()
        }

    has_legacy_single_head = any(
        k.startswith("final_layer.") for k in state_dict.keys()
    )
    has_split_heads = any(
        k.startswith("final_layer_flow.") or k.startswith("final_layer_jump.")
        for k in state_dict.keys()
    )

    if has_legacy_single_head and not has_split_heads:
        # Map legacy final_layer.* to final_layer_flow.*
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("final_layer."):
                new_key = k.replace("final_layer.", "final_layer_flow.")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        print("Detected legacy single-head checkpoint (final_layer.*). Mapped to final_layer_flow.*")

    # Check and remove shape mismatches before strict=False loading.
    model_state = model.state_dict()
    mismatched_keys = []
    for k in list(state_dict.keys()):
        if k in model_state:
            if state_dict[k].shape != model_state[k].shape:
                mismatched_keys.append(k)
                del state_dict[k]

    if len(mismatched_keys) > 0:
        print(f"Detected {len(mismatched_keys)} shape mismatches. Removing them from state_dict.")

    incompatible = model.load_state_dict(state_dict, strict=False)

    if len(incompatible.missing_keys) > 0:
        print(f"Checkpoint load missing keys: {len(incompatible.missing_keys)}")
    if len(incompatible.unexpected_keys) > 0:
        print(f"Checkpoint load unexpected keys: {len(incompatible.unexpected_keys)}")


def main(mode, args):
    if mode in ["MIXED", "JUMP+FLOW"]:
        mode = "JUMP_FLOW"

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download."
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = False

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
        num_bins=getattr(args, 'num_bins', 128),
        jump_range=getattr(args, 'jump_range', 4.0),
    ).to(device)
    
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    if args.ckpt is not None and args.ckpt.lower() == "none":
        print("Skipping checkpoint loading, using randomly initialized model for testing...")
    else:
        ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
        state_dict = find_model(ckpt_path)
        load_pretrained_compatible(model, state_dict)
        
    model.eval()  # important!
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        bregman_type=args.bregman_type,
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse,
                jump_alpha=args.jump_alpha
            )
            
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    elif mode == "JUMP_FLOW":
        sample_fn = sampler.sample_jump_flow(
            num_steps=args.num_sampling_steps,
            stochastic_jump=args.stochastic_jump,
            jump_y_noise_scale=args.jump_y_noise_scale,
            jump_alpha=args.jump_alpha,
        )
    

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Load labels:
    if args.label_path and os.path.exists(args.label_path):
        with open(args.label_path, 'r') as f:
            data = json.load(f)
        all_labels = data['labels']
        print(f"Loaded {len(all_labels)} labels from {args.label_path}")
        
        # Distribute labels across GPUs if world_size > 1
        if args.world_size > 1:
            chunk_size = (len(all_labels) + args.world_size - 1) // args.world_size
            start_idx = args.rank * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_labels))
            class_labels = all_labels[start_idx:end_idx]
            print(f"Rank {args.rank} processing labels {start_idx} to {end_idx-1} (count: {len(class_labels)})")
        else:
            class_labels = all_labels
    else:
        # Default fallback labels
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
        print(f"Using default fallback labels: {class_labels}")
    
    if not class_labels:
        print(f"No labels for rank {args.rank}, exiting.")
        return

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    start_time = time()
    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
    # forward_with_cfg returns 2N batch: [guided_conditional | unconditional].
    # We must take only the first N guided samples for decoding.
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save images to the requested location.
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, str(out_path), nrow=8, normalize=True, value_range=(-1, 1))
    print(f"Saved samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE", "JUMP_FLOW", "MIXED", "JUMP+FLOW"], \
        "Invalid mode. Please choose 'ODE', 'SDE', or 'JUMP_FLOW'"
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--jump-range", type=float, default=3.0)
    parser.add_argument("--jump-alpha", type=float, default=0.5,
                        help="The alpha parameter used for continuous flow scaling.")
    parser.add_argument("--stochastic-jump", action=argparse.BooleanOptionalAction, default=False,
                        help="Sample jump landings from the learned Gaussian jump kernel.")
    parser.add_argument("--jump-y-noise-scale", type=float, default=1.0,
                        help="Scale multiplier for the learned jump std used when --stochastic-jump is enabled.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint.")
    parser.add_argument("--out-file", type=str, default="sample.png",
                        help="Output image path for this sampling run.")
    parser.add_argument("--label-path", type=str, default=None,
                        help="Path to used_labels.json")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current worker")
    parser.add_argument("--world-size", type=int, default=1, help="Total number of workers")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)
    elif mode in ["JUMP_FLOW", "MIXED", "JUMP+FLOW"]:
        parse_ode_args(parser)
    
    args = parser.parse_known_args()[0]
    main(mode, args)

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import json

from torch.utils.data import Dataset

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

class CustomDataset(Dataset):
    def __init__(self, features_dir):
        json_path = os.path.join(features_dir, "file_list.json")
        if os.path.exists(json_path):
            print(f"---> Loading file list from {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.features_dir = data['features_dir']
            self.labels_dir = data['labels_dir']
            self.features_files = data['features_files']
            self.labels_files = data['labels_files']
        else:
            # Fallback to slow os.listdir
            L = os.listdir(features_dir)
            print(f'---> Folders in {features_dir}: {L}')
            for name in L:
                if name.endswith('_features'):
                    self.features_dir = os.path.join(features_dir, name)
                elif name.endswith('_labels'):
                    self.labels_dir = os.path.join(features_dir, name)

            # Updated sorting for 0_0.npy style
            def sort_key(x):
                try:
                    parts = x.split('_')
                    batch_idx = int(parts[0])
                    rank_idx = int(parts[1].split('.')[0])
                    return batch_idx * 1000 + rank_idx
                except:
                    return x

            self.features_files = sorted(os.listdir(self.features_dir), key=sort_key)
            self.labels_files = sorted(os.listdir(self.labels_dir), key=sort_key)
            
            # Optionally cache to json here for next time?
            # data = {"features_dir": self.features_dir, "labels_dir": self.labels_dir, ...}
            # with open(json_path, 'w') as f: json.dump(...)

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, is_main_process):
    """
    Create a logger that writes to a log file and stdout.
    """
    if is_main_process:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Initialize Accelerator:
    accelerator = Accelerator(
        mixed_precision=getattr(args, 'mixed_precision', 'no'),
        log_with="wandb" if args.wandb else None
    )
    device = accelerator.device
    
    # Setup DDP:
    rank = accelerator.process_index
    seed = args.global_seed * accelerator.num_processes + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={accelerator.num_processes}.")
    local_batch_size = int(args.global_batch_size // accelerator.num_processes)

    # Setup an experiment folder:
    experiment_dir = None
    checkpoint_dir = None
    
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, True)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.wandb:
            entity = os.environ.get("ENTITY", "default")
            project = os.environ.get("PROJECT", "SiT-GM-moe")
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None, False)
    
    # Broadcast experiment_dir and checkpoint_dir to all processes
    # (Actually, we can just compute them on all processes if we handle the index carefully, 
    # but broadcasting is safer. Alternatively, just compute 'experiment_name' on all.)
    
    # Simpler: All processes can compute the names, but only rank 0 creates the dir.
    # To ensure index 000, 001 match, we should probably synchronize or let rank 0 decide.
    
    # Let's just gather the experiment_dir from rank 0:
    experiment_dir_list = [experiment_dir]
    dist.broadcast_object_list(experiment_dir_list, src=0)
    experiment_dir = experiment_dir_list[0]
    checkpoint_dir = f"{experiment_dir}/checkpoints"

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_bins=getattr(args, 'num_bins', 128),
        jump_range=getattr(args, 'jump_range', 4.0),
    ).to(device)

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model)  # Create an EMA of the model for use after training

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)


    requires_grad(ema, False)
    
    # We don't wrap EMA in Accelerator as it doesn't need optimization or distribution,
    # but we do want it on the correct device.
    ema = ema.to(device)
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    if args.feature_path is None or args.wandb:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    else:
        vae = None
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    if args.feature_path is None:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
        if hasattr(logger, 'info'):
            logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    else:
        if hasattr(logger, 'info'):
            logger.info(f"---> Preload Imagenet VAE features at {args.feature_path}...")
        dataset = CustomDataset(args.feature_path)
        if hasattr(logger, 'info'):
            logger.info(f"Dataset contains {len(dataset):,} features ({args.feature_path})")
    
    # Note: DistributedSampler is handled by Accelerator if we pass shuffle=True to DataLoader
    # but since this code manually creates it, we'll keep it and let Accelerator handle the preparation.
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    if args.feature_path is None:
        loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        # Features are pre-batched in .npy files (e.g. 32 per file)
        # We need to flatten the batch dimension when loading them
        def custom_collate(batch):
            # batch is a list of tuples: [(features_b1, labels_b1), (features_b2, labels_b2), ...]
            features = torch.cat([b[0] for b in batch], dim=0)
            labels = torch.cat([b[1] for b in batch], dim=0)
            return features, labels
            
        # Calculate how many pre-batched "files" to load per step to reach local_batch_size
        # Assuming each file has 32 items
        files_per_batch = max(1, local_batch_size // 32)
        
        loader = DataLoader(
            dataset,
            batch_size=files_per_batch,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=True
        )

    # Prepare everything with Accelerator:
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Prepare models for training:
    update_ema(ema, accelerator.unwrap_model(model), decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    start_epoch = 0
    log_steps = 0
    running_loss = 0
    running_loss_flow = 0
    running_loss_jump = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    
    # Resume or Initialize from checkpoint:
    if args.ckpt is not None:
        if os.path.isdir(args.ckpt):
            logger.info(f"Resuming from accelerate state folder: {args.ckpt}")
            accelerator.load_state(args.ckpt)
            
            # Load EMA separately (custom tracking)
            ema_path = os.path.join(args.ckpt, "ema.pt")
            if os.path.exists(ema_path):
                logger.info(f"Loading EMA from {ema_path}")
                ema.load_state_dict(torch.load(ema_path, map_location=device))
            else:
                logger.warning("EMA checkpoint not found in state directory.")
            
            # Restore training progress metadata
            state_info_path = os.path.join(args.ckpt, "training_state.json")
            if os.path.exists(state_info_path):
                with open(state_info_path, "r") as f:
                    state_info = json.load(f)
                    train_steps = state_info.get("train_steps", 0)
                    start_epoch = state_info.get("epoch", 0)
                    logger.info(f"Restored train_steps={train_steps}, epoch={start_epoch}")
            else:
                try:
                    train_steps = int(os.path.basename(args.ckpt))
                    logger.info(f"Extracted train_steps={train_steps} from directory name.")
                except:
                    pass
        elif os.path.isfile(args.ckpt):
            logger.info(f"Initializing from checkpoint file: {args.ckpt}")
            checkpoint = torch.load(args.ckpt, map_location=device)
            # Handle different checkpoint formats
            if "model" in checkpoint:
                accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
                if "ema" in checkpoint:
                    ema.load_state_dict(checkpoint["ema"])
                if "opt" in checkpoint:
                    opt.load_state_dict(checkpoint["opt"])
                # Extract steps from filename if possible
                try:
                    train_steps = int(os.path.basename(args.ckpt).split('.')[0])
                except:
                    pass
            else:
                # Direct weight loading (e.g. pretrained model)
                accelerator.unwrap_model(model).load_state_dict(checkpoint)
                update_ema(ema, accelerator.unwrap_model(model), decay=0)
        else:
            logger.error(f"Checkpoint path not found: {args.ckpt}")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            # Accelerator handles device placement
            if args.feature_path is None:
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            else:
                # Features are already encoded and scaled if they were generated correctly by fastdit script,
                # but depending on generation script they might not have the 0.18215 scale applied.
                # fast-DiT scale 0.18215 is applied during save. We assume they are ready for diffusion.
                pass
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, accelerator.unwrap_model(model))

            # Log loss values:
            running_loss += loss.item()
            if "loss_flow" in loss_dict:
                running_loss_flow += loss_dict["loss_flow"].item()
            if "loss_jump" in loss_dict:
                running_loss_jump += loss_dict["loss_jump"].item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Collective average across all GPUs (mathematically same as original all_reduce)
                log_stats = torch.tensor([running_loss, running_loss_flow, running_loss_jump], device=device)
                dist.all_reduce(log_stats, op=dist.ReduceOp.SUM)
                log_stats = log_stats / (accelerator.num_processes * log_steps)
                
                avg_loss, avg_loss_flow, avg_loss_jump = log_stats.tolist()

                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} (Flow: {avg_loss_flow:.4f}, Jump: {avg_loss_jump:.4f}), Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { 
                            "train loss": avg_loss, 
                            "train loss flow": avg_loss_flow,
                            "train loss jump": avg_loss_jump,
                            "train steps/sec": steps_per_sec 
                        },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                running_loss_flow = 0
                running_loss_jump = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}"
                accelerator.save_state(checkpoint_path)
                
                if accelerator.is_main_process:
                    # 1. Save EMA separately (for accelerate resume)
                    ema_path = f"{checkpoint_path}/ema.pt"
                    torch.save(ema.state_dict(), ema_path)
                    
                    # 2. Save training metadata
                    state_info = {"train_steps": train_steps, "epoch": epoch}
                    with open(f"{checkpoint_path}/training_state.json", "w") as f:
                        json.dump(state_info, f)
                        
                    # 3. CRITICAL: Save a standalone .pt file for direct inference (compatible with sample.py)
                    # This contains only the EMA weights, which is what find_model/sample.py expects
                    inference_ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(ema.state_dict(), inference_ckpt_path)
                    
                    logger.info(f"Saved accelerator state to {checkpoint_path}")
                    logger.info(f"Saved inference-ready model to {inference_ckpt_path}")
                
                accelerator.wait_for_everyone()
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    if getattr(args, 'sampler_type', 'ode') == "jump_flow":
                        sample_fn = transport_sampler.sample_jump_flow(num_steps=50)
                    else:
                        sample_fn = transport_sampler.sample_ode()
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    accelerator.wait_for_everyone()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    
                    if vae is not None:
                        # Decode in chunks to avoid OOM
                        decoded_samples = []
                        chunk_size = 8
                        for i in range(0, samples.shape[0], chunk_size):
                            chunk = samples[i:i+chunk_size]
                            decoded_chunk = vae.decode(chunk / 0.18215).sample
                            decoded_samples.append(decoded_chunk)
                        samples = torch.cat(decoded_samples, dim=0)

                        out_samples = accelerator.gather(samples)
                    else:
                        out_samples = None

                if args.wandb and out_samples is not None:
                    wandb_utils.log_image(out_samples, train_steps)
                logging.info("Generating EMA samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--feature-path", type=str, default=None, help="Path to precomputed VAE features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--jump-range", type=float, default=3.0)
    parser.add_argument("--sampler-type", type=str, default="ode",
                        choices=["ode", "jump_flow"])

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)

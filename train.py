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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
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

class RepeatedDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n
    def __len__(self):
        return len(self.dataset) * self.n
    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

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


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
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

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = max(1, int(args.global_batch_size // dist.get_world_size()))

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.wandb:
            entity = os.environ.get("ENTITY", "default")
            project = os.environ.get("PROJECT", "SiT-GM-moe")
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_bins=getattr(args, 'num_bins', 128),
        jump_range=getattr(args, 'jump_range', 4.0),
    ).to(device)

    # Freeze unused heads based on sampler-type:
    if args.sampler_type == "ode":
        print("Training ODE ONLY: freezing jump head.")
        requires_grad(model.final_layer_jump, False)
    elif args.sampler_type == "jump":
        print("Training JUMP ONLY: freezing flow head.")
        requires_grad(model.final_layer_flow, False)
    elif args.sampler_type == "jump_flow":
        print("Training BOTH flow and jump heads.")

    # Note that parameter initialization is done within the SiT constructor
    # No EMA needed: Schedule-Free handles internal weight averaging

    # Setup optimizer: ProdigyPlusScheduleFree (auto-tunes LR, no scheduler needed)
    opt = ProdigyPlusScheduleFree(
        model.parameters(), lr=1.0, betas=(0.95, 0.99),
        weight_decay=0.0, d0=1e-6, d_coef=1.0,
        use_stableadamw=True, use_schedulefree=True,
        split_groups=True, factored=True,
    )

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict["model"])
        if "opt" in state_dict:
            try:
                opt.load_state_dict(state_dict["opt"])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Starting with fresh optimizer.")
        old_args = args
        args = state_dict["args"]
        if not hasattr(args, 'sampler_type'):
            args.sampler_type = getattr(old_args, 'sampler_type', 'ode')
        if not hasattr(args, 'num_bins'):
            args.num_bins = getattr(old_args, 'num_bins', 128)
        if not hasattr(args, 'jump_range'):
            args.jump_range = getattr(old_args, 'jump_range', 4.0)
    
    model = DDP(model, device_ids=[device])
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

    # Limit dataset size if requested:
    if args.max_train_samples is not None:
        if args.feature_path is None:
            num_samples = min(len(dataset), args.max_train_samples)
        else:
            # Each item in CustomDataset is 32 images (pre-batched files)
            num_samples = min(len(dataset), max(1, args.max_train_samples // 32))
        
        # Use Subset for raw images, but for CustomDataset we can just slice files if we want,
        # however Subset is more generic for the sampler.
        from torch.utils.data import Subset
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)
        if hasattr(logger, 'info'):
            logger.info(f"Limited dataset to {num_samples} items (~{args.max_train_samples} samples)")

    # Repeat dataset if requested:
    if getattr(args, 'dataset_repeat', 1) > 1:
        dataset = RepeatedDataset(dataset, args.dataset_repeat)
        if hasattr(logger, 'info'):
            logger.info(f"Repeated dataset {args.dataset_repeat} times. Total items: {len(dataset):,}")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
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
            # If the cat-ed files have more samples than local_batch_size (e.g. 32 > 1), slice it.
            if features.shape[0] > local_batch_size:
                features = features[:local_batch_size]
                labels = labels[:local_batch_size]
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

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    opt.train()  # Schedule-Free: switch to training mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
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
        model_fn = model.module.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = model.module.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
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
            sampler_type = getattr(args, 'sampler_type', 'ode')
            if sampler_type == "ode":
                loss = loss_dict["loss_flow"].mean()
            elif sampler_type == "jump":
                loss = loss_dict["loss_jump"].mean()
            else:  # "jump_flow"
                loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # No EMA update needed: Schedule-Free handles averaging internally

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
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss_flow = torch.tensor(running_loss_flow / log_steps, device=device)
                avg_loss_jump = torch.tensor(running_loss_jump / log_steps, device=device)
                
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_flow, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_jump, op=dist.ReduceOp.SUM)
                
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_loss_flow = avg_loss_flow.item() / dist.get_world_size()
                avg_loss_jump = avg_loss_jump.item() / dist.get_world_size()
                
                # Fetch Prodigy Schedule-Free dynamic learning rate correctly
                group = opt.param_groups[0]
                d_val = group.get('d', 1.0)
                effective_lr = group.get('effective_lr', group.get('lr', 1.0))
                current_lr = d_val * effective_lr
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} (Flow: {avg_loss_flow:.4f}, Jump: {avg_loss_jump:.4f}), Train Steps/Sec: {steps_per_sec:.2f}, LR: {current_lr:.6e}")
                if args.wandb:
                    wandb_utils.log(
                        { 
                            "train loss": avg_loss, 
                            "train loss flow": avg_loss_flow,
                            "train loss jump": avg_loss_jump,
                            "train steps/sec": steps_per_sec,
                            "lr": current_lr
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
                # Schedule-Free: must call opt.eval() before saving model weights
                # so that model.state_dict() reflects the averaged weights (not training buffer)
                opt.eval()
                model.eval()
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
                model.train()
                opt.train()  # Schedule-Free: switch back to training mode
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating samples...")
                opt.eval()  # Schedule-Free: switch to eval mode for sampling
                model.eval()
                with torch.no_grad():
                    sampler_type = getattr(args, 'sampler_type', 'ode')
                    if sampler_type in ["jump_flow", "jump"]:
                        sample_fn = transport_sampler.sample_jump_flow(num_steps=50)
                    else:  # "ode"
                        sample_fn = transport_sampler.sample_ode()
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

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

                        out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                        dist.all_gather_into_tensor(out_samples, samples)
                    else:
                        out_samples = None

                if args.wandb and out_samples is not None:
                    wandb_utils.log_image(out_samples, train_steps)
                model.train()
                opt.train()  # Schedule-Free: switch back to training mode
                logging.info("Generating samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    opt.eval()  # Schedule-Free: switch to eval mode
    # do any sampling/FID calculation/etc. with model in eval mode ...

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
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit the number of training samples (e.g. 512)")
    parser.add_argument("--dataset-repeat", type=int, default=1,
                        help="Repeat the dataset N times for longer epochs")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--jump-range", type=float, default=3.0)
    parser.add_argument("--sampler-type", type=str, default="ode",
                        choices=["ode", "jump_flow", "jump"])

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)

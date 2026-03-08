# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using HF Accelerate.
"""
import wandb_utils
from train_utils import parse_transport_args
from diffusers.models import AutoencoderKL
from transport import create_transport, Sampler
from download import find_model
from models import SiT_models
from torch.utils.data import Dataset
import json
import os
import logging
import argparse
from time import time
from glob import glob
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
from PIL import Image
from collections import OrderedDict
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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

            self.features_files = sorted(
                os.listdir(self.features_dir), key=sort_key)
            self.labels_files = sorted(
                os.listdir(self.labels_dir), key=sort_key)

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
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def load_pretrained_compatible(model, state_dict, logger):
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
        # Keep backbone weights and keep new heads initialized by current model init.
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("final_layer.")
        }
        logger.info(
            "Detected legacy single-head checkpoint (final_layer.*). "
            "Loaded backbone only; keeping final_layer_flow/jump as current init."
        )

    # Check and remove shape mismatches before strict=False loading.
    # strict=False handles missing/unexpected keys, but crashes on shape mismatch.
    model_state = model.state_dict()
    mismatched_keys = []
    for k in list(state_dict.keys()):
        if k in model_state:
            if state_dict[k].shape != model_state[k].shape:
                mismatched_keys.append(k)
                del state_dict[k]

    if len(mismatched_keys) > 0:
        logger.info(
            f"Detected {len(mismatched_keys)} shape mismatches (e.g., {mismatched_keys[0]}). "
            "These keys were removed from state_dict; keeping current model initialization for them."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)

    if len(incompatible.missing_keys) > 0:
        logger.info(
            f"Checkpoint load missing keys: {len(incompatible.missing_keys)} (expected for head/layout changes)."
        )
    if len(incompatible.unexpected_keys) > 0:
        logger.info(
            f"Checkpoint load unexpected keys: {len(incompatible.unexpected_keys)} (ignored)."
        )


def create_logger(logging_dir, is_main_process):
    """
    Create a logger that writes to a log file and stdout.
    """
    if is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(
                f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
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
    Trains a new SiT model using HF Accelerate.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup Accelerate:
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device
    set_seed(args.global_seed)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    assert args.global_batch_size % world_size == 0, \
        f"Batch size {args.global_batch_size} must be divisible by world size {world_size}."
    local_batch_size = max(1, int(args.global_batch_size // world_size))

    print(
        f"Starting rank={rank}, seed={args.global_seed}, world_size={world_size}.")

    # Setup an experiment folder:
    if is_main:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
            f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, is_main)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.wandb:
            entity = os.environ.get("ENTITY", "default")
            project = os.environ.get("PROJECT", "SiT-GM-moe")
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None, is_main)
        experiment_dir = None
        checkpoint_dir = None

    # Broadcast experiment_dir and checkpoint_dir to all processes
    # We use a simple file-based approach: main process writes, others read
    import torch.distributed as dist
    if accelerator.num_processes > 1:
        # Share the paths via broadcast
        if is_main:
            path_info = [experiment_dir, checkpoint_dir]
        else:
            path_info = [None, None]
        # Use accelerator's gather or a simple object broadcast
        import pickle
        if is_main:
            path_bytes = pickle.dumps(path_info)
            path_tensor = torch.tensor(
                list(path_bytes), dtype=torch.uint8, device=device)
            size_tensor = torch.tensor(
                [len(path_bytes)], dtype=torch.long, device=device)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)

        dist.broadcast(size_tensor, src=0)
        size = size_tensor.item()

        if not is_main:
            path_tensor = torch.zeros(size, dtype=torch.uint8, device=device)
        dist.broadcast(path_tensor, src=0)

        if not is_main:
            path_info = pickle.loads(bytes(path_tensor.cpu().tolist()))
            experiment_dir, checkpoint_dir = path_info

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_bins=getattr(args, 'num_bins', 128),
        jump_range=getattr(args, 'jump_range', 4.0),
    )
    if args.gradient_checkpointing and hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
        logger.info("Enabled gradient checkpointing.")

    # Freeze unused heads based on sampler-type:
    if args.sampler_type == "ode":
        print("Training ODE ONLY: freezing jump head.")
        requires_grad(model.final_layer_jump, False)
    elif args.sampler_type == "jump":
        print("Training JUMP ONLY: freezing flow head.")
        requires_grad(model.final_layer_flow, False)
    elif args.sampler_type == "jump_flow":
        print("Training BOTH flow and jump heads.")

    # Setup optimizer: ProdigyPlusScheduleFree
    opt = ProdigyPlusScheduleFree(
        model.parameters(), lr=1.0, betas=(0.95, 0.99),
        weight_decay=0.0, d0=1e-6, d_coef=1.0,
        use_stableadamw=True, use_schedulefree=True,
        split_groups=True, factored=True,
    )

    # Resume from Accelerate checkpoint directory
    train_steps = 0
    start_epoch = 0
    if args.resume is not None:
        # Load metadata first (train_steps, epoch)
        meta_path = os.path.join(args.resume, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            train_steps = meta.get("train_steps", 0)
            start_epoch = meta.get("epoch", 0)
            logger.info(
                f"Resuming from step {train_steps}, epoch {start_epoch}")

    # Load pretrained weights (without optimizer state, for fine-tuning)
    ckpt_path = args.ckpt
    if ckpt_path is None and args.resume is None:
        assert args.model == "SiT-XL/2", \
            "Only SiT-XL/2 is available for default auto-download. Pass --ckpt for custom models."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        # Current downloader only provides the 256x256 checkpoint.
        assert args.image_size == 256, \
            "Default auto-download currently supports only 256x256. Pass --ckpt for other sizes."
        ckpt_path = f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
        logger.info(
            f"No --ckpt provided. Auto-downloading default pre-trained checkpoint: {ckpt_path}")

    if ckpt_path is not None:
        state_dict = find_model(ckpt_path)
        load_pretrained_compatible(model, state_dict, logger)
        logger.info(f"Loaded pretrained weights from {ckpt_path}")

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        bregman_type=args.bregman_type,
        time_schedule=args.time_schedule,
    )
    transport_sampler = Sampler(transport)
    if args.feature_path is None or args.wandb:
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    else:
        vae = None
    logger.info(
        f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    if args.feature_path is None:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(
                pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
                                 0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
        logger.info(
            f"Dataset contains {len(dataset):,} images ({args.data_path})")
    else:
        logger.info(
            f"---> Preload Imagenet VAE features at {args.feature_path}...")
        dataset = CustomDataset(args.feature_path)
        logger.info(
            f"Dataset contains {len(dataset):,} features ({args.feature_path})")

    subset_indices = None
    # Limit dataset size if requested:
    if args.max_train_samples is not None:
        if args.feature_path is None:
            num_samples = min(len(dataset), args.max_train_samples)
            approx_samples = num_samples
        else:
            example_features, _ = dataset[0]
            samples_per_file = max(1, int(example_features.shape[0]))
            num_files = max(
                1, (args.max_train_samples + samples_per_file - 1) // samples_per_file)
            num_samples = min(len(dataset), num_files)
            approx_samples = num_samples * samples_per_file

        from torch.utils.data import Subset
        subset_generator = torch.Generator()
        subset_generator.manual_seed(args.global_seed)
        indices = torch.randperm(
            len(dataset), generator=subset_generator).tolist()[:num_samples]
        subset_indices = [int(i) for i in indices]
        dataset = Subset(dataset, indices)
        logger.info(
            f"Limited dataset to {num_samples} items (~{approx_samples} samples), random subset with seed={args.global_seed}")

    # Record labels used by the limited random subset for reproducibility/debugging.
    if is_main and args.max_train_samples is not None and subset_indices is not None:
        try:
            used_labels = []
            if args.feature_path is None:
                for idx in subset_indices:
                    _, label = dataset.dataset[idx]
                    used_labels.append(int(label))
            else:
                base_dataset = dataset.dataset
                for idx in subset_indices:
                    label_file = base_dataset.labels_files[idx]
                    label_path = os.path.join(
                        base_dataset.labels_dir, label_file)
                    labels_np = np.load(label_path)
                    used_labels.extend(np.asarray(
                        labels_np).reshape(-1).astype(np.int64).tolist())

            used_labels_path = os.path.join(experiment_dir, "used_labels.json")
            with open(used_labels_path, "w") as f:
                json.dump(
                    {
                        "max_train_samples": int(args.max_train_samples),
                        "subset_indices": subset_indices,
                        "num_labels": int(len(used_labels)),
                        "labels": [int(v) for v in used_labels],
                    },
                    f,
                    indent=2,
                )
            logger.info(
                f"Saved used labels to {used_labels_path} (num_labels={len(used_labels):,})")
        except Exception as e:
            logger.warning(f"Failed to record used labels: {e}")

    # Repeat dataset if requested:
    if getattr(args, 'dataset_repeat', 1) > 1:
        dataset = RepeatedDataset(dataset, args.dataset_repeat)
        logger.info(
            f"Repeated dataset {args.dataset_repeat} times. Total items: {len(dataset):,}")

    if args.feature_path is None:
        loader_kwargs = dict(
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(
            dataset,
            **loader_kwargs
        )
    else:
        def custom_collate(batch):
            features = torch.cat([b[0] for b in batch], dim=0)
            labels = torch.cat([b[1] for b in batch], dim=0)
            if features.shape[0] <= 0:
                raise ValueError(
                    "Loaded empty feature batch from feature files.")

            if features.shape[0] >= local_batch_size:
                selected = torch.randperm(features.shape[0])[:local_batch_size]
            else:
                extra = torch.randint(
                    0, features.shape[0], (local_batch_size - features.shape[0],))
                selected = torch.cat(
                    [torch.arange(features.shape[0]), extra], dim=0)

            return features[selected], labels[selected]

        example_features, _ = dataset[0]
        samples_per_file = max(1, int(example_features.shape[0]))
        files_per_batch = max(
            1, (local_batch_size + samples_per_file - 1) // samples_per_file)

        loader_kwargs = dict(
            batch_size=files_per_batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=True,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(
            dataset,
            **loader_kwargs
        )

    # Prepare with Accelerate (handles DDP wrapping, device placement, dataloader sharding)
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Compile the base model to speed up training
    base_model = accelerator.unwrap_model(model)
    if not hasattr(base_model, "_orig_mod"):
        # Compile if not already compiled
        import torch._dynamo as dynamo
        dynamo.config.suppress_errors = True
        base_model = torch.compile(base_model, mode="max-autotune")

        # We need to re-wrap the compiled model with DDP
        # For simplicity in Accelerate, we often just assign the compiled block back to the DDP module if possible,
        # but the safest way in standard PyTorch 2 is to compile AFTER DDP, which we just did on the unwrapped model.
        # So we update the DDP module's wrapped model:
        if hasattr(model, "module"):
            model.module = base_model
        else:
            model = base_model
    logger.info("Model compiled with torch.compile().")

    # Load Accelerate state (after prepare)
    if args.resume is not None:
        accelerator.load_state(args.resume)
        logger.info(f"Loaded Accelerate state from {args.resume}")
        group = opt.param_groups[0]
        d_val = group.get('d', 1.0)
        effective_lr = group.get('effective_lr', group.get('lr', 1.0))
        logger.info(
            "Resume optimizer state (group0): "
            f"lr={group.get('lr', None)}, d={d_val}, "
            f"effective_lr={effective_lr}, d*effective_lr={d_val * effective_lr:.9e}"
        )

    # Prepare models for training:
    model.train()
    opt.train()  # Schedule-Free: switch to training mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_loss_flow = 0
    running_loss_jump = 0
    running_loss_jump_lambda = 0
    running_loss_jump_mu = 0
    running_mae = 0
    running_lambda_theta = 0
    running_lambda_target = 0
    start_time = time()

    # Keep periodic training-time sampling deterministic and guidance-free.
    # This stays fixed at CFG=1.0 regardless of CLI --cfg-scale.
    training_sample_cfg_scale = 1.0
    use_cfg = training_sample_cfg_scale > 1.0
    # Keep latent noise fixed across periodic samples for easier visual comparison.
    base_zs = torch.randn(local_batch_size, 4, latent_size,
                          latent_size, device=device)
    # Using the compiled unwrapped model
    model_fn = base_model.forward_with_cfg if use_cfg else base_model.forward
    # For periodic sampling during training, prefer the uncompiled module when available.
    # This avoids torch.compile/CUDAGraph buffer reuse issues across repeated model calls.
    sample_base_model = base_model._orig_mod if hasattr(
        base_model, "_orig_mod") else base_model
    sample_model_fn = sample_base_model.forward_with_cfg if use_cfg else sample_base_model.forward

    logger.info(
        f"Training for {args.epochs} epochs (resuming from step {train_steps})...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if args.feature_path is None:
                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)
            with accelerator.accumulate(model):
                loss_dict = transport.training_losses(model, x, model_kwargs)
                sampler_type = getattr(args, 'sampler_type', 'ode')
                if sampler_type == "ode":
                    loss = loss_dict["loss_flow"].mean()
                elif sampler_type == "jump":
                    loss = loss_dict["loss_jump"].mean()
                else:  # "jump_flow"
                    loss = loss_dict["loss"].mean()

                opt.zero_grad()
                accelerator.backward(loss)
                opt.step()

            # Log loss values:
            running_loss += loss.item()
            if "loss_flow" in loss_dict:
                running_loss_flow += loss_dict["loss_flow"].item()
            if "loss_jump" in loss_dict:
                running_loss_jump += loss_dict["loss_jump"].item()
            if "loss_jump_lambda" in loss_dict:
                running_loss_jump_lambda += loss_dict["loss_jump_lambda"].item()
            if "loss_jump_mu" in loss_dict:
                running_loss_jump_mu += loss_dict["loss_jump_mu"].item()
            if "mae" in loss_dict:
                running_mae += loss_dict["mae"].item()
            if "lambda_theta" in loss_dict:
                running_lambda_theta += loss_dict["lambda_theta"].item()
            if "lambda_target" in loss_dict:
                running_lambda_target += loss_dict["lambda_target"].item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                # Use local rank values directly (no cross-GPU reduce)
                # This avoids 8 extra AllReduce ops per log interval.
                # Local averages are statistically representative enough.
                avg_loss = running_loss / log_steps
                avg_loss_flow = running_loss_flow / log_steps
                avg_loss_jump = running_loss_jump / log_steps
                avg_loss_jump_lambda = running_loss_jump_lambda / log_steps
                avg_loss_jump_mu = running_loss_jump_mu / log_steps
                avg_mae = running_mae / log_steps
                avg_lambda_theta = running_lambda_theta / log_steps
                avg_lambda_target = running_lambda_target / log_steps

                # Fetch Prodigy Schedule-Free dynamic learning rate correctly
                group = opt.param_groups[0]
                d_val = group.get('d', 1.0)
                effective_lr = group.get('effective_lr', group.get('lr', 1.0))
                current_lr = d_val * effective_lr

                logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f} (Flow: {avg_loss_flow:.4f}, Jump: {avg_loss_jump:.4f}), L_lam: {avg_loss_jump_lambda:.4f}, L_mu: {avg_loss_jump_mu:.4f}, lam: {avg_lambda_theta:.2f}/{avg_lambda_target:.2f}, mae: {avg_mae:.2f}, LR: {current_lr:.2e}")
                if args.wandb:
                    wandb_utils.log(
                        {
                            "train loss": avg_loss,
                            "train loss flow": avg_loss_flow,
                            "train loss jump": avg_loss_jump,
                            "train loss jump lambda": avg_loss_jump_lambda,
                            "train loss jump mu": avg_loss_jump_mu,
                            "train mae": avg_mae,
                            "train lambda theta": avg_lambda_theta,
                            "train lambda target": avg_lambda_target,
                            "train_steps_per_sec": steps_per_sec,
                            "lr": current_lr
                        },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                running_loss_flow = 0
                running_loss_jump = 0
                running_loss_jump_lambda = 0
                running_loss_jump_mu = 0
                running_mae = 0
                running_lambda_theta = 0
                running_lambda_target = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                # Schedule-Free: must call opt.eval() before saving model weights
                opt.eval()
                model.eval()

                # Save Accelerate state (model, optimizer, dataloader, RNG)
                ckpt_dir = f"{checkpoint_dir}/{train_steps:07d}"
                accelerator.save_state(ckpt_dir)

                # Save metadata (train_steps, epoch) for resume
                if is_main:
                    meta = {"train_steps": train_steps, "epoch": epoch}
                    with open(os.path.join(ckpt_dir, "metadata.json"), 'w') as f:
                        json.dump(meta, f)

                    # Also save standalone .pt for sampling compatibility
                    # NOTE:
                    # accelerate.unwrap_model() may hit KeyError('_orig_mod')
                    # when torch.compile regions exist but the top-level wrapper
                    # does not expose _orig_mod. Fall back safely.
                    try:
                        unwrapped_model = accelerator.unwrap_model(model)
                    except KeyError as e:
                        if "_orig_mod" not in str(e):
                            raise
                        logger.warning(
                            "unwrap_model hit KeyError('_orig_mod'); "
                            "falling back to raw module for standalone save."
                        )
                        if hasattr(model, "module"):
                            unwrapped_model = model.module
                        else:
                            unwrapped_model = base_model

                    # If wrapped by torch.compile, save the original module state.
                    if hasattr(unwrapped_model, "_orig_mod"):
                        unwrapped_model = unwrapped_model._orig_mod

                    standalone_path = os.path.join(ckpt_dir, "model.pt")
                    torch.save(unwrapped_model.state_dict(), standalone_path)
                    logger.info(
                        f"Saved checkpoint to {ckpt_dir} (Accelerate state + model.pt)")

                accelerator.wait_for_everyone()
                model.train()
                opt.train()  # Schedule-Free: switch back to training mode
                # Exclude checkpoint I/O time from subsequent step/sec measurement.
                log_steps = 0
                start_time = time()

            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating samples...")
                opt.eval()
                model.eval()
                with torch.no_grad():
                    # Force sampling labels to come from real training labels.
                    if y.shape[0] >= local_batch_size:
                        label_idx = torch.randperm(y.shape[0], device=y.device)[
                            :local_batch_size]
                    else:
                        label_idx = torch.randint(
                            0, y.shape[0], (local_batch_size,), device=y.device)
                    ys = y[label_idx]

                    if use_cfg:
                        zs = torch.cat([base_zs, base_zs], 0)
                        y_null = torch.full(
                            (local_batch_size,), args.num_classes, device=device, dtype=ys.dtype)
                        sample_model_kwargs = dict(
                            y=torch.cat([ys, y_null], 0),
                            cfg_scale=training_sample_cfg_scale,
                        )
                    else:
                        zs = base_zs
                        sample_model_kwargs = dict(y=ys)

                    sampler_type = getattr(args, 'sampler_type', 'ode')
                    if sampler_type == "jump":
                        sample_fn = transport_sampler.sample_jump_flow(
                            num_steps=250,
                            pure_jump=True,
                            stochastic_jump=False)
                    elif sampler_type == "jump_flow":
                        sample_fn = transport_sampler.sample_jump_flow(
                            num_steps=250,
                            pure_jump=False,
                            stochastic_jump=False)
                    else:
                        sample_fn = transport_sampler.sample_ode()
                    samples = sample_fn(
                        zs, sample_model_fn, **sample_model_kwargs)[-1]
                    accelerator.wait_for_everyone()

                    if use_cfg:
                        samples, _ = samples.chunk(2, dim=0)

                    if vae is not None:
                        decoded_samples = []
                        chunk_size = 8
                        for i in range(0, samples.shape[0], chunk_size):
                            chunk = samples[i:i+chunk_size]
                            decoded_chunk = vae.decode(chunk / 0.18215).sample
                            decoded_samples.append(decoded_chunk)
                        samples = torch.cat(decoded_samples, dim=0)

                        out_samples = torch.zeros(
                            (args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                        # Gather samples from all processes
                        gathered = accelerator.gather(samples)
                        if is_main:
                            out_samples = gathered
                    else:
                        out_samples = None

                if args.wandb and out_samples is not None and is_main:
                    wandb_utils.log_image(out_samples, train_steps)
                model.train()
                opt.train()
                logger.info("Generating samples done.")
                # Exclude sampling/visualization time from subsequent step/sec measurement.
                log_steps = 0
                start_time = time()

    model.eval()
    opt.eval()

    logger.info("Done!")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--feature-path", type=str, default=None,
                        help="Path to precomputed VAE features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str,
                        choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int,
                        choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str,
                        choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit the number of training samples (e.g. 512)")
    parser.add_argument("--dataset-repeat", type=int, default=1,
                        help="Repeat the dataset N times for longer epochs")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to pretrained model weights (.pt)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an Accelerate checkpoint directory to resume training from")
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--jump-range", type=float, default=3.0)
    parser.add_argument("--sampler-type", type=str, default="ode",
                        choices=["ode", "jump_flow", "jump"])
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce GPU memory usage.")
    parser.add_argument("--mixed-precision", type=str, default=None,
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training. Defaults to 'no' or what's in 'accelerate launch'.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating.")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)

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
from copy import deepcopy
from time import time
from glob import glob
from PIL import Image
from collections import OrderedDict
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True # 开启 cuDNN 自带的算法搜索
import torch._inductor.config as inductor_config


#################################################################################
#                             Optimizer Functions                               #
#################################################################################

try:
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
except ImportError:
    # Fallback placeholder if not installed
    MuonWithAuxAdam = None
    SingleDeviceMuonWithAuxAdam = None

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


class MemmapDataset(Dataset):
    def __init__(self, features_file, labels_file):
        print(f"Loading MemmapDataset from {features_file}")
        self.features = np.load(features_file, mmap_mode='r')
        self.labels = np.load(labels_file, mmap_mode='r')

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Using .copy() avoids the "non-writable" warning and is safe for small slices
        return torch.from_numpy(self.features[idx].copy()), torch.as_tensor(self.labels[idx])


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
        # Map legacy final_layer.* to final_layer_flow.*
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("final_layer."):
                new_key = k.replace("final_layer.", "final_layer_flow.")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        logger.info(
            "Detected legacy single-head checkpoint (final_layer.*). "
            "Mapped to final_layer_flow.*; keeping final_layer_jump as current init."
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device
    set_seed(args.global_seed)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    assert args.global_batch_size % (world_size * args.gradient_accumulation_steps) == 0, \
        f"Batch size {args.global_batch_size} must be divisible by world size {world_size} * accumulation {args.gradient_accumulation_steps}."
    local_batch_size = max(1, int(
        args.global_batch_size // (world_size * args.gradient_accumulation_steps)))

    print(
        f"Starting rank={rank}, seed={args.global_seed}, world_size={world_size}. cuDNN={torch.backends.cudnn.version()}")

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

    # Print model detail and backend status
    print(f"\n{'='*50}\nModel Data Type: {next(model.parameters()).dtype}\n{'='*50}")

    # Since we are not using torch.compile, we are in Eager Mode
    print(f"\n{'#'*50}")
    print(f"# Execution Mode: Eager Mode (Standard PyTorch)")
    print(f"# cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"# cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"# Note: Standard cuDNN/cuBLAS kernels are used for acceleration.")
    print(f"{'#'*50}\n")

    # Check Attention Suitability based on PyTorch Blog
    gpu_name = torch.cuda.get_device_name(device)
    is_blackwell = "Blackwell" in gpu_name or "B100" in gpu_name or "B200" in gpu_name
    is_hopper = "H100" in gpu_name or "H200" in gpu_name
    
    print(f"\n{'#'*50}")
    print("# Attention Interface Suitability Check (Ref: pytorch.org/blog/flexattention-flashattention-4)")
    print(f"{'#'*50}")
    print(f"# GPU: {gpu_name}")
    print("# Model Attention Pattern: Standard Bidirectional (Dense/Noop)")
    print("# Analysis:")
    
    if is_blackwell:
        print("#  [CRITICAL] Blackwell GPU detected.")
        print("#  - Insight: Triton-based FlexAttention is significantly slower on Blackwell.")
        print("#  - Insight: 'FlexAttention (Flash Backend) matches cuDNN performance for Noop'.")
        print("#  - Insight: 'Forward pass: Noop matches cuDNN closely'.")
        print("#")
        print("#  >>> RECOMMENDATION: Use SDPA (current default) OR FlexAttention with BACKEND='FLASH'.")
        print("#      Do NOT use Triton backend for FlexAttention.")
    elif is_hopper:
        print("#  [INFO] Hopper GPU detected.")
        print("#  - Insight: FlashAttention-3/4 provides best performance.")
        print("#  >>> RECOMMENDATION: SDPA (uses FlashAttn internally) or FlexAttention (FLASH backend).")
    else:
        print("#  [INFO] Pre-Hopper GPU detected.")
        print("#  - Insight: Standard SDPA or FlashAttention-2 is sufficient.")
    
    print(f"{'#'*50}\n")

    if args.gradient_checkpointing and hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
        logger.info("Enabled gradient checkpointing.")

    # Freeze unused heads based on sampler-type:
    if args.sampler_type == "ode":
        print("Training ODE ONLY: freezing jump heads.")
        requires_grad(model.final_layer_jump, False)
    elif args.sampler_type == "jump":
        print("Training JUMP ONLY: freezing flow head.")
        requires_grad(model.final_layer_flow, False)
    elif args.sampler_type == "jump_flow":
        print("Training BOTH flow and jump heads.")

    # Filter parameters for layered learning rates (standard format for muon library)
    muon_params = []
    embedding_params = []
    head_params = []
    aux_adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        if "embed" in name:
            embedding_params.append(p)
        elif "final_layer" in name:
            head_params.append(p)
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            aux_adam_params.append(p)

    # Re-organize into official param_groups format
    param_groups = [
        {
            "params": muon_params, 
            "lr": args.muon_lr, 
            "momentum": args.muon_momentum, 
            "weight_decay": args.muon_weight_decay, 
            "use_muon": True
        },
        {
            "params": embedding_params, 
            "lr": args.embed_lr, 
            "betas": (args.aux_adam_beta1, args.aux_adam_beta2), 
            "eps": args.aux_adam_eps, 
            "weight_decay": args.aux_adam_weight_decay, 
            "use_muon": False
        },
        {
            "params": head_params, 
            "lr": args.head_lr, 
            "betas": (args.aux_adam_beta1, args.aux_adam_beta2), 
            "eps": args.aux_adam_eps, 
            "weight_decay": args.aux_adam_weight_decay, 
            "use_muon": False
        },
        {
            "params": aux_adam_params, 
            "lr": args.aux_adam_lr, 
            "betas": (args.aux_adam_beta1, args.aux_adam_beta2), 
            "eps": args.aux_adam_eps, 
            "weight_decay": args.aux_adam_weight_decay, 
            "use_muon": False
        },
    ]

    # Initialize the official hybrid optimizer
    if accelerator.num_processes > 1:
        opt = MuonWithAuxAdam(param_groups)
        logger.info(f"Initialized Distributed MuonWithAuxAdam (momentum={args.muon_momentum})")
    else:
        opt = SingleDeviceMuonWithAuxAdam(param_groups)
        logger.info(f"Initialized SingleDeviceMuonWithAuxAdam (momentum={args.muon_momentum})")
    
    logger.info(f"Layered LRs: Muon: {args.muon_lr}, Embed: {args.embed_lr}, Head: {args.head_lr}, Aux: {args.aux_adam_lr}")

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
    if ckpt_path is not None:
        state_dict = find_model(ckpt_path)
        load_pretrained_compatible(model, state_dict, logger)
        logger.info(f"Loaded pretrained weights from {ckpt_path}")
    elif args.resume is None:
        logger.info(
            "No --ckpt and no --resume provided. Training from scratch.")

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        bregman_type=args.bregman_type,
    )
    transport_sampler = Sampler(transport)
    logger.info(
        "Using fixed 1:1 Flow/Jump weighting for Markov Superposition."
    )
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
        merged_features_path = os.path.join(
            args.feature_path, "merged_features.npy")
        merged_labels_path = os.path.join(
            args.feature_path, "merged_labels.npy")
        if os.path.exists(merged_features_path) and os.path.exists(merged_labels_path):
            dataset = MemmapDataset(merged_features_path, merged_labels_path)
            logger.info(
                f"Using high-performance MemmapDataset: {len(dataset):,} total features")
        else:
            logger.info(
                f"---> Preload Imagenet VAE features at {args.feature_path}...")
            dataset = CustomDataset(args.feature_path)
            logger.info(
                f"Dataset contains {len(dataset):,} features ({args.feature_path})")

    # Limit dataset size if requested:
    if args.max_train_samples is not None:
        num_samples = min(len(dataset), args.max_train_samples)

        from torch.utils.data import Subset
        import random
        subset_generator = torch.Generator()
        subset_generator.manual_seed(args.global_seed)
        indices = torch.randperm(len(dataset), generator=subset_generator).tolist()[
            :num_samples]
        subset_indices = [int(i) for i in indices]
        dataset = Subset(dataset, indices)
        logger.info(
            f"Limited dataset to {num_samples} items, random subset seed={args.global_seed}")

    # Safety Guard: Check if repeat factor causes massive index lists (> 5M items)
    base_len = len(dataset)
    repeat_factor = getattr(args, 'dataset_repeat', 1)
    if repeat_factor > 1:
        if base_len > 200_000:
            logger.warning(f"Dataset is too large ({base_len:,}) to repeat {repeat_factor}x. "
                           "To prevent memory OOM in Sampler, forcing repeat=1. "
                           "Please increase --epochs instead.")
            args.dataset_repeat = 1
        elif base_len * repeat_factor > 5_000_000:
            new_repeat = 5_000_000 // base_len
            logger.warning(f"Repeat factor {repeat_factor} results in {base_len * repeat_factor:,} items. "
                           f"Capping repeat to {new_repeat} to save memory.")
            args.dataset_repeat = new_repeat

    # Repeat dataset for longer epochs
    if getattr(args, 'dataset_repeat', 1) > 1:
        dataset = RepeatedDataset(dataset, args.dataset_repeat)
        logger.info(
            f"Repeating dataset {args.dataset_repeat} times. Effective length: {len(dataset):,}")

    # Record labels used by the limited random subset for reproducibility/debugging.
    if is_main and args.max_train_samples is not None and subset_indices is not None:
        try:
            used_labels = []
            # Find the actual data-holding dataset
            _base = dataset
            while hasattr(_base, 'dataset'):
                _base = _base.dataset

            if args.feature_path is None:
                for idx in subset_indices:
                    _, label = _base[idx]
                    used_labels.append(int(label))
            else:
                for idx in subset_indices:
                    if hasattr(_base, 'labels_files'):  # CustomDataset
                        label_file = _base.labels_files[idx]
                        label_path = os.path.join(_base.labels_dir, label_file)
                        labels_np = np.load(label_path)
                    else:  # MemmapDataset
                        labels_np = _base.labels[idx]

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

    _base_dataset = dataset
    while hasattr(_base_dataset, 'dataset'):
        _base_dataset = _base_dataset.dataset

    is_memmap_or_image = args.feature_path is None or isinstance(
        _base_dataset, MemmapDataset)

    if is_memmap_or_image:
        loader_kwargs = dict(
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
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

    # torch.compile — must happen BEFORE accelerator.prepare() so that the
    # compiled graph sees the raw module, not the DDP/DeepSpeed wrapper.
    if getattr(args, 'compile', False):
        compile_mode = getattr(args, 'compile_mode', 'reduce-overhead')

        # Apply inductor tuning knobs only if supported by the current torch build.
        inductor_toggles = {
            "max_autotune": True,
            "cudnn_prologue_fusion": True,
            "epilogue_fusion": True,
            "shape_padding": True,
        }
        for key, value in inductor_toggles.items():
            if hasattr(inductor_config, key):
                setattr(inductor_config, key, value)
            else:
                logger.warning(
                    f"torch._inductor.config.{key} is unavailable in this torch version; skipping."
                )
        
        logger.info(f"Compiling model with torch.compile (mode='{compile_mode}') ...")
        # model = torch.compile(model, mode=compile_mode)
        logger.info("torch.compile() done.")

    # Prepare with Accelerate (handles DDP wrapping, device placement, dataloader sharding)
    model, opt, loader = accelerator.prepare(model, opt, loader)
    
    # Check and log the actual model precision in a very prominent way
    if is_main:
        try:
            model_dtype = next(model.parameters()).dtype
            logger.info("\n" + "="*80 + f"\n\n    ACTUAL MODEL PRECISION: {model_dtype}\n\n" + "="*80 + "\n")
        except Exception as e:
            logger.warning(f"Could not determine model dtype: {e}")

    # Enable gradient communication compression hook (massive PCIe bandwidth savings)
    if accelerator.num_processes > 1 and hasattr(model, 'register_comm_hook'):
        import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as comm_hooks
        if accelerator.mixed_precision == 'bf16':
            model.register_comm_hook(
                state=None, hook=comm_hooks.bf16_compress_hook)
            logger.info(
                "Enabled BF16 gradient communication compression hook (payload reduced by 50%).")
        elif accelerator.mixed_precision == 'fp16':
            model.register_comm_hook(
                state=None, hook=comm_hooks.fp16_compress_hook)
            logger.info(
                "Enabled FP16 gradient communication compression hook (payload reduced by 50%).")

    # Extract the base model for compilation tracking or raw sampling functions later.
    # accelerator.unwrap_model() can raise KeyError('_orig_mod') when torch.compile
    # is used *before* accelerator.prepare(), because the DDP wrapper does not expose
    # the compiled module's _orig_mod attribute at its own __dict__ level.
    try:
        base_model = accelerator.unwrap_model(model)
    except KeyError as e:
        if "_orig_mod" not in str(e):
            raise
        logger.warning(
            "unwrap_model hit KeyError('_orig_mod'); "
            "falling back to model.module for base_model."
        )
        if hasattr(model, "module"):
            base_model = model.module
        else:
            base_model = model

    # Load Accelerate state (after prepare)
    if args.resume is not None:
        accelerator.load_state(args.resume)
        logger.info(f"Loaded Accelerate state from {args.resume}")
        # group = opt.param_groups[0]
        # d_val = group.get('d', 1.0)
        # effective_lr = group.get('effective_lr', group.get('lr', 1.0))
        # logger.info(
        #     "Resume optimizer state (group0): "
        #     f"lr={group.get('lr', None)}, d={d_val}, "
        #     f"effective_lr={effective_lr}, d*effective_lr={d_val * effective_lr:.9e}"
        # )

    # Prepare models for training:
    model.train()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_loss_flow = 0
    running_loss_jump = 0
    running_jump_rmse = 0
    running_lambda_theta = 0
    running_lambda_target = 0
    running_jump_var_theta = 0
    running_jump_var_target = 0
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

    # Keep an EMA copy for more stable evaluation/sampling checkpoints.
    ema_model = None
    if args.ema:
        ema_model = deepcopy(sample_base_model).to(device)
        requires_grad(ema_model, False)
        ema_model.eval()
        if args.resume is not None and getattr(args, "ema_resume", True):
            # Try to load EMA from .safetensors first, then fall back to .pt
            ema_safetensors_path = os.path.join(args.resume, "ema.safetensors")
            ema_pt_path = os.path.join(args.resume, "ema.pt")
            
            ema_state = None
            if os.path.exists(ema_safetensors_path):
                from safetensors.torch import load_file
                ema_state = load_file(ema_safetensors_path, device=str(device))
                logger.info(f"Loaded EMA weights from {ema_safetensors_path} (.safetensors)")
            elif os.path.exists(ema_pt_path):
                ema_state = torch.load(ema_pt_path, map_location=device, weights_only=True)
                logger.info(f"Loaded EMA weights from {ema_pt_path} (.pt)")
                
            if ema_state is not None:
                ema_model.load_state_dict(ema_state, strict=False)
            else:
                logger.warning(
                    f"EMA checkpoint not found in {args.resume}; initializing EMA from current model weights."
                )

    sample_runtime_model = ema_model if (args.sample_use_ema and ema_model is not None) else sample_base_model
    sample_model_fn = sample_runtime_model.forward_with_cfg if use_cfg else sample_runtime_model.forward

    logger.info(
        f"Training for {args.epochs} epochs (resuming from step {train_steps})...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            target_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else (
                torch.float16 if accelerator.mixed_precision == 'fp16' else torch.float32)
            x = x.to(device, dtype=target_dtype)
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
                if accelerator.sync_gradients and args.grad_clip_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                opt.step()
                if ema_model is not None:
                    update_ema(ema_model, sample_base_model, decay=args.ema_decay)

            # Log loss values:
            running_loss += loss.item()
            if "loss_flow" in loss_dict:
                running_loss_flow += loss_dict["loss_flow"].item()
            if "loss_jump" in loss_dict:
                running_loss_jump += loss_dict["loss_jump"].item()
            if "jump_rmse" in loss_dict:
                running_jump_rmse += loss_dict["jump_rmse"].item()
            if "lambda_theta" in loss_dict:
                running_lambda_theta += loss_dict["lambda_theta"].item()
            if "lambda_target" in loss_dict:
                running_lambda_target += loss_dict["lambda_target"].item()
            if "jump_var_theta" in loss_dict:
                running_jump_var_theta += loss_dict["jump_var_theta"].item()
            if "jump_var_target" in loss_dict:
                running_jump_var_target += loss_dict["jump_var_target"].item()
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
                avg_jump_rmse = running_jump_rmse / log_steps
                avg_lambda_theta = running_lambda_theta / log_steps
                avg_lambda_target = running_lambda_target / log_steps
                avg_jump_var_theta = running_jump_var_theta / log_steps
                avg_jump_var_target = running_jump_var_target / log_steps
                avg_lambda_ratio = avg_lambda_theta / (avg_lambda_target + 1e-8)

                # MuonWithAuxAdam has two groups: Muon group then aux-Adam group.
                muon_lr = opt.param_groups[0].get('lr', 0.0)
                adam_lr = opt.param_groups[1].get('lr', 0.0) if len(opt.param_groups) > 1 else 0.0

                logger.info(
                    f"(step={train_steps:07d}) Loss: {avg_loss:.4f} "
                    f"(Flow: {avg_loss_flow:.4f}, Jump: {avg_loss_jump:.4f}), "
                    f"lam: {avg_lambda_theta:.2f}/{avg_lambda_target:.2f} "
                    f"(ratio {avg_lambda_ratio:.2f}), "
                    f"jump_var: {avg_jump_var_theta:.3f}/{avg_jump_var_target:.3f}, "
                    f"jump_rmse: {avg_jump_rmse:.2f}, Muon LR: {muon_lr:.2e}, Adam LR: {adam_lr:.2e}"
                )
                if args.wandb:
                    wandb_utils.log(
                        {
                            "train loss": avg_loss,
                            "train loss flow": avg_loss_flow,
                            "train loss jump": avg_loss_jump,
                            "train jump rmse": avg_jump_rmse,
                            "train lambda theta": avg_lambda_theta,
                            "train lambda target": avg_lambda_target,
                            "train lambda ratio": avg_lambda_ratio,
                            "train jump var theta": avg_jump_var_theta,
                            "train jump var target": avg_jump_var_target,
                            "train_steps_per_sec": steps_per_sec,
                            "lr/muon": muon_lr,
                            "lr/adam": adam_lr
                        },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                running_loss_flow = 0
                running_loss_jump = 0
                running_jump_rmse = 0
                running_lambda_theta = 0
                running_lambda_target = 0
                running_jump_var_theta = 0
                running_jump_var_target = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
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

                    # Save EMA model if present (sufficient for Diffusion Transformers evaluation)
                    if ema_model is not None:
                        ema_state_dict = ema_model.state_dict()
                        # Save in .safetensors format only
                        from safetensors.torch import save_file
                        save_file(ema_state_dict, os.path.join(ckpt_dir, "ema.safetensors"), metadata={"format": "pt"})
                        
                        logger.info(f"Saved checkpoint to {ckpt_dir} (Accelerate state + ema.safetensors)")
                    else:
                        logger.warning(f"Saved checkpoint to {ckpt_dir} (Accelerate state only; No EMA model found!)")

                accelerator.wait_for_everyone()
                model.train()
                # Exclude checkpoint I/O time from subsequent step/sec measurement.
                log_steps = 0
                start_time = time()

            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating samples...")
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
                logger.info("Generating samples done.")
                # Exclude sampling/visualization time from subsequent step/sec measurement.
                log_steps = 0
                start_time = time()

    model.eval()

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
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile() for faster GPU kernel fusion.")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode. 'reduce-overhead' suits fixed-shape training loops.")
    parser.add_argument("--muon-lr", type=float, default=0.01,
                        help="Learning rate for Muon parameter group (2D hidden weights).")
    parser.add_argument("--muon-momentum", type=float, default=0.95,
                        help="Momentum for Muon parameter group.")
    parser.add_argument("--muon-weight-decay", type=float, default=0.0,
                        help="Weight decay for Muon parameter group.")
    parser.add_argument("--embed-lr", type=float, default=0.1,
                        help="Learning rate for Embedding parameter group.")
    parser.add_argument("--head-lr", type=float, default=0.004,
                        help="Learning rate for the final output layer group.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0,
                        help="Global norm for gradient clipping. Set <= 0 to disable clipping.")
    parser.add_argument("--aux-adam-lr", type=float, default=1e-4,
                        help="Learning rate for Aux-Adam parameter group.")
    parser.add_argument("--aux-adam-beta1", type=float, default=0.9,
                        help="Beta1 for Aux-Adam parameter group.")
    parser.add_argument("--aux-adam-beta2", type=float, default=0.95,
                        help="Beta2 for Aux-Adam parameter group.")
    parser.add_argument("--aux-adam-eps", type=float, default=1e-10,
                        help="Epsilon for Aux-Adam parameter group.")
    parser.add_argument("--aux-adam-weight-decay", type=float, default=0.0,
                        help="Weight decay for Aux-Adam parameter group.")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True,
                        help="Maintain an EMA copy of model weights during training.")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay factor.")
    parser.add_argument("--ema-resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Load EMA weights from checkpoint during resume.")
    parser.add_argument("--sample-use-ema", action=argparse.BooleanOptionalAction, default=True,
                        help="Use EMA weights for periodic sampling during training.")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)

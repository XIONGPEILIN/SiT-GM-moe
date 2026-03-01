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
        mixed_precision='no',  # FP32
        gradient_accumulation_steps=1,
    )
    device = accelerator.device
    set_seed(args.global_seed)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    assert args.global_batch_size % world_size == 0, \
        f"Batch size must be divisible by world size."
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
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
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

    # Limit dataset size if requested:
    if args.max_train_samples is not None:
        if args.feature_path is None:
            num_samples = min(len(dataset), args.max_train_samples)
        else:
            num_samples = min(len(dataset), max(
                1, args.max_train_samples // 32))

        from torch.utils.data import Subset
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)
        logger.info(
            f"Limited dataset to {num_samples} items (~{args.max_train_samples} samples)")

    # Repeat dataset if requested:
    if getattr(args, 'dataset_repeat', 1) > 1:
        dataset = RepeatedDataset(dataset, args.dataset_repeat)
        logger.info(
            f"Repeated dataset {args.dataset_repeat} times. Total items: {len(dataset):,}")

    if args.feature_path is None:
        loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        def custom_collate(batch):
            features = torch.cat([b[0] for b in batch], dim=0)
            labels = torch.cat([b[1] for b in batch], dim=0)
            if features.shape[0] > local_batch_size:
                features = features[:local_batch_size]
                labels = labels[:local_batch_size]
            return features, labels

        files_per_batch = max(1, local_batch_size // 32)

        loader = DataLoader(
            dataset,
            batch_size=files_per_batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=True
        )

    # Prepare with Accelerate (handles DDP wrapping, device placement, dataloader sharding)
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Load Accelerate state (after prepare)
    if args.resume is not None:
        accelerator.load_state(args.resume)
        logger.info(f"Loaded Accelerate state from {args.resume}")

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
    running_loss_jump_mu_raw = 0
    running_loss_jump_var = 0
    start_time = time()

    # Labels to condition the model with (for periodic sampling):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        base_model = accelerator.unwrap_model(model)
        model_fn = base_model.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        base_model = accelerator.unwrap_model(model)
        model_fn = base_model.forward

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
            if "loss_jump_mu_raw" in loss_dict:
                running_loss_jump_mu_raw += loss_dict["loss_jump_mu_raw"].item()
            if "loss_jump_var" in loss_dict:
                running_loss_jump_var += loss_dict["loss_jump_var"].item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(
                    running_loss / log_steps, device=device)
                avg_loss_flow = torch.tensor(
                    running_loss_flow / log_steps, device=device)
                avg_loss_jump = torch.tensor(
                    running_loss_jump / log_steps, device=device)
                avg_loss_jump_lambda = torch.tensor(
                    running_loss_jump_lambda / log_steps, device=device)
                avg_loss_jump_mu = torch.tensor(
                    running_loss_jump_mu / log_steps, device=device)
                avg_loss_jump_mu_raw = torch.tensor(
                    running_loss_jump_mu_raw / log_steps, device=device)
                avg_loss_jump_var = torch.tensor(
                    running_loss_jump_var / log_steps, device=device)

                # Use accelerate gather for reduction
                avg_loss = accelerator.reduce(avg_loss, reduction="mean")
                avg_loss_flow = accelerator.reduce(
                    avg_loss_flow, reduction="mean")
                avg_loss_jump = accelerator.reduce(
                    avg_loss_jump, reduction="mean")
                avg_loss_jump_lambda = accelerator.reduce(
                    avg_loss_jump_lambda, reduction="mean")
                avg_loss_jump_mu = accelerator.reduce(
                    avg_loss_jump_mu, reduction="mean")
                avg_loss_jump_mu_raw = accelerator.reduce(
                    avg_loss_jump_mu_raw, reduction="mean")
                avg_loss_jump_var = accelerator.reduce(
                    avg_loss_jump_var, reduction="mean")

                avg_loss = avg_loss.item()
                avg_loss_flow = avg_loss_flow.item()
                avg_loss_jump = avg_loss_jump.item()
                avg_loss_jump_lambda = avg_loss_jump_lambda.item()
                avg_loss_jump_mu = avg_loss_jump_mu.item()
                avg_loss_jump_mu_raw = avg_loss_jump_mu_raw.item()
                avg_loss_jump_var = avg_loss_jump_var.item()

                # Fetch Prodigy Schedule-Free dynamic learning rate correctly
                group = opt.param_groups[0]
                d_val = group.get('d', 1.0)
                effective_lr = group.get('effective_lr', group.get('lr', 1.0))
                current_lr = d_val * effective_lr

                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} (Flow: {avg_loss_flow:.4f}, Jump: {avg_loss_jump:.4f}, L_lam: {avg_loss_jump_lambda:.4f}, L_mu: {avg_loss_jump_mu:.4f}, L_mu_raw: {avg_loss_jump_mu_raw:.4f}, L_var: {avg_loss_jump_var:.4f}), Train Steps/Sec: {steps_per_sec:.2f}, LR: {current_lr:.6e}")
                if args.wandb:
                    wandb_utils.log(
                        {
                            "train loss": avg_loss,
                            "train loss flow": avg_loss_flow,
                            "train loss jump": avg_loss_jump,
                            "train loss jump lambda": avg_loss_jump_lambda,
                            "train loss jump mu": avg_loss_jump_mu,
                            "train loss jump mu raw": avg_loss_jump_mu_raw,
                            "train loss jump var": avg_loss_jump_var,
                            "train steps/sec": steps_per_sec,
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
                running_loss_jump_mu_raw = 0
                running_loss_jump_var = 0
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
                    unwrapped_model = accelerator.unwrap_model(model)
                    standalone_path = os.path.join(ckpt_dir, "model.pt")
                    torch.save(unwrapped_model.state_dict(), standalone_path)
                    logger.info(
                        f"Saved checkpoint to {ckpt_dir} (Accelerate state + model.pt)")

                accelerator.wait_for_everyone()
                model.train()
                opt.train()  # Schedule-Free: switch back to training mode

            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating samples...")
                opt.eval()
                model.eval()
                with torch.no_grad():
                    sampler_type = getattr(args, 'sampler_type', 'ode')
                    if sampler_type in ["jump_flow", "jump"]:
                        sample_fn = transport_sampler.sample_jump_flow(
                            num_steps=50)
                    else:
                        sample_fn = transport_sampler.sample_ode()
                    samples = sample_fn(
                        zs, model_fn, **sample_model_kwargs)[-1]
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
    parser.add_argument("--sample-every", type=int, default=10_000)
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

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)

import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import math


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    wandb_key = os.environ.get("WANDB_KEY", None)
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        wandb.login()  # uses cached credentials from `wandb login`
    run_id = os.environ.get("WANDB_RUN_ID")
    init_kwargs = {
        "project": project_name,
        "entity": entity,
        "name": exp_name,
        "config": config_dict,
    }
    # Default: let W&B generate a fresh run id to avoid collisions with deleted runs.
    # Optional resume path: provide WANDB_RUN_ID explicitly.
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"
    wandb.init(**init_kwargs)


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x

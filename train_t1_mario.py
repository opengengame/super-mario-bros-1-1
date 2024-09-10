"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_P2P_DISABLE=1 HF_HOME=/mnt/store/kmei1/HF_HOME accelerate launch --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    train_t1_mario.py \
    --num-workers 8 \
    --model t1 \
    --dataset action_video \
    --ckpt-every 1000 \
    --dataset action_video \
    --vae ema \
    --config-path configs/mario_t1_v0.yaml

HF_HOME=/mnt/store/kmei1/HF_HOME accelerate launch --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 \
    train_t1_mario.py \
    --num-workers 8 \
    --model t1 \
    --dataset action_video \
    --ckpt-every 2000 \
    --dataset action_video \
    --vae ema \
    --config-path configs/mario_t1_v0.yaml
"""

import importlib

import math
import torch
import argparse
import logging
import os
from copy import deepcopy
from glob import glob
from time import time

import psutil
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion import create_diffusion
from utils import requires_grad, update_cpu_ema

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

accelerator = Accelerator(step_scheduler_with_optimizer=False)
device = accelerator.device


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if accelerator.is_main_process:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    configs = OmegaConf.load(args.config_path)

    if accelerator.is_main_process:
        os.makedirs(
            args.results_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Running command: {psutil.Process(os.getpid()).cmdline()}")
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    model = importlib.import_module(f"models.{args.model}").Model(
        **configs.get("model", {})
    )
    pretrained = torch.load("./results/DiT-XL-2-256x256.pt", map_location="cpu")
    _current_model = model.state_dict()
    new_state_dict={k:v if v.size()==_current_model[k].size() else _current_model[k] for k,v in zip(_current_model.keys(), pretrained.values())}
    model.load_state_dict(new_state_dict, strict=False)
    # model.load_state_dict(torch.load("results/002-t1/checkpoints/0003000.pt", map_location="cpu")['model'])

    ema = deepcopy(model)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    diffusion = create_diffusion(
        timestep_respacing=""
    )  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=configs.learning_rate, weight_decay=0)
    # opt.load_state_dict(torch.load("results/002-t1/checkpoints/0003000.pt", map_location="cpu")['opt'])

    # Setup data:
    dataset = importlib.import_module(f"datasets.{args.dataset}").Dataset(
        **configs.get("dataset", {})
    )
    loader = DataLoader(
        dataset,
        batch_size=int(configs.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,}")

    # Prepare models for training:
    update_cpu_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    def lr_lambda(current_step: int):
        if current_step < configs.warmup_steps:
            return float(current_step) / float(max(1, configs.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model, opt, loader, scheduler = accelerator.prepare(model, opt, loader, scheduler)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")

    # scaler = torch.GradScaler()
    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for data in loader:
            x, y, pos = data

            x = x.to(device)
            pos = pos.to(device)

            with torch.no_grad():
                T = x.shape[2]
                x = rearrange(x, "N C T H W -> (N T) C H W")
                x = vae.encode(x).latent_dist.sample().mul_(0.13025)
                x = rearrange(x, "(N T) C H W -> N C T H W", T=T)

            # inject the last frame
            past_frame, x = x[:, :, :-1], x[:, :, -1:]
            past_pos, pos = pos[:, :, :-1], pos[:, :, -1:]

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(
                pos=pos, past_frame=past_frame, past_pos=past_pos
            )

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            scheduler.step()

            if train_steps % configs.ema_iter == 0:
                update_cpu_ema(ema, model, decay=math.pow(0.999, configs.ema_iter))

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % configs.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / accelerator.num_processes
                dynamic_lr = opt.param_groups[0]["lr"]
                logger.info(f"(step={train_steps:07d}) Train LR: {dynamic_lr:.6f} Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "ema": ema.state_dict(),
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default="ema"
    )  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=2_000)
    args = parser.parse_args()
    main(args)

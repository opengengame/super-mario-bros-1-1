"""
A minimal training script for DiT using PyTorch DDP.

References:
- https://github.com/facebookresearch/DiT/blob/main/train.py

Running exmaples:
$ HF_HOME=/mnt/store/kmei1/HF_HOME NCCL_P2P_DISABLE=1 torchrun --master_port 29502 --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    train_runjump.py \
    --num-workers 4 \
    --model dit_celebvt \
    --dataset celebvt \
    --log-every 50 \
    --ckpt-every 1000 \
    --global-batch-size 8 \
    --image-size 256 \
    --video-length 16 \
    --dataset text_video \
    --vae ema \
    --data-path /mnt/store/kmei1/projects/t1/datasets/godmodeanimation_runjump/runjump_dataset

$ HF_HOME=/mnt/store/kmei1/HF_HOME NCCL_P2P_DISABLE=1 accelerate launch --multi_gpu \
    --num_processes 8 \
    --mixed_precision fp16 train_runjump.py \
    --num-workers 4 \
    --model dit_celebvt \
    --dataset celebvt \
    --log-every 50 \
    --ckpt-every 1000 \
    --global-batch-size 8 \
    --image-size 256 \
    --video-length 16 \
    --dataset text_video \
    --vae ema \
    --data-path /mnt/store/kmei1/projects/t1/datasets/godmodeanimation_runjump/runjump_dataset

$ HF_HOME=/mnt/store/kmei1/HF_HOME NCCL_P2P_DISABLE=1 accelerate launch --multi_gpu \
    --num_machines 2 \
    --same_network \
    --num_processes 8 \
    --machine_rank 0 \
    --main_process_ip 10.99.134.98 \
    --main_process_port 29500 \
    --mixed_precision fp16 \
    train_runjump.py \
    --num-workers 4 \
    --model dit_celebvt \
    --dataset celebvt \
    --log-every 50 \
    --ckpt-every 1000 \
    --global-batch-size 16 \
    --image-size 256 \
    --video-length 16 \
    --dataset text_video \
    --vae ema \
    --data-path /mnt/store/kmei1/projects/t1/datasets/godmodeanimation_runjump/runjump_dataset

$ HF_HOME=/mnt/store/kmei1/HF_HOME NCCL_P2P_DISABLE=1 accelerate launch --multi_gpu \
    --num_machines 2 \
    --same_network \
    --num_processes 8 \
    --machine_rank 1 \
    --main_process_ip 10.99.134.98 \
    --main_process_port 29500 \
    --mixed_precision fp16 \
    train_runjump.py \
    --num-workers 4 \
    --model dit_celebvt \
    --dataset celebvt \
    --log-every 50 \
    --ckpt-every 1000 \
    --global-batch-size 16 \
    --image-size 256 \
    --video-length 16 \
    --dataset text_video \
    --vae ema \
    --data-path /mnt/store/kmei1/projects/t1/datasets/godmodeanimation_runjump/runjump_dataset

"""

import importlib
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
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import psutil

from einops import rearrange
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from t5 import t5_encode_text

from accelerate import Accelerator

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup DDP:
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # seed = args.global_seed * dist.get_world_size() + rank
    # torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    # if rank == 0:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Running command: {psutil.Process(os.getpid()).cmdline()}")
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    model = importlib.import_module(f"models.{args.model}").Model(
        input_size=(40, 64),
        learn_sigma=False,
        depth=20,
        hidden_size=1152,
        num_heads=8,
        condition_channels=2048,    # useless here
        patch_size=4,
        use_y_embedder=False,
    )
    # model = model.to(device)
    ema = deepcopy(model)           # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # model.load_state_dict(torch.load("results/010-dit_celebvt/checkpoints/0048000.pt", map_location="cpu")['model'], strict=False)
    # Note that parameter initialization is done within the DiT constructor
    # model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(
        timestep_respacing="",
        predict_xstart=True,
        learn_sigma=False,
        sigma_small=True
    )  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    # opt.load_state_dict(torch.load("results/010-dit_celebvt/checkpoints/0048000.pt", map_location="cpu")['opt'])

    # Setup data:
    dataset = importlib.import_module(f"datasets.{args.dataset}").Dataset(
        args.data_path, 
        args.image_size,
        video_length=args.video_length,
        dataset_name="UCF-101",
        subset_split="train",
        clip_step=1,
        temporal_transform="rand_clips",
        cfg_random_null_text_ratio=0.1,
        latent_scale=32,
    )
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        # sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    model, opt, loader = accelerator.prepare(model, opt, loader)

    model.to(device)
    ema.to(device)

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
                # Map input images to latent space + normalize latents:
                # y = t5_encode_text(y).to(torch.float32)
                # if np.random.rand() < 0.1:
                #     y = t5_encode_text([""]).to(torch.float32)
                x = rearrange(x, "N C T H W -> (N T) C H W")
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, "(N T) C H W -> N C T H W", T=args.video_length)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # pos = torch.cat([pos, pos[:, :1]], dim=1)
            # print("===>", x.shape, pos.shape)
            model_kwargs = dict(y=None, pos=pos, first_frame=x[:,:,:1], first_pos=pos[:, :, :1])

            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # loss.backward()
            # scaler.scale(loss).backward()
            # opt.step()
            # scaler.step(opt)
            # scaler.update()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / accelerator.num_processes
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                # dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    # cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--video-length", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2_000)
    args = parser.parse_args()
    main(args)

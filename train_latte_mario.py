"""
Running exmaples:
$ HF_HOME=/mnt/store/kmei1/HF_HOME NCCL_P2P_DISABLE=1 torchrun --master_port 29502 --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    train_latte_mario.py \
    --num-workers 4 \
    --ckpt-every 500 \
    --image-size 256 \
    --dataset action_video \
    --vae ema \
    --config-path configs/mario_latte_v0.yaml
"""

import importlib
import math
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import psutil
import torch.nn.functional as F


from einops import rearrange
from diffusers.models import AutoencoderKL

from omegaconf import OmegaConf

from diffusers import DDPMScheduler

from contextlib import nullcontext

from models.modeling_latte import LatteT2V

from safetensors.torch import load as safetensors_load

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
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


@torch.no_grad()
def update_cpu_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.detach().cpu().data, alpha=1 - decay)


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
    configs = OmegaConf.load(args.config_path)
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert configs.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Running command: {psutil.Process(os.getpid()).cmdline()}")
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    model = LatteT2V(
        **configs.get("model", {})
    )
    model.gradient_checkpointing = True

    # mismatching loading
    with open("results/open_sora_v1_1_f64.safetensors", "rb") as f:
        data = f.read()
    keys_vin = safetensors_load(data)
    _current_model = model.state_dict()
    new_state_dict={k:v if v.size()==_current_model[k].size() else _current_model[k] for k,v in zip(_current_model.keys(), keys_vin.values())}
    model.load_state_dict(new_state_dict, strict=False)

    ema = deepcopy(model)           # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Note that parameter initialization is done within the DiT constructor
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}", torch_dtype=torch.float16).to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=configs.learning_rate, weight_decay=0)
    # opt.load_state_dict(torch.load("results/010-dit_celebvt/checkpoints/0048000.pt", map_location="cpu")['opt'])

    # diffusion related logics
    noise_scheduler = DDPMScheduler(**configs.get("noise_scheduler", {}))

    # Setup data:
    dataset = importlib.import_module(f"datasets.{args.dataset}").Dataset(
        **configs.get("dataset", {})
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(configs.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    update_cpu_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    def lr_lambda(current_step: int):
        if current_step < configs.warmup_steps:
            return float(current_step) / float(max(1, configs.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")

    scaler = torch.GradScaler()
    # with torch.profiler.profile(
    # schedule=torch.profiler.schedule(
    #     wait=2,
    #     warmup=2,
    #     active=6,
    #     repeat=1),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    # with_stack=False
    # ) as profiler:
    with nullcontext():
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            logger.info(f"Beginning epoch {epoch}...")
            for data in loader:

                x, y, pos = data
                bsz = x.shape[0]

                x = x.to(device).to(torch.float16)
                pos = pos.to(device).to(torch.float16)

                with torch.no_grad():
                    T = x.shape[2]
                    x = rearrange(x, "N C T H W -> (N T) C H W")
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    x = rearrange(x, "(N T) C H W -> N C T H W", T=T)
                
                # inject the first frame
                first_frame, x = x[:, :, :1], x[:, :, 1:]
                noise = torch.randn_like(x, dtype=x.dtype)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                )
                noisy_model_input = noise_scheduler.add_noise(x, noise, timesteps)
                noisy_model_input = torch.cat([first_frame, x], dim=2)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    model_pred = model(
                        noisy_model_input,
                        encoder_hidden_states=None,
                        timestep=timesteps,
                        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                        enable_temporal_attentions=True,
                        return_dict=False
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(x, noise, timesteps)
                    elif noise_scheduler.config.prediction_type == "sample":
                        target = x
                        model_pred = model_pred - noise
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    target = torch.cat([first_frame, target], dim=2)

                    loss = torch.mean((target - model_pred) ** 2)

                scaler.scale(loss).backward()
                if (train_steps + 1) % configs.iters_to_accumulate == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    scheduler.step()
                # loss.backward()
                # opt.step()
                # profiler.step()

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
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    dynamic_lr = opt.param_groups[0]["lr"]
                    logger.info(f"(step={train_steps:07d}) Train LR: {dynamic_lr:.6f} Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save DiT checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=2_000)
    args = parser.parse_args()
    main(args)

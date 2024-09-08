"""
Sample new images from a pre-trained DiT.

References:
- https://github.com/facebookresearch/DiT/blob/main/sample.py

Running exmaples:

CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/store/kmei1/HF_HOME NCCL_P2P_DISABLE=1 python sample_latte_marrio.py \
    --dataset action_video \
    --image-size 256 \
    --ckpt results/017/checkpoints/0005000.pt \
    --num-sampling-steps 20 \
    --config-path configs/mario_latte_t1_v0.yaml
"""
import importlib
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.models import AutoencoderKL
import argparse
from t5 import t5_encode_text
from einops import rearrange
import torchvision
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from models.modeling_latte import LatteT2V
from safetensors.torch import load as safetensors_load
from ipdb import launch_ipdb_on_exception, set_trace

def main(args):
    # Setup PyTorch:
    configs = OmegaConf.load(args.config_path)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    model = LatteT2V(
        **configs.get("model", {})
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    checkpoint = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    # checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    # with open("results/open_sora_v1_1_f64.safetensors", "rb") as f:
    #     data = f.read()
    # model.load_state_dict(safetensors_load(data), strict=False)
    model.eval()  # important!
    model = model.to(device, dtype=torch.float16)
    # model.to()

    noise_scheduler = DDIMScheduler(**configs.get("noise_scheduler", {}))
    # noise_scheduler = DDPMScheduler(**configs.get("noise_scheduler", {}))

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    dataset = importlib.import_module(f"datasets.{args.dataset}").Dataset(
        **configs.get("dataset", {})
    )

    data = dataset[0]
    x, y, pos = data
    # print("============>x", x.max(), x.min())

    pos = torch.from_numpy(pos).to(device)

    with torch.no_grad():
        first_frame_latent = torch.from_numpy(x[None, :, 0]).to(device)
        first_frame_latent = vae.encode(first_frame_latent).latent_dist.sample().mul_(0.18215)
        first_frame_latent = first_frame_latent[:, :, None]

    z = torch.randn((1, 4, 16, 32, 32), device=device).to(torch.float16)
    first_pos, pos = pos[None, :, :1], pos[None, :, 1:]

    # with torch.no_grad():
    #     y = t5_encode_text([y]).to(torch.float16)

    dtype = z.dtype
    noise_scheduler.set_timesteps(args.num_sampling_steps, device=device)
    timesteps = noise_scheduler.timesteps

    latent_model_input = z
    for t in tqdm(timesteps):
        print("======>", latent_model_input.max(), latent_model_input.min())
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                # print("=======> latent_model_input", t, latent_model_input.max(), latent_model_input.min())
                noise_pred = model(
                    torch.cat([first_frame_latent.repeat(1, 1, latent_model_input.size(2), 1, 1), latent_model_input], dim=1),
                    encoder_hidden_states=None,
                    timestep=t[None],
                    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                    enable_temporal_attentions=True,
                    return_dict=False,
                    t1_pos = pos if configs.get("use_t1", False) else None
                )[0]
                print("======> noise_pred", noise_pred.max(), noise_pred.min())
        latent_model_input = noise_scheduler.step(noise_pred, t, latent_model_input, return_dict=False)[0]
    samples = latent_model_input
    samples = torch.cat([first_frame_latent, samples], dim=2)

    _samples = rearrange(samples, "N C T H W -> (N T) C H W")
    with torch.no_grad():
        samples = []
        for frame in _samples:
            samples.append(vae.decode(frame.unsqueeze(0) / 0.18215).sample)
        samples = torch.cat(samples)
    samples = torch.clamp(samples, -1, 1)

    samples = samples.detach().cpu()

    print("samples", samples.shape)

    # Save samples to disk as individual .png files
    # for i, sample in enumerate(samples):
    video = 255 * (samples.clip(-1, 1) / 2 + 0.5)
    torchvision.io.write_video(
        f"samples.mp4",
        video.permute(0, 2, 3, 1).numpy(),
        fps=18,
        video_codec="h264",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    with launch_ipdb_on_exception():
        main(args)

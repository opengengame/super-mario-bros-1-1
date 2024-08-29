"""
Sample new images from a pre-trained DiT.

References:
- https://github.com/facebookresearch/DiT/blob/main/sample.py

Running exmaples:
$ HF_HOME=/mnt/store/kmei1/HF_HOME NCCL_P2P_DISABLE=1 python sample_runjump.py \
    --model dit_celebvt \
    --dataset text_video \
    --image-size 256 \
    --data-path /mnt/store/kmei1/projects/t1/datasets/godmodeanimation_runjump/runjump_dataset \
    --ckpt results/007-dit_celebvt/checkpoints/0003000.pt \
    --num-sampling-steps 50
"""
import importlib
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import argparse
from t5 import t5_encode_text
from einops import rearrange
import torchvision
import numpy as np
from PIL import Image


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    # model = importlib.import_module(f"models.{args.model}").Model(
    #     input_size=(40, 64),
    #     learn_sigma=False,
    #     depth=20,
    #     hidden_size=1152,
    #     num_heads=8,
    #     condition_channels=2048,    # useless here
    #     patch_size=4,
    #     use_y_embedder=False,
    # ).to(device)
    model = importlib.import_module(f"models.{args.model}").Model(
        input_size=(40, 64),
        learn_sigma=False,
        depth=28,
        hidden_size=1152,
        num_heads=16,
        condition_channels=2048,    # useless here
        patch_size=4,
        use_y_embedder=False,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    checkpoint = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(checkpoint)
    model.eval()  # important!
    model.to(device)

    diffusion = create_diffusion(
        timestep_respacing=str(args.num_sampling_steps),
        predict_xstart=True,
        learn_sigma=False,
        sigma_small=True
    )  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    dataset = importlib.import_module(f"datasets.{args.dataset}").Dataset(
        args.data_path, 
        256,                                # not useful
        video_length=16,
        dataset_name="UCF-101",
        subset_split="train",
        clip_step=1,
        temporal_transform="rand_clips",
        cfg_random_null_text_ratio=0.0,
        latent_scale=32,
    )

    data = dataset[12]
    x, _, pos = data
    # print("generating: ", y)
    pos = torch.from_numpy(pos).to(device).unsqueeze(0)
    # with torch.no_grad():
    #     y = t5_encode_text([y])
    #     y_null = t5_encode_text([""])

    N, _, T = pos.shape[:3]
    z = torch.randn(N, 4, 16, 40, 64, device=device)
    # z = torch.cat(pos.shape[2] * [z], 2)

    # Setup classifier-free guidance:
    # y = torch.cat([y, y_null], 0)
    # z = torch.cat([z, z], 0)
    # pos = torch.cat([pos, pos], 0)
    with torch.no_grad():
        first_frame = x[None, :, 0].to(device)       # rearrange(x, "N C T H W -> (N T) C H W")
        # print("======>", x.shape, first_frame.shape)
        first_frame = vae.encode(first_frame).latent_dist.sample().mul_(0.18215)
        first_frame = first_frame[:, :, None]

    model_kwargs = dict(y=None, pos=pos.to(device), first_frame=first_frame.to(device), first_pos=pos[:, :, :1].to(device))

    # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    samples = diffusion.p_sample_loop(
        model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    T = samples.shape[2]
    _samples = rearrange(samples, "N C T H W -> (N T) C H W")
    with torch.no_grad():
        samples = []
        for frame in _samples:
            samples.append(vae.decode(frame.unsqueeze(0) / 0.18215).sample)
        samples = torch.cat(samples)
    samples = torch.clamp(samples, -1, 1)

    samples = samples.cpu()
    print("samples", samples.shape)

    # Save samples to disk as individual .png files
    # for i, sample in enumerate(samples):
    video = 255 * (samples.clip(-1, 1) / 2 + 0.5)
    torchvision.io.write_video(
        f"samples.mp4",
        video.detach().permute(0, 2, 3, 1).numpy(),
        fps=18,
        video_codec="h264",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)

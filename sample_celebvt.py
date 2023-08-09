"""
Sample new images from a pre-trained DiT.

References:
- https://github.com/facebookresearch/DiT/blob/main/sample.py

Running exmaples:
$ HF_HOME=/data/kmei1/HF_HOME CUDA_VISIBLE_DEVICES=2 python sample_celebvt.py \
    --model dit_celebvt \
    --dataset celebvt \
    --image-size 256 \
    --data-path /data/kmei1/datasets/CelebV-Text/frames \
    --ckpt results/005-dit_celebvt/checkpoints/0060000.pt \
    --cfg-scale 8.5 \
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


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = importlib.import_module(f"models.{args.model}").Model(
        input_size=latent_size,
        learn_sigma=False,
        depth=12,
        hidden_size=1152,
        num_heads=12,
        condition_channels=1024,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    # checkpoint = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    model.eval()  # important!
    diffusion = create_diffusion(
        timestep_respacing=str(args.num_sampling_steps),
        predict_xstart=True,
        learn_sigma=False,
        sigma_small=True
    )  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    dataset = importlib.import_module(f"datasets.{args.dataset}").Dataset(
        args.data_path, 
        size=args.image_size,
        range=[0, 16],
        nframes=128,
        latent_scale=16,
    )

    data = dataset[15]
    _, y, pos = data
    y = ": He wears goatee and eyeglasses."
    print("generating: ", y)
    pos = torch.from_numpy(pos).to(device).unsqueeze(0)
    with torch.no_grad():
        y = t5_encode_text([y])
        y_null = t5_encode_text([""])

    N, _, T = pos.shape[:3]
    z = torch.randn(N, 4, 1, latent_size, latent_size, device=device)
    z = torch.cat(pos.shape[2] * [z], 2)
    
    # Setup classifier-free guidance:
    y = torch.cat([y, y_null], 0)
    z = torch.cat([z, z], 0)
    pos = torch.cat([pos, pos], 0)
    model_kwargs = dict(y=y, pos=pos, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    # samples = diffusion.p_sample_loop(
    #     model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
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

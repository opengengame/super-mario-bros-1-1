"""
HF_HOME=/mnt/store/kmei1/HF_HOME/ python testing.py
"""
from omegaconf import OmegaConf

import torch
import torchvision
from diffusers.models import AutoencoderKL
from einops import rearrange
from models.t1 import Model as T1Model
from datasets.action_video import Dataset as ActionDataset
from diffusion import create_diffusion
import gradio as gr
import numpy as np
import PIL
from tqdm import tqdm

device = "cuda:7"

class ActionGameDemo:
    nframes: int = 129
    memory_frames: int = 16
    def __init__(self, config_path, ckpt_path) -> None:
        configs = OmegaConf.load(config_path)
        configs.dataset.video_length = self.nframes
        configs.dataset.data_root = "/data/kmei1/projects/t1/datasets/super-mario-bros-reinforcement-learning/test_clips"
        model = T1Model(**configs.get("model", {}))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["ema"])
        model = model.to(device)
        self.model = model

        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
        dataset = ActionDataset(
            **configs.get("dataset", {})
        )

        data = dataset[11]
        x, action, pos = data

        pos = torch.from_numpy(pos[None]).to(device)
        x =  torch.from_numpy(x[None]).to(device)
        action = torch.tensor(action).to(device)
        with torch.no_grad():
            T = x.shape[2]
            x = rearrange(x, "N C T H W -> (N T) C H W")
            x = self.vae.encode(x).latent_dist.sample().mul_(0.13025)
            x = rearrange(x, "(N T) C H W -> N C T H W", T=T)

        self.all_frames = x
        self.all_poses = pos
        self.all_actions = action

        self.diffusion = create_diffusion(
            timestep_respacing="ddim20",
        )

        _video = x[0]
        grid_h = np.arange(_video.shape[2] // 2, dtype=np.float32)
        grid_w = np.arange(_video.shape[3] // 2, dtype=np.float32)
        grid = np.meshgrid(grid_h, grid_w, indexing='ij')  # here w goes first
        self.grid = torch.from_numpy(np.stack(grid, axis=0)[None]).to(device)

    def next_frame_predict(self, i, action):
        z = torch.randn(1, 4, 1, 32, 32, device=device)
        past_pos = torch.cat([
            torch.tensor([i for i in range(self.memory_frames)], device=device)[None, None, :, None, None].expand(1, 1, self.memory_frames, *self.all_poses.shape[-2:]),
            self.grid[:,:,None,:,:].expand(1, 2, self.memory_frames, *self.all_poses.shape[-2:]),
            self.all_actions[None, None, -self.memory_frames:, None, None].expand(1, 1, self.memory_frames, *self.all_poses.shape[-2:]),
        ], dim=1)
        pos = torch.cat([
            torch.tensor([self.memory_frames], device=device)[None, None, :, None, None].expand(1, 1, 1, *self.all_poses.shape[-2:]),
            self.grid[:,:,None,:,:].expand(1, 2, 1, *self.all_poses.shape[-2:]),
            torch.tensor([action], device=device)[None, None, :, None, None].expand(1, 1, 1, *self.all_poses.shape[-2:]),
        ], dim=1)
        model_kwargs = dict(
            pos=pos,
            past_frame=self.all_frames[:, :, i-self.memory_frames:i],
            past_pos=past_pos
        )
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                samples = self.diffusion.p_sample_loop(
                    self.model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
        mse = torch.mean((self.all_frames[:,:,i:i+1] - samples) ** 2)
        psnr =  20 * torch.log10(2 / torch.sqrt(mse))
        self.all_frames[:,:,i:i+1] = samples
        return psnr


    def output_video(self):
        _samples = rearrange(self.all_frames, "N C T H W -> (N T) C H W")
        with torch.no_grad():
            vidoes = []
            for frame in _samples:
                vidoes.append(self.vae.decode(frame.unsqueeze(0) / 0.13025).sample)
            vidoes = torch.cat(vidoes)
        vidoes = torch.clamp(vidoes, -1, 1)
        del _samples
        video = 255 * (vidoes.clip(-1, 1) / 2 + 0.5)
        
        screeshoot = PIL.Image.fromarray(np.uint8(rearrange(video.cpu().numpy(), "T C H W -> H (T W) C")))
        screeshoot.save("samples.png")

        torchvision.io.write_video(
            "samples.mp4",
            video.permute(0, 2, 3, 1).cpu().numpy(),
            fps=8,
            video_codec="h264",
        )
        return "samples.mp4"

    def create_interface(self):
        with gr.Blocks() as demo:
            gr.Markdown("#  GenGame - OpenGenGAME/super-mario-bros-rl-1-1")
            gr.Markdown("Try to control the game: Right, A, Right + A, Left")

            with gr.Row():
                image_output = gr.Image(label="Next Frame")
                video_output = gr.Video(label="All Frames", autoplay=True, loop=True)

            with gr.Row():
                # left_btn = gr.Button("←")
                right_btn = gr.Button("→")
                # a_btn = gr.Button("A")
                right_a_btn = gr.Button("→ A")

            play_btn = gr.Button("conver video")
            log_output = gr.Textbox(label="Actions Log", lines=5, interactive=False)

            # sampling_steps = gr.Slider(minimum=10, maximum=50, value=10, step=1, label="Sampling Steps")

            # left_btn.click(lambda: self.next_frame_predict(6), outputs=[image_output, log_output])
            right_btn.click(lambda: self.next_frame_predict(3), outputs=[image_output, log_output])
            # a_btn.click(lambda: self.next_frame_predict(5), outputs=[image_output, log_output])
            right_a_btn.click(lambda: self.next_frame_predict(4), outputs=[image_output, log_output])
            play_btn.click(self.output_video, outputs=video_output)
        return demo

if __name__ == "__main__":
    action_game = ActionGameDemo(
        "configs/mario_t1_v0.yaml",
        "results/002-t1/checkpoints/0180000.pt",
    )
    # demo = action_game.create_interface()
    # demo.launch()
    # for i in tqdm(range(64)):
    #     action_game.next_frame_predict(4)
    # action_game.output_video()

    # next_frame = action_game.next_frame_predict(6)
    avg_psnr = []
    for i, action in enumerate(action_game.all_actions[16:]):
        psnr = action_game.next_frame_predict(i + action_game.memory_frames, action)
        avg_psnr.append(psnr.item())
        print(i, psnr, np.mean(np.array(avg_psnr)))
        action_game.output_video()

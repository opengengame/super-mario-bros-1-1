import os
import random
from typing import List

import av
import av.logging

import PIL
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from einops import rearrange
from cache_decorator import Cache

import torch.nn.functional as F
import torchvision.transforms.functional as TF

import cv2


class VideoFolder(Dataset):
    IMG_EXTENSIONS = [
        ".png",
        ".PNG",
        ".jpg",
        ".JPG"
    ]
    VIDEO_EXTENSIONS = [".mp4", ".MP4", ".avi", ".AVI"]

    def __init__(
        self,
        path: str,
        size: List[int],
        nframes: int = 128,
    ):
        if isinstance(size, (list, tuple)):
            if len(size) not in [1, 2]:
                raise ValueError(
                    f"Size must be an int or a 1 or 2 element tuple/list, not a {len(size)} element tuple/list"
                )

        if isinstance(size, int):
            size = [size, size]

        def _find_all_path(_path):
            _all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=_path)
                for root, _dirs, files in os.walk(_path)
                for fname in files
            }
            _video_fnames = sorted(
                fname
                for fname in _all_fnames
                if self._file_ext(fname) in self.VIDEO_EXTENSIONS
            ) + sorted(
                list(
                    set(
                        (
                            os.path.dirname(fname)
                            for fname in _all_fnames
                            if self._file_ext(fname) in self.IMG_EXTENSIONS
                        )
                    )
                )
            )
            _video_fnames = sorted(_video_fnames)
            return _video_fnames

        _video_fnames = _find_all_path(path)

        self.path = path
        self.size = size
        self.nframes = nframes

        self._video_fnames = _video_fnames
        self._total_size = len(self._video_fnames)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    @Cache(
        cache_path="/data/kmei1/caches/{_hash}.pkl",
    )
    def _read_video_opencv(self, video_path, nframes, size):
        video = []
        if os.path.isdir(video_path):
            _all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=video_path)
                for root, _dirs, files in os.walk(video_path)
                for fname in files
            }
            _video_fnames = sorted(
                fname
                for fname in _all_fnames
                if self._file_ext(fname) in self.IMG_EXTENSIONS
            )
            for fname in _video_fnames:
                with open(os.path.join(video_path, fname), "rb") as f:
                    video.append(
                        np.array(
                            PIL.Image.open(f)
                            .convert("RGB")
                            .resize(
                                size, resample=3
                            )  # PIL.Image.Resampling.LANCZOS = 1 PIL.Image.Resampling.BICUBIC = 3
                        )
                    )
        else:
            video = []
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                success, image = cap.read()
                if success:
                    video.append(
                        np.asarray(
                            cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)[
                                :, :, ::-1
                            ]
                        )
                    )
                else:
                    break
            cap.release()

            if len(video) != nframes:
                frame_scale = len(video) / nframes
                frame_scaled_idxs = [int(i * frame_scale) for i in range(nframes)]
                video = [video[i] for i in range(len(video)) if i in frame_scaled_idxs]

        # for cache
        video = np.stack(video).astype(np.uint8)
        return video

    @Cache(
        cache_path="/data/kmei1/caches/{_hash}.pkl",
    )
    def _read_video(self, video_path, nframes, size):
        video = []
        if os.path.isdir(video_path):
            _all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=video_path)
                for root, _dirs, files in os.walk(video_path)
                for fname in files
            }
            _video_fnames = sorted(
                fname
                for fname in _all_fnames
                if self._file_ext(fname) in self.IMG_EXTENSIONS
            )
            for fname in _video_fnames:
                with open(os.path.join(video_path, fname), "rb") as f:
                    video.append(
                        np.array(
                            PIL.Image.open(f)
                            .convert("RGB")
                            .resize(
                                self.size, resample=1
                            )  # PIL.Image.Resampling.LANCZOS = 1 PIL.Image.Resampling.BICUBIC = 3
                        )
                    )
        else:
            with av.open(video_path) as container:
                container.streams.video[0].thread_type = "AUTO"
                container.streams.video[0].thread_count = 2
                total_frames = container.streams.video[0].frames

                frame_scale = total_frames / nframes
                frame_scaled_idxs = [int(i * frame_scale) for i in range(total_frames)]

                for idx, frame in enumerate(container.decode(video=0)):
                    if idx in frame_scaled_idxs:
                        video.append(
                            np.asarray(
                                frame.to_image().resize(
                                    size, resample=1
                                )  # PIL.Image.Resampling.LANCZOS = 1 PIL.Image.Resampling.BICUBIC = 3
                            ).clip(0, 255)
                        )
                container.close()

        frame_scale = len(video) / nframes
        frame_scaled_idxs = [int(i * frame_scale) for i in range(nframes)]
        video = [video[i] for i in range(len(video)) if i in frame_scaled_idxs]
        video = np.stack(video).astype(np.uint8)    # for cache
        return video

    @Cache(
        cache_path="/data/kmei1/caches/{_hash}.pkl",
    )
    def _read_video_metric(self, video_path, nframes, size):
        video = []
        if os.path.isdir(video_path):
            _all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=video_path)
                for root, _dirs, files in os.walk(video_path)
                for fname in files
            }
            _video_fnames = sorted(
                fname
                for fname in _all_fnames
                if self._file_ext(fname) in self.IMG_EXTENSIONS
            )
            for fname in _video_fnames:
                with open(os.path.join(video_path, fname), "rb") as f:
                    video.append(
                        np.array(
                            PIL.Image.open(f)
                            .convert("RGB")
                            .resize(
                                self.size, resample=1
                            )  # PIL.Image.Resampling.LANCZOS = 1 PIL.Image.Resampling.BICUBIC = 3
                        )
                    )
        else:
            with av.open(video_path) as container:
                container.streams.video[0].thread_type = "AUTO"
                container.streams.video[0].thread_count = 2
                total_frames = container.streams.video[0].frames

                frame_scale = total_frames / nframes
                frame_scaled_idxs = [int(i * frame_scale) for i in range(total_frames)]

                for idx, frame in enumerate(container.decode(video=0)):
                    if idx in frame_scaled_idxs:
                        frame = F.interpolate(TF.pil_to_tensor(frame.to_image()).unsqueeze(0), size=size[0], mode='bilinear', align_corners=False)[0].numpy().clip(0, 255)
                        video.append(frame)
                container.close()

        frame_scale = len(video) / nframes
        frame_scaled_idxs = [int(i * frame_scale) for i in range(nframes)]
        video = [video[i] for i in range(len(video)) if i in frame_scaled_idxs]
        # for cache
        video = np.stack(video).astype(np.uint8)
        return video

    def __getitem__(self, index):
        video_path = os.path.join(self.path, self._video_fnames[index])
        try:
            video = self._read_video_metric(
                video_path=video_path, nframes=self.nframes, size=self.size
            )
        except Exception as e:
            print("=> error with loading video", video_path, e)
            video = self.__getitem__(index + 1)

        if video.shape[0] != self.nframes: print("=> unconsisitent video frames", video_path, video.shape[0], "v.s.", self.nframes)

        video = video.astype(np.float32)
        video = (video - 127.5) / 127.5

        return video

    def __len__(self):
        return self._total_size


class Dataset(VideoFolder):
    def __init__(
        self,
        path: str,
        size: List[int],
        range: List[int] = None,
        nframes: int = 128,
        latent_scale = 8,
    ):
        super().__init__(path=path, size=size, nframes=nframes)
        textbase = pd.read_csv(
            "/mnt/store/kmei1/projects/video-generation/datasets/webvid/results_2M_train.csv"
        )[["videoid", "name"]]
        textbase["videoid"] = textbase["videoid"].astype(str)
        textbase = textbase.set_index('videoid')['name'].to_dict()
        self.textbase = textbase

        if range is not None:
            self._video_fnames = self._video_fnames[range[0]:range[1]]
        self._total_size = len(self._video_fnames)
        self.latent_scale = latent_scale

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getitem__(self, index):
        index = index % self._total_size
        video_path = os.path.join(self.path, self._video_fnames[index])
        
        try:
            video = self._read_video(video_path=video_path, nframes=self.nframes, size=self.size)
            video = video.astype(np.float32)
            video = (video - 127.5) / 127.5
            text = self.textbase[os.path.basename(self._video_fnames[index])[:-4]]

            if video.shape[0] != self.nframes:
                raise ValueError(
                    f"less than {self.nframes} frames only have {video.shape[0]}"
                )
    
        except Exception as e:
            print("=> error with loading video", video_path, e)
            video, text, grid = self.__getitem__(index + 1)
            return (video, text, grid)

        video = rearrange(video, 'T H W C -> C T H W')
        grid_size = [self.nframes, video.shape[2] // self.latent_scale, video.shape[3] // self.latent_scale]
        grid_t = np.arange(grid_size[0], dtype=np.float32)
        grid_h = np.arange(grid_size[1], dtype=np.float32)
        grid_w = np.arange(grid_size[2], dtype=np.float32)
        grid = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij')  # here w goes first
        grid = np.stack(grid, axis=0)

        return (video, text, grid)

    def __len__(self):
        return self._total_size * 10000

import os
from timeit import repeat
import imageio
import numpy as np
from typing import Union

import torch
import torchvision

from tqdm import tqdm
from einops import rearrange, repeat
import decord
decord.bridge.set_bridge('torch')
  

  
def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        # if rescale:
        #     x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        # x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


if __name__ == '__main__':

    paths = ["/mnt/bn/qy-dcar-valume/animate-anything/spider_man_skiing.mp4", "/mnt/bn/qy-dcar-valume/animate-anything/pink_man_skiing.mp4", "/mnt/bn/qy-dcar-valume/animate-anything/woman.mp4"]
    samples = []
    for path in paths:
        video_reader = decord.VideoReader(path)
        sample_index = list(range(0, len(video_reader), 1))[:24]
        video = video_reader.get_batch(sample_index) # f h w c -> b c f h w
        
        video = repeat(video, "f h w c -> b f h w c", b = 1)
        video = rearrange(video, "b f h w c -> b c f h w")
        samples.append(video)
    
    samples = torch.concat(samples)
    save_path = "./merged.mp4"
    save_videos_grid(samples, save_path)
    

    



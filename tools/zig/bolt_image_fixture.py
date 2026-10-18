"""Fixtures for nemotron_image.zig: torch bicubic antialiased resizes and CLIP normalization of a synthetic image."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bolt_vision_reference import target_patches  # noqa: E402

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
CFG = {"patch_size": 16, "min_num_patches": 1024, "max_num_patches": 13312, "downsample_ratio": 0.5,
       "max_model_len": 16384}


def image(w, h):
    y, x, c = np.meshgrid(np.arange(h), np.arange(w), np.arange(3), indexing="ij")
    return ((7 * x + 13 * y + 29 * c + (x * y) % 17) % 256).astype(np.uint8)


def prep(rgb, ow, oh):
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(oh, ow), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return ((t / 255.0 - mean) / std)[0].numpy().astype("<f4")


def main():
    rgb = image(37, 23)
    out = b"".join(prep(rgb, ow, oh).tobytes() for ow, oh in [(48, 32), (16, 16), (24, 8)])
    Path(sys.argv[1]).write_bytes(out)
    for w, h in [(2172, 724), (640, 480), (37, 23), (4000, 3000), (16, 16), (100, 5000), (1024, 1024), (512, 512),
                 (3000, 100), (1920, 1080), (800, 600)]:
        print(w, h, target_patches(w, h, CFG))


if __name__ == "__main__":
    main()

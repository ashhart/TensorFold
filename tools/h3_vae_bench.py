"""Time the H3 video VAE decode with and without the int8 kernels and compare the decoded frames.

    PYTHONPATH=src:<minimax-h3-mlx> <minimax-h3-mlx>/.venv/bin/python tools/h3_vae_bench.py \
        ~/h3-models/MiniMax-H3 [--latents z.safetensors] [--png cmp.png]

The decoder is minimax-h3-mlx's; `tensorfold.families.h3.vae_fast.accelerate` swaps its projections in place.
Reports warm decode seconds, PSNR against the unmodified decode, and the attention / FFN split of each variant.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from tensorfold.families.h3 import config as h3
from tensorfold.families.h3.vae_fast import accelerate

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def to_frames(decoded: mx.array) -> np.ndarray:
    """(1, 3, F, H, W) decoder output to (F, H, W, 3) uint8, as the render path does."""

    frames = np.array(decoded.astype(mx.float32))
    frames = frames * np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    frames = frames + np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    return (np.clip(frames, 0.0, 1.0)[0].transpose(1, 2, 3, 0) * 255.0 + 0.5).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    error = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if error == 0 else 10.0 * np.log10(255.0**2 / error)


class Timed:
    """Wraps a block stage, forcing evaluation around it so its time can be read."""

    def __init__(self, stage, totals: dict, name: str):
        self.stage, self.totals, self.name = stage, totals, name

    def __call__(self, *args):
        mx.eval(*[a for a in args if isinstance(a, mx.array)])
        started = time.perf_counter()
        out = self.stage(*args)
        mx.eval(out)
        self.totals[self.name] = self.totals.get(self.name, 0.0) + time.perf_counter() - started
        return out


def split(vae, z: mx.array) -> dict:
    """Seconds spent in attention (with its projections) and in the FFN over one decode."""

    totals: dict = {}
    blocks = vae.decoder.transformer_blocks
    saved = [(block.attn, block.ff) for block in blocks]
    for block in blocks:
        block.attn, block.ff = Timed(block.attn, totals, "attention"), Timed(block.ff, totals, "ffn")
    started = time.perf_counter()
    mx.eval(vae.decode(z))
    totals["total"] = time.perf_counter() - started
    for block, (attention, ffn) in zip(blocks, saved, strict=True):
        block.attn, block.ff = attention, ffn
    return {name: round(seconds, 2) for name, seconds in totals.items()}


def measure(vae, z: mx.array, repeats: int) -> tuple[float, np.ndarray]:
    mx.eval(vae.decode(z))  # compiles kernels and touches the weights
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        out = vae.decode(z)
        mx.eval(out)
        seconds.append(time.perf_counter() - started)
    return float(np.median(seconds)), to_frames(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--latents", default=None, help="safetensors with `z`, the (1, 24, F, H, W) decoder input")
    parser.add_argument("--shape", default="37,30,54", help="latent frames, height, width for random latents")
    parser.add_argument("--png", default=None, help="write a comparison of three frames here")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    from minimax_h3_mlx.load import load_video_vae

    vae = load_video_vae(h3.pipeline_root(args.model_dir) / "video_vae")
    if args.latents:
        z = mx.load(args.latents)["z"].astype(mx.float32)
    else:
        mx.random.seed(0)
        frames, height, width = (int(v) for v in args.shape.split(","))
        z = mx.random.normal((1, vae.config.latent_channels, frames, height, width)).astype(mx.float32)
    mx.eval(z)
    print(f"[tensorfold] vae decode of {tuple(z.shape)} at {vae.decode_dtype}, batch {vae.decode_batch}, "
          f"{'real' if args.latents else 'random'} latents", flush=True)

    seconds, reference = measure(vae, z, args.repeats)
    print(f"[tensorfold] reference: {seconds:.2f}s  split {split(vae, z)}", flush=True)
    shown = [("reference", reference)]
    for name, options in (("int8 ffn", {"mlp": True, "projections": False}),
                          ("int8 ffn + projections", {"mlp": False, "projections": True})):
        changed = accelerate(vae, **options)
        seconds, frames = measure(vae, z, args.repeats)
        print(f"[tensorfold] {name}: {seconds:.2f}s  PSNR {psnr(frames, reference):.1f} dB  "
              f"split {split(vae, z)}  swapped {changed}", flush=True)
        shown.append((name, frames))

    if args.png:
        from PIL import Image

        picks = [reference.shape[0] // 6, reference.shape[0] // 2, reference.shape[0] * 5 // 6]
        rows = [np.concatenate([frames[i] for _, frames in shown], axis=1) for i in picks]
        Image.fromarray(np.concatenate(rows, axis=0)).save(args.png)
        print(f"[tensorfold] wrote {args.png}: columns {[name for name, _ in shown]}, frames {picks}", flush=True)


if __name__ == "__main__":
    main()

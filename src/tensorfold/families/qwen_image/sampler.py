"""Qwen-Image-2.1 denoising loop: Euler over the shifted schedule, no guidance."""

from __future__ import annotations

from .config import latent_grid
from .schedule import sigmas as make_sigmas


def start_noise(seed: int, width: int, height: int, channels: int = 64):
    """The starting latents, (1, rows * columns, channels) bfloat16; the draw mflux makes for the same seed."""

    import mlx.core as mx

    rows, columns = latent_grid(width, height)
    return mx.random.normal(shape=[1, rows * columns, channels], key=mx.random.key(seed)).astype(mx.bfloat16)


def denoise(dit, text, width: int, height: int, steps: int = 40, seed: int = 0, nodes=None, on_step=None,
            latents=None):
    """Image latents (1, channels, rows, columns) float32 from prompt embeddings ``text`` (1, tokens, dim).

    ``nodes`` are fixed raw noise levels for a distilled adapter; ``steps`` is ignored when they are given.
    ``on_step(index, total)`` is called after each step.
    """

    import mlx.core as mx

    rows, columns = latent_grid(width, height)
    levels = make_sigmas(steps, width, height, nodes)
    prefix = dit.prefix(text, rows, columns)
    x = start_noise(seed, width, height, dit.config.in_channels) if latents is None else latents
    x = x.astype(mx.float32)
    total = len(levels) - 1
    for index in range(total):
        velocity = dit(x, float(levels[index]), prefix)
        x = x + velocity.astype(mx.float32) * float(levels[index + 1] - levels[index])
        mx.eval(x)
        if on_step is not None:
            on_step(index, total)
    return x.reshape(1, rows, columns, -1).transpose(0, 3, 1, 2)

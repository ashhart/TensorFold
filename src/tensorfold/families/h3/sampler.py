"""The H3 joint denoise loop: one forward per step moves video and audio rows down their own schedules."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import config as h3
from .packing import AUDIO_CHANNELS, KEYFRAME_NOISE, Layout, layout, patchify, timestep_plan
from .schedule import AUDIO_SHIFT, VIDEO_SHIFT, Schedule


@dataclass
class Latents:
    """Denoised rows and the geometry needed to decode them."""

    video_rows: mx.array
    audio_rows: mx.array
    packed: Layout
    latent_frames: int
    latent_height: int
    latent_width: int
    audio_latents: int
    step_seconds: list[float]


def start_noise(config, latent_frames: int, latent_height: int, latent_width: int, audio_latents: int, seed: int,
                condition: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """Video then audio noise from one seed, in the reference's draw order.

    Keyframe ``condition`` rows are noised first, from the same generator, to ``KEYFRAME_NOISE`` and placed ahead
    of the generated video rows; they keep that level for the whole run.
    """

    mx.random.seed(seed)
    if condition is not None:
        noise = mx.random.normal(condition.shape).astype(mx.float32)
        level = float(np.float32(KEYFRAME_NOISE))
        condition = level * condition.astype(mx.float32) + float(np.float32(1.0) - np.float32(KEYFRAME_NOISE)) * noise
    video = mx.random.normal((1, config.latents_dim, latent_frames, latent_height, latent_width))
    audio = mx.random.normal((audio_latents * AUDIO_CHANNELS, config.audio_latents_dim))
    video = patchify(video.astype(mx.float32), config.patch_size)
    if condition is not None:
        video = mx.concatenate([condition, video])
    return video, audio.astype(mx.float32)


def check_noise(name: str, rows: mx.array) -> None:
    """Refuse a modality that does not start from noise.

    Video and audio denoise in one sequence, so a constant or zero audio start corrupts the picture as well as
    the sound; the engine this guards against shipped exactly that.
    """

    mean, std = float(mx.mean(rows).item()), float(mx.std(rows).item())
    if not (abs(mean) < 0.05 and 0.9 < std < 1.1):
        raise ValueError(f"{name} start is not unit noise (mean {mean:.3f}, std {std:.3f})")


def denoise(dit, text, text_tags, width: int, height: int, frames: int, points: int, seed: int = 0,
            subset: tuple[int, ...] | None = None, on_step=None, forward=None, release: bool = False,
            condition: mx.array | None = None, keyframes: tuple[str, ...] = (),
            audio_shift: float | None = None) -> Latents:
    """Denoise one clip. ``points`` is the number of sigma points, so ``points - 1`` forwards (fewer with
    ``subset``). ``forward`` replaces the plain DiT call. The AdaLN tables for the whole run are projected once
    up front; ``release`` then frees the projection weights. ``condition`` holds the encoded keyframe rows for
    ``keyframes`` (``first`` or ``last`` each); they condition every step and are not denoised. ``audio_shift``
    replaces the audio schedule's sigma shift (3 in the released model)."""

    config = dit.config
    latent_frames = h3.latent_frames(frames)
    latent_height, latent_width = height // h3.VAE_SPATIAL_RATIO, width // h3.VAE_SPATIAL_RATIO
    audio_latents = h3.audio_latents(frames)
    packed = layout(text_tags, latent_frames, latent_height, latent_width, audio_latents, config.patch_size,
                    keyframes)
    held = packed.condition_video_rows
    if (condition is None) != (held == 0) or (condition is not None and condition.shape[0] != held):
        raise ValueError(f"{len(keyframes)} keyframes need {held} condition rows")
    video_rows, audio_rows = start_noise(config, latent_frames, latent_height, latent_width, audio_latents, seed,
                                         condition)
    check_noise("video", video_rows[held:])
    check_noise("audio", audio_rows)

    video_schedule = Schedule(VIDEO_SHIFT, points, subset)
    audio_schedule = Schedule(AUDIO_SHIFT if audio_shift is None else audio_shift, points, subset)
    table, plan = timestep_plan(packed, video_schedule.timesteps, audio_schedule.timesteps)
    text = text.astype(mx.bfloat16)
    dit.cache_modulation(table, release=release)
    run = forward or dit
    seconds = []
    for index in range(len(video_schedule)):
        started = time.perf_counter()
        video_velocity, audio_velocity = run(video_rows[None], audio_rows[None], text, table, plan[index],
                                             packed.tags, packed.position_ids, packed.video_rows,
                                             packed.audio_rows, packed.text_rows)
        stepped = video_schedule.step(index, video_velocity[0, held:], video_rows[held:])
        video_rows = mx.concatenate([video_rows[:held], stepped]) if held else stepped
        audio_rows = audio_schedule.step(index, audio_velocity[0], audio_rows)
        mx.eval(video_rows, audio_rows)
        seconds.append(time.perf_counter() - started)
        if on_step is not None:
            on_step(index + 1, len(video_schedule), seconds[-1])
    return Latents(video_rows[held:], audio_rows, packed, latent_frames, latent_height, latent_width, audio_latents,
                   seconds)

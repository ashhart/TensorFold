"""The packed H3 sequence: row order, rotary positions, modality tags and latent patching."""

# Adapted from minimax-h3-mlx (Apache-2.0, https://github.com/mrbizarro/minimax-h3-mlx, revision 7919020).

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2
AUDIO_CHANNELS = 2
KEYFRAME_NOISE = 0.999  # conditioning rows are held at this timestep for every step

# One latent frame spans 5/3 of the frames it covers, in the VAE's 17-frames-to-5-latents grouping; the
# spatial axes are normalised by the square root of the latent area and scaled by 32.
_FRAME_RESCALE = 5.0 / 3.0
_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_SPATIAL_SCALE = 32


@dataclass
class Layout:
    """Rows of one packed sequence, in the order [text | keyframe conditions | audio | video]."""

    rows: int
    position_ids: mx.array  # (rows, 3) float32 over (t, h, w)
    tags: mx.array  # (rows,) modality per row
    video_rows: mx.array  # conditioning rows first, then generated rows
    audio_rows: mx.array
    text_rows: mx.array
    condition_video_rows: int


def patchify(latents: mx.array, patch: tuple[int, int, int]) -> mx.array:
    """(B, C, F, H, W) video latents to (rows, C * patch volume), frame-major then row-major."""

    pt, ph, pw = patch
    b, c, f, h, w = latents.shape
    x = latents.reshape(b, c, f // pt, pt, h // ph, ph, w // pw, pw).transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * pt * ph * pw)


def unpatchify(rows: mx.array, frames: int, height: int, width: int, channels: int,
               patch: tuple[int, int, int]) -> mx.array:
    """Inverse of :func:`patchify`: (B, C, F, H, W)."""

    pt, ph, pw = patch
    x = rows.reshape(-1, frames // pt, height // ph, width // pw, channels, pt, ph, pw)
    return x.transpose(0, 4, 1, 5, 2, 6, 3, 7).reshape(-1, channels, frames, height, width)


def unpack_audio(rows: mx.array, latents: int) -> mx.array:
    """Channel-major audio rows to (2, latent channels, latents): one batch item per stereo channel."""

    return rows.reshape(AUDIO_CHANNELS, latents, rows.shape[-1]).transpose(0, 2, 1)


def _spatial_grid(size: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = size / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, size // patch, endpoint=False) * _SPATIAL_SCALE


def _frame_times(frames: int, origin: float) -> np.ndarray:
    spans = np.array([_FRAME_RESCALE * _FRAMES_PER_LATENT[i % 5] for i in range(frames)], dtype=np.float64)
    return origin + np.concatenate([np.zeros(1), np.cumsum(spans[:-1])])


def _time_span(frames: int) -> float:
    # summed pairwise, as the reference computes the keyframe anchor; sequential summation differs in the last
    # ulp from 16 latent frames on
    spans = np.ones(frames, dtype=np.float64) * _FRAME_RESCALE
    for i in range(5):
        spans[i::5] *= _FRAMES_PER_LATENT[i]
    return float(spans.sum())


def layout(text_tags, latent_frames: int, latent_height: int, latent_width: int, audio_latents: int,
           patch: tuple[int, int, int], keyframes: tuple[str, ...] = ()) -> Layout:
    """Build the packed layout. ``text_tags`` is the modality of every text row (a keyframe's vision block is
    tagged video); ``keyframes`` names each conditioning block, ``first`` or ``last``."""

    _, ph, pw = patch
    text_tags = np.asarray(text_tags, dtype=np.int64)
    per_frame = (latent_height // ph) * (latent_width // pw)
    text = int(text_tags.shape[0])
    condition = len(keyframes) * per_frame
    audio = audio_latents * AUDIO_CHANNELS
    video = latent_frames * per_frame
    rows = text + condition + audio + video
    audio_start = text + condition
    video_start = audio_start + audio

    # text rows sit on the time axis at their own index and the media rows continue from there
    positions = np.zeros((rows, 3), dtype=np.float64)
    positions[:text, 0] = np.arange(text)
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_grid(latent_height, ph, sqrt_area)
    width_grid = _spatial_grid(latent_width, pw, sqrt_area)
    hh, ww = np.meshgrid(height_grid, width_grid, indexing="ij")
    frame = np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)

    for index, anchor in enumerate(keyframes):
        if anchor not in ("first", "last"):
            raise ValueError(f"a keyframe anchor is 'first' or 'last', got {anchor!r}")
        when = float(text) if anchor == "first" else float(text) + _time_span(latent_frames) - _FRAME_RESCALE
        lo = text + index * per_frame
        positions[lo:lo + per_frame, 0] = when
        positions[lo:lo + per_frame, 1:] = frame

    # audio shares the video clock (40 latents a second equals 24 fps * 5/3), has no height and sits at the
    # two ends of the width grid, one per stereo channel
    positions[audio_start:video_start, 0] = np.tile(float(text) + np.arange(audio_latents), AUDIO_CHANNELS)
    positions[audio_start:video_start, 2] = np.concatenate([np.full(audio_latents, width_grid[0]),
                                                            np.full(audio - audio_latents, width_grid[-1])])
    video_positions = np.empty((latent_frames, per_frame, 3), dtype=np.float64)
    video_positions[:, :, 0] = _frame_times(latent_frames, float(text))[:, None]
    video_positions[:, :, 1:] = frame[None]
    positions[video_start:] = video_positions.reshape(-1, 3)

    video_rows = np.concatenate([np.arange(text, audio_start), np.arange(video_start, rows)])
    audio_rows = np.arange(audio_start, video_start)
    tags = np.empty(rows, dtype=np.int64)
    tags[:text] = text_tags
    tags[audio_rows] = TAG_AUDIO
    tags[video_rows] = TAG_VIDEO
    return Layout(rows, mx.array(positions.astype(np.float32)), mx.array(tags.astype(np.int32)),
                  mx.array(video_rows.astype(np.int32)), mx.array(audio_rows.astype(np.int32)),
                  mx.array(np.arange(text, dtype=np.int32)), condition)


def timestep_plan(packed: Layout, video_timesteps, audio_timesteps) -> tuple[mx.array, list[mx.array]]:
    """One table of the distinct timesteps of a run and, for every step, each row's index into it.

    Generated video and audio rows follow their own schedules inside one forward; conditioning video rows stay
    at ``max(t, KEYFRAME_NOISE)``; text rows never reach an output and take the video timestep.
    """

    video_index = np.asarray(packed.video_rows.tolist(), dtype=np.int64)
    audio_index = np.asarray(packed.audio_rows.tolist(), dtype=np.int64)
    steps = []
    for video_t, audio_t in zip(video_timesteps, audio_timesteps, strict=True):
        per_row = np.full(packed.rows, np.float32(video_t), dtype=np.float32)
        per_row[video_index[:packed.condition_video_rows]] = np.float32(max(float(video_t), KEYFRAME_NOISE))
        per_row[audio_index] = np.float32(audio_t)
        steps.append(per_row)
    table = np.unique(np.concatenate(steps))
    return mx.array(table), [mx.array(np.searchsorted(table, per_row).astype(np.int32)) for per_row in steps]

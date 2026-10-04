"""H3 checkpoint configuration and sequence geometry, read from the checkpoint's own JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FPS = 24
FRAMES_PER_CHUNK = 17  # the video VAE consumes 17-frame clips ...
LATENTS_PER_CHUNK = 5  # ... and emits 5 latent frames for each, plus 2 for the trailing 5 frames
AUDIO_LATENTS_PER_SECOND = 40
VAE_SPATIAL_RATIO = 16
PARTITIONS = ("FL2VA", "Ref2VA")


def pipeline_root(model_dir) -> Path | None:
    """The folder holding ``model_index.json`` for an H3 pipeline: ``model_dir`` or its FL2VA partition."""

    base = Path(model_dir)
    for candidate in (base, *(base / part for part in PARTITIONS)):
        index = candidate / "model_index.json"
        if not index.is_file():
            continue
        with open(index) as handle:
            if json.load(handle).get("_class_name") == "MiniMaxH3Pipeline" and (candidate / "transformer").is_dir():
                return candidate
    return None


@dataclass(frozen=True)
class DiTConfig:
    """The diffusion transformer, with field names as in ``transformer/config.json``."""

    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 96768
    final_adaln_out_features: int = 10752
    rope_inv_freq_len: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @property
    def inner_dim(self) -> int:
        """Attention width; wider than the residual stream (7168 against 5376) in the released model."""

        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_dict(cls, raw: dict) -> DiTConfig:
        known = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        if "patch_size" in known:
            known["patch_size"] = tuple(known["patch_size"])
        return cls(**known)

    @classmethod
    def from_checkpoint(cls, model_dir) -> DiTConfig:
        root = pipeline_root(model_dir)
        if root is None:
            raise FileNotFoundError(f"{model_dir} is not a MiniMax H3 pipeline folder")
        with open(root / "transformer" / "config.json") as handle:
            return cls.from_dict(json.load(handle))


def align_frames(frames: int) -> int:
    """The largest valid frame count not above ``frames``: H3 clips are ``17 n + 5`` frames long."""

    if frames < LATENTS_PER_CHUNK + FRAMES_PER_CHUNK:
        raise ValueError(f"H3 needs at least {LATENTS_PER_CHUNK + FRAMES_PER_CHUNK} frames, got {frames}")
    return (frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * FRAMES_PER_CHUNK + LATENTS_PER_CHUNK


def latent_frames(frames: int) -> int:
    """Latent frames for a ``17 n + 5`` frame clip: ``5 n + 2``."""

    if frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"frames must be 17 n + 5, got {frames}")
    return (frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latents(frames: int) -> int:
    return round(frames / FPS * AUDIO_LATENTS_PER_SECOND)


def generated_rows(width: int, height: int, frames: int, patch_size=(1, 2, 2)) -> tuple[int, int]:
    """(video rows, audio rows) the transformer denoises for one clip; text rows come on top.

    Audio is stereo with one row per channel per latent.
    """

    step_t, step_h, step_w = patch_size
    cell_h, cell_w = VAE_SPATIAL_RATIO * step_h, VAE_SPATIAL_RATIO * step_w
    if width % cell_w or height % cell_h:
        raise ValueError(f"canvas must be a multiple of {cell_w}x{cell_h}, got {width}x{height}")
    return latent_frames(frames) // step_t * (height // cell_h) * (width // cell_w), 2 * audio_latents(frames)

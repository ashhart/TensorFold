"""Decoding for h3_generate.py: frames and sound from the latents, and the turnaround sheet's five views."""

from __future__ import annotations

import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from tensorfold.families.h3.packing import unpack_audio, unpatchify

AUDIO_PEAK = 0.89  # -1 dBFS: where the loudest sample of a clip is brought to when the decoder overshoots


def unclipped_audio(audio_vae, latents, hard_clip: bool = False):
    """The audio decoder's waveform without its hard clip at +-1, scaled down whole when it peaks over AUDIO_PEAK."""

    import minimax_h3_mlx.audio_vae as module

    class _Passthrough:
        def __getattr__(self, name):
            return getattr(mx, name)

        @staticmethod
        def clip(x, *_):
            return x

    original = module.mx
    module.mx = _Passthrough()
    try:
        wave = np.array(audio_vae.decode(latents))[:, 0, :].astype(np.float32)
    finally:
        module.mx = original
    peak = float(np.abs(wave).max())
    over = int((np.abs(wave) > 1.0).sum())
    if hard_clip:  # the reference decoder's behaviour, kept for comparisons
        print(f"[tensorfold] audio peak {peak:.2f}, {over} samples hard-clipped at full scale", flush=True)
        return np.clip(wave, -1.0, 1.0)
    gain = min(1.0, AUDIO_PEAK / peak) if peak > 0 else 1.0
    print(f"[tensorfold] audio peak {peak:.2f} before any clip, {over} samples past full scale, gain "
          f"{20 * np.log10(gain):.1f} dB", flush=True)
    return wave * np.float32(gain)


def decode(model_dir, root: Path, latents, config, int8: bool = True, upscale_decoder=None, hard_clip: bool = False):
    """Frames from TensorFold's video decoder; the audio decoder is still minimax-h3-mlx's."""

    from minimax_h3_mlx.load import load_audio_vae

    from tensorfold.families.h3.vae_video import load_video_decoder

    parts, mark = {}, time.perf_counter()

    def lap(name):
        nonlocal mark
        parts[name] = round(time.perf_counter() - mark, 2)
        mark = time.perf_counter()

    video_decoder = load_video_decoder(model_dir, int8=int8, upscale_decoder=upscale_decoder)
    audio_vae = load_audio_vae(root / "audio_vae")
    lap("load_vaes")
    video = unpatchify(latents.video_rows, latents.latent_frames, latents.latent_height, latents.latent_width,
                       video_decoder.config.latent_channels, config.patch_size)
    frames = video_decoder.frames(video)
    lap("video_decode")
    acfg = audio_vae.config
    audio = unpack_audio(latents.audio_rows, latents.audio_latents)
    amean = mx.array(np.array(acfg.latents_mean, np.float32)).reshape(1, -1, 1)
    astd = mx.array(np.array(acfg.latents_std, np.float32)).reshape(1, -1, 1)
    wave = unclipped_audio(audio_vae, (audio * astd + amean).astype(mx.float32), hard_clip)
    lap("audio_decode")
    print(f"[tensorfold] decode_parts {parts}", flush=True)
    return frames, wave, acfg.sampling_rate

# Turnaround sheet: five stills ride where a 17-frame clip's five latent steps go; audio is its 28 silent latents
SHEET_SLOTS, SHEET_AUDIO = 5, 28


def write_sheet(model_dir, latents, config, prefix: str, int8: bool = True) -> list[str]:
    """Decode the five slots one by one (each as a still clip) into PREFIX_sheet.png and PREFIX_view1..5.png."""

    from PIL import Image

    from tensorfold.families.h3.vae_video import load_video_decoder

    decoder = load_video_decoder(model_dir, int8=int8)
    video = unpatchify(latents.video_rows, latents.latent_frames, latents.latent_height, latents.latent_width,
                       decoder.config.latent_channels, config.patch_size)
    vae = decoder.config
    shortest = max(2, 2 * math.ceil(vae.clip_length / vae.temporal_ratio) - vae.token_drop)   # latent steps of a clip
    views = []
    for slot in range(latents.latent_frames):
        one = video[:, :, slot:slot + 1]
        views.append(decoder.frames(mx.concatenate([one] * shortest, axis=2))[0])
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [str(base.parent / f"{base.name}_sheet.png")] + [str(base.parent / f"{base.name}_view{i + 1}.png") for i in range(len(views))]
    Image.fromarray(np.concatenate(views, axis=1)).save(outputs[0])
    for view, target in zip(views, outputs[1:]):
        Image.fromarray(view).save(target)
    return outputs

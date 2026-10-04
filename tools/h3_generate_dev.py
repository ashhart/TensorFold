"""Development render: TensorFold's H3 transformer and sampler between a reference text encoder and decoders.

    PYTHONPATH=src:<minimax-h3-mlx> <minimax-h3-mlx>/.venv/bin/python tools/h3_generate_dev.py \
        ~/h3-models/MiniMax-H3 --prompt-file prompt.txt -o out.mp4

The text encoder, the audio decoder and the MP4 writer come from minimax-h3-mlx until the family has its own;
the transformer, sampler, adapters and video decoder are `tensorfold.families.h3`. `--parity` also runs
the reference transformer on the first step's inputs and reports the difference.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from tensorfold.families.h3 import config as h3
from tensorfold.families.h3.lora import merge, settle
from tensorfold.families.h3.packing import unpack_audio, unpatchify
from tensorfold.families.h3.sampler import denoise
from tensorfold.families.h3.schedule import parse_subset
from tensorfold.families.h3.weights import int8_attention, int8_mlp, load_dit


def encode_text(root: Path, prompt: str, image=None):
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    encoder = MiniMaxH3TextEncoder(root / "text_encoder", dtype=mx.bfloat16, load_vision=image is not None)
    if image is not None:
        # the checkpoint's processor folder asks for PyTorch; the image processor built from the vision config
        # needs only numpy, and the image arrives already on the render canvas
        from minimax_h3_mlx.text_encoder import _FallbackProcessor

        encoder._processor = _FallbackProcessor(encoder._build_image_processor())
    embeds, tags = encoder.encode(prompt, [image] if image is not None else None)
    mx.eval(embeds)
    del encoder
    gc.collect()
    mx.clear_cache()
    return embeds, tags


def encode_first_frame(root: Path, image, width: int, height: int, patch):
    """Conditioning rows for a first frame, from minimax-h3-mlx's video VAE encoder (not ported)."""

    from minimax_h3_mlx.load import load_video_vae
    from minimax_h3_mlx.pipeline import encode_keyframe_rows

    vae = load_video_vae(root / "video_vae")
    rows = encode_keyframe_rows(vae, [image], height, width, patch)
    mx.eval(rows)
    del vae
    gc.collect()
    mx.clear_cache()
    return rows


def decode(model_dir, root: Path, latents, config, int8: bool = True, upscale_decoder=None):
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
    wave = np.array(audio_vae.decode((audio * astd + amean).astype(mx.float32)))[:, 0, :].astype(np.float32)
    lap("audio_decode")
    print(f"[tensorfold] decode_parts {parts}", flush=True)
    return frames, wave, acfg.sampling_rate


def parity(root: Path, dit, args, text, tags):
    """Largest and relative difference between this transformer and the reference on one real forward."""

    from minimax_h3_mlx.load import load_dit as load_reference

    from tensorfold.families.h3.packing import layout, timestep_plan
    from tensorfold.families.h3.sampler import start_noise
    from tensorfold.families.h3.schedule import AUDIO_SHIFT, VIDEO_SHIFT, Schedule

    config = dit.config
    frames, lat_h, lat_w = h3.latent_frames(args.frames), args.height // 16, args.width // 16
    packed = layout(tags, frames, lat_h, lat_w, h3.audio_latents(args.frames), config.patch_size)
    video, audio = start_noise(config, frames, lat_h, lat_w, h3.audio_latents(args.frames), args.seed)
    table, plan = timestep_plan(packed, Schedule(VIDEO_SHIFT, args.points).timesteps,
                                Schedule(AUDIO_SHIFT, args.points).timesteps)
    # the reference rounds the latent rows to bfloat16 on the way in; do the same so the comparison is like for like
    call = (video[None].astype(mx.bfloat16), audio[None].astype(mx.bfloat16), text.astype(mx.bfloat16), table,
            plan[0], packed.tags,
            packed.position_ids, packed.video_rows, packed.audio_rows, packed.text_rows)
    ours = dit(*call)
    mx.eval(*ours)
    reference = load_reference(root / "transformer")
    theirs = reference(*call)
    mx.eval(*theirs)
    for name, a, b in zip(("video", "audio"), ours, theirs, strict=True):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        diff = float(mx.max(mx.abs(a - b)).item())
        rel = float((mx.linalg.norm(a - b) / mx.linalg.norm(b)).item())
        print(f"[tensorfold] parity {name}: max abs diff {diff:.3e}, relative {rel:.3e}, "
              f"reference rms {float(mx.sqrt(mx.mean(b * b)).item()):.3f}", flush=True)
    del reference
    gc.collect()
    mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--points", type=int, default=21, help="sigma points; one fewer forwards")
    parser.add_argument("--subset", default=None, help="N:i0,i1,... keeps those points of an N-point grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora", action="append", default=[], help="PATH or PATH:STRENGTH, merged at load")
    parser.add_argument("--int8-mlp", action="store_true", help="run the block MLPs through the int8 kernel")
    parser.add_argument("--int8-qkv", action="store_true", help="int8 QKV projection")
    parser.add_argument("--int8-out", action="store_true", help="int8 attention-output projection")
    parser.add_argument("--unfused-qkv", action="store_true", help="int8 QKV without the fused norm and rotation")
    parser.add_argument("--keep-adaln", action="store_true", help="keep the AdaLN projection weights loaded")
    parser.add_argument("--upscale-vae", help="safetensors of a packed-head (2x) video decoder; frames come out larger")
    parser.add_argument("--float-vae", action="store_true", help="video decoder in float32, without int8 kernels")
    parser.add_argument("--first-frame", default=None, help="image the clip starts from (image to video)")
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()

    root = h3.pipeline_root(args.model_dir)
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    started = time.perf_counter()
    image = None
    if args.first_frame:
        from minimax_h3_mlx.packing import prepare_keyframe_image
        from PIL import Image

        image = prepare_keyframe_image(Image.open(args.first_frame).convert("RGB"), args.height, args.width,
                                       stretch=True)
    text, tags = encode_text(root, prompt, image)
    condition = None
    if image is not None:
        condition = encode_first_frame(root, image, args.width, args.height,
                                       h3.DiTConfig.from_checkpoint(args.model_dir).patch_size)
    text_seconds = time.perf_counter() - started
    print(f"[tensorfold] text: {text.shape[1]} rows in {text_seconds:.1f}s", flush=True)

    started = time.perf_counter()
    dit = load_dit(args.model_dir)
    for spec in args.lora:
        path, _, strength = spec.partition(":")
        print(f"[tensorfold] {merge(dit, path, float(strength or 1.0))}", flush=True)
    if args.int8_mlp:
        print(f"[tensorfold] int8 MLP in {int8_mlp(dit)} blocks", flush=True)
    if args.int8_qkv or args.int8_out:
        changed = int8_attention(dit, qkv=args.int8_qkv, out=args.int8_out, fused=not args.unfused_qkv)
        print(f"[tensorfold] int8 attention projections: {changed}", flush=True)
    rounded = settle(dit)
    if rounded and args.lora:
        print(f"[tensorfold] {rounded} adapted projections rounded to bfloat16 (not on an int8 kernel)", flush=True)
    load_seconds = time.perf_counter() - started
    if args.parity:
        parity(root, dit, args, text, tags)

    points, subset = args.points, None
    if args.subset:
        points, subset = parse_subset(args.subset)
    started = time.perf_counter()
    latents = denoise(dit, text, tags, args.width, args.height, args.frames, points, args.seed, subset,
                      release=not args.keep_adaln, condition=condition,
                      keyframes=("first",) if condition is not None else (),
                      on_step=lambda i, n, s: print(f"[tensorfold] step {i}/{n} {s:.2f}s", flush=True))
    denoise_seconds = time.perf_counter() - started
    del dit
    gc.collect()
    mx.clear_cache()

    started = time.perf_counter()
    frames, wave, rate = decode(args.model_dir, root, latents, h3.DiTConfig.from_checkpoint(args.model_dir),
                                int8=not args.float_vae, upscale_decoder=args.upscale_vae)
    from minimax_h3_mlx.media import save_mp4

    mux_started = time.perf_counter()
    save_mp4(args.output, frames, h3.FPS, audio=wave, sample_rate=rate)
    print(f"[tensorfold] mux {time.perf_counter() - mux_started:.2f}s", flush=True)
    decode_seconds = time.perf_counter() - started
    report = {"output": args.output, "rows": latents.packed.rows, "forwards": len(latents.step_seconds),
              "text_s": round(text_seconds, 1), "load_s": round(load_seconds, 1),
              "denoise_s": round(denoise_seconds, 1), "decode_s": round(decode_seconds, 1),
              "per_forward_s": round(float(np.median(latents.step_seconds)), 2),
              "peak_gib": round(mx.get_peak_memory() / 2**30, 1)}
    print("[tensorfold] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

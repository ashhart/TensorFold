#!/usr/bin/env python3
"""One reference case for the native FastH3 transformer (zig/src/families/h3), and the decode of what it returns.

Writes case.safetensors from the Python 0.6 family, float projections and FastVideo's reference routing:
  text                 the prompt's rows after the condition projection and the token refiner (they never change)
  video, audio         the starting noise rows
  condition            with --first-frame, the image's rows at the keyframe noise level; they are never stepped
  cos, sin             rotary tables for every packed row, in the order [text | keyframe | audio | video]
  adaln, times         per step, each row's line in a block's modulation tables and in the final layer's
  tables.N, final      every block's six modulation tables for the run's timesteps, and the final layer's
  video_step, audio_step  per step (sigma seen, Euler ratio)
  tile_slot, tile_sizes, geometry   FastVideo's tile map: each row's padded slot, each tile's rows, the counts
  the small float projections (patches in, heads out) and the first forward's two velocities
The block projections are read by the native tool from the checkpoint itself.
`--decode PREFIX` instead decodes PREFIX.video.f32 and PREFIX.audio.f32 (raw rows) into --mp4.
`--revoice N` with it first makes the audio rows again with the released model (no distillation) in N steps against
the finished picture, as families/h3/sampler.py `revoice` does; it needs the prompt (--prompt-file) and --seed.
`--first-frame IMAGE` starts the clip from an image: its vision tokens join the prompt's rows and its encoded rows sit
between the text and the audio, as tools/h3_generate_dev.py --first-frame does (the image is stretched onto the canvas).
`--steps N` exports N forwards instead of the checkpoint's own count, rungs spread evenly from its first rung.
`--at K` lets the reference take steps 0..K-1 and writes step K's rows and velocities (the native tool runs from=K).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="the MiniMax H3 pipeline folder (text encoder and decoders)")
    parser.add_argument("fasth3", help="the FastH3 checkpoint folder")
    parser.add_argument("out_dir")
    parser.add_argument("--tools", required=True, help="folder holding h3_generate_dev.py")
    parser.add_argument("--prompt-file")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode", help="prefix of the raw float32 rows to decode into --mp4")
    parser.add_argument("--mp4")
    parser.add_argument("--crop", help="with --decode: WxH, centre-crop the decoded frames before the MP4 is written")
    parser.add_argument("--steps", type=int, default=0, help="forwards, when not the checkpoint's trained count")
    parser.add_argument("--revoice", type=int, default=0, metavar="STEPS",
                        help="with --decode: audio made again by the released model in this many steps")
    parser.add_argument("--first-frame", help="image the clip starts from (image to video)")
    parser.add_argument("--at", type=int, default=0,
                        help="the reference takes steps 0..AT-1 first; the case holds step AT (the native tool runs from=AT)")
    parser.add_argument("--no-reference", action="store_true",
                        help="skip the reference forward (its velocities are written as zeros): for renders, not parity")
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, args.tools)
    import h3_generate_dev as tool

    from tensorfold.families.h3 import config as h3
    from tensorfold.families.h3 import fasth3
    from tensorfold.families.h3.dit import MODALITIES, rotary_tables, timestep_embedding
    from tensorfold.families.h3.packing import layout, timestep_plan
    from tensorfold.families.h3.sampler import start_noise
    from tensorfold.families.h3.schedule import Schedule

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = h3.pipeline_root(args.model_dir)
    frames_l = h3.latent_frames(args.frames)
    lat_h, lat_w = args.height // h3.VAE_SPATIAL_RATIO, args.width // h3.VAE_SPATIAL_RATIO
    audio_l = h3.audio_latents(args.frames)
    config = fasth3.config(args.fasth3)

    if args.decode:
        video = np.fromfile(args.decode + ".video.f32", dtype=np.float32).reshape(-1, config.latents_dim * 4)
        audio = np.fromfile(args.decode + ".audio.f32", dtype=np.float32).reshape(-1, config.audio_latents_dim)
        latents = SimpleNamespace(video_rows=mx.array(video), audio_rows=mx.array(audio), latent_frames=frames_l,
                                  latent_height=lat_h, latent_width=lat_w, audio_latents=audio_l)
        if args.revoice:
            import gc

            from tensorfold.families.h3.sampler import revoice

            # the native run keeps no layout: rebuild it from the prompt, as the export did
            text, tags = tool.encode_text(root, Path(args.prompt_file).read_text())
            latents.packed = layout(tags, frames_l, lat_h, lat_w, audio_l, config.patch_size)
            mark = time.perf_counter()
            base = tool.load_dit(args.model_dir)
            tool.int8_mlp(base)
            tool.int8_attention(base, qkv=True, out=True, fused=True)
            latents.audio_rows = revoice(base, text, latents, args.revoice + 1, args.seed, release=True,
                                         on_step=lambda i, n, t: print(f"[tensorfold] voice {i}/{n} {t:.2f}s", flush=True))
            mx.eval(latents.audio_rows)
            print(f"[tensorfold] re-voiced in {time.perf_counter() - mark:.1f}s", flush=True)
            del base
            gc.collect()
            mx.clear_cache()
        frames, wave, rate = tool.decode(args.model_dir, root, latents, h3.DiTConfig.from_checkpoint(args.model_dir))
        from minimax_h3_mlx.media import save_mp4

        if args.crop:
            crop_w, crop_h = (int(v) for v in args.crop.lower().split("x"))
            top, left = (frames.shape[1] - crop_h) // 2, (frames.shape[2] - crop_w) // 2
            frames = np.ascontiguousarray(frames[:, top : top + crop_h, left : left + crop_w])
        save_mp4(args.mp4, frames, h3.FPS, audio=wave, sample_rate=rate)
        print(json.dumps({"mp4": args.mp4, "frames": int(frames.shape[0])}))
        return 0

    image, condition, keyframes = None, None, ()
    if args.first_frame:
        from minimax_h3_mlx.packing import prepare_keyframe_image
        from PIL import Image

        image = prepare_keyframe_image(Image.open(args.first_frame).convert("RGB"), args.height, args.width, stretch=True)
    text, tags = tool.encode_text(root, Path(args.prompt_file).read_text(), image)
    if image is not None:
        condition, keyframes = tool.encode_first_frame(root, image, args.width, args.height, config.patch_size), ("first",)
    dit, gates, fast = fasth3.load_fasth3(args.fasth3)
    packed = layout(tags, frames_l, lat_h, lat_w, audio_l, config.patch_size, keyframes)
    held = packed.condition_video_rows
    # the keyframe's rows come first among the video rows, already at their noise level
    video, audio = start_noise(config, frames_l, lat_h, lat_w, audio_l, args.seed, condition)
    prepare = fasth3.route(dit, gates, fast.sparsity, fast.tile, "reference")
    geometry = prepare(packed, frames_l, lat_h, lat_w)
    nodes = fast.nodes
    if args.steps and args.steps != len(nodes):
        nodes = tuple(nodes[0] * (1 - i / args.steps) for i in range(args.steps))
    schedules = [Schedule(fast.video_shift, 0, None, nodes), Schedule(fast.audio_shift, 0, None, nodes)]
    table, plan = timestep_plan(packed, schedules[0].timesteps, schedules[1].timesteps)
    text = text.astype(mx.bfloat16)
    rows = dit.token_refiner(dit.condition_proj(text.astype(dit.condition_proj.weight.dtype)).astype(mx.bfloat16))
    temb = dit.time_embedder(timestep_embedding(table, config.timestep_input_dim))
    cos, sin = rotary_tables(config, packed.position_ids)
    lines = np.maximum(np.asarray(packed.tags), 0)
    case = {
        "text": rows[0].astype(mx.bfloat16), "video": video[held:].astype(mx.float32), "audio": audio.astype(mx.float32),
        "cos": cos.astype(mx.float32), "sin": sin.astype(mx.float32),
        "adaln": mx.array(np.stack([np.asarray(p) * MODALITIES + lines for p in plan]).astype(np.int32)),
        "times": mx.array(np.stack([np.asarray(p) for p in plan]).astype(np.int32)),
        "final": dit.final_layer.adaln_proj(temb).astype(mx.bfloat16),
        "tile_slot": mx.array(geometry.untile_combined_index.astype(np.int32)),
        "tile_sizes": mx.array(geometry.variable_block_sizes.astype(np.int32)),
        "geometry": mx.array(np.asarray([geometry.num_prefix_tiles, geometry.num_video_tiles,
                                         fasth3_keep(fast.sparsity, geometry.num_video_tiles),
                                         int(packed.text_rows.shape[0]), int(packed.audio_rows.shape[0]),
                                         int(packed.video_rows.shape[0]) - held, held][: 7 if held else 6],
                                        dtype=np.int32)),
        "video_in.weight": dit.video_patch_proj.weight.astype(mx.float32),
        "video_in.bias": dit.video_patch_proj.bias.astype(mx.float32),
        "audio_in.weight": dit.audio_patch_proj.weight.astype(mx.float32),
        "audio_in.bias": dit.audio_patch_proj.bias.astype(mx.float32),
        "video_out.weight": dit.final_layer.video_out.weight.astype(mx.float32),
        "video_out.bias": dit.final_layer.video_out.bias.astype(mx.float32),
        "audio_out.weight": dit.final_layer.audio_out.weight.astype(mx.float32),
        "audio_out.bias": dit.final_layer.audio_out.bias.astype(mx.float32),
        "final_norm.weight": dit.final_layer.norm.weight.astype(mx.bfloat16),
    }
    if held:
        case["condition"] = video[:held].astype(mx.float32)
    for name, schedule in zip(("video_step", "audio_step"), schedules, strict=True):
        steps = [(float(np.float32(1.0) - schedule.timesteps[i]), float(schedule.sigmas[i + 1] / schedule.sigmas[i]))
                 for i in range(len(schedule))]
        case[name] = mx.array(np.asarray(steps, dtype=np.float32))
    for index, block in enumerate(dit.blocks):
        case[f"tables.{index}"] = mx.stack(block.adaln_proj.tables(temb)).astype(mx.bfloat16)   # (6, lines, hidden)
    mx.eval(*case.values())
    mark = time.perf_counter()
    if args.no_reference:
        case["video_velocity"], case["audio_velocity"] = mx.zeros_like(video[held:]), mx.zeros_like(audio)
    else:
        pairs = (np.asarray(case["video_step"]), np.asarray(case["audio_step"]))
        for i in range(args.at + 1):
            first = dit(video[None], audio[None], text, table, plan[i], packed.tags, packed.position_ids,
                        packed.video_rows, packed.audio_rows, packed.text_rows)
            mx.eval(*first)
            if i == args.at:
                break
            # the Euler step the native tool takes, on the generated rows only
            (seen_v, ratio_v), (seen_a, ratio_a) = (float(x) for x in pairs[0][i]), (float(x) for x in pairs[1][i])
            moved = video[held:].astype(mx.float32)
            moved = ratio_v * moved + (1 - ratio_v) * (moved + seen_v * first[0][0, held:].astype(mx.float32))
            video = mx.concatenate([video[:held].astype(mx.float32), moved]) if held else moved
            audio = audio.astype(mx.float32)
            audio = ratio_a * audio + (1 - ratio_a) * (audio + seen_a * first[1][0].astype(mx.float32))
            mx.eval(video, audio)
        case["video"], case["audio"] = video[held:].astype(mx.float32), audio.astype(mx.float32)
        case["video_velocity"] = first[0][0, held:].astype(mx.float32)
        case["audio_velocity"] = first[1][0].astype(mx.float32)
    mx.save_safetensors(str(out / "case.safetensors"), case)
    print(json.dumps({"rows": packed.rows, "text": int(packed.text_rows.shape[0]), "audio": int(packed.audio_rows.shape[0]),
                      "video": int(packed.video_rows.shape[0]) - held, "keyframe": held, "timesteps": int(table.shape[0]),
                      "tiles": geometry.num_tiles, "prefix_tiles": geometry.num_prefix_tiles,
                      "hidden": config.hidden_size, "layers": config.num_layers,
                      "reference_forward_s": round(time.perf_counter() - mark, 1)}))
    return 0


def fasth3_keep(sparsity: float, video_tiles: int) -> int:
    from tensorfold.families.h3.vendor import fastvideo_vsa as vsa

    return vsa.compute_topk(sparsity, video_tiles)


if __name__ == "__main__":
    raise SystemExit(main())

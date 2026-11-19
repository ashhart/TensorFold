#!/usr/bin/env python3
"""A Ref2VA case for the native H3 transformer: the released MiniMax-H3 Ref2VA model with reference pictures, videos
and audio clips (families/h3/packing.py `reference_layout`, families/h3/sampler.py `denoise` with references).

The reference rows are read and encoded by the Studio's h3_generate.py (`--studio` names its folder), as the Studio
does for a Ref2VA clip. The rows are then reordered into the native engine's order, [text | reference video rows |
audio, reference audio first | video], with every per-row table: attention does not depend on the order once each
row keeps its position, modality and tile. `--at K` lets the reference take steps 0..K-1 first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="the MiniMax H3 Ref2VA pipeline folder")
    parser.add_argument("out_dir")
    parser.add_argument("--studio", required=True, help="folder holding the Studio's h3_generate.py")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--video-sound", action="store_true", help="a reference video's own sound is a reference too")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--at", type=int, default=0)
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, args.studio)
    import h3_generate as studio

    from tensorfold.families.h3 import config as h3
    from tensorfold.families.h3.dit import MODALITIES, rotary_tables, timestep_embedding
    from tensorfold.families.h3.packing import reference_layout, timestep_plan
    from tensorfold.families.h3.sampler import start_noise
    from tensorfold.families.h3.schedule import AUDIO_SHIFT, VIDEO_SHIFT, Schedule
    from tensorfold.families.h3.vendor import fastvideo_vsa as vsa
    from tensorfold.families.h3.weights import load_dit

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = h3.pipeline_root(args.model_dir)
    frames_l = h3.latent_frames(args.frames)
    lat_h, lat_w = args.height // h3.VAE_SPATIAL_RATIO, args.width // h3.VAE_SPATIAL_RATIO
    audio_l = h3.audio_latents(args.frames)
    config = h3.DiTConfig.from_checkpoint(args.model_dir)
    with open(root / "audio_vae" / "metadata.json") as handle:
        rate = int(json.load(handle)["metadata"]["kwargs"]["sample_rate"])
    items = studio.prepare_reference_media(args.image, args.video, args.audio, args.width, args.height, args.frames,
                                           rate, video_sound=args.video_sound)
    text, tags = studio.encode_reference_text(root, Path(args.prompt_file).read_text(), items, None, True)
    held_video, held_audio, references = studio.encode_reference_rows(root, items, config.patch_size)
    packed = reference_layout(tags, references, frames_l, lat_h, lat_w, audio_l, config.patch_size)
    video, audio = start_noise(config, frames_l, lat_h, lat_w, audio_l, args.seed, held_video)
    held = packed.condition_video_rows
    if held_audio is not None:
        audio = mx.concatenate([held_audio.astype(mx.float32), audio])
    video_schedule = Schedule(VIDEO_SHIFT, args.steps + 1)
    audio_schedule = Schedule(AUDIO_SHIFT, args.steps + 1)
    table, plan = timestep_plan(packed, video_schedule.timesteps, audio_schedule.timesteps)

    dit = load_dit(args.model_dir)
    text_rows = int(packed.text_rows.shape[0])
    audio_rows = int(packed.audio_rows.shape[0])
    shape = (frames_l // config.patch_size[0], lat_h // config.patch_size[1], lat_w // config.patch_size[2])
    # one prefix segment for the references: their rows sit between the text and the clip's audio
    references_rows = held + packed.condition_audio_rows
    segments = tuple(n for n in (text_rows, references_rows, audio_rows - packed.condition_audio_rows) if n > 0)
    geometry = vsa.build_h3_tile_geometry(segments, shape, 64)
    video_index = np.asarray(packed.video_rows.tolist(), dtype=np.int64)
    audio_index = np.asarray(packed.audio_rows.tolist(), dtype=np.int64)
    text_index = np.asarray(packed.text_rows.tolist(), dtype=np.int64)
    order = np.concatenate([text_index, video_index[:held], audio_index, video_index[held:]])

    text = text.astype(mx.bfloat16)
    rows = dit.token_refiner(dit.condition_proj(text.astype(dit.condition_proj.weight.dtype)).astype(mx.bfloat16))
    temb = dit.time_embedder(timestep_embedding(table, config.timestep_input_dim))
    cos, sin = rotary_tables(config, packed.position_ids)
    lines = np.maximum(np.asarray(packed.tags), 0)
    counts = [geometry.num_prefix_tiles, geometry.num_video_tiles, geometry.num_video_tiles, text_rows,
              audio_rows, int(video_index.shape[0]) - held, held, packed.condition_audio_rows]

    def steps(schedule):
        return mx.array(np.asarray([(float(np.float32(1.0) - schedule.timesteps[i]),
                                     float(schedule.sigmas[i + 1] / schedule.sigmas[i]))
                                    for i in range(len(schedule))], dtype=np.float32))

    case = {
        "text": rows[0].astype(mx.bfloat16), "video": video[held:].astype(mx.float32), "audio": audio,
        "cos": mx.array(np.asarray(cos)[order]), "sin": mx.array(np.asarray(sin)[order]),
        "adaln": mx.array(np.stack([(np.asarray(p) * MODALITIES + lines)[order] for p in plan]).astype(np.int32)),
        "times": mx.array(np.stack([np.asarray(p)[order] for p in plan]).astype(np.int32)),
        "final": dit.final_layer.adaln_proj(temb).astype(mx.bfloat16),
        "tile_slot": mx.array(geometry.untile_combined_index.astype(np.int32)[order]),
        "tile_sizes": mx.array(geometry.variable_block_sizes.astype(np.int32)),
        "geometry": mx.array(np.asarray(counts, dtype=np.int32)),
        "video_in.weight": dit.video_patch_proj.weight.astype(mx.float32),
        "video_in.bias": dit.video_patch_proj.bias.astype(mx.float32),
        "audio_in.weight": dit.audio_patch_proj.weight.astype(mx.float32),
        "audio_in.bias": dit.audio_patch_proj.bias.astype(mx.float32),
        "video_out.weight": dit.final_layer.video_out.weight.astype(mx.float32),
        "video_out.bias": dit.final_layer.video_out.bias.astype(mx.float32),
        "audio_out.weight": dit.final_layer.audio_out.weight.astype(mx.float32),
        "audio_out.bias": dit.final_layer.audio_out.bias.astype(mx.float32),
        "final_norm.weight": dit.final_layer.norm.weight.astype(mx.bfloat16),
        "video_step": steps(video_schedule), "audio_step": steps(audio_schedule),
    }
    if held:
        case["condition"] = video[:held].astype(mx.float32)
    for index, block in enumerate(dit.blocks):
        case[f"tables.{index}"] = mx.stack(block.adaln_proj.tables(temb)).astype(mx.bfloat16)
    mx.eval(*case.values())
    mark = time.perf_counter()
    held_audio_rows = packed.condition_audio_rows
    for i in range(args.at + 1):
        started = time.perf_counter()
        first = dit(video[None], audio[None], text, table, plan[i], packed.tags, packed.position_ids,
                    packed.video_rows, packed.audio_rows, packed.text_rows)
        mx.eval(*first)
        print(f"forward {i}: {time.perf_counter() - started:.2f} s", flush=True)
        if i < args.at:
            stepped_video = video_schedule.step(i, first[0][0, held:], video[held:])
            stepped_audio = audio_schedule.step(i, first[1][0, held_audio_rows:], audio[held_audio_rows:])
            video = mx.concatenate([video[:held], stepped_video]) if held else stepped_video
            audio = mx.concatenate([audio[:held_audio_rows], stepped_audio]) if held_audio_rows else stepped_audio
            mx.eval(video, audio)
    case["video"], case["audio"] = video[held:].astype(mx.float32), audio.astype(mx.float32)
    case["video_velocity"] = first[0][0, held:].astype(mx.float32)
    case["audio_velocity"] = first[1][0].astype(mx.float32)
    mx.save_safetensors(str(out / "case.safetensors"), case)
    print(json.dumps({"rows": packed.rows, "text": text_rows, "reference_video": held,
                      "reference_audio": held_audio_rows, "audio": audio_rows, "timesteps": int(table.shape[0]),
                      "tiles": geometry.num_tiles, "prefix_tiles": geometry.num_prefix_tiles, "at": args.at,
                      "reference_s": round(time.perf_counter() - mark, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

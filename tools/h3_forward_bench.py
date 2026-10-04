"""Time the H3 transformer forward on MLX at a real canvas, with real weights and synthetic inputs.

    python tools/h3_forward_bench.py ~/h3-models/MiniMax-H3 --width 768 --height 448 --frames 124

Reports seconds per forward and a per-stage split of one block, for comparison with a reference engine's
denoise time per step at the same canvas.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

from tensorfold.families.h3 import config as h3
from tensorfold.families.h3.weights import load_dit


def inputs(config, width, height, frames, text_rows, seed=0):
    video_rows, audio_rows = h3.generated_rows(width, height, frames, config.patch_size)
    lat_t, lat_h, lat_w = h3.latent_frames(frames), height // 32, width // 32
    mx.random.seed(seed)
    rows = text_rows + video_rows + audio_rows
    t, y, x = mx.meshgrid(mx.arange(lat_t), mx.arange(lat_h), mx.arange(lat_w), indexing="ij")
    video_pos = mx.stack([t.reshape(-1), y.reshape(-1), x.reshape(-1)], axis=-1)
    audio_pos = mx.stack([mx.arange(audio_rows), mx.zeros(audio_rows, dtype=mx.int32),
                          mx.zeros(audio_rows, dtype=mx.int32)], axis=-1)
    position_ids = mx.concatenate([mx.zeros((text_rows, 3), dtype=mx.int32), video_pos, audio_pos])
    tags = mx.concatenate([mx.full((text_rows,), 1), mx.full((video_rows,), 0), mx.full((audio_rows,), 2)])
    timestep_rows = mx.concatenate([mx.zeros(text_rows + video_rows, dtype=mx.int32),
                                    mx.ones(audio_rows, dtype=mx.int32)])
    patch = config.latents_dim * config.patch_size[1] * config.patch_size[2]
    batch = {
        "video": mx.random.normal((1, video_rows, patch)),
        "audio": mx.random.normal((1, audio_rows, config.audio_latents_dim)),
        "text": mx.random.normal((1, text_rows, config.text_dim)),
        "timestep": mx.array([0.7, 0.5]),
        "timestep_rows": timestep_rows, "tags": tags, "position_ids": position_ids,
        "text_rows": mx.arange(text_rows),
        "video_rows": mx.arange(text_rows, text_rows + video_rows),
        "audio_rows": mx.arange(text_rows + video_rows, rows),
    }
    return batch, rows


def timed(fn, repeats):
    out = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        out.append(time.perf_counter() - start)
    return statistics.median(out), min(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--text-rows", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=None, help="load only the first N blocks")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--int8", default="", help="comma list of mlp,qkv,out to run through int8 kernels")
    parser.add_argument("--half", default=None, help="recast bfloat16 weights, e.g. float16")
    args = parser.parse_args()

    start = time.perf_counter()
    model = load_dit(args.model_dir, blocks=args.blocks, half=args.half)
    config = model.config
    chosen = {part for part in args.int8.split(",") if part}
    if "mlp" in chosen:
        from tensorfold.families.h3.weights import int8_mlp

        int8_mlp(model)
    if chosen & {"qkv", "out"}:
        from tensorfold.families.h3.weights import int8_attention

        int8_attention(model, qkv="qkv" in chosen, out="out" in chosen)
    print(f"[tensorfold] h3 dit: {config.num_layers} blocks loaded in {time.perf_counter() - start:.1f}s, "
          f"active {mx.get_active_memory() / 2**30:.1f} GiB")
    batch, rows = inputs(config, args.width, args.height, args.frames, args.text_rows)
    print(f"[tensorfold] canvas {args.width}x{args.height}x{args.frames}: {rows:,} rows "
          f"({batch['video'].shape[1]:,} video, {batch['audio'].shape[1]} audio, {args.text_rows} text)")

    mx.eval(model(**batch))  # first call compiles kernels and touches every weight
    median, best = timed(lambda: model(**batch), args.repeats)
    print(f"[tensorfold] forward: {median:.2f}s median, {best:.2f}s best over {args.repeats} "
          f"({median / config.num_layers * 1000:.0f} ms per block), peak {mx.get_peak_memory() / 2**30:.1f} GiB")

    x, adaln_rows, rotary = model.pack(**batch)
    block = model.blocks[0]
    tables = model.modulation(batch["timestep"])[0][0]
    temb = model.time_embedder(__import__("tensorfold.families.h3.dit", fromlist=["timestep_embedding"]).timestep_embedding(batch["timestep"], config.timestep_input_dim))
    h = block.norm1(x) * (1.0 + tables[1][adaln_rows]) + tables[0][adaln_rows]
    mx.eval(x, h, *tables, *rotary)
    q, k, v = block.attn.qkv(h, rotary)
    mx.eval(q, k, v)
    mixed = block.attn.mix(q, k, v).astype(x.dtype)
    mx.eval(mixed)
    stages = {
        "adaln projection": lambda: block.adaln_proj.tables(temb),
        "norm + modulate": lambda: block.norm1(x) * (1.0 + tables[1][adaln_rows]) + tables[0][adaln_rows],
        "qkv + q/k norm + rotary": lambda: block.attn.qkv(h, rotary),
        "attention": lambda: block.attn.mix(q, k, v),
        "attention out": lambda: block.attn.out_proj(mixed),
        "mlp": lambda: block.mlp(h),
    }
    total = 0.0
    for name, fn in stages.items():
        mx.eval(fn())
        median, _ = timed(fn, max(args.repeats, 5))
        total += median
        print(f"[tensorfold]   {name:<26} {median * 1000:7.1f} ms")
    print(f"[tensorfold]   {'sum of stages':<26} {total * 1000:7.1f} ms per block")


if __name__ == "__main__":
    main()

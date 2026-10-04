"""Time and check the int8 H3 MLP kernel against the bf16 MLP on block 0's released weights.

    python tools/h3_mlp_kernel_bench.py ~/h3-models/MiniMax-H3

The input is block 0's real MLP input for a synthetic packed sequence; ``--noise`` uses unit-RMS noise at fixed
row counts instead.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx import nn

from tensorfold.families.h3 import config as h3
from tensorfold.kernels.minimax.h3.v1.mlp_int8 import FC2_GROUP, Int8MLP

CANVASES = ((768, 448, 124), (864, 480, 124))
NOISE_ROWS = (12895, 15448)   # the packed sequences of those canvases with 256 text rows


def block0_mlp_input(model, width, height, frames, text_rows=256):
    """The tensor block 0 hands its MLP for random latents: norm2 of the post-attention stream, modulated."""

    config = model.config
    video_rows, audio_rows = h3.generated_rows(width, height, frames, config.patch_size)
    lat_t, lat_h, lat_w = h3.latent_frames(frames), height // 32, width // 32
    mx.random.seed(0)
    t, y, x = mx.meshgrid(mx.arange(lat_t), mx.arange(lat_h), mx.arange(lat_w), indexing="ij")
    video_pos = mx.stack([t.reshape(-1), y.reshape(-1), x.reshape(-1)], axis=-1)
    zeros = mx.zeros(audio_rows, dtype=mx.int32)
    audio_pos = mx.stack([mx.arange(audio_rows), zeros, zeros], axis=-1)
    rows = text_rows + video_rows + audio_rows
    patch = config.latents_dim * config.patch_size[1] * config.patch_size[2]
    x, temb, adaln_rows, rotary = model.pack(
        video=mx.random.normal((1, video_rows, patch)),
        audio=mx.random.normal((1, audio_rows, config.audio_latents_dim)),
        text=mx.random.normal((1, text_rows, config.text_dim)),
        timestep=mx.array([0.7, 0.5]),
        timestep_rows=mx.concatenate([mx.zeros(text_rows + video_rows, dtype=mx.int32),
                                      mx.ones(audio_rows, dtype=mx.int32)]),
        tags=mx.concatenate([mx.full((text_rows,), 1), mx.full((video_rows,), 0), mx.full((audio_rows,), 2)]),
        position_ids=mx.concatenate([mx.zeros((text_rows, 3), dtype=mx.int32), video_pos, audio_pos]),
        video_rows=mx.arange(text_rows, text_rows + video_rows),
        audio_rows=mx.arange(text_rows + video_rows, rows),
        text_rows=mx.arange(text_rows))
    block = model.blocks[0]
    shift_a, scale_a, gate_a, shift_m, scale_m, _ = block.adaln_proj.tables(temb)
    h = block.norm1(x) * (1.0 + scale_a[adaln_rows]) + shift_a[adaln_rows]
    x = x + gate_a[adaln_rows] * block.attn(h, rotary)
    out = (block.norm2(x) * (1.0 + scale_m[adaln_rows]) + shift_m[adaln_rows])[0]
    mx.eval(out)
    return out


def swiglu(fused, width):
    return nn.silu(fused[:, :width]) * fused[:, width:]


def timed(fn, repeats):
    mx.eval(fn())
    mx.eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000, min(samples) * 1000


def compare(reference: mx.array, other: mx.array) -> tuple[float, float]:
    a, b = reference.astype(mx.float32), other.astype(mx.float32)
    error = mx.sqrt(mx.sum((a - b) ** 2) / mx.sum(a**2)).item()
    cosine = (mx.sum(a * b) / mx.sqrt(mx.sum(a**2) * mx.sum(b**2))).item()
    return error, cosine


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--noise", action="store_true", help="unit-RMS noise input instead of block 0's own")
    parser.add_argument("--fc2-group", type=int, default=FC2_GROUP)
    args = parser.parse_args()

    from tensorfold.families.h3.weights import load_dit

    model = load_dit(args.model_dir, blocks=1)
    mlp = model.blocks[0].mlp
    fast = Int8MLP(mlp.fc1.weight, mlp.fc2.weight, fc2_group=args.fc2_group)
    print(f"[tensorfold] h3 mlp: fc1 {tuple(mlp.fc1.weight.shape)}, fc2 {tuple(mlp.fc2.weight.shape)}, "
          f"fc2 activation scale per {fast.fc2_group} channels")
    for (width, height, frames), noise_rows in zip(CANVASES, NOISE_ROWS):
        if args.noise:
            x = mx.random.normal((noise_rows, model.config.hidden_size)).astype(mx.bfloat16)
        else:
            x = block0_mlp_input(model, width, height, frames)
        mx.eval(x)
        reference = mlp(x)
        result = fast(x)
        mx.eval(reference, result)
        error, cosine = compare(reference, result)
        base, base_best = timed(lambda x=x: mlp(x), args.repeats)
        quick, quick_best = timed(lambda x=x: fast(x), args.repeats)
        rows = x.shape[0]
        print(f"[tensorfold] {width}x{height}x{frames}, {rows:,} rows: bf16 {base:.1f} ms (best {base_best:.1f}), "
              f"int8 {quick:.1f} ms (best {quick_best:.1f}), {base / quick:.2f}x; "
              f"relative L2 error {error:.4f}, cosine {cosine:.5f}")

    # where the int8 time goes, at the last canvas
    from tensorfold.kernels.minimax.h3.v1 import mlp_int8 as k

    x2 = x.reshape(-1, fast.hidden)
    xq, xs, rows = k.quantize_rows(x2, fast.hidden)
    hidden = k.int8_swiglu(xq, xs, fast.w1, fast.s1, rows)
    hq, hs, _ = k.quantize_rows(hidden, fast.fc2_group)
    mx.eval(xq, xs, hidden, hq, hs)
    stages = {
        "quantize fc1 input": lambda: k.quantize_rows(x2, fast.hidden)[:2],
        "fc1 + swiglu int8": lambda: k.int8_swiglu(xq, xs, fast.w1, fast.s1, rows),
        "quantize fc2 input": lambda: k.quantize_rows(hidden, fast.fc2_group)[:2],
        "fc2 int8": lambda: k.int8_linear(hq, hs, fast.w2, fast.s2, rows, fast.fc2_group),
        "bf16 fc1 + swiglu": lambda: swiglu(mlp.fc1(x2), fast.width),
        "bf16 fc2": lambda: mlp.fc2(hidden),
    }
    for name, fn in stages.items():
        median, _ = timed(fn, args.repeats)
        print(f"[tensorfold]   {name:<20} {median:6.1f} ms")

if __name__ == "__main__":
    main()

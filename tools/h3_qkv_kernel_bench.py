"""Time and check the fused int8 H3 QKV kernel against the bf16 and the unfused int8 paths on block 0.

    python tools/h3_qkv_kernel_bench.py ~/h3-models/MiniMax-H3

The input is block 0's real attention input for a synthetic packed sequence at two canvases.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

from tensorfold.families.h3 import config as h3
from tensorfold.families.h3.dit import apply_rotary
from tensorfold.kernels.minimax.h3.v1.mlp_int8 import Int8Linear
from tensorfold.kernels.minimax.h3.v1.qkv_int8 import Int8QKV

CANVASES = ((768, 448, 124), (864, 480, 124))


def block0_attention_input(model, width, height, frames, text_rows=256):
    """(h, cos, sin): what block 0 hands its attention for random latents, and the rotary tables."""

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
    shift, scale = block.adaln_proj.tables(temb)[:2]
    h = block.norm1(x) * (1.0 + scale[adaln_rows]) + shift[adaln_rows]
    mx.eval(h, *rotary)
    return h, rotary[0], rotary[1]


def unfused(attention, projection, x, cos, sin):
    """`Attention.qkv` with ``projection`` in place of the bf16 linear."""

    batch, rows, _ = x.shape
    qkv = projection(x).reshape(batch, rows, attention.heads, 3, attention.head_dim)
    q = attention.q_norm(qkv[:, :, :, 0]).transpose(0, 2, 1, 3)
    k = attention.k_norm(qkv[:, :, :, 1]).transpose(0, 2, 1, 3)
    return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin), qkv[:, :, :, 2].transpose(0, 2, 1, 3)


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
    return mx.max(mx.abs(a - b)).item(), mx.sqrt(mx.sum((a - b) ** 2) / mx.sum(a**2)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    from tensorfold.families.h3.weights import load_dit

    model = load_dit(args.model_dir, blocks=1)
    attention = model.blocks[0].attn
    config = model.config
    plain = Int8Linear(attention.qkv_proj.weight, group=config.hidden_size)
    fused = Int8QKV(attention.qkv_proj.weight, attention.q_norm.weight, attention.k_norm.weight,
                    config.num_attention_heads, config.attention_head_dim, config.qk_norm_eps)
    for width, height, frames in CANVASES:
        h, cos, sin = block0_attention_input(model, width, height, frames)
        reference = attention.qkv(h, (cos, sin))
        before = unfused(attention, plain, h, cos, sin)
        after = fused(h, cos, sin)
        mx.eval(*reference, *before, *after)
        print(f"[tensorfold] {width}x{height}x{frames}, {h.shape[1]:,} rows", flush=True)
        for name, a, b, c in zip("qkv", reference, before, after, strict=True):
            near, near_l2 = compare(b, c)
            far, far_l2 = compare(a, c)
            print(f"[tensorfold]   {name}: fused vs unfused int8 max abs {near:.4f}, relative L2 {near_l2:.5f}; "
                  f"vs bf16 max abs {far:.4f}, relative L2 {far_l2:.5f} (rms {mx.sqrt(mx.mean(a.astype(mx.float32) ** 2)).item():.3f})")
        base, base_best = timed(lambda h=h, cos=cos, sin=sin: attention.qkv(h, (cos, sin)), args.repeats)
        mid, mid_best = timed(lambda h=h, cos=cos, sin=sin: unfused(attention, plain, h, cos, sin), args.repeats)
        quick, quick_best = timed(lambda h=h, cos=cos, sin=sin: fused(h, cos, sin), args.repeats)
        print(f"[tensorfold]   bf16 {base:.1f} ms (best {base_best:.1f}), int8 unfused {mid:.1f} ms (best "
              f"{mid_best:.1f}), int8 fused {quick:.1f} ms (best {quick_best:.1f}); {base / quick:.2f}x over bf16, "
              f"{mid / quick:.2f}x over unfused", flush=True)


if __name__ == "__main__":
    main()

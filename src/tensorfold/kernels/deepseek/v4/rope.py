"""RMSNorm (optional weight) then RoPE on the last dims of each head row, in one dispatch: a simdgroup a head row."""

from __future__ import annotations

import mlx.core as mx

from tensorfold.kernels.deepseek.v4.attention import _scale
from tensorfold.kernels.glm.flash.v1 import kernels as GK

# A simdgroup a head row, lane l dims l*PER ..: the norm rounded to bf16, then the last PE dims roped (precise trig)
_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint n = threadgroup_position_in_grid.y;
  const int r = int(n) / H;
  constexpr int PER = D / 32;
  const device bfloat* x = X + size_t(n) * D + lane * PER;
  float v[PER];
  for (int i = 0; i < PER; i++) v[i] = float(x[i]);
  if (NORM) {
    float ss = 0.0f;
    for (int i = 0; i < PER; i++) ss += v[i] * v[i];
    ss = simd_sum(ss);
    const float inv = metal::precise::rsqrt(ss / float(D) + EPS[0]);
    for (int i = 0; i < PER; i++) {
      float y = v[i] * inv;
      if (WEIGHTED) y = y * float(W[lane * PER + i]);
      v[i] = float(bfloat(y));
    }
  }
  const int base = int(lane) * PER;
  if (base + PER > D - PE) {
    const float pos = float(POS[r]);
    for (int i = 0; i < PER; i += 2) {
      const int d = base + i;
      if (d >= D - PE) {
        const float th = pos * INV[(d - (D - PE)) / 2];
        const float c = metal::precise::cos(th);
        const float s = INVERSE ? -metal::precise::sin(th) : metal::precise::sin(th);
        const float a = v[i], b = v[i + 1];
        v[i] = a * c - b * s;
        v[i + 1] = a * s + b * c;
      }
    }
  }
  device bfloat* o = OUT + size_t(n) * D + lane * PER;
  for (int i = 0; i < PER; i++) o[i] = bfloat(v[i]);
"""


def fits(x: mx.array) -> bool:
    return GK.metal() and x.dtype == mx.bfloat16 and int(x.shape[-1]) % 64 == 0


def norm_rope(x: mx.array, positions: mx.array, inv_freq: mx.array, *, weight: mx.array | None = None,
              eps: float = 1e-6, norm: bool = True, inverse: bool = False) -> mx.array:
    """x [R, D] or [R, H, D] bf16: head rows RMS-normed (times ``weight``), last dims roped at their row's position."""

    rows, dims = int(x.shape[0]), int(x.shape[-1])
    heads = int(x.size) // (rows * dims)
    kernel = GK._kernel("ds4_norm_rope", _SOURCE, ["X", "W", "POS", "INV", "EPS"], ["OUT"])
    pos = positions if positions.dtype == mx.int32 else positions.astype(mx.int32)
    out = kernel(inputs=[mx.contiguous(x), inv_freq if weight is None else weight, pos, inv_freq, _scale(eps)],
                 template=[("D", dims), ("H", heads), ("PE", 2 * int(inv_freq.shape[0])), ("NORM", int(norm)),
                           ("WEIGHTED", int(weight is not None)), ("INVERSE", int(inverse))],
                 grid=(32, rows * heads, 1), threadgroup=(32, 1, 1),
                 output_shapes=[(rows * heads, dims)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(x.shape)

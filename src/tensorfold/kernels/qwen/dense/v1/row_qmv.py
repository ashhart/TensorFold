"""A 4-bit matvec for 1 to 8 rows that gives a row the same bits at any row count, for Macs without the M5's
tensor units.

MLX 0.32's quantized matmul sums a row differently when 2 to 8 rows ride together: on an M3 Ultra even row 0 of
a 2-row call differs from the 1-row call (2026-09-26), so a drafted window's rows would not reproduce serial
decoding. This kernel is Flash Next's ``qmv`` (``kernels/qwen/flash_next/v1``) for groups of 32 or 64: MLX's
qmv_fast inner loop (lane l reads 16 inputs of each 512-input step, the weights, scales and biases as MLX packs
them), run once per row in a fixed order, each weight word read once for all the rows. Serial decoding goes
through it too (``install``), so it defines the reference the drafted rounds reproduce.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

MAX_ROWS = 8
RPS = 4          # output rows a simdgroup
SG = 2           # simdgroups a threadgroup

_HEADER = r"""
// one lane's 16 inputs, pre-scaled for the nibble masks below, and their sum (bf16 adds, as MLX's load_vector)
inline float load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
// one lane's 16 inputs times its 8 bytes of one weight row (MLX's qdot)
inline float qdot16w(const thread uint16_t* ws, const thread float* xt, float scale, float bias, float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
"""

_SOURCE = r"""
  // threadgroup b has SG simdgroups of RPS output rows each; a group of GS inputs spans GS / 16 lanes. Each
  // 512-input step loads the RPS weight rows' words once, then takes the R input rows in turn (one row's 16
  // inputs live at a time: holding all R rows' inputs spilled registers from 3 rows on an M3 Ultra; converting the
  // nibbles once for all rows, kept as halves, gave the same bits but ran slower).
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / (GS / 16);
  const device bfloat* bi = B + size_t(row0) * KG + lane / (GS / 16);
  float acc[R][RPS];
  for (int r = 0; r < R; r++) for (int j = 0; j < RPS; j++) acc[r][j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    uint16_t ws[RPS][4];
    float s[RPS], b[RPS];
    for (int j = 0; j < RPS; j++) {
      const device uint16_t* wp = (const device uint16_t*)(w + j * KB);
      for (int i = 0; i < 4; i++) ws[j][i] = wp[i];
      s[j] = float(sc[j * KG]);
      b[j] = float(bi[j * KG]);
    }
    for (int r = 0; r < R; r++) {
      float xt[16];
      const float sum = load16(X + r * K + k0 + lane * 16, xt);
      for (int j = 0; j < RPS; j++) acc[r][j] += qdot16w(ws[j], xt, s[j], b[j], sum);
    }
    w += 256; sc += 512 / GS; bi += 512 / GS;
  }
  for (int r = 0; r < R; r++)
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
    }
"""

_ORIG: Any = None
enabled = False
# One-row calls through MLX's own kernel: set where this kernel's rows are MLX's one-row bits (``matches_mlx``),
# so serial decoding keeps MLX's speed
mlx_one_row = False


_kernel: Any = None


def _compiled() -> Any:
    global _kernel
    if _kernel is None:
        name = "row_qmv_" + hashlib.sha256((_HEADER + _SOURCE).encode()).hexdigest()[:16]
        _kernel = mx.fast.metal_kernel(name=name, input_names=["X", "W", "S", "B"], output_names=["OUT"],
                                       source=_SOURCE, header=_HEADER)
    return _kernel


def fits(module: Any) -> bool:
    """Whether a quantized linear has the layout the kernel reads: 4-bit, groups of 32 or 64, bf16 scales,
    inputs a multiple of 512 and outputs a multiple of SG * RPS."""

    weight = module["weight"]
    return (module.bits == 4 and module.group_size in (32, 64) and module["scales"].dtype == mx.bfloat16
            and weight.ndim == 2 and (int(weight.shape[1]) * 8) % 512 == 0 and int(weight.shape[0]) % (SG * RPS) == 0
            and getattr(module, "mode", "affine") == "affine")


def qmv(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> mx.array:
    """x [..., K] bf16 (at most MAX_ROWS rows) @ W.T for 4-bit weights [N, K / 8] -> [..., N] bf16; a row's bits
    do not depend on how many rows ride with it."""

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = int(x2.shape[0]), int(x2.shape[1])
    n = int(weight.shape[0])
    out = _compiled()(inputs=[x2, weight, scales, biases],
                      template=[("K", dims), ("N", n), ("R", rows), ("GS", int(group_size)), ("RPS", RPS), ("SG", SG)],
                      grid=(32 * SG, n // (SG * RPS), 1), threadgroup=(32 * SG, 1, 1),
                      output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*shape[:-1], n)


def _call(self: Any, x: mx.array) -> mx.array:
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    low = 2 if mlx_one_row else 1
    if not (enabled and low <= rows <= MAX_ROWS and x.dtype == mx.bfloat16 and getattr(self, "_row_qmv", False)):
        return _ORIG(self, x)
    y = qmv(x, self["weight"], self["scales"], self["biases"], self.group_size)
    if "bias" in self:
        y = y + self["bias"]
    return y


def matches_mlx(model: Any, *, seed: int = 0) -> bool:
    """Whether every covered linear's one-row output here equals MLX's own one-row call bit for bit."""

    import mlx.nn as nn

    key = mx.random.key(seed)
    seen: set[tuple[int, int, int]] = set()
    for _, module in model.named_modules():
        if not (isinstance(module, nn.QuantizedLinear) and getattr(module, "_row_qmv", False)):
            continue
        weight = module["weight"]
        shape = (int(weight.shape[0]), int(weight.shape[1]), int(module.group_size))
        if shape in seen:
            continue
        seen.add(shape)
        key, sub = mx.random.split(key)
        x = (mx.random.normal((1, shape[1] * 8), key=sub) * 0.5).astype(mx.bfloat16)
        ours = qmv(x, weight, module["scales"], module["biases"], module.group_size)
        if "bias" in module:
            ours = ours + module["bias"]
        theirs = _ORIG(module, x)
        if not bool(mx.array_equal(ours, theirs).item()):
            return False
    return bool(seen)


def install(model: Any) -> int:
    """Route every call of at most MAX_ROWS rows of the model's fitting 4-bit linears through ``qmv``, one-row
    calls (serial decoding) included. Returns how many linears it covers. Idempotent."""

    global _ORIG, enabled
    import mlx.nn as nn

    if _ORIG is None:
        _ORIG = nn.QuantizedLinear.__call__
        nn.QuantizedLinear.__call__ = _call
    count = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and fits(module):
            object.__setattr__(module, "_row_qmv", True)
            count += 1
    enabled = True
    return count


__all__ = ["MAX_ROWS", "fits", "install", "qmv"]

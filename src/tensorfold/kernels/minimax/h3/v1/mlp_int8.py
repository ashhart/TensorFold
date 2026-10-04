"""H3 SwiGLU MLP with int8 weights and int8 activations on the M5 tensor units."""

# The scheme follows antirez's h3.c (MIT, https://github.com/antirez/h3.c, revision 8974cc0): activations are
# quantized at run time with one scale per row (per row and 1,024 channels for the fc2 input), weights carry
# one scale per output channel, and 128x128x128 int8 tiles run through Metal 4 tensor operations. The kernels
# here are written for `mx.fast.metal_kernel`; no h3.c source is copied.

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

TILE = 128            # rows, output columns and K per tensor-unit tile
FC2_GROUP = 1024      # input channels sharing one activation scale on the fc2 input
QMAX = 127.0

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_PRELUDE = r"""
  constexpr int T = 128;
  constexpr int KT = GROUP / T;                              // K tiles sharing one activation scale
  constexpr int KG = K / GROUP;
  constexpr int CAP = T * T / 256;                           // elements a thread owns
  const int M = mdims[0], MP = mdims[1];
  const int n0 = threadgroup_position_in_grid.x * T;
  const int r0 = threadgroup_position_in_grid.y * T;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> x((device int8_t*)X, dextents<int32_t, 2>(K, MP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> w((device int8_t*)W, dextents<int32_t, 2>(K, N));
  constexpr auto desc = matmul2d_descriptor(T, T, T, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<8>> op;
  auto a0 = x.slice<T, T>(0, r0);
  auto b0 = w.slice<T, T>(0, n0);
  // this tile's activation scales, staged once: [row in tile][group]
  threadgroup float sc[T * KG];
  for (int j = thread_position_in_threadgroup.x; j < T * KG; j += 256) sc[j] = XS[r0 * KG + j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# Y[row, n] = sum over K groups of (int8 X . int8 W) * xscale[row, group] * wscale[n]
# X: (MP, K) int8, rows padded to a multiple of TILE.  W: (N, K) int8.  XS: (MP, K / GROUP) float.  WS: (N,) float.
_LINEAR = _PRELUDE + r"""
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  float total[CAP];
  short erow[CAP], ecol[CAP];
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < CAP; i++) {
    total[i] = 0.0f; acc[i] = 0;
    auto ids = acc.get_multidimensional_index(i);
    ecol[i] = ids[0]; erow[i] = ids[1];
  }
  for (int g = 0; g < KG; g++) {
    for (int t = 0; t < KT; t++) {
      auto a = x.slice<T, T>((g * KT + t) * T, r0);
      auto b = w.slice<T, T>((g * KT + t) * T, n0);
      op.run(a, b, acc);
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < CAP; i++) { total[i] = fma(float(acc[i]), sc[erow[i] * KG + g], total[i]); acc[i] = 0; }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < CAP; i++) {
    const int row = r0 + erow[i];
    if (row < M) Y[(int64_t)row * N + n0 + ecol[i]] = static_cast<bfloat>(total[i] * WS[n0 + ecol[i]]);
  }
"""

# H[row, n] = silu(gate) * value for the fused fc1: W is (2 N, K) with gate rows first, one scale per row of X.
_SWIGLU = _PRELUDE.replace("dextents<int32_t, 2>(K, N));", "dextents<int32_t, 2>(K, 2 * N));") + r"""
  auto gate = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  auto value = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < CAP; i++) { gate[i] = 0; value[i] = 0; }
  for (int t = 0; t < K / T; t++) {
    auto a = x.slice<T, T>(t * T, r0);
    auto bg = w.slice<T, T>(t * T, n0);
    auto bv = w.slice<T, T>(t * T, N + n0);
    op.run(a, bg, gate);
    op.run(a, bv, value);
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < CAP; i++) {
    auto ids = gate.get_multidimensional_index(i);
    const int row = r0 + ids[1], n = n0 + ids[0];
    if (row >= M) continue;
    const float xs = sc[ids[1]];
    const float g = float(gate[i]) * xs * WS[n];
    const float v = float(value[i]) * xs * WS[N + n];
    Y[(int64_t)row * N + n] = static_cast<bfloat>(g / (1.0f + exp(-g)) * v);
  }
"""

# The same two kernels with one float bias per output channel (BS), added before the SiLU in the fused fc1.
_LINEAR_BIAS = _LINEAR.replace("total[i] * WS[n0 + ecol[i]])", "total[i] * WS[n0 + ecol[i]] + BS[n0 + ecol[i]])")
_SWIGLU_BIAS = _SWIGLU.replace("xs * WS[n];", "xs * WS[n] + BS[n];").replace("xs * WS[N + n];", "xs * WS[N + n] + BS[N + n];")

# One threadgroup quantizes one (row, group): the largest magnitude sets the scale, values round to int8.
_QUANTIZE = r"""
  constexpr int PER = GROUP / 256;
  constexpr int KG = K / GROUP;
  const int M = mdims[0];
  const int row = threadgroup_position_in_grid.y;
  const int g = threadgroup_position_in_grid.x;
  const int first = g * GROUP + thread_position_in_threadgroup.x * PER;
  float v[PER];
  float top = 0.0f;
  for (int j = 0; j < PER; j++) {
    v[j] = row < M ? float(X[(int64_t)row * K + first + j]) : 0.0f;
    top = max(top, abs(v[j]));
  }
  threadgroup float tops[8];
  top = simd_max(top);
  if (thread_index_in_simdgroup == 0) tops[simdgroup_index_in_threadgroup] = top;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  top = 1e-12f;
  for (int j = 0; j < 8; j++) top = max(top, tops[j]);
  const float scale = top / 127.0f;
  if (thread_position_in_threadgroup.x == 0) XS[row * KG + g] = scale;
  const float inverse = 1.0f / scale;
  for (int j = 0; j < PER; j++)
    Q[(int64_t)row * K + first + j] = (int8_t)clamp(int(rint(v[j] * inverse)), -127, 127);
"""

_compiled: dict[tuple, Any] = {}
_dims: dict[tuple[int, int], mx.array] = {}


def _kernel(kind: str, n: int, k: int, group: int) -> Any:
    key = (kind, n, k, group)
    run = _compiled.get(key)
    if run is None:
        body = {"linear": _LINEAR, "swiglu": _SWIGLU, "quantize": _QUANTIZE, "linear_bias": _LINEAR_BIAS,
                "swiglu_bias": _SWIGLU_BIAS}[kind]
        source = f"  constexpr int N = {n};\n  constexpr int K = {k};\n  constexpr int GROUP = {group};\n" + body
        name = f"h3_int8_{kind}_" + hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        if kind == "quantize":
            inputs, outputs = ["X", "mdims"], ["Q", "XS"]
        elif kind.endswith("_bias"):
            inputs, outputs = ["X", "XS", "W", "WS", "BS", "mdims"], ["Y"]
        else:
            inputs, outputs = ["X", "XS", "W", "WS", "mdims"], ["Y"]
        run = _compiled[key] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs,
                                                    source=source, header=_HEADER)
    return run


def _mdims(rows: int, padded: int) -> mx.array:
    key = (rows, padded)
    if key not in _dims:
        _dims[key] = mx.array([rows, padded], dtype=mx.int32)
    return _dims[key]


def available() -> bool:
    """Whether this GPU compiles and runs the int8 tensor-unit kernel."""

    try:
        x = mx.zeros((TILE, TILE), dtype=mx.int8)
        mx.eval(int8_linear(x, mx.ones((TILE, 1)), x, mx.ones((TILE,)), TILE, TILE))
        mx.eval(*quantize_rows(mx.ones((2, 256), dtype=mx.bfloat16), 256)[:2])
    except (RuntimeError, ValueError):
        return False
    return True


def quantize_weight(weight: mx.array) -> tuple[mx.array, mx.array]:
    """(N, K) float weight -> int8 weight and one float32 scale per output channel."""

    weight = weight.astype(mx.float32)
    scale = mx.maximum(mx.max(mx.abs(weight), axis=1), 1e-12) / QMAX
    q = mx.clip(mx.round(weight / scale[:, None]), -QMAX, QMAX).astype(mx.int8)
    return q, scale


def quantize_rows(x: mx.array, group: int) -> tuple[mx.array, mx.array, int]:
    """(M, K) activations -> (MP, K) int8 padded to whole tiles, (MP, K / group) float32 scales, and M.

    Each group of ``group`` channels in a row is scaled by its own largest magnitude.
    """

    rows, k = x.shape
    if k % group or group % 256:
        raise ValueError(f"the scale group must divide K and be a multiple of 256, got K {k}, group {group}")
    padded = rows + -rows % TILE
    q, scale = _kernel("quantize", 0, k, group)(
        inputs=[x.astype(mx.bfloat16), _mdims(rows, padded)], grid=(k // group * 256, padded, 1),
        threadgroup=(256, 1, 1), output_shapes=[(padded, k), (padded, k // group)],
        output_dtypes=[mx.int8, mx.float32])
    return q, scale, rows


def _check(padded: int, k: int, n: int, group: int) -> None:
    if padded % TILE or k % TILE or n % TILE or group % TILE or k % group:
        raise ValueError(f"int8 tiles need multiples of {TILE}: rows {padded}, K {k}, N {n}, group {group}")


def int8_linear(xq: mx.array, xscale: mx.array, wq: mx.array, wscale: mx.array, rows: int, group: int,
                bias: mx.array | None = None) -> mx.array:
    """bf16 (rows, N) from int8 activations (MP, K) and int8 weights (N, K) with their scales, plus ``bias``."""

    padded, k = xq.shape
    n = wq.shape[0]
    _check(padded, k, n, group)
    kind, extra = ("linear", []) if bias is None else ("linear_bias", [bias])
    return _kernel(kind, n, k, group)(inputs=[xq, xscale, wq, wscale, *extra, _mdims(rows, padded)],
                                      grid=(n // TILE * 256, padded // TILE, 1), threadgroup=(256, 1, 1),
                                      output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]


def int8_swiglu(xq: mx.array, xscale: mx.array, wq: mx.array, wscale: mx.array, rows: int,
                bias: mx.array | None = None) -> mx.array:
    """bf16 (rows, N) ``silu(gate) * value`` from int8 activations and the fused (2 N, K) int8 fc1.

    ``bias`` is the fused (2 N,) bias, gate half first, added before the SiLU.
    """

    padded, k = xq.shape
    n = wq.shape[0] // 2
    _check(padded, k, n, k)
    kind, extra = ("swiglu", []) if bias is None else ("swiglu_bias", [bias])
    return _kernel(kind, n, k, k)(inputs=[xq, xscale, wq, wscale, *extra, _mdims(rows, padded)],
                                  grid=(n // TILE * 256, padded // TILE, 1), threadgroup=(256, 1, 1),
                                  output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]


class Int8MLP:
    """``fc2(silu(gate) * value)`` with ``[gate; value] = fc1(x)``, both projections in int8."""

    def __init__(self, fc1_weight: mx.array, fc2_weight: mx.array, fc2_group: int = FC2_GROUP,
                 fc1_bias: mx.array | None = None, fc2_bias: mx.array | None = None) -> None:
        self.width = fc2_weight.shape[1]
        self.hidden = fc1_weight.shape[1]
        if fc1_weight.shape[0] != 2 * self.width:
            raise ValueError(f"fc1 must give [gate; value] of 2 x {self.width}, got {fc1_weight.shape[0]}")
        self.fc2_group = fc2_group if self.width % fc2_group == 0 else self.width
        if self.hidden % 256 or self.fc2_group % 256:
            raise ValueError(f"the int8 MLP needs widths in multiples of 256, got {self.hidden} and {self.fc2_group}")
        self.w1, self.s1 = quantize_weight(fc1_weight)
        self.w2, self.s2 = quantize_weight(fc2_weight)
        self.b1 = None if fc1_bias is None else fc1_bias.astype(mx.float32)
        self.b2 = None if fc2_bias is None else fc2_bias.astype(mx.float32)
        mx.eval(self.w1, self.s1, self.w2, self.s2, *[b for b in (self.b1, self.b2) if b is not None])

    def __call__(self, x: mx.array) -> mx.array:
        lead = x.shape[:-1]
        xq, xs, rows = quantize_rows(x.reshape(-1, self.hidden), self.hidden)
        hidden = int8_swiglu(xq, xs, self.w1, self.s1, rows, self.b1)
        hq, hs, rows = quantize_rows(hidden, self.fc2_group)
        out = int8_linear(hq, hs, self.w2, self.s2, rows, self.fc2_group, self.b2)
        return out.reshape(*lead, -1).astype(x.dtype)


class Int8Linear:
    """A projection in int8: activations quantized per row and ``group`` channels, weights per output channel."""

    def __init__(self, weight: mx.array, group: int = FC2_GROUP, bias: mx.array | None = None) -> None:
        self.width = weight.shape[1]
        self.group = group if self.width % group == 0 else self.width
        if self.width % 256 or self.group % 256 or weight.shape[0] % TILE:
            raise ValueError(f"the int8 projection needs {TILE}-aligned outputs and 256-aligned inputs, got "
                             f"{tuple(weight.shape)} with group {self.group}")
        self.w, self.s = quantize_weight(weight)
        self.b = None if bias is None else bias.astype(mx.float32)
        mx.eval(self.w, self.s, *([] if self.b is None else [self.b]))

    def __call__(self, x: mx.array) -> mx.array:
        lead = x.shape[:-1]
        xq, xs, rows = quantize_rows(x.reshape(-1, self.width), self.group)
        return int8_linear(xq, xs, self.w, self.s, rows, self.group, self.b).reshape(*lead, -1).astype(x.dtype)

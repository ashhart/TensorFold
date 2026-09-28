"""Packed affine projections that read each weight block once a window and keep every row's bits at any width."""

from __future__ import annotations

from typing import Any

BITS = (2, 3, 4, 5, 6, 8)
GROUP_SIZES = (32, 64, 128)
MAX_ROWS = 1 << 16
SG = 8                  # simdgroups a threadgroup: 256 threads, within every M1/M2 pipeline's limit

_HEADER = r"""
#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
// code i of a 32-code block held in BITS words (i is a compile-time constant after unrolling)
template <int BITS>
inline uint code_at(const thread uint* w, const int i) {
  const int bit = i * BITS, word = bit >> 5, shift = bit & 31;
  uint v = w[word] >> shift;
  if (shift + BITS > 32) v |= w[word + 1] << (32 - shift);
  return v & ((1u << BITS) - 1u);
}
// the bf16 in the low (h = 0) or high (h = 1) half of v, as fp32
inline float bf_half(uint v, int h) { return as_type<float>(h ? (v & 0xFFFF0000u) : (v << 16)); }
"""

_SOURCE = r"""
  // A simdgroup: OPS outputs of up to RT rows. Lane l takes 32-code blocks l, l + 32, ... of each output: it reads the
  // block's BITS words once, dequantizes each code once (fma(scale, q, bias)) and folds it into every row's own fma
  // chain, then one simd_sum. A row's bits never depend on OPS, RT or the rows beside it.
  constexpr int BLOCKS = K / 32, GROUPS = K / GS, BPG = GS / 32, WPR = K * BITS / 32, STEPS = (BLOCKS + 31) / 32;
  const int lane = int(thread_index_in_simdgroup);
  const int n0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * OPS;
  const int r0 = int(threadgroup_position_in_grid.y) * RT;
  const int rows = min(RT, int(X_shape[0]) - r0);
  const device uint* xw = (const device uint*)X;
  int nn[OPS];
  float acc[OPS][RT];
  PRAGMA_UNROLL
  for (int u = 0; u < OPS; u++) {
    nn[u] = min(n0 + u, N - 1);
    PRAGMA_UNROLL
    for (int r = 0; r < RT; r++) acc[u][r] = 0.0f;
  }
  for (int s = 0; s < STEPS; s++) {
    const int b = s * 32 + lane;
    if (b >= BLOCKS) break;
    uint wd[OPS][BITS];
    float sc[OPS], bi[OPS];
    PRAGMA_UNROLL
    for (int u = 0; u < OPS; u++) {
      const device uint* p = W + size_t(nn[u]) * WPR + size_t(b) * BITS;
      PRAGMA_UNROLL
      for (int j = 0; j < BITS; j++) wd[u][j] = p[j];
      const size_t g = size_t(nn[u]) * GROUPS + b / BPG;
      sc[u] = float(SC[g]);
      bi[u] = float(BI[g]);
    }
    PRAGMA_UNROLL
    for (int c = 0; c < 4; c++) {
      float wv[OPS][8];
      PRAGMA_UNROLL
      for (int u = 0; u < OPS; u++)
        PRAGMA_UNROLL
        for (int i = 0; i < 8; i++) wv[u][i] = fma(sc[u], float(code_at<BITS>(wd[u], 8 * c + i)), bi[u]);
      PRAGMA_UNROLL
      for (int r = 0; r < RT; r++) {
        if (r < rows) {
          const size_t at = (size_t(r0 + r) * K + size_t(b) * 32 + 8 * c) / 2;
          float xv[8];
          PRAGMA_UNROLL
          for (int h = 0; h < 4; h++) {
            const uint v = xw[at + h];
            xv[2 * h] = bf_half(v, 0);
            xv[2 * h + 1] = bf_half(v, 1);
          }
          PRAGMA_UNROLL
          for (int u = 0; u < OPS; u++)
            PRAGMA_UNROLL
            for (int i = 0; i < 8; i++) acc[u][r] = fma(xv[i], wv[u][i], acc[u][r]);
        }
      }
    }
  }
  PRAGMA_UNROLL
  for (int r = 0; r < RT; r++) {
    if (r < rows) {
      PRAGMA_UNROLL
      for (int u = 0; u < OPS; u++) {
        const float total = simd_sum(acc[u][r]);
        if (lane == 0 && n0 + u < N) OUT[size_t(r0 + r) * N + n0 + u] = bfloat(total);
      }
    }
  }
"""


def readable(bits: int, group_size: int, mode: str = "affine") -> bool:
    return mode == "affine" and bits in BITS and group_size in GROUP_SIZES


def shape(weight: Any, scales: Any, biases: Any, group_size: int, bits: int) -> tuple[int, int]:
    """Validate complete groups and tightly packed uint32 rows before a kernel can address them."""
    if not readable(bits, group_size):
        raise ValueError("Affine rows require 2/3/4/5/6/8 bits and groups of 32/64/128")
    if weight.ndim != 2 or scales.ndim != 2 or biases.shape != scales.shape:
        raise ValueError("Affine rows require 2-D packed weights with matching scale and bias shapes")
    n, words = (int(value) for value in weight.shape)
    if n < 1 or words < 1 or words * 32 % bits:
        raise ValueError("Affine weight rows must contain a whole number of packed values")
    k = words * 32 // bits
    if k % group_size or tuple(scales.shape) != (n, k // group_size):
        raise ValueError("Affine weight, scale and bias shapes do not match the declared quantization")
    return n, k


def fits(module: Any) -> bool:
    """Accept unpacked-layout affine linears whose inputs can use this bf16 row path."""
    import mlx.core as mx

    if not readable(getattr(module, "bits", 0), getattr(module, "group_size", 0), getattr(module, "mode", "affine")):
        return False
    if getattr(module, "_lane_tiled", False):
        return False
    try:
        weight, scales, biases = module["weight"], module["scales"], module["biases"]
        shape(weight, scales, biases, int(module.group_size), int(module.bits))
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return (weight.dtype == mx.uint32 and scales.dtype in (mx.bfloat16, mx.float16, mx.float32)
            and biases.dtype == scales.dtype)


def launch(rows: int) -> tuple[int, int]:
    """(outputs a simdgroup, rows a threadgroup) for ``rows`` rows; every launch has the same per-row arithmetic."""

    # past 8 rows, tiles of 8 rows beat one tile of 16 on the M3 and the M5: the weights come back from cache
    return 2, (1 if rows == 1 else 2 if rows == 2 else 4 if rows <= 4 else 8)


_kernels: dict[tuple, Any] = {}


def _kernel(k: int, n: int, bits: int, group_size: int, ops: int, rt: int) -> Any:
    """The kernel for one shape and launch, its constants in the source (MLX parses template args on every call)."""

    import hashlib

    import mlx.core as mx

    key = (k, n, bits, group_size, ops, rt)
    kernel = _kernels.get(key)
    if kernel is None:
        consts = (("K", k), ("N", n), ("BITS", bits), ("GS", group_size), ("OPS", ops), ("RT", rt), ("SG", SG))
        source = "".join(f"  constexpr int {name} = {value};\n" for name, value in consts) + _SOURCE
        digest = hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        kernel = _kernels[key] = mx.fast.metal_kernel(name=f"tensorfold_affine_rows_{digest}",
                                                      input_names=["X", "W", "SC", "BI"], output_names=["OUT"],
                                                      header=_HEADER, source=source)
    return kernel


def qmm(x: Any, weight: Any, scales: Any, biases: Any, group_size: int = 64, bits: int = 4) -> Any:
    """Read MLX's little-endian packed bit stream directly and emit bf16 rows without dequantizing weights."""
    import mlx.core as mx

    from tensorfold.kernels.inputs import padded

    n, k = shape(weight, scales, biases, group_size, bits)
    if x.ndim < 2 or int(x.shape[-1]) != k or x.dtype != mx.bfloat16:
        raise ValueError(f"Affine rows require bf16 inputs shaped (..., {k})")
    if (weight.dtype != mx.uint32 or scales.dtype not in (mx.bfloat16, mx.float16, mx.float32)
            or biases.dtype != scales.dtype):
        raise ValueError("Affine rows require uint32 weights and matching floating-point scales and biases")
    rows = int(x.size) // k
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"Affine rows take 1..{MAX_ROWS} input rows")
    ops, rt = launch(rows)
    inputs = [mx.contiguous(x.reshape(rows, k)), padded(mx.contiguous(weight)), padded(mx.contiguous(scales)),
              padded(mx.contiguous(biases))]
    output = _kernel(k, n, bits, group_size, ops, rt)(
        inputs=inputs, grid=(-(-n // (SG * ops)) * 32 * SG, -(-rows // rt), 1), threadgroup=(32 * SG, 1, 1),
        output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]
    return output.reshape(*x.shape[:-1], n)


__all__ = ["BITS", "GROUP_SIZES", "MAX_ROWS", "SG", "fits", "launch", "qmm", "readable", "shape"]

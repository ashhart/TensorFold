"""Prompt-chunk matmuls (64+ rows): MLX's own 4-bit kernels with other tiles, so MLX's bits."""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import mlx.core as mx
import mlx.nn as nn

MIN_ROWS = 64
_INCLUDE = os.path.join(os.path.dirname(mx.__file__), "include")
# already in every custom kernel: MLX prefixes its utils.h (and what that includes)
_SKIP = {f"mlx/backend/metal/kernels/{n}" for n in ("utils.h", "bf16.h", "bf16_math.h", "complex.h", "defines.h",
                                                    "logging.h")}


def _inline(path: str, seen: set[str]) -> str:
    if path in seen or path in _SKIP:
        return ""
    seen.add(path)
    with open(os.path.join(_INCLUDE, path)) as f:
        text = f.read()
    out = []
    for line in text.splitlines():
        m = re.match(r'\s*#include\s+"(.+)"', line)
        if m:
            out.append(_inline(m.group(1), seen))
        elif line.strip() != "#pragma once":
            out.append(line)
    return "\n".join(out)


_header_cache: dict[str, str] = {}


def _header() -> str:
    h = _header_cache.get("base")
    if h is None:
        seen: set[str] = set()
        h = _header_cache["base"] = "\n".join(_inline(f"mlx/backend/metal/kernels/{p}", seen)
                                              for p in ("steel/gemm/gemm.h", "quantized_utils.h", "quantized.h"))
    return h


_QMM_BODY = """
  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);
  threadgroup bfloat16_t Xs[BM * BK_padded];
  threadgroup bfloat16_t Ws[BN * BK_padded];
  qmm_t_impl<bfloat16_t, 32, 4, ALIGNED != 0, BM, BK, BN>(W, S, B, X, Y, Xs, Ws, KK[0], NN[0], MM[0], KK[0],
      threadgroup_position_in_grid, thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
"""

# First row of each expert in the sorted rows (and M at the end): a lower bound a thread.
_OFFSETS = """
  const int g = int(thread_position_in_grid.x);
  if (g > EE[0]) return;
  int lo = 0, hi = MM[0];
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (int(IDX[mid]) < g) lo = mid + 1; else hi = mid;
  }
  OFF[g] = lo;
"""

# A tile's expert and rows from one simdgroup scan of tile counts, then one K pass with MLX's loaders, MMA and K order
_TILES_FN = """
template <typename T, const int group_size, const int bits, const int BM, const int BN, const int BK, const int WM,
          const int WN>
METAL_FUNC void tf_gather_qmm_tiles(
    threadgroup T* Xs, threadgroup T* Ws, const device T* x, const device uint32_t* w, const device T* scales,
    const device T* biases, const device int* off, device T* y, const int M, const int N, const int K, const int E,
    uint3 tid, uint simd_group_id, uint simd_lane_id) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, true, WM * WN * SIMD_SIZE, group_size, bits>;

  // tile tid.y: lane l counts the tiles of experts [l * EPL, (l + 1) * EPL)
  const int t = int(tid.y);
  const int EPL = (E + 31) / 32;
  int mine = 0;
  for (int i = 0; i < EPL; i++) {
    const int e = int(simd_lane_id) * EPL + i;
    if (e < E) mine += (off[e + 1] - off[e] + BM - 1) / BM;
  }
  const int before = simd_prefix_exclusive_sum(mine);
  const int total = simd_shuffle(before + mine, ushort(31));
  if (t >= total) return;
  int hit_e = 0, hit_row = 0, hit_n = 0;
  if (t >= before && t < before + mine) {
    int acc = before;
    for (int i = 0; i < EPL; i++) {
      const int e = int(simd_lane_id) * EPL + i;
      const int rows = off[e + 1] - off[e];
      const int tiles = (rows + BM - 1) / BM;
      if (t < acc + tiles) {
        hit_e = e;
        hit_row = off[e] + (t - acc) * BM;
        hit_n = min(BM, rows - (t - acc) * BM);
        break;
      }
      acc += tiles;
    }
  }
  const uint32_t index = uint32_t(simd_max(hit_e));  // one lane holds the tile; the others hold zeros
  const int y_row = simd_max(hit_row);
  const short rows = short(simd_max(hit_n));

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int y_col = int(tid.x) * BN;
  const short tgp_bm = short(min(BM, M - y_row));       // rows loadable from memory (the tile's are the first)
  const short tgp_bn = short(min(BN, N - y_col));
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);
  auto wl = (const device uint8_t*)w;
  x += size_t(y_row) * K;
  y += size_t(y_row) * N + y_col;
  wl += size_t(y_col) * K_w + index * stride_w;
  scales += size_t(y_col) * K_g + index * stride_s;
  biases += size_t(y_col) * K_g + index * stride_s;
  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl, scales, biases, K, Ws, simd_group_id, simd_lane_id);
  if (tgp_bm == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
  } else if (tgp_bm == BM) {
    gemm_loop_unaligned<true, false, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
  } else {
    gemm_loop_unaligned<false, false, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
  }
  if (k_remain) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
  }
  if (rows == BM && tgp_bn == BN) mma_op.store_result(y, N);
  else mma_op.store_result_slice(y, N, short2(0, 0), short2(tgp_bn, rows));
}
"""

_GATHER_BODY = """
  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);
  threadgroup bfloat16_t Xs[BM * BK_padded];
  threadgroup bfloat16_t Ws[BN * BK_padded];
  tf_gather_qmm_tiles<bfloat16_t, GS, 4, BM, BN, BK, WM, WN>(Xs, Ws, X, W, S, B, OFF, Y, MM[0], NN[0], KK[0], EE[0],
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_kernels: dict[str, Any] = {}


def _k(name: str, source: str, inputs: list[str], outputs: list[str], header: str = "") -> Any:
    k = _kernels.get(name)
    if k is None:
        k = _kernels[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source,
                                                  header=header)
    return k


_ints: dict[int, mx.array] = {}


def _int(v: int) -> mx.array:
    a = _ints.get(v)
    if a is None:
        a = _ints[v] = mx.array([v], dtype=mx.int32)
    return a


def fast_prefill() -> bool:
    """Whether prefill takes the fast path: on the GPU, unless TF_FLASH_PREFILL=0."""

    return os.environ.get("TF_FLASH_PREFILL", "1") != "0" and mx.default_device() == mx.gpu


def active(rows: int) -> bool:
    """Whether a call on ``rows`` rows goes through this module's kernels."""

    return rows >= MIN_ROWS and fast_prefill()


def _tensor_units() -> bool:
    """Whether this GPU has the M5 generation's tensor units (applegpu_g17 and later)."""

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    digits = "".join(ch for ch in str(info.get("architecture", "")).removeprefix("applegpu_g") if ch.isdigit())
    return bool(digits) and int(digits) >= 17


_tiles: list[bool] = []


def prefill_identity() -> str:
    """What decides the prefill matmuls' bits here, for snapshot keys (reading it builds no kernel)."""

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    architecture = str(info.get("architecture", "unknown"))
    state = "pending" if not _tiles else ("custom" if _tiles[0] else "native")
    return f"architecture={architecture};matmul={state}"             # the source itself is in kernel_version


def tiles() -> bool:
    """Whether the tiles serve here: no tensor units, and they gave MLX's bits once this process (then fixed)."""

    if not _tiles:
        ok = False
        if not _tensor_units():
            try:
                ok = _self_check()
            except Exception as e:  # noqa: BLE001 - a kernel that does not build means MLX's matmuls, not a crash
                print(f"[tensorfold] prefill matmul kernels unavailable ({type(e).__name__}: {e}); using MLX's",
                      file=sys.stderr)
            else:
                if not ok:
                    print("[tensorfold] prefill matmul kernels differ from MLX's; using MLX's", file=sys.stderr)
        _tiles.append(ok)
    return _tiles[0]


def _self_check() -> bool:
    """``qmm`` (both tiles) and ``gather_sorted`` (every shape) against MLX on small random products (own key)."""

    keys = mx.random.split(mx.random.key(20260926), 4)

    def weights(key, lead, n, k):
        w = mx.random.randint(0, 2**31, (*lead, n, k // 8), dtype=mx.uint32, key=key)
        s = (mx.random.normal((*lead, n, k // 32), key=mx.random.split(key)[0]) * 0.02).astype(mx.bfloat16)
        b = (mx.random.normal((*lead, n, k // 32), key=mx.random.split(key)[1]) * 0.02).astype(mx.bfloat16)
        return w, s, b

    same = []
    for (m, n), key in (((512, 640), keys[0]), ((128, 8192), keys[1])):  # 64 x 32 and 64 x 64 tiles, no split-K
        x = mx.random.normal((m, 256), key=keys[3]).astype(mx.bfloat16)
        w, s, b = weights(key, (), n, 256)
        ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=32, bits=4)
        same.append(mx.array_equal(qmm(x, w, s, b), ref))
    x = mx.random.normal((400, 256), key=keys[3]).astype(mx.bfloat16)
    w, s, b = weights(keys[2], (16,), 64, 256)
    idx = mx.sort(mx.random.randint(0, 16, (400,), key=keys[2])).astype(mx.uint32)
    ref = mx.gather_qmm(x[:, None], w, s, b, rhs_indices=idx, transpose=True, group_size=32, bits=4,
                        sorted_indices=True)[:, 0]
    same.extend(mx.array_equal(gather_sorted(x, w, s, b, idx, shape), ref) for shape in SHAPES)
    mx.eval(same)
    return all(bool(v.item()) for v in same)


def _mlx_splits_k(m: int, n: int, k: int) -> bool:
    """Whether MLX runs this product split-K (under ~512 32 x 32 tiles): other sums, so ``qmm`` steps aside."""

    split = max(1, 512 // (-(-n // 32) * -(-m // 32)))
    split = min(split, k // 32)
    while split > 1 and k % (split * 32):
        split -= 1
    return split > 1


def _q4(layer: Any) -> bool:
    """A 4-bit, group-32 affine-quantized layer without a bias (what these kernels read)."""

    return (getattr(layer, "bits", None) == 4 and getattr(layer, "group_size", None) == 32
            and getattr(layer, "mode", "affine") == "affine" and "scales" in layer and "biases" in layer
            and "bias" not in layer and layer.weight.dtype == mx.uint32)


def qmm(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """x [M, K] @ w.T (4-bit, groups of 32) -> [M, N] bf16 on MLX's qmm kernel with a 64-row tile: its bits."""

    m, k = x.shape
    n = int(w.shape[0])
    bm, bn = (64, 64) if n >= 8192 else (64, 32)
    kern = _k("tf_prefill_qmm", _QMM_BODY, ["X", "W", "S", "B", "KK", "NN", "MM"], ["Y"], _header())
    return kern(inputs=[x, w, scales, biases, _int(k), _int(n), _int(m)],
                template=[("BM", bm), ("BN", bn), ("BK", 32), ("ALIGNED", int(n % bn == 0))],
                grid=(-(-n // bn) * 128, -(-m // bm), 1), threadgroup=(128, 1, 1),
                output_shapes=[(m, n)], output_dtypes=[mx.bfloat16])[0]


def matmul(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """mx.quantized_matmul(x, w, scales, biases) for 4-bit group-32 weights: ``qmm`` where it gives the same bits."""

    m, k = x.shape
    if active(m) and not _mlx_splits_k(m, int(w.shape[0]), k) and tiles():
        return qmm(x, w, scales, biases)
    return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=32, bits=4)


def linear(layer: Any, x: mx.array) -> mx.array:
    """``layer(x)``: a 4-bit group-32 QuantizedLinear on 64+ bf16 rows through ``qmm`` (the same bits), else MLX."""

    rows, n, k = x.size // x.shape[-1], int(layer.weight.shape[0]), int(x.shape[-1])
    if (not isinstance(layer, nn.QuantizedLinear) or not _q4(layer) or x.dtype != mx.bfloat16 or n < 32
            or not active(rows) or _mlx_splits_k(rows, n, k) or not tiles()):
        return layer(x)
    lead = x.shape[:-1]
    y = qmm(x.reshape(-1, x.shape[-1]), layer.weight, layer.scales, layer.biases)
    return y.reshape(*lead, y.shape[-1])


# (BM, BN, WM, WN): bits are the same for every shape; M3 Ultra sweeps: 16 rows best below ~56 rows an expert, then 32
SHAPES = ((16, 32, 1, 2), (32, 32, 1, 2))


def shape_for(rows: int, experts: int) -> tuple[int, int, int, int]:
    """The tile shape for ``rows`` sorted rows over ``experts`` experts."""

    return SHAPES[0] if rows < 56 * max(1, experts) else SHAPES[1]


def gather_sorted(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, idx: mx.array,
                  shape: tuple[int, int, int, int] | None = None) -> mx.array:
    """x [M, K] bf16 sorted by expert, idx [M] uint32 -> [M, N]: row i times expert idx[i], one K pass a row."""

    m, k = x.shape
    experts, n = int(w.shape[0]), int(w.shape[1])
    group = k // int(scales.shape[-1])
    bm, bn, wm, wn = shape or shape_for(m, experts)
    count = _int(experts)
    offsets = _k("tf_expert_offsets", _OFFSETS, ["IDX", "MM", "EE"], ["OFF"])(
        inputs=[idx, _int(m), count], grid=(experts + 1, 1, 1), threadgroup=(min(256, experts + 1), 1, 1),
        output_shapes=[(experts + 1,)], output_dtypes=[mx.int32])[0]
    most = min(m, -(-m // bm) + experts)                                # tiles past the last exit at once
    kern = _k("tf_gather_qmm_tiles", _GATHER_BODY, ["X", "W", "S", "B", "OFF", "MM", "NN", "KK", "EE"], ["Y"],
              _header() + _TILES_FN)
    return kern(inputs=[x, w, scales, biases, offsets, _int(m), _int(n), _int(k), count],
                template=[("GS", group), ("BM", bm), ("BN", bn), ("BK", 32), ("WM", wm), ("WN", wn)],
                grid=(-(-n // bn) * 32, most * wn, wm), threadgroup=(32, wn, wm),
                output_shapes=[(m, n)], output_dtypes=[mx.bfloat16])[0]


# MLX's sorted gather kernel on M5 keeps row offsets in 16 bits (to 0.32.2): a call takes at most this many rows
MAX_SORTED_ROWS = 32768


def _experts(x: mx.array, layer: Any, idx: mx.array) -> mx.array:
    """A QuantizedSwitchLinear on rows sorted by expert: ``gather_sorted``, or MLX's sorted gather_qmm."""

    # MLX's sorted gather_qmm runs QMV below 4 routes an expert, and QMV sums in another order: keep its dispatch
    if x.shape[0] // int(layer.weight.shape[0]) >= 4 and tiles():
        return gather_sorted(x, layer.weight, layer.scales, layer.biases, idx)
    return _mlx_experts(x, layer, idx)


def _mlx_experts(x: mx.array, layer: Any, idx: mx.array) -> mx.array:
    """MLX's sorted gather_qmm, in balanced slices of at most MAX_SORTED_ROWS rows (a fixed function of the rows)."""

    rows = int(x.shape[0])
    if rows > MAX_SORTED_ROWS:
        size = -(-rows // -(-rows // MAX_SORTED_ROWS))
        return mx.concatenate([_mlx_experts(x[a:a + size], layer, idx[a:a + size]) for a in range(0, rows, size)])
    return mx.gather_qmm(x[:, None], layer.weight, layer.scales, layer.biases, rhs_indices=idx, transpose=True,
                         group_size=32, bits=4, sorted_indices=True)[:, 0]


def deltanet_in(g: Any, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """GatedDeltaNet's input projections (qkv, z [B, L, NV, DV], b, a): one stacked matmul for a prefill chunk."""

    batch, length, _ = x.shape
    stacked = g.__dict__.get("stacked")                                 # the fused decode's stacked rows
    if stacked is not None and active(batch * length):
        cuts = [g.conv_dim, g.conv_dim + g.value_dim, g.conv_dim + g.value_dim + g.nv]
        qkv, z, b, a = mx.split(linear(stacked, x), cuts, axis=-1)
    else:
        qkv, z, b, a = g.in_proj_qkv(x), g.in_proj_z(x), g.in_proj_b(x), g.in_proj_a(x)
    return qkv, z.reshape(batch, length, g.nv, g.dv), b, a


def moe_applies(module: Any, x: mx.array) -> bool:
    """Whether ``moe`` serves model.SparseMoE on x: batch 1, 64+ rows, bf16, 4-bit group-32 experts."""

    sw = module.switch_mlp
    return (x.ndim == 3 and x.shape[0] == 1 and x.dtype == mx.bfloat16 and active(int(x.shape[1]))
            and all(_q4(p) for p in (sw.gate_proj, sw.up_proj, sw.down_proj)))


def moe(module: Any, x: mx.array, *, route: Any = None, switch: Any = None) -> mx.array:
    """model.SparseMoE on a prompt chunk x [1, L, D]: the reference's routing, sums and shared expert, MLX's bits."""

    batch, length, dims = x.shape
    k = module.top_k
    experts, weights = module.route(x) if route is None else route      # [1, L, k] each
    flat = experts.reshape(-1)
    order = mx.argsort(flat)
    idx = flat[order].astype(mx.uint32)
    pos = mx.argsort(order).astype(mx.int32)                            # a route's row among the sorted ones
    xs = x.reshape(length, dims)[order // k]                            # [L k, D], sorted by expert
    sw = module.switch_mlp if switch is None else switch
    g = _experts(xs, sw.gate_proj, idx)
    u = _experts(xs, sw.up_proj, idx)
    act = sw.activation(u, g)                                           # SwitchGLU: activation(x_up, x_gate)
    y = _experts(act, sw.down_proj, idx)
    # the reference's unsort, bf16 product and MLX's sum: a sequential fp32 sum of the products changes bits
    routed = (y[pos].reshape(batch, length, k, dims) * weights[..., None]).sum(axis=-2)
    routed = routed.reshape(length, dims)
    se = module.shared_expert
    xf = x.reshape(length, dims)
    shared = linear(se.down_proj, nn.silu(linear(se.gate_proj, xf)) * linear(se.up_proj, xf))
    shared = shared * mx.sigmoid(linear(module.shared_expert_gate, xf))
    return (routed + shared).reshape(batch, length, dims)

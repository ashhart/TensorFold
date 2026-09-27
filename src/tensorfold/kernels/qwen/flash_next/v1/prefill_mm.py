"""Matmuls for Flash Next prefill chunks (64+ rows): MLX's own 4-bit kernels with tiles that suit these shapes."""

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
        h = "\n".join(_inline(f"mlx/backend/metal/kernels/{p}", seen)
                      for p in ("steel/gemm/gemm.h", "quantized_utils.h", "quantized.h"))
        # affine_gather_qmm_rhs as a helper: threadgroup buffers passed in, alignment as template arguments
        with open(os.path.join(_INCLUDE, "mlx/backend/metal/kernels/quantized.h")) as f:
            src = f.read()
        at = src.index("[[kernel]] void affine_gather_qmm_rhs(")
        start = src.rindex("template <", 0, at)
        depth, end = 0, src.index("{", at)
        while True:
            depth += {"{": 1, "}": -1}.get(src[end], 0)
            if depth == 0 and src[end] == "}":
                break
            end += 1
        fn = src[start:end + 1]
        fn = fn.replace("    bool transpose>", "    bool transpose,\n    bool align_M,\n    bool align_N,\n    bool align_K>", 1)
        fn = fn.replace("[[kernel]] void affine_gather_qmm_rhs(",
                        "METAL_FUNC void tf_gather_qmm_rhs_impl(\n    threadgroup T* Xs,\n    threadgroup T* Ws,", 1)
        fn = re.sub(r"\s*\[\[[a-z_]+(\(\d+\))?\]\]", "", fn)
        fn = re.sub(r"\n\s*threadgroup T (Xs|Ws)\[[^\]]*\];", "", fn)
        h = _header_cache["base"] = h + "\n" + fn
    return h


_QMM_BODY = """
  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);
  threadgroup bfloat16_t Xs[BM * BK_padded];
  threadgroup bfloat16_t Ws[BN * BK_padded];
  qmm_t_impl<bfloat16_t, 32, 4, ALIGNED != 0, BM, BK, BN>(W, S, B, X, Y, Xs, Ws, KK[0], NN[0], MM[0], KK[0],
      threadgroup_position_in_grid, thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
"""
_GATHER_BODY = """
  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);
  threadgroup bfloat16_t Xs[BM * BK_padded];
  threadgroup bfloat16_t Ws[BN * BK_padded];
  tf_gather_qmm_rhs_impl<bfloat16_t, 32, 4, BM, BN, BK, WM, WN, true, AM != 0, AN != 0, AK != 0>(
      Xs, Ws, X, W, S, B, IDX, Y, MM[0], NN[0], KK[0],
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
_WSUM = """
  // Thread (d, r): sum_j bf16(y[route r*k+j] * w[r][j]) in j order (fp32), from the expert-sorted outputs Y via POS
  // (a route's row in Y); the shared expert is added by the caller.
  const int d = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  float acc = 0.0f;
  for (int j = 0; j < TOPK; j++) {
    const int route = r * TOPK + j;
    const float p = float(bfloat16_t(float(Y[size_t(POS[route]) * D + d]) * float(WT[route])));
    acc += p;
  }
  OUT[size_t(r) * D + d] = bfloat16_t(acc);
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


def tiles() -> bool:
    """Whether ``qmm`` and ``gather_sorted`` serve prefill here: not on GPUs with tensor units, and only once both
    have given MLX's bits on small products in this process (decided on first use, then fixed)."""

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
    """``qmm`` (both tiles) and ``gather_sorted`` against MLX on small random products (their own PRNG key)."""

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
    same.append(mx.array_equal(gather_sorted(x, w, s, b, idx), ref))
    mx.eval(same)
    return all(bool(v.item()) for v in same)


def _mlx_splits_k(m: int, n: int, k: int) -> bool:
    """Whether mx.quantized_matmul runs this product split-K (quantized.cpp, qmm_splitk: fewer than ~512 32 x 32
    threadgroups). Its sums then run in another order, so ``qmm`` (which gives MLX's qmm bits) steps aside there."""

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
    """x [M, K] bf16 @ dequantized w [N, K / 8] (4-bit, groups of 32).T -> [M, N] bf16 through MLX's kernel with a
    tuned tile; mx.quantized_matmul's bits unless MLX would split K (``matmul`` picks between them)."""

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
    """``layer(x)`` for x [..., K]: a 4-bit, group-32 QuantizedLinear on 64+ bf16 rows through ``qmm`` (the same
    bits), anything else through the layer itself."""

    rows, n, k = x.size // x.shape[-1], int(layer.weight.shape[0]), int(x.shape[-1])
    if (not isinstance(layer, nn.QuantizedLinear) or not _q4(layer) or x.dtype != mx.bfloat16 or n < 32
            or not active(rows) or _mlx_splits_k(rows, n, k) or not tiles()):
        return layer(x)
    lead = x.shape[:-1]
    y = qmm(x.reshape(-1, x.shape[-1]), layer.weight, layer.scales, layer.biases)
    return y.reshape(*lead, y.shape[-1])


def gather_sorted(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, idx: mx.array) -> mx.array:
    """x [M, K] bf16 with rows sorted by expert, idx [M] uint32 (sorted) -> [M, N]: row i times expert idx[i]."""

    m, k = x.shape
    n = int(w.shape[1])
    bm, bn, wm, wn = 16, 32, 1, 2
    kern = _k("tf_prefill_gather_qmm", _GATHER_BODY, ["X", "W", "S", "B", "IDX", "MM", "NN", "KK"], ["Y"], _header())
    return kern(inputs=[x, w, scales, biases, idx, _int(m), _int(n), _int(k)],
                template=[("BM", bm), ("BN", bn), ("BK", 32), ("WM", wm), ("WN", wn),
                          ("AM", int(m % bm == 0)), ("AN", int(n % bn == 0)), ("AK", int(k % 32 == 0))],
                grid=(-(-n // bn) * 32, -(-m // bm) * wn, wm), threadgroup=(32, wn, wm),
                output_shapes=[(m, n)], output_dtypes=[mx.bfloat16])[0]


def _experts(x: mx.array, layer: Any, idx: mx.array) -> mx.array:
    """A QuantizedSwitchLinear on rows sorted by expert: ``gather_sorted``, or MLX's sorted gather_qmm."""

    if tiles():
        return gather_sorted(x, layer.weight, layer.scales, layer.biases, idx)
    return mx.gather_qmm(x[:, None], layer.weight, layer.scales, layer.biases, rhs_indices=idx, transpose=True,
                         group_size=32, bits=4, sorted_indices=True)[:, 0]


def moe_applies(module: Any, x: mx.array) -> bool:
    """Whether ``moe`` serves model.SparseMoE on x: batch 1, 64+ rows, bf16, 4-bit group-32 experts."""

    sw = module.switch_mlp
    return (x.ndim == 3 and x.shape[0] == 1 and x.dtype == mx.bfloat16 and active(int(x.shape[1]))
            and all(_q4(p) for p in (sw.gate_proj, sw.up_proj, sw.down_proj)))


def moe(module: Any, x: mx.array) -> mx.array:
    """model.SparseMoE on x [1, L, D] for a prefill chunk (see ``moe_applies``): the reference routing, the three
    expert matmuls on rows sorted by expert (``_experts``), a weighted sum over the sorted outputs, the shared
    expert."""

    batch, length, dims = x.shape
    k = module.top_k
    experts, weights = module.route(x)                                  # [1, L, k] each
    flat = experts.reshape(-1)
    order = mx.argsort(flat)
    idx = flat[order].astype(mx.uint32)
    pos = mx.argsort(order).astype(mx.int32)                            # a route's row among the sorted ones
    xs = x.reshape(length, dims)[order // k]                            # [L k, D], sorted by expert
    sw = module.switch_mlp
    g = _experts(xs, sw.gate_proj, idx)
    u = _experts(xs, sw.up_proj, idx)
    act = sw.activation(u, g)                                           # SwitchGLU: activation(x_up, x_gate)
    y = _experts(act, sw.down_proj, idx)
    wsum = _k("tf_prefill_moe_wsum", _WSUM, ["Y", "POS", "WT"], ["OUT"])
    routed = wsum(inputs=[y, pos, weights.reshape(-1)], template=[("TOPK", k), ("D", dims)],
                  grid=(dims, length, 1), threadgroup=(256, 1, 1), output_shapes=[(length, dims)],
                  output_dtypes=[x.dtype])[0]
    se = module.shared_expert
    xf = x.reshape(length, dims)
    shared = linear(se.down_proj, nn.silu(linear(se.gate_proj, xf)) * linear(se.up_proj, xf))
    shared = shared * mx.sigmoid(linear(module.shared_expert_gate, xf))
    return (routed + shared).reshape(batch, length, dims)

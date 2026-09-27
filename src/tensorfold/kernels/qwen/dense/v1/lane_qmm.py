"""Row-exact 4-bit projections on the M5 tensor units: the lane matmul (groups of 32 or 64).

One arithmetic for every row count: for weight group g (GS inputs; scale s, bias b per column),
P[m, n, g] = x[m, g] . q[n, g] on the tensor unit (bf16 x uint4 -> fp32), then
y[m, n] = sum over g, in order, of fma(s, P, fma(b, xs[m, g], y)), with xs the group's fp32 input
sum. The groups split into SK slices by weight shape only, added in slice order. So a row's bits
never depend on the other rows: rows 1..M of any call equal the same rows computed one at a time.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.inputs import ints

MAX_ROWS = 128         # rows the lane kernel accepts in one call
ROW_BLOCK = 32         # rows per threadgroup above 32 rows (one 32-row op per weight group)
NT = 32                # output columns per simdgroup tile

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_XSUM = r"""
  const int M = mdims[0], MP = mdims[1];
  const uint m = thread_position_in_grid.y;
  const uint g = thread_position_in_grid.x;
  if (g >= K / GS || int(m) >= MP) return;
  float acc = 0.0f;
  if (int(m) < M) for (int i = 0; i < GS; i++) acc += float(X[m * K + g * GS + i]);
  XS[g * MP + m] = acc;
"""

_MAIN = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / GS;
  constexpr int NF = NT / 16;
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;   // first row of this threadgroup's row block
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;

  // one op for all TMR 16-row blocks: each row gets the 16-row op's bits
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq, dextents<int32_t, 2>(K, N));

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  const device uint4* sbv = (const device uint4*)SBt;   // (s, b) bf16 pairs, [g][n]
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
  for (int g = g_begin; g < g_end; g++) {
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * GS, 0);
    auto b = tB.slice(g * GS, n0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    for (int t = 0; t < TMR; t++) {
      const float xs0 = XS[g * MP + rb + t * 16 + fm];
      const float xs1 = XS[g * MP + rb + t * 16 + fm + 8];
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  // K slices are added in slice order, one 16-row block at a time
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int n = n0 + f * 16 + fn;
          if (m < M && n < N)
            for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
"""

# Tiled weights (``tile_weight``): column tile t's group g is one contiguous NT x GS block. Same values, same op.
_MAIN_TILED = _MAIN.replace(
    "    auto b = tB.slice(g * GS, n0);\n",
    "    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(\n"
    "        (device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (NT * GS / 2), dextents<int32_t, 2>(GS, NT));\n")
assert _MAIN_TILED != _MAIN

# 64-wide tiles: two simdgroups run each group's 16 TMR x 64 op together; every output keeps _MAIN_TILED's arithmetic.
_COOP = r"""
  const ushort sg = simdgroup_index_in_threadgroup;
  const ushort slice = sg >> 1;                                  // K slice: a pair of simdgroups each
  const ushort tip = ushort(thread_position_in_threadgroup.x) - slice * 64;   // thread within its pair
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / GS;
  const int n0 = threadgroup_position_in_grid.x * 64;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;       // first row of this threadgroup's row block
  const int g_begin = (slice * KG) / SK;
  const int g_end = ((slice + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, 64, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<2>> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  auto a0 = tA.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0((device uchar*)Wq, dextents<int32_t, 2>(GS, 64));
  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  constexpr int CAP = 16 * TMR;                                  // 16 TMR x 64 outputs over 64 threads
  short ecol[CAP], erow[CAP];
  for (int i = 0; i < CAP; i++) { auto ids = P.get_multidimensional_index(i); ecol[i] = ids[0]; erow[i] = ids[1]; }
  float C[CAP];
  for (int i = 0; i < CAP; i++) C[i] = 0.0f;
  const device uint* sbw = (const device uint*)SBt;              // (s, b) bf16 pairs, [g][n]
  for (int g = g_begin; g < g_end; g++) {
    auto a = tA.slice(g * GS, 0);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (64 * GS / 2), dextents<int32_t, 2>(GS, 64));
    op.run(a, b, P);
    for (int i = 0; i < CAP; i++) {
      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(sbw[g * N + n0 + ecol[i]]);
      const float xs = XS[g * MP + rb + erow[i]];
      C[i] = fma(float(sb[0]), P[i], fma(float(sb[1]), xs, C[i]));
    }
  }
  // K slices added in slice order, 16 outputs a thread at a time (the buffer stays within 28 KB at 8 slices)
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * 16 * 64];
  if (SK > 1)
    for (int c0 = 0; c0 < CAP; c0 += 16) {
      if (slice > 0) for (int i = 0; i < 16; i++) part[((slice - 1) * 16 + i) * 64 + tip] = C[c0 + i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (slice == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < 16; i++) C[c0 + i] += part[((s2 - 1) * 16 + i) * 64 + tip];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  if (slice == 0)
    for (int i = 0; i < CAP; i++) {
      const int m = rb + erow[i], n = n0 + ecol[i];
      if (m < M) Y[m * N + n] = static_cast<bfloat>(C[i]);
    }
"""
AB_FLAG = [False]                                    # a live A/B flips this every few rounds (engine side)

_kernels: dict[str, Any] = {}


def _named(base: str, source: str) -> str:
    """Kernel names carry a hash of their source: MLX caches compiled kernels by name."""

    import hashlib

    return f"{base}_{hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]}"


class _Baked:
    """A kernel called like MLX's, its ``template`` integers written into the source as constants: MLX runs a
    std::regex over template arguments on every call (lane_matmul built."""

    def __init__(self, base: str, body: str, inputs: list[str], outputs: list[str]) -> None:
        self.base, self.body, self.inputs, self.outputs = base, body, inputs, outputs
        self.compiled: dict[tuple, Any] = {}

    def __call__(self, *, template: Any = (), **kwargs: Any) -> Any:
        key = tuple(template)
        run = self.compiled.get(key)
        if run is None:
            source = "".join(f"  constexpr int {k} = {int(v)};\n" for k, v in key) + self.body
            run = self.compiled[key] = mx.fast.metal_kernel(name=_named(self.base, source), input_names=self.inputs,
                                                            output_names=self.outputs, source=source, header=_HEADER)
        return run(**kwargs)


def _kernel(name: str) -> Any:
    if name not in _kernels:
        if name == "xsum":
            _kernels[name] = _Baked("lane_qmm_xsum", _XSUM, ["X", "mdims"], ["XS"])
        else:
            source = {"coop": _COOP, "main_tiled": _MAIN_TILED, "main": _MAIN}[name]
            _kernels[name] = _Baked("lane_qmm_" + name, source, ["X", "XS", "Wq", "SBt", "mdims"], ["Y"])
    return _kernels[name]


_mdims_cache: dict[tuple[int, int], mx.array] = {}
_xs_cache: dict[Any, tuple[mx.array, mx.array]] = {}


def _mdims(m: int, mp: int) -> mx.array:
    key = (m, mp)
    if key not in _mdims_cache:
        _mdims_cache[key] = ints((m, mp))
    return _mdims_cache[key]


def split_k(n: int, k: int) -> int:
    """K slices for an (n, k) weight: fixed by the shape, never by the row count."""

    tiles = -(-n // NT)
    sk = 1
    while sk < 8 and tiles * sk < 1024 and (k // 64) // (sk * 2) >= 8:
        sk *= 2
    return sk


def pack_scales(scales: mx.array, biases: mx.array) -> mx.array:
    """(N, K/GS) scales and biases -> (K/GS, N, 2) bf16 pairs, group-major."""

    return mx.stack([scales.T, biases.T], axis=-1).astype(mx.bfloat16)


def tile_weight(weight: mx.array, nt: int = NT, group: int = 64) -> mx.array:
    """MLX's packed (N, K/8) weight -> the same shape and bytes, regrouped [N/nt][K/group][nt columns x group/2 bytes]."""

    n, k8, w = int(weight.shape[0]), int(weight.shape[1]), group // 8
    return mx.contiguous(weight.reshape(n // nt, nt, k8 // w, w).transpose(0, 2, 1, 3).reshape(n, k8))


def untile_weight(weight: mx.array, nt: int = NT, group: int = 64) -> mx.array:
    """``tile_weight`` undone: MLX's packed layout again."""

    n, k8, w = int(weight.shape[0]), int(weight.shape[1]), group // 8
    return mx.contiguous(weight.reshape(n // nt, k8 // w, nt, w).transpose(0, 2, 1, 3).reshape(n, k8))


def supports(weight: mx.array, scales: mx.array, x: mx.array, bits: int, group_size: int, mode: str) -> bool:
    if bits != 4 or group_size not in (32, 64) or mode != "affine":
        return False
    if x.dtype != mx.bfloat16 or scales.dtype != mx.bfloat16 or weight.dtype != mx.uint32 or weight.ndim != 2:
        return False
    k = int(x.shape[-1])
    n = int(weight.shape[0])
    return k % 64 == 0 and int(weight.shape[1]) * 8 == k and n % 4 == 0


def lane_matmul(x: mx.array, weight: mx.array, sbt: mx.array, *, tiled: bool = False,
                sk: int | None = None, nt: int = NT, group: int = 64) -> mx.array:
    """x (..., K) bf16 times the packed 4-bit ``weight`` (N, K/8) transposed; rows <= MAX_ROWS.

    ``tiled``: ``weight`` is in ``tile_weight``'s layout (N a multiple of ``nt``, 32 or 64); same bits either way.
    ``sk``: the K slices instead of ``split_k(N, K)``; stacked weights keep their own bits when it is theirs.
    ``group``: inputs a scale covers (32 or 64).
    """

    K = int(x.shape[-1])
    N = int(weight.shape[0])
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = int(x2.shape[0])
    if M > MAX_ROWS:
        raise ValueError(f"lane_matmul takes at most {MAX_ROWS} rows, got {M}")
    MP = 16 * ((M + 15) // 16)
    KG = K // group
    mdims = _mdims(M, MP)
    # inputs shared by several projections: their group sums once (the entry holds x alive)
    key = id(x) if group == 64 else (id(x), group)
    hit = _xs_cache.get(key)
    if hit is not None and hit[0] is x:
        xs = hit[1]
    else:
        xs = _kernel("xsum")(inputs=[x2, mdims], template=[("K", K), ("GS", group)], grid=(KG, MP, 1),
                             threadgroup=(min(KG, 256), 1, 1), output_shapes=[(KG, MP)],
                             output_dtypes=[mx.float32])[0]
        _xs_cache[key] = (x, xs)
        while len(_xs_cache) > 4:
            _xs_cache.pop(next(iter(_xs_cache)))
    sk = int(sk) if sk else split_k(N, K)
    nt = int(nt) if tiled else NT
    block = MP if MP <= ROW_BLOCK else ROW_BLOCK
    if nt == 64:
        y = _kernel("coop")(inputs=[x2, xs, weight, sbt, mdims],
                            template=[("TMR", block // 16), ("N", N), ("K", K), ("SK", sk), ("GS", group)],
                            grid=((N // 64) * 64 * sk, -(-MP // block), 1), threadgroup=(64 * sk, 1, 1),
                            output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
        return y.reshape(*lead, N)
    if tiled and N % nt:
        raise ValueError(f"tiled weights need N to be a multiple of {nt}, got {N}")
    y = _kernel("main_tiled" if tiled else "main")(inputs=[x2, xs, weight, sbt, mdims],
                        template=[("TMR", block // 16), ("N", N), ("K", K), ("NT", nt), ("SK", sk), ("GS", group)],
                        grid=(-(-N // nt) * 32 * sk, -(-MP // block), 1), threadgroup=(32 * sk, 1, 1),
                        output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(*lead, N)


# -- routing the model's projections --------------------------------------------------------
_ORIG: Any = None
enabled = False
max_rows = MAX_ROWS


_tiled_modules: list[Any] = []   # modules whose weight install() regrouped (uninstall() restores them)


def _call(self: Any, x: mx.array) -> mx.array:
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    tiled = getattr(self, "_lane_tiled", False)
    nt = getattr(self, "_lane_nt", NT)
    if enabled and rows <= max_rows and supports(self["weight"], self["scales"], x, self.bits,
                                                 self.group_size, getattr(self, "mode", "affine")):
        sbt = getattr(self, "_lane_sbt", None)
        if sbt is None:
            sbt = pack_scales(self["scales"], self["biases"])
            mx.eval(sbt)
            object.__setattr__(self, "_lane_sbt", sbt)
        y = lane_matmul(x, self["weight"], sbt, tiled=tiled, nt=nt, group=self.group_size)
    elif tiled:
        # wider than the lane kernel takes (MLX's chunked prefill): MLX's layout, rebuilt for this call
        y = mx.quantized_matmul(x, untile_weight(self["weight"], nt, self.group_size), self["scales"], self["biases"],
                                transpose=True, group_size=self.group_size, bits=self.bits)
    else:
        return _ORIG(self, x)
    if "bias" in self:
        y = y + self["bias"]
    return y


# Kept 32 columns wide under ``wide``: in_proj_z stacks with in_proj_b/a (48 rows each), which only 32-column tiles divide
NARROW = ("in_proj_z",)


def install(model: Any = None, *, rows: int = MAX_ROWS, tile: bool = True, wide: bool = False) -> None:
    """Route every 4-bit QuantizedLinear call of at most ``rows`` rows through the lane matmul.

    With ``model``: interleaved scales built up front, and (``tile``) each weight whose row count is a
    multiple of NT regrouped in place (``wide``: 64 columns where it divides). Idempotent.
    """

    global _ORIG, enabled, max_rows
    import mlx.nn as nn

    if _ORIG is None:
        _ORIG = nn.QuantizedLinear.__call__
    nn.QuantizedLinear.__call__ = _call
    enabled = True
    max_rows = min(int(rows), MAX_ROWS)
    if model is not None:
        built, pending = [], 0
        for name, module in model.named_modules():
            if not (isinstance(module, nn.QuantizedLinear) and module.bits == 4 and module.group_size in (32, 64)
                    and module["scales"].dtype == mx.bfloat16):
                continue
            if getattr(module, "_lane_sbt", None) is None:
                sbt = pack_scales(module["scales"], module["biases"])
                object.__setattr__(module, "_lane_sbt", sbt)
                built.append(sbt)
                pending += sbt.nbytes
            weight = module["weight"]
            if (tile and not getattr(module, "_lane_tiled", False) and weight.dtype == mx.uint32
                    and weight.ndim == 2 and int(weight.shape[0]) % NT == 0 and int(weight.shape[1]) % 8 == 0):
                nt = 64 if (wide and int(weight.shape[0]) % 64 == 0 and not name.endswith(NARROW)) else NT
                module.weight = tile_weight(weight, nt, module.group_size)
                object.__setattr__(module, "_lane_tiled", True)
                object.__setattr__(module, "_lane_nt", nt)
                _tiled_modules.append(module)
                built.append(module["weight"])
                pending += 2 * weight.nbytes             # the old layout lives until this batch is evaluated
            if pending >= 2 * 1024**3:
                mx.eval(built)
                built, pending = [], 0
        if built:
            mx.eval(built)
        mx.clear_cache()      # the old layouts' buffers would otherwise sit in MLX's buffer cache


def warm(model: Any, *, rows: tuple[int, ...] = (1, 17, 33)) -> int:
    """Compile every kernel variant the model's projections will use (one per shape and row tile)."""

    import mlx.nn as nn

    seen: set[tuple[int, int, int]] = set()
    outs = []
    for _, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear) or getattr(module, "_lane_sbt", None) is None:
            continue
        n, k = int(module["weight"].shape[0]), int(module["weight"].shape[1]) * 8
        if (n, k, module.group_size) in seen:
            continue
        seen.add((n, k, module.group_size))
        for m in rows:
            outs.append(lane_matmul(mx.zeros((m, k), dtype=mx.bfloat16), module["weight"], module._lane_sbt,
                                    tiled=getattr(module, "_lane_tiled", False), nt=getattr(module, "_lane_nt", NT),
                                    group=module.group_size))
    mx.eval(outs)
    return len(seen)


def uninstall() -> None:
    """MLX's own kernels again, with the weights back in MLX's layout."""

    global enabled
    import mlx.nn as nn

    enabled = False
    while _tiled_modules:
        module = _tiled_modules.pop()
        module.weight = untile_weight(module["weight"], getattr(module, "_lane_nt", NT), module.group_size)
        object.__setattr__(module, "_lane_tiled", False)
        object.__setattr__(module, "_lane_nt", NT)
        mx.eval(module["weight"])
    if _ORIG is not None:
        nn.QuantizedLinear.__call__ = _ORIG


__all__ = ["MAX_ROWS", "install", "lane_matmul", "pack_scales", "split_k", "supports", "tile_weight", "uninstall",
           "untile_weight", "warm"]

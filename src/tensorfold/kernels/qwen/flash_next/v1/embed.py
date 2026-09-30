"""Token and n-gram embedding rows, and the (1 + w) RMSNorm over rows."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import AFFINE_HEADER, QDOT_HEADER, edited, kernel, padded

_PLE_LOOKUP = r"""
  // Thread (d, h, r): dim d of head h of row r. Row id IDS[r][h] lies in one of 8 table groups (row starts GSTART);
  // its 4-bit value q, scale and bias give bf16(bf16(scale * q) + bias) (mx.dequantize on bf16 scales).
  const int d = int(thread_position_in_grid.x);
  const int h = int(thread_position_in_grid.y);
  const int r = int(thread_position_in_grid.z);
  const uint id = IDS[r * H + h];
  int g = 0;
  for (int j = 1; j < 8; j++) g += id >= GSTART[j] ? 1 : 0;
  const size_t row = size_t(id - GSTART[g]);
  const device uint32_t* W; const device bfloat* SC; const device bfloat* BI;
  switch (g) {
    case 0: W = W0; SC = S0; BI = B0; break;
    case 1: W = W1; SC = S1; BI = B1; break;
    case 2: W = W2; SC = S2; BI = B2; break;
    case 3: W = W3; SC = S3; BI = B3; break;
    case 4: W = W4; SC = S4; BI = B4; break;
    case 5: W = W5; SC = S5; BI = B5; break;
    case 6: W = W6; SC = S6; BI = B6; break;
    default: W = W7; SC = S7; BI = B7; break;
  }
  const uint word = W[row * (DIMS / 8) + d / 8];
  const bfloat q = bfloat(float((word >> (4 * (d % 8))) & 0xFu));
  const bfloat sc = SC[row * (DIMS / 32) + d / 32], bi = BI[row * (DIMS / 32) + d / 32];
  OUT[(r * H + h) * DIMS + d] = sc * q + bi;
"""

_PLE_ROWS = r"""
  // Thread (d, i): dim d of gathered row i (its words, bf16 scales and biases copied from the host table), with
  // q4_ple_lookup's arithmetic: bf16(bf16(scale * q) + bias).
  const int d = int(thread_position_in_grid.x);
  const size_t i = size_t(thread_position_in_grid.y);
  const uint word = W[i * (DIMS / 8) + d / 8];
  const bfloat q = bfloat(float((word >> (4 * (d % 8))) & 0xFu));
  const bfloat sc = SC[i * (DIMS / 32) + d / 32], bi = BI[i * (DIMS / 32) + d / 32];
  OUT[i * DIMS + d] = sc * q + bi;
"""

_EMBED_ROWS = r"""
  // Thread (d, r): dim d of token row r (quantized embedding, mx.dequantize's bf16(bf16(scale * q) + bias)),
  // written to each of the TILE copies of the row (the residual streams start as copies of the embedding).
  const int d = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  const size_t row = size_t(IDS[r]);
  const uint word = W[row * (DIMS / 8) + d / 8];
  const bfloat q = bfloat(float((word >> (4 * (d % 8))) & 0xFu));
  const bfloat v = SC[row * (DIMS / 32) + d / 32] * q + BI[row * (DIMS / 32) + d / 32];
  for (int t = 0; t < TILE; t++) OUT[(r * TILE + t) * DIMS + d] = v;
"""

_RMS_ROWS = r"""
  // One threadgroup of 1024 threads per row (per group of G features when G < W): bf16((x * rinv) * scale), the
  // sum of squares in fp32 (each thread's features in order, then the simdgroups in order).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.y);
  const int grp = int(threadgroup_position_in_grid.x);
  const size_t base = size_t(r) * W + size_t(grp) * G;
  threadgroup float part[32];
  float ss = 0.0f;
  for (int i = int(t); i < G; i += 1024) { const float v = float(X[base + i]); ss = fma(v, v, ss); }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < 32; k++) total += part[k];
  const float rinv = metal::rsqrt(total / float(G) + eps[0]);
  for (int i = int(t); i < G; i += 1024)
    OUT[base + i] = bfloat((float(X[base + i]) * rinv) * SCALE[(grp * G + i) % SW]);
"""


_PLE_GATE = r"""
  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  threadgroup float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = simd_sum(sk); sq = simd_sum(sq);
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = simd_sum(dot);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[2][sg] = dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / float(bfloat(metal::precise::sqrt(float(D))))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = simd_sum(sc);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[0][sg] = sc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(v[i]);
    NORMED[size_t(r) * W + e] = bfloat((v[i] * rc) * CS[e]);
  }
"""

_PLE_CONV = r"""
  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb * bsig(yb)));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)));
"""

# The lookups for any MLX affine width: value d of a row's bit stream, its group's scale and bias.
_CODE = ("const bfloat q = bfloat(float((word >> (4 * (d % 8))) & 0xFu));",
         "const bfloat q = bfloat(float(code_at<BITS>(W + ROW * (DIMS * BITS / 32), d)));")
_PLE_LOOKUP_Q = edited(_PLE_LOOKUP, [("  const uint word = W[row * (DIMS / 8) + d / 8];\n", ""),
                                     (_CODE[0], _CODE[1].replace("ROW", "row")), (
    "const bfloat sc = SC[row * (DIMS / 32) + d / 32], bi = BI[row * (DIMS / 32) + d / 32];",
    "const bfloat sc = SC[row * (DIMS / GS) + d / GS], bi = BI[row * (DIMS / GS) + d / GS];")])
_PLE_ROWS_Q = edited(_PLE_ROWS, [("  const uint word = W[i * (DIMS / 8) + d / 8];\n", ""),
                                 (_CODE[0], _CODE[1].replace("ROW", "i")), (
    "const bfloat sc = SC[i * (DIMS / 32) + d / 32], bi = BI[i * (DIMS / 32) + d / 32];",
    "const bfloat sc = SC[i * (DIMS / GS) + d / GS], bi = BI[i * (DIMS / GS) + d / GS];")])
_EMBED_ROWS_Q = edited(_EMBED_ROWS, [("  const uint word = W[row * (DIMS / 8) + d / 8];\n", ""),
                                     (_CODE[0], _CODE[1].replace("ROW", "row")), (
    "const bfloat v = SC[row * (DIMS / 32) + d / 32] * q + BI[row * (DIMS / 32) + d / 32];",
    "const bfloat v = SC[row * (DIMS / GS) + d / GS] * q + BI[row * (DIMS / GS) + d / GS];")])


def _lookup(name: str, q4: Any, generic: Any, inputs: list[str], bits: int, group: int) -> tuple[Any, list]:
    """The 4-bit group-32 kernel, or the any-width one with its format as template constants."""

    if (bits, group) == (4, 32):
        return kernel(f"q4_{name}", q4, inputs, ["OUT"]), []
    return (kernel(f"qa_{name}", generic, inputs, ["OUT"], header=QDOT_HEADER + AFFINE_HEADER),
            [("BITS", bits), ("GS", group)])


class PleTables:
    """Keep n-gram shards as views into eight GPU groups, or use the host table with its checkpoint memory-mapped."""

    groups = 8

    def __init__(self, emb: Any) -> None:
        self.dims = int(emb.dims)
        self.bits, self.group = int(getattr(emb, "quant_bits", 4)), int(getattr(emb, "quant_group", 32))
        self.scale = float(getattr(emb, "table_scale", 1.0))
        self.host = getattr(emb, "host", None)
        if self.host is not None:
            return
        shards = emb.shards
        if any((int(sh.bits), int(sh.group_size)) != (self.bits, self.group) for sh in shards):
            raise ValueError("the n-gram shards must share one quantization format")
        per = -(-len(shards) // self.groups)
        self.weights, self.scales, self.biases, starts = [], [], [], [0]
        for g in range(self.groups):
            part = shards[g * per:(g + 1) * per]
            w = mx.concatenate([sh.weight for sh in part])
            sc = mx.concatenate([sh.scales for sh in part])
            bi = mx.concatenate([sh.biases for sh in part])
            mx.eval(w, sc, bi)
            at = 0
            for sh in part:
                n = int(sh.weight.shape[0])
                sh.weight, sh.scales, sh.biases = w[at:at + n], sc[at:at + n], bi[at:at + n]
                mx.eval(sh.weight, sh.scales, sh.biases)
                at += n
            self.weights.append(w)
            self.scales.append(sc)
            self.biases.append(bi)
            starts.append(starts[-1] + at)
            mx.clear_cache()
        self.starts = mx.array(starts[:-1], dtype=mx.uint32)
        mx.eval(self.starts)

def scaled_rows(rows: mx.array, scale: float) -> mx.array:
    """Looked-up bf16 rows times the table's scale, rounded once to bf16 (the identity for scale 1)."""

    if scale == 1.0:
        return rows
    return (rows.astype(mx.float32) * scale).astype(rows.dtype)


def ple_lookup(ids: Any, tables: PleTables) -> mx.array:
    """Dequantized rows [R, H * DIMS] bf16 for global n-gram row ids [R, H], times the table's scale."""

    return scaled_rows(_ple_lookup(ids, tables), getattr(tables, "scale", 1.0))


def _ple_lookup(ids: Any, tables: PleTables) -> mx.array:
    """Dequantized rows [R, H * DIMS] bf16 for global n-gram row ids [R, H] (the shards' concatenated order)."""

    import numpy as np

    ids = np.asarray(ids).reshape(-1, np.asarray(ids).shape[-1])
    rows, heads = ids.shape
    if tables.host is not None:
        words, scales, biases = tables.host.gather(ids)
        run, fmt = _lookup("ple_rows", _PLE_ROWS, _PLE_ROWS_Q, ["W", "SC", "BI"], tables.bits, tables.group)
        return run(inputs=[mx.array(words), mx.array(scales).view(mx.bfloat16), mx.array(biases).view(mx.bfloat16)],
                   template=[("DIMS", tables.dims), *fmt], grid=(tables.dims, rows * heads, 1),
                   threadgroup=(tables.dims, 1, 1), output_shapes=[(rows, heads * tables.dims)],
                   output_dtypes=[mx.bfloat16])[0]
    names = ["IDS", "GSTART"] + [f"{k}{g}" for g in range(8) for k in ("W", "S", "B")]
    run, fmt = _lookup("ple_lookup", _PLE_LOOKUP, _PLE_LOOKUP_Q, names, tables.bits, tables.group)
    arrays = [mx.array(ids.astype(np.uint32)), tables.starts]
    for g in range(8):
        arrays += [tables.weights[g], tables.scales[g], tables.biases[g]]
    return run(inputs=arrays, template=[("H", heads), ("DIMS", tables.dims), *fmt],
                  grid=(tables.dims, heads, rows), threadgroup=(tables.dims, 1, 1),
                  output_shapes=[(rows, heads * tables.dims)], output_dtypes=[mx.bfloat16])[0]

def embed_rows(ids: Any, embedding: Any, *, tile: int = 1) -> mx.array:
    """Repeat each dequantized token row tile times into bf16 [R, tile * DIMS], matching mx.dequantize bit for bit."""

    import numpy as np

    if not isinstance(ids, mx.array):
        ids = mx.array(np.asarray(ids, dtype=np.uint32).reshape(-1))
    rows = int(ids.size)
    bits, group = int(getattr(embedding, "bits", 4)), int(getattr(embedding, "group_size", 32))
    dims = int(embedding.weight.shape[1]) * 32 // bits
    run, fmt = _lookup("embed_rows", _EMBED_ROWS, _EMBED_ROWS_Q, ["IDS", "W", "SC", "BI"], bits, group)
    return run(inputs=[padded(ids.reshape(-1).astype(mx.uint32)), embedding.weight, embedding.scales, embedding.biases],
                  template=[("DIMS", dims), ("TILE", tile), *fmt], grid=(dims, rows, 1),
                  threadgroup=(min(dims, 256), 1, 1),
                  output_shapes=[(rows, tile * dims)], output_dtypes=[mx.bfloat16])[0]

def rms_norm_rows(x: mx.array, scale: mx.array, eps: mx.array, *, group: int | None = None) -> mx.array:
    """CenteredRMSNorm's (1 + w) RMSNorm over each row (or each run of ``group`` features) of x [R, W] -> bf16."""

    rows, width = x.shape
    g = int(group or width)
    run = kernel("q4_rms_rows", _RMS_ROWS, ["X", "SCALE", "eps"], ["OUT"])
    return run(inputs=[x, scale, eps], template=[("W", width), ("G", g), ("SW", int(scale.shape[-1]))],
                  grid=(1024 * (width // g), rows, 1), threadgroup=(1024, 1, 1),
                  output_shapes=[(rows, width)], output_dtypes=[mx.bfloat16])[0]


def ple_gate(kv: mx.array, h: mx.array, key_scale: mx.array, query_scale: mx.array, conv_scale: mx.array,
             eps: mx.array, *, streams: int) -> tuple[mx.array, mx.array]:
    """PLE's gate from the stacked key|value rows [R, S*D + D] and the streams [R, S*D]: (gated, conv-normed)."""

    rows, wide = h.shape
    dims = wide // streams
    if dims % 256:
        raise ValueError("ple_gate: D must be a multiple of 256")
    from tensorfold.kernels.qwen.flash_next.v1.base import QDOT_HEADER

    run = kernel("q4_ple_gate", _PLE_GATE, ["KV", "H", "KS", "QS", "CS", "eps"], ["GATED", "NORMED"],
                 header=QDOT_HEADER)
    return tuple(run(inputs=[kv, h, key_scale, query_scale, conv_scale, eps], template=[("S", streams), ("D", dims)],
                     grid=(256 * streams, rows, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(rows, wide), (rows, wide)], output_dtypes=[mx.bfloat16, mx.bfloat16]))


def ple_conv(conv_in: mx.array, weight: mx.array, gated: mx.array, h: mx.array, *, streams: int,
             dilation: int) -> mx.array:
    """h + gated + SiLU(depthwise conv) for the rows after ``conv_in``'s tail: h_new [R, S*D] bf16."""

    rows, wide = h.shape
    taps = int(weight.shape[-1])
    from tensorfold.kernels.qwen.flash_next.v1.base import QDOT_HEADER

    run = kernel("q4_ple_conv", _PLE_CONV, ["CIN", "CW", "GATED", "H"], ["HOUT"], header=QDOT_HEADER)
    return run(inputs=[conv_in, weight, gated, h],
               template=[("S", streams), ("D", wide // streams), ("TAPS", taps), ("DIL", dilation)],
               grid=(wide, rows, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, wide)],
               output_dtypes=[mx.bfloat16])[0]

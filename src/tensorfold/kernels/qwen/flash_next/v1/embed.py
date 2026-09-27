"""Token and n-gram embedding rows, and the (1 + w) RMSNorm over rows."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import kernel, padded

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


class PleTables:
    """Keep n-gram shards as views into eight GPU groups, or use the host table with its checkpoint memory-mapped."""

    groups = 8

    def __init__(self, emb: Any) -> None:
        self.dims = int(emb.dims)
        self.host = getattr(emb, "host", None)
        if self.host is not None:
            return
        shards = emb.shards
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

def ple_lookup(ids: Any, tables: PleTables) -> mx.array:
    """Dequantized rows [R, H * DIMS] bf16 for global n-gram row ids [R, H] (the shards' concatenated order)."""

    import numpy as np

    ids = np.asarray(ids).reshape(-1, np.asarray(ids).shape[-1])
    rows, heads = ids.shape
    if tables.host is not None:
        words, scales, biases = tables.host.gather(ids)
        run = kernel("q4_ple_rows", _PLE_ROWS, ["W", "SC", "BI"], ["OUT"])
        return run(inputs=[mx.array(words), mx.array(scales).view(mx.bfloat16), mx.array(biases).view(mx.bfloat16)],
                   template=[("DIMS", tables.dims)], grid=(tables.dims, rows * heads, 1),
                   threadgroup=(tables.dims, 1, 1), output_shapes=[(rows, heads * tables.dims)],
                   output_dtypes=[mx.bfloat16])[0]
    names = ["IDS", "GSTART"] + [f"{k}{g}" for g in range(8) for k in ("W", "S", "B")]
    run = kernel("q4_ple_lookup", _PLE_LOOKUP, names, ["OUT"])
    arrays = [mx.array(ids.astype(np.uint32)), tables.starts]
    for g in range(8):
        arrays += [tables.weights[g], tables.scales[g], tables.biases[g]]
    return run(inputs=arrays, template=[("H", heads), ("DIMS", tables.dims)],
                  grid=(tables.dims, heads, rows), threadgroup=(tables.dims, 1, 1),
                  output_shapes=[(rows, heads * tables.dims)], output_dtypes=[mx.bfloat16])[0]

def embed_rows(ids: Any, embedding: Any, *, tile: int = 1) -> mx.array:
    """Repeat each dequantized token row tile times into bf16 [R, tile * DIMS], matching mx.dequantize bit for bit."""

    import numpy as np

    if not isinstance(ids, mx.array):
        ids = mx.array(np.asarray(ids, dtype=np.uint32).reshape(-1))
    rows = int(ids.size)
    dims = int(embedding.weight.shape[1]) * 8
    run = kernel("q4_embed_rows", _EMBED_ROWS, ["IDS", "W", "SC", "BI"], ["OUT"])
    return run(inputs=[padded(ids.reshape(-1).astype(mx.uint32)), embedding.weight, embedding.scales, embedding.biases],
                  template=[("DIMS", dims), ("TILE", tile)], grid=(dims, rows, 1), threadgroup=(min(dims, 256), 1, 1),
                  output_shapes=[(rows, tile * dims)], output_dtypes=[mx.bfloat16])[0]

def rms_norm_rows(x: mx.array, scale: mx.array, eps: mx.array, *, group: int | None = None) -> mx.array:
    """CenteredRMSNorm's (1 + w) RMSNorm over each row (or each run of ``group`` features) of x [R, W] -> bf16."""

    rows, width = x.shape
    g = int(group or width)
    run = kernel("q4_rms_rows", _RMS_ROWS, ["X", "SCALE", "eps"], ["OUT"])
    return run(inputs=[x, scale, eps], template=[("W", width), ("G", g), ("SW", int(scale.shape[-1]))],
                  grid=(1024 * (width // g), rows, 1), threadgroup=(1024, 1, 1),
                  output_shapes=[(rows, width)], output_dtypes=[mx.bfloat16])[0]

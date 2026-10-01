"""Sparse attention: q/k norms and RoPE, the block selection past 2,048 keys, attention over each row's keys."""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import MAX_STREAMS, consts, ints, kernel, log2, padded, pick

_ATTN_PREP = r"""
  // one threadgroup of HD threads per (row, head): heads [0, NQ) are queries (from the stacked projection's
  // [q | gate] pairs), [NQ, NQ + NKV) keys, then NI indexer queries (IHD dims each, after the values). RMSNorm
  // with (1 + w) in fp32, bf16 out, then RoPE on the first RD dims (non-interleaved halves), angles in fp32 at the
  // row's position.
  const int d = int(thread_position_in_threadgroup.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const bool isq = head < NQ;
  const bool isi = head >= NQ + NKV;
  const int width = isi ? IHD : HD;
  const bool live = d < width;
  int src;
  if (isq) src = r * PW + head * 2 * HD + d;
  else if (!isi) src = r * PW + NQ * 2 * HD + (head - NQ) * HD + d;
  else src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD + d;
  threadgroup float part[HD / 32];
  threadgroup float normed[HD];
  const float x = live ? float(P[src]) : 0.0f;
  float ss = simd_sum(x * x);
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < width / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!live) return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    // as mx.fast.rope: inv_freq = exp2(-(i / half) * log2(base)), fast cos/sin of position * inv_freq
    const float freq = metal::exp2(-(float(i) / float(hr)) * LOG2BASE[0]);
    const float angle = float(POS[r]) * freq;
    const float c = metal::fast::cos(angle), s = metal::fast::sin(angle);
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) Q[(r * NQ + head) * HD + d] = bfloat(out);
  else if (isi) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(out);
  else Kout[(r * NKV + head - NQ) * HD + d] = bfloat(out);
"""

_IDX_POOL = r"""
  // Threadgroup j (DI threads): block START + j's pooled indexer key: the mean of its 4 raw keys (fp32 in order,
  // bf16), RMSNorm with (1 + w) (fp32, bf16), RoPE (RD dims, non-interleaved halves) at the block's first position.
  const int d = int(thread_position_in_threadgroup.x);
  const int j = int(threadgroup_position_in_grid.y);
  const int b = START[0] + j;
  threadgroup float part[DI / 32];
  threadgroup float normed[DI];
  const device bfloat* src = RAW + size_t(4 * b) * DI + d;
  float m = float(src[0]);
  for (int k = 1; k < 4; k++) m += float(src[k * DI]);
  const float x = float(bfloat(m * 0.25f));
  float ss = simd_sum(x * x);
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < DI / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(DI) + eps[0]);
  normed[d] = float(bfloat((x * inv) * W[d]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const float freq = metal::exp2(-(float(i) / float(hr)) * LOG2BASE[0]);
    const float angle = float(4 * b) * freq;
    const float c = metal::fast::cos(angle), s = metal::fast::sin(angle);
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  OUT[j * DI + d] = bfloat(out);
"""

# _IDX_POOL with raw rows counted from block START's first key (a chained draft's rows sit apart from the cache)
_IDX_POOL_REL = _IDX_POOL.replace("const device bfloat* src = RAW + size_t(4 * b) * DI + d;",
                                  "const device bfloat* src = RAW + size_t(4 * j) * DI + d;")

_SELECT_HEADER = r"""
inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
"""

_IDX_SCORES = r"""
  // Simdgroup s of threadgroup (x, y) scores blocks (8 x + s) BB .. + BB - 1 for rows RB y .. RB y + RB - 1, each
  // block's keys read once for those rows: block b's score for row r is the sum over the HI indexer heads (in order)
  // of relu(q . pooled b) (fp32: a lane's DI / 32 dims in order, then simd_sum), over sqrt(DI). Only rows past TOP
  // complete blocks, and only their complete blocks, are scored.
  const uint lane = thread_index_in_simdgroup;
  const int b0 = (int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup)) * BB;
  const int r0 = int(threadgroup_position_in_grid.y) * RB;
  const int nb = int(POOLED_shape[0]), r1 = metal::min(int(Q_shape[0]), r0 + RB);
  constexpr int PER = DI / 32;
  for (int b = b0; b < b0 + BB && b < nb; b++) {
    const device bfloat* pb = POOLED + size_t(b) * DI + lane * PER;
    float p[PER];
    for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
    for (int r = r0; r < r1; r++) {
      const int complete = COMPLETE[r];
      if (complete <= TOP || b >= complete) continue;
      float s = 0.0f;
      for (int h = 0; h < HI; h++) {
        const device bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
        float dot = 0.0f;
        for (int i = 0; i < PER; i++) dot = fma(float(qh[i]), p[i], dot);
        s += metal::max(simd_sum(dot), 0.0f);
      }
      if (lane == 0) SC[size_t(r) * nb + b] = s / metal::precise::sqrt(float(DI));
    }
  }
"""
SCORE_ROWS = 8      # rows a threadgroup scores: one read of a block's keys serves them


def score_blocks(blocks: int, rows: int) -> int:
    """Blocks a simdgroup scores: 8 at 16k+ blocks, fewer below so ~256 threadgroups stay busy; one for one row."""

    return 1 if rows == 1 or blocks < 4096 else min(8, 1 << ((blocks // 2048).bit_length() - 1))

_IDX_SELECT = r"""
  // One threadgroup (1024 threads) a row past TOP complete blocks: its TOP best blocks by score (radix select over
  // order-preserving keys, 8 bits a pass; among scores equal to the cut, the lowest block ids), written as the keys
  // they cover (4 a block) in position order, then the row's tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) return;
  const int ends = ENDS[r];
  const int stride = SC_shape[1];
  const device float* sc = SC + size_t(r) * stride;
  device int* keys = KEYS + size_t(r) * KW;
  threadgroup atomic_uint hist[256];
  threadgroup uint cut_t, need_t;
  threadgroup int tot_a[32], tot_e[32];
  uint prefix = 0u, mask = 0u, need = TOP;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = int(t); b < nb; b += 1024) {
      const uint k = tf_key(sc[b]);
      if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {                            // the cut bin: lane l scans bins 255 - 8l down to 248 - 8l
      uint c[8], mine = 0u;
      for (int i = 0; i < 8; i++) {
        c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
        mine += c[i];
      }
      uint above = simd_prefix_exclusive_sum(mine);
      if (above < need && above + mine >= need) {
        int bin = 248 - 8 * int(lane);
        for (int i = 0; i < 8; i++) {
          if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
          above += c[i];
        }
        cut_t = prefix | (uint(bin) << shift);
        need_t = need - above;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = cut_t;
    need = need_t;
    mask |= 255u << shift;
  }
  // `prefix` is the cut score's key: every block above it is taken, and the first `need` equal to it
  const int chunk = (nb + 1023) / 1024;
  const int lo = min(nb, int(t) * chunk), hi = min(nb, lo + chunk);
  int n_above = 0, n_equal = 0;
  for (int b = lo; b < hi; b++) {
    const uint k = tf_key(sc[b]);
    n_above += k > prefix ? 1 : 0;
    n_equal += k == prefix ? 1 : 0;
  }
  int pa = simd_prefix_exclusive_sum(n_above), pe = simd_prefix_exclusive_sum(n_equal);
  if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const int a = tot_a[lane], e = tot_e[lane];
    tot_a[lane] = simd_prefix_exclusive_sum(a);
    tot_e[lane] = simd_prefix_exclusive_sum(e);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  pa += tot_a[sg];
  pe += tot_e[sg];
  int out = pa + min(pe, int(need));
  for (int b = lo; b < hi; b++) {
    const uint k = tf_key(sc[b]);
    bool take = k > prefix;
    if (k == prefix) { take = pe < int(need); pe++; }
    if (take) {
      for (int j = 0; j < 4; j++) keys[out * 4 + j] = 4 * b + j;
      out++;
    }
  }
  if (t == 0)
    for (int k = 4 * nb; k < ends; k++) keys[4 * TOP + (k - 4 * nb)] = k;
"""

_ATTN_PARTS = r"""
  // Threadgroup (h, r, p): query head h of row r over part p of the row's key list (SPARSE[r]: the NK[r] ids
  // IDS[r]; else keys 0 .. NK[r] - 1), entries [p n / P, (p + 1) n / P); 8 simdgroups, simdgroup g taking every
  // 8th entry from the part's start, a lane D / 32 dims. fp32: scores q . k with q pre-scaled, an online softmax
  // per simdgroup, the simdgroups combined in order into the part's (max, sum, output).
  constexpr int PER = D / 32;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int h = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int part = int(threadgroup_position_in_grid.z);
  const int kvh = h / (H / KVH);
  const int n = NK[r];
  const int lo = int((long(part) * n) / P), hi = int((long(part + 1) * n) / P);
  const bool sparse = SPARSE[r] != 0;
  const size_t cap = size_t(Kc_shape[2]);
  const device bfloat* kb = Kc + size_t(kvh) * cap * D + lane * PER;
  const device bfloat* vb = Vc + size_t(kvh) * cap * D + lane * PER;
  const auto ids = IDS + size_t(r) * IDS_shape[1];      // device, or constant when MLX binds a small array so
  const device bfloat* qp = Q + (size_t(r) * H + h) * D + lane * PER;
  float q[PER], o[PER];
  for (int i = 0; i < PER; i++) { q[i] = SCALE[0] * float(qp[i]); o[i] = 0.0f; }
  float m = -INFINITY, l = 0.0f;
  for (int j = lo + int(g); j < hi; j += 8) {
    const size_t key = size_t(sparse ? ids[j] : j) * D;
    float sc = 0.0f;
    for (int i = 0; i < PER; i++) sc = fma(q[i], float(kb[key + i]), sc);
    sc = simd_sum(sc);
    const float mn = metal::max(m, sc);
    const float f = metal::exp(m - mn), e = metal::exp(sc - mn);
    l = fma(l, f, e);
    for (int i = 0; i < PER; i++) o[i] = fma(e, float(vb[key + i]), o[i] * f);
    m = mn;
  }
  threadgroup float ms[8], ls[8];
  threadgroup float tile[8][D];
  if (lane == 0) { ms[g] = m; ls[g] = l; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float top = -INFINITY;
  for (int k = 0; k < 8; k++) top = metal::max(top, ms[k]);
  const float mine = m == -INFINITY ? 0.0f : metal::exp(m - top);
  for (int i = 0; i < PER; i++) tile[g][lane * PER + i] = o[i] * mine;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const size_t at = (size_t(r) * H + h) * P + part;
  for (int d = int(thread_position_in_threadgroup.x); d < D; d += 256) {
    float acc = 0.0f;
    for (int k = 0; k < 8; k++) acc += tile[k][d];
    PO[at * D + d] = acc;
  }
  if (thread_position_in_threadgroup.x == 0) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += ms[k] == -INFINITY ? 0.0f : ls[k] * metal::exp(ms[k] - top);
    PM[at * 2] = top;
    PM[at * 2 + 1] = total;
  }
"""

_ATTN_MERGE = r"""
  // Threadgroup (h, r), a thread a dim: the P parts of head h of row r combined in part order.
  const int d = int(thread_position_in_threadgroup.x);
  const int h = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const size_t at = (size_t(r) * H + h) * P;
  float top = -INFINITY;
  for (int k = 0; k < P; k++) top = metal::max(top, PM[(at + k) * 2]);
  float total = 0.0f, acc = 0.0f;
  for (int k = 0; k < P; k++) {
    const float mk = PM[(at + k) * 2];
    const float w = mk == -INFINITY ? 0.0f : metal::exp(mk - top);
    total = fma(PM[(at + k) * 2 + 1], w, total);
    acc = fma(PO[(at + k) * D + d], w, acc);
  }
  OUT[(size_t(r) * H + h) * D + d] = bfloat(acc / total);
"""

# the merge, then the output gate: bf16(bf16(merged) * sigmoid(gate)) (bf16 ops), gate from the [q | gate] pairs
_ATTN_MERGE_GATE = _ATTN_MERGE.replace(
    "  OUT[(size_t(r) * H + h) * D + d] = bfloat(acc / total);\n",
    "  const float g = float(GP[r * PW + h * 2 * D + D + d]);\n"
    "  OUT[(size_t(r) * H + h) * D + d] = bfloat(float(bfloat(acc / total)) * bsig(g));\n")


def attn_prep(projected: mx.array, positions: mx.array, q_norm: mx.array, k_norm: mx.array, index_norm: mx.array,
              eps: mx.array, *, q_heads: int, kv_heads: int, head_dim: int, index_heads: int, index_dim: int,
              rotary_dim: int, base: float) -> tuple[mx.array, mx.array, mx.array]:
    """Normalize and rotate projected q/k/indexer q using int32 row positions and (1 + w) norm scales."""

    rows, width = projected.shape
    run = kernel("q4_attn_prep", _ATTN_PREP, ["P", "POS", "QW", "KW", "IW", "eps", "LOG2BASE"],
                     ["Q", "Kout", "IQ"])
    return tuple(run(inputs=[projected, padded(positions), q_norm, k_norm, index_norm, eps, log2(base)],
                        template=[("NQ", q_heads), ("NKV", kv_heads), ("HD", head_dim), ("RD", rotary_dim),
                                  ("PW", width), ("NI", index_heads), ("IHD", index_dim)],
                        grid=(head_dim, q_heads + kv_heads + index_heads, rows), threadgroup=(head_dim, 1, 1),
                        output_shapes=[(rows, q_heads, head_dim), (rows, kv_heads, head_dim),
                                       (rows, index_heads, index_dim)],
                        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16]))

def index_pool(raw: mx.array, start: int, stop: int, norm: mx.array, eps: mx.array, *, rotary_dim: int,
               base: float, relative: bool = False) -> mx.array:
    """Pooled keys [stop - start, DI] of blocks [start, stop) from raw keys; ``relative``: raw row 0 is key 4 start."""

    dims = int(raw.shape[-1])
    if relative:
        run = kernel("q4_idx_pool_rel", _IDX_POOL_REL, ["RAW", "START", "W", "eps", "LOG2BASE"], ["OUT"])
    else:
        run = kernel("q4_idx_pool", _IDX_POOL, ["RAW", "START", "W", "eps", "LOG2BASE"], ["OUT"])
    return run(inputs=[raw, mx.array([start], dtype=mx.int32), norm, eps, log2(base)],
                  template=[("DI", dims), ("RD", rotary_dim)],
                  grid=(dims, stop - start, 1), threadgroup=(dims, 1, 1),
                  output_shapes=[(stop - start, dims)], output_dtypes=[mx.bfloat16])[0]

def index_scores(q: mx.array, pooled: mx.array, complete: Sequence[int], *, top: int) -> mx.array:
    """Block scores [R, NB] (fp32) of each row past top complete blocks, over its complete blocks only (rest unset)."""

    rows, heads, dims = q.shape
    nb = int(pooled.shape[0])
    bb = score_blocks(nb, rows)
    score = kernel("q4_idx_scores", _IDX_SCORES, ["Q", "POOLED", "COMPLETE"], ["SC"])
    return score(inputs=[q, pooled, ints(complete)],
                 template=[("HI", heads), ("DI", dims), ("TOP", top), ("BB", bb), ("RB", SCORE_ROWS)],
                 grid=(-(-nb // (8 * bb)) * 256, -(-rows // SCORE_ROWS), 1), threadgroup=(256, 1, 1),
                 output_shapes=[(rows, nb)], output_dtypes=[mx.float32])[0]


def index_select(q: mx.array, pooled: mx.array, complete: list[int], ends: list[int], *, top: int) -> mx.array:
    """Each row's top complete blocks in position order, then its unfinished tail; rows at or below top are skipped."""

    return select_blocks(index_scores(q, pooled, complete, top=top), complete, ends, top=top)


def select_blocks(scores: mx.array, complete: list[int], ends: list[int], *, top: int) -> mx.array:
    """``index_select`` from block scores [R, NB] (fp32): each row's best ``top`` of its complete blocks."""

    rows = int(scores.shape[0])
    width = 4 * top + 3
    select = kernel("q4_idx_select", _IDX_SELECT, ["SC", "COMPLETE", "ENDS"], ["KEYS"], header=_SELECT_HEADER)
    return select(inputs=[scores, ints(complete), ints(ends)],
                  template=[("TOP", top), ("KW", width)],
                  grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1),
                  output_shapes=[(rows, width)], output_dtypes=[mx.int32])[0]

def attention_rows(q: mx.array, keys: mx.array, values: mx.array, counts: list[int], ids: mx.array | None,
                   sparse: list[bool], scale: float, *, parts: int = 16, gate: mx.array | None = None) -> mx.array:
    """Each row over its sparse ids or dense prefix, parts merged in order; ``gate``: the projected rows' gate."""

    rows, heads, dims = q.shape
    kv_heads = int(keys.shape[1])
    if ids is None:
        ids = consts.get(("no ids", rows))
        if ids is None:
            ids = consts[("no ids", rows)] = mx.zeros((max(rows, 8), 1), dtype=mx.int32)
    scale_arr = consts.get(("scale", scale))
    if scale_arr is None:
        scale_arr = consts[("scale", scale)] = mx.array([scale], dtype=mx.float32)
    first = kernel("q4_attn_parts", _ATTN_PARTS, ["Q", "Kc", "Vc", "IDS", "NK", "SPARSE", "SCALE"], ["PO", "PM"])
    po, pm = first(inputs=[q, keys, values, ids, ints(counts),
                           ints([int(bool(x)) for x in sparse]), scale_arr],
                   template=[("H", heads), ("KVH", kv_heads), ("D", dims), ("P", parts)],
                   grid=(256 * heads, rows, parts), threadgroup=(256, 1, 1),
                   output_shapes=[(rows, heads, parts, dims), (rows, heads, parts, 2)],
                   output_dtypes=[mx.float32, mx.float32])
    return _merge(po, pm, gate, rows, heads, dims, parts)


def _split_source() -> str:
    """_ATTN_PARTS with key ids from SPLIT[0] on read from the side buffers Ks / Vs (the same arithmetic)."""

    swaps = [
        ("  const device bfloat* vb = Vc + size_t(kvh) * cap * D + lane * PER;\n",
         "  const device bfloat* vb = Vc + size_t(kvh) * cap * D + lane * PER;\n"
         "  const int split = SPLIT[0], held = int(Ks_shape[2]);\n"
         "  const device bfloat* ks = Ks + size_t(kvh) * held * D + lane * PER;\n"
         "  const device bfloat* vs = Vs + size_t(kvh) * held * D + lane * PER;\n"),
        ("    const size_t key = size_t(sparse ? ids[j] : j) * D;\n",
         "    const int id = sparse ? ids[j] : j;\n"
         "    const device bfloat* kp = id >= split ? ks + size_t(id - split) * D : kb + size_t(id) * D;\n"
         "    const device bfloat* vp = id >= split ? vs + size_t(id - split) * D : vb + size_t(id) * D;\n"),
        ("    for (int i = 0; i < PER; i++) sc = fma(q[i], float(kb[key + i]), sc);\n",
         "    for (int i = 0; i < PER; i++) sc = fma(q[i], float(kp[i]), sc);\n"),
        ("    for (int i = 0; i < PER; i++) o[i] = fma(e, float(vb[key + i]), o[i] * f);\n",
         "    for (int i = 0; i < PER; i++) o[i] = fma(e, float(vp[i]), o[i] * f);\n"),
    ]
    src = _ATTN_PARTS
    for old, new in swaps:
        if src.count(old) != 1:
            raise RuntimeError(f"attention source changed; cannot derive the split variant at: {old!r}")
        src = src.replace(old, new)
    return src


def attention_rows_split(q: mx.array, keys: mx.array, values: mx.array, side_keys: mx.array, side_values: mx.array,
                         split: int, counts: list[int], ids: mx.array | None, sparse: list[bool], scale: float, *,
                         parts: int = 16, gate: mx.array | None = None) -> mx.array:
    """``attention_rows`` with keys and values of positions ``split`` on from side buffers [1, KVH, n, D]."""

    rows, heads, dims = q.shape
    if ids is None:
        ids = mx.zeros((max(rows, 8), 1), dtype=mx.int32)
    first = kernel("q4_attn_parts_split", _split_source, ["Q", "Kc", "Vc", "Ks", "Vs", "IDS", "NK", "SPARSE", "SCALE",
                                                          "SPLIT"], ["PO", "PM"])
    po, pm = first(inputs=[q, keys, values, side_keys, side_values, ids, ints(counts),
                           ints([int(bool(x)) for x in sparse]), mx.array([scale], dtype=mx.float32), ints([split])],
                   template=[("H", heads), ("KVH", int(keys.shape[1])), ("D", dims), ("P", parts)],
                   grid=(256 * heads, rows, parts), threadgroup=(256, 1, 1),
                   output_shapes=[(rows, heads, parts, dims), (rows, heads, parts, 2)],
                   output_dtypes=[mx.float32, mx.float32])
    return _merge(po, pm, gate, rows, heads, dims, parts)


def _merge(po: mx.array, pm: mx.array, gate: mx.array | None, rows: int, heads: int, dims: int, parts: int) -> mx.array:
    """The parts merged in order: [R, H, D], or with ``gate`` (the projected rows) gated, [R, H * D]."""

    if gate is None:
        merge = kernel("q4_attn_merge", _ATTN_MERGE, ["PO", "PM"], ["OUT"])
        return merge(inputs=[po, pm], template=[("H", heads), ("D", dims), ("P", parts)],
                     grid=(dims, heads, rows), threadgroup=(dims, 1, 1),
                     output_shapes=[(rows, heads, dims)], output_dtypes=[mx.bfloat16])[0]
    merge = kernel("q4_attn_merge_gate", _ATTN_MERGE_GATE, ["PO", "PM", "GP"], ["OUT"])
    return merge(inputs=[po, pm, gate], template=[("H", heads), ("D", dims), ("P", parts),
                                                  ("PW", int(gate.shape[-1]))],
                 grid=(dims, heads, rows), threadgroup=(dims, 1, 1),
                 output_shapes=[(rows, heads * dims)], output_dtypes=[mx.bfloat16])[0]


def _attn_source(streams: int) -> str:
    src = _ATTN_PARTS
    swaps = [
        ("  const size_t cap = size_t(Kc_shape[2]);\n"
         "  const device bfloat* kb = Kc + size_t(kvh) * cap * D + lane * PER;\n"
         "  const device bfloat* vb = Vc + size_t(kvh) * cap * D + lane * PER;\n",
         "  const int sb = SROW[r];\n"
         "  const size_t cap = size_t(CAPS[sb]);\n"
         f"  const device bfloat* kb = {pick('Kc', streams, 'sb')} + size_t(kvh) * cap * D + lane * PER;\n"
         f"  const device bfloat* vb = {pick('Vc', streams, 'sb')} + size_t(kvh) * cap * D + lane * PER;\n"),
    ]
    for old, new in swaps:
        if src.count(old) != 1:
            raise RuntimeError(f"attention source changed; cannot derive the multi-stream variant at: {old!r}")
        src = src.replace(old, new)
    return src

def attention_rows_multi(q: mx.array, keys: Sequence[mx.array], values: Sequence[mx.array], stream_of_row: Sequence[int],
                         counts: Sequence[int], ids: mx.array | None, sparse: Sequence[bool], scale: float, *,
                         parts: int = 16, gate: mx.array | None = None) -> mx.array:
    """``attention_rows`` with row r reading the cache of its stream, stream_of_row[r]."""

    rows, heads, dims = q.shape
    streams = len(keys)
    if not 1 <= streams <= MAX_STREAMS or len(values) != streams:
        raise ValueError(f"attention_rows_multi: 1-{MAX_STREAMS} streams, keys and values each")
    kv_heads = int(keys[0].shape[1])
    if ids is None:
        ids = mx.zeros((max(rows, 8), 1), dtype=mx.int32)
    names = (["Q"] + [f"Kc{b}" for b in range(streams)] + [f"Vc{b}" for b in range(streams)]
             + ["IDS", "NK", "SPARSE", "SCALE", "SROW", "CAPS"])
    first = kernel(f"q4_attn_parts_multi{streams}", lambda: _attn_source(streams), names, ["PO", "PM"])
    caps = ints([int(k.shape[2]) for k in keys])
    po, pm = first(inputs=[q, *keys, *values, ids, ints(counts),
                           ints([int(bool(x)) for x in sparse]),
                           mx.array([scale], dtype=mx.float32), ints(stream_of_row),
                           caps],
                   template=[("H", heads), ("KVH", kv_heads), ("D", dims), ("P", parts)],
                   grid=(256 * heads, rows, parts), threadgroup=(256, 1, 1),
                   output_shapes=[(rows, heads, parts, dims), (rows, heads, parts, 2)],
                   output_dtypes=[mx.float32, mx.float32])
    return _merge(po, pm, gate, rows, heads, dims, parts)

def _scores_source(streams: int) -> str:
    src = _IDX_SCORES
    load = ("    const device bfloat* pb = POOLED + size_t(b) * DI + lane * PER;\n"
            "    float p[PER];\n"
            "    for (int i = 0; i < PER; i++) p[i] = float(pb[i]);\n")
    rows = ("    for (int r = r0; r < r1; r++) {\n"
            "      const int complete = COMPLETE[r];\n"
            "      if (complete <= TOP || b >= complete) continue;\n")
    swaps = [
        ("  const int nb = int(POOLED_shape[0]), r1 = metal::min(int(Q_shape[0]), r0 + RB);\n",
         "  const int nb = STRIDE[0], r1 = metal::min(int(Q_shape[0]), r0 + RB);\n"),
        (load + rows,                                   # a row reads its own stream's blocks
         rows + "      const int sb = SROW[r];\n"
         f"      const device bfloat* pb = {pick('POOLED', streams, 'sb')} + size_t(b) * DI + lane * PER;\n"
         "      float p[PER];\n"
         "      for (int i = 0; i < PER; i++) p[i] = float(pb[i]);\n"),
    ]
    for old, new in swaps:
        if src.count(old) != 1:
            raise RuntimeError(f"index score source changed; cannot derive the multi-stream variant at: {old!r}")
        src = src.replace(old, new)
    return src

def index_scores_multi(q: mx.array, pooled: Sequence[mx.array], stream_of_row: Sequence[int],
                       complete: Sequence[int], *, top: int) -> mx.array:
    """``index_scores`` with row r scored against its stream's pooled blocks, stream_of_row[r]."""

    rows, heads, dims = q.shape
    streams = len(pooled)
    if not 1 <= streams <= MAX_STREAMS:
        raise ValueError(f"index_select_multi: 1-{MAX_STREAMS} streams")
    nb = max(int(p.shape[0]) for p in pooled)
    bb = score_blocks(nb, rows)
    names = ["Q"] + [f"POOLED{b}" for b in range(streams)] + ["COMPLETE", "SROW", "STRIDE"]
    score = kernel(f"q4_idx_scores_multi{streams}", lambda: _scores_source(streams), names, ["SC"])
    return score(inputs=[q, *pooled, ints(complete), ints(stream_of_row), mx.array([nb], dtype=mx.int32)],
                 template=[("HI", heads), ("DI", dims), ("TOP", top), ("BB", bb), ("RB", SCORE_ROWS)],
                 grid=(-(-nb // (8 * bb)) * 256, -(-rows // SCORE_ROWS), 1), threadgroup=(256, 1, 1),
                 output_shapes=[(rows, nb)], output_dtypes=[mx.float32])[0]


def index_select_multi(q: mx.array, pooled: Sequence[mx.array], stream_of_row: Sequence[int],
                       complete: Sequence[int], ends: Sequence[int], *, top: int) -> mx.array:
    """Score each row against its stream's pooled blocks; rows at or below top complete blocks are skipped."""

    scores = index_scores_multi(q, pooled, stream_of_row, complete, top=top)
    return select_blocks(scores, list(complete), list(ends), top=top)


def warm_decode(*, heads: int, kv_heads: int, dims: int, index_heads: int, index_dims: int, top: int, scale: float,
                width: int, norm: mx.array, eps: mx.array, rotary_dim: int, base: float) -> None:
    """Build the sparse decode kernels' variants once at load: a long request would compile each on first use."""

    outs = []
    for blocks in (4 * top + 4, 4096, 8192, 16384):           # every score_blocks variant past top blocks
        pooled = mx.zeros((blocks, index_dims), dtype=mx.bfloat16)
        for rows in (1, 2):
            q = mx.zeros((rows, index_heads, index_dims), dtype=mx.bfloat16)
            outs.append(index_select(q, pooled, [blocks] * rows, [4 * blocks + 1] * rows, top=top))
    q = mx.zeros((1, heads, dims), dtype=mx.bfloat16)
    keys = mx.zeros((1, kv_heads, 16, dims), dtype=mx.bfloat16)
    side = mx.zeros((1, kv_heads, 1, dims), dtype=mx.bfloat16)
    gate = mx.zeros((1, width), dtype=mx.bfloat16)
    outs.append(attention_rows_split(q, keys, keys, side, side, 8, [9], None, [False], scale, gate=gate))
    raw = mx.zeros((8, index_dims), dtype=mx.bfloat16)
    outs.append(index_pool(raw, 1, 3, norm, eps, rotary_dim=rotary_dim, base=base, relative=True))
    mx.eval(outs)

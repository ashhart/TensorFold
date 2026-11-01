// DSpark's Markov step, compiled after lfm2.metal's text (its includes and constants).

// DSpark's Markov scan, W2 as 8-bit codes (BF16 false) or bf16 as released (true): a threadgroup's best of 512 tokens.
struct lfm2_markov_args { uint stride, rank, vocab, prev_stride; };
constant constexpr uint MARKOV_SPAN = 512;

// Eight bf16 in a uint4 as floats (a bf16 is the high half of its float).
inline void lfm2_unpack8(uint4 v, thread float* f) {
  for (int j = 0; j < 4; ++j) {
    f[2 * j] = as_type<float>(v[j] << 16);
    f[2 * j + 1] = as_type<float>(v[j] & 0xffff0000u);
  }
}
template <bool BF16>
inline void lfm2_markov_scan_t(const device bfloat* logits, const device bfloat* w1, const device uint8_t* w2,
                               const device uint* prev, constant lfm2_markov_args& p, device float* best,
                               device uint* at, const device bfloat* w2s, const device bfloat* w2b, uint2 tg,
                               uint2 grid, uint sg, uint lane, threadgroup float* e, threadgroup float* sv,
                               threadgroup uint* si) {
  const uint s = tg.y, t = sg * 32 + lane, id = prev[size_t(s) * p.prev_stride];
  for (uint i = t; i < p.rank; i += 256) e[i] = id < p.vocab ? float(w1[size_t(id) * p.rank + i]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device bfloat* row = logits + size_t(s) * p.stride;
  float mv = -INFINITY;
  uint mi = 0xffffffffu;
  bool nan = false;
  const uint first = tg.x * MARKOV_SPAN, last = min(first + MARKOV_SPAN, p.vocab);
  for (uint v = first + sg; v < last; v += 8) {
    float d = 0.0f;
    for (uint i = 8 * lane; i < p.rank; i += 256) {
      if (BF16) {
        float f[8];
        lfm2_unpack8(*(const device uint4*)((const device bfloat*)w2 + size_t(v) * p.rank + i), f);
        for (int j = 0; j < 8; ++j) d = fma(f[j], e[i + j], d);
        continue;
      }
      const size_t g = (size_t(v) * p.rank + i) / 64;
      const float sc = w2s[g], bi = w2b[g];
      const uchar4 lo = *(const device uchar4*)(w2 + size_t(v) * p.rank + i);
      const uchar4 hi = *(const device uchar4*)(w2 + size_t(v) * p.rank + i + 4);
      for (int j = 0; j < 4; ++j) d = fma(fma(sc, float(lo[j]), bi), e[i + j], d);
      for (int j = 0; j < 4; ++j) d = fma(fma(sc, float(hi[j]), bi), e[i + 4 + j], d);
    }
    d = simd_sum(d);
    const float sc = float(static_cast<bfloat>(float(row[v]) + float(static_cast<bfloat>(d))));
    nan = nan || isnan(sc);
    if (sc > mv || (mi == 0xffffffffu && sc == mv)) {
      mv = sc;
      mi = v;
    }
  }
  if (lane == 0) {
    sv[sg] = nan ? NAN : mv;
    si[sg] = mi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  float bv = sv[0];
  uint bi = si[0];
  for (uint g = 1; g < 8; ++g) {
    if (isnan(bv) || isnan(sv[g])) bv = NAN;
    else if (sv[g] > bv || (sv[g] == bv && si[g] < bi)) {
      bv = sv[g];
      bi = si[g];
    }
  }
  best[size_t(s) * grid.x + tg.x] = bv;
  at[size_t(s) * grid.x + tg.x] = bi;
}

[[kernel]] void lfm2_markov_scan(const device bfloat* logits [[buffer(0)]], const device bfloat* w1 [[buffer(1)]],
                                 const device uint8_t* w2 [[buffer(2)]], const device uint* prev [[buffer(3)]],
                                 constant lfm2_markov_args& p [[buffer(4)]], device float* best [[buffer(5)]],
                                 device uint* at [[buffer(6)]], const device bfloat* w2s [[buffer(7)]],
                                 const device bfloat* w2b [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 grid [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float e[256];
  threadgroup float sv[8];
  threadgroup uint si[8];
  lfm2_markov_scan_t<false>(logits, w1, w2, prev, p, best, at, w2s, w2b, tg, grid, sg, lane, e, sv, si);
}

// The same scan over the released bf16 W2 [vocab, rank] (buffers 7 and 8 unused).
[[kernel]] void lfm2_markov_scan_bf16(const device bfloat* logits [[buffer(0)]], const device bfloat* w1 [[buffer(1)]],
                                      const device uint8_t* w2 [[buffer(2)]], const device uint* prev [[buffer(3)]],
                                      constant lfm2_markov_args& p [[buffer(4)]], device float* best [[buffer(5)]],
                                      device uint* at [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                      uint2 grid [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float e[256];
  threadgroup float sv[8];
  threadgroup uint si[8];
  lfm2_markov_scan_t<true>(logits, w1, w2, prev, p, best, at, w1, w1, tg, grid, sg, lane, e, sv, si);
}

// DSpark's Markov step pick: the scan's best over a stream's threadgroups (ties to the lowest token), as a Pick.
[[kernel]] void lfm2_markov_pick(const device float* best [[buffer(0)]], const device uint* at [[buffer(1)]],
                                 constant uint& groups [[buffer(2)]], device uint* picks [[buffer(3)]],
                                 uint s [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tv[256];
  threadgroup uint ti[256];
  threadgroup uint tn[256];
  float bv = -INFINITY;
  uint bi = 0xffffffffu, nan = 0;
  for (uint g = t; g < groups; g += 256) {
    const float v = best[size_t(s) * groups + g];
    const uint i = at[size_t(s) * groups + g];
    if (isnan(v)) nan = 1;
    else if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  tv[t] = bv;
  ti[t] = bi;
  tn[t] = nan;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint w = 128; w > 0; w >>= 1) {
    if (t < w) {
      const float v = tv[t + w];
      const uint i = ti[t + w];
      if (v > tv[t] || (v == tv[t] && i < ti[t])) {
        tv[t] = v;
        ti[t] = i;
      }
      tn[t] |= tn[t + w];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) {
    picks[2 * s] = ti[0];
    picks[2 * s + 1] = tn[0];
  }
}

// Held drafts from picks[i * stride] into ids[i]; a nonfinite or past-vocabulary pick is token 0 and sets bad.
[[kernel]] void lfm2_draft_ids(const device uint* picks [[buffer(0)]], constant uint4& p [[buffer(1)]],
                               device uint* ids [[buffer(2)]], device atomic_uint* bad [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= p.y) return;
  const uint token = picks[size_t(i) * p.x];
  const bool ok = token < p.z && picks[size_t(i) * p.x + 1] == 0u;
  ids[i] = ok ? token : 0u;
  if (!ok) atomic_store_explicit(bad, 1u, memory_order_relaxed);
}

// DSpark's confidence logit of draft i of stream s: proj . [x row s * n + i, W1[previous token]] in fp32, no bias.
struct lfm2_confidence_args { uint n, pass, vocab, rank, hidden; };
[[kernel]] void lfm2_dspark_confidence(const device bfloat* x [[buffer(0)]], const device bfloat* w1 [[buffer(1)]],
                                       const device uint* ids [[buffer(2)]], const device uint* picks [[buffer(3)]],
                                       const device bfloat* proj [[buffer(4)]],
                                       constant lfm2_confidence_args& p [[buffer(5)]], device float* out [[buffer(6)]],
                                       uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const uint i = tg.x, s = tg.y;
  // the previous token: the block's pending id for the first draft, else the Markov step's pick before it
  const uint id = i == 0 ? ids[size_t(s) * p.n] : picks[2 * (size_t(i - 1) * p.pass + s)];
  const device bfloat* row = x + (size_t(s) * p.n + i) * p.hidden;
  float d = 0.0f;
  for (uint k = t; k < p.hidden; k += 256) d = fma(float(row[k]), float(proj[k]), d);
  if (id < p.vocab)
    for (uint k = t; k < p.rank; k += 256) d = fma(float(w1[size_t(id) * p.rank + k]), float(proj[p.hidden + k]), d);
  d = simd_sum(d);
  if (lane == 0) part[sg] = d;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  float sum = 0.0f;
  for (uint g = 0; g < 8; ++g) sum += part[g];
  out[size_t(i) * p.pass + s] = sum;
}

// DSpark's bf16 linears (the released matrices as stored): y [rows, n] = x [rows, k] W [n, k]^T, fp32 sums, bf16 out.
struct lfm2_bf16_args { uint rows, n, k; };
constant constexpr uint BF16_ROWS = 16;

// Few rows: a simdgroup a column of W over 16 rows, so W is read once a 16 rows. k a multiple of 8.
[[kernel]] void lfm2_bf16_rows(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                               constant lfm2_bf16_args& p [[buffer(2)]], device bfloat* y [[buffer(3)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint col = tg.x * 8 + sg, r0 = tg.y * BF16_ROWS;
  if (col >= p.n) return;
  const uint rows = min(BF16_ROWS, p.rows - r0);
  float acc[BF16_ROWS];
  for (uint r = 0; r < BF16_ROWS; ++r) acc[r] = 0.0f;
  const device bfloat* wr = w + size_t(col) * p.k;
  const device bfloat* xr = x + size_t(r0) * p.k;
  for (uint i = 8 * lane; i < p.k; i += 256) {
    float wv[8];
    lfm2_unpack8(*(const device uint4*)(wr + i), wv);
    for (uint r = 0; r < BF16_ROWS; ++r) {
      if (r >= rows) break;
      float xv[8];
      lfm2_unpack8(*(const device uint4*)(xr + size_t(r) * p.k + i), xv);
      for (int j = 0; j < 8; ++j) acc[r] = fma(wv[j], xv[j], acc[r]);
    }
  }
  for (uint r = 0; r < BF16_ROWS; ++r) {
    const float sum = simd_sum(acc[r]);
    if (lane == 0 && r < rows) y[size_t(r0 + r) * p.n + col] = static_cast<bfloat>(sum);
  }
}

// Many rows: 64x64 tiles of y, four simdgroups of 32x32, k in steps of 32. k a multiple of 32.
[[kernel]] void lfm2_bf16_mma(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                              constant lfm2_bf16_args& p [[buffer(2)]], device bfloat* y [[buffer(3)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat xs[64 * 32];
  threadgroup bfloat ws[64 * 32];
  threadgroup float out[4][32 * 32];
  const uint c0 = tg.x * 64, r0 = tg.y * 64;
  simdgroup_matrix<float, 8, 8> acc[4][4];
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
  const uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
  for (uint k0 = 0; k0 < p.k; k0 += 32) {
    for (uint i = t; i < 64 * 4; i += 128) { // 64 rows of 32 bf16 as 4 chunks of 8, 16 bytes a load
      const uint r = i / 4, q = (i % 4) * 8;
      const uint4 xv = r0 + r < p.rows ? *(const device uint4*)(x + size_t(r0 + r) * p.k + k0 + q) : uint4(0);
      const uint4 wv = c0 + r < p.n ? *(const device uint4*)(w + size_t(c0 + r) * p.k + k0 + q) : uint4(0);
      *(threadgroup uint4*)(xs + r * 32 + q) = xv;
      *(threadgroup uint4*)(ws + r * 32 + q) = wv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32; kk += 8) {
      simdgroup_matrix<bfloat, 8, 8> a[4], b[4];
      for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], xs + (sr + 8 * i) * 32 + kk, 32);
      for (uint j = 0; j < 4; ++j) simdgroup_load(b[j], ws + (sc + 8 * j) * 32 + kk, 32, ulong2(0, 0), true);
      for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], out[sg] + 8 * i * 32 + 8 * j, 32);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = t % 32; i < 32 * 32; i += 32) {
    const uint r = i / 32, c = i % 32;
    const uint row = r0 + sr + r, col = c0 + sc + c;
    if (row < p.rows && col < p.n) y[size_t(row) * p.n + col] = static_cast<bfloat>(out[sg][i]);
  }
}

// A drafter block's attention: lfm2_attn_mma where every row sees all keys, its 128 threads reserved for M1 and M2.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void lfm2_attn_block(
    const device bfloat* queries [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
    const device bfloat* values [[buffer(2)]], constant lfm2_attn_args& p [[buffer(3)]], device float* pm [[buffer(4)]],
    device float* pl [[buffer(5)]], device float* po [[buffer(6)]], constant uint& rows [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint ks[2][HD * (KT / 2)];
  threadgroup bfloat vs[2][KB * KP];
  lfm2_attn_rows<1, true>(queries, keys, values, p, pm, pl, po, rows, tg, tid, sg, lane, ks, vs);
}

[[kernel]] void lfm2_norm_tap(device bfloat* h [[buffer(0)]], const device bfloat* r [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                          constant float& eps [[buffer(3)]], constant uint& add [[buffer(4)]], device bfloat* x [[buffer(5)]], device bfloat* taps [[buffer(6)]], constant uint2& tap_at [[buffer(7)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * D + 4 * t;
  float v[4];
  bfloat4 saved;
  for (int j = 0; j < 4; ++j) {
    bfloat hv = h[base + j];
    if (add != 0) {
      hv = hv + r[base + j];
      h[base + j] = hv;
    }
    v[j] = hv;
    saved[j] = hv;
  }
  *(device bfloat4*)(taps + size_t(row) * tap_at.x + tap_at.y + 4 * t) = saved;
  threadgroup float sums[32], inv[1];
  lfm2_rms(v, w, eps, x, base, t, sg, lane, sums, inv);
}

[[kernel]] void lfm2_combine_norm_tap(const device bfloat* y [[buffer(0)]], const device bfloat* wts [[buffer(1)]], device bfloat* h [[buffer(2)]],
                                  const device bfloat* w [[buffer(3)]], constant float& eps [[buffer(4)]], device bfloat* x [[buffer(5)]], device bfloat* taps [[buffer(6)]], constant uint2& tap_at [[buffer(7)]],
                                  uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * D + 4 * t;
  float v[4];
  bfloat4 saved;
  for (int j = 0; j < 4; ++j) {
    float acc = 0.0f;
    for (uint k = 0; k < TOPK; ++k) {
      const size_t slot = size_t(row) * TOPK + k;
      acc += float(static_cast<bfloat>(float(y[slot * D + 4 * t + j]) * float(wts[slot])));
    }
    const bfloat hv = h[base + j] + static_cast<bfloat>(acc);
    h[base + j] = hv;
    v[j] = hv;
    saved[j] = hv;
  }
  *(device bfloat4*)(taps + size_t(row) * tap_at.x + tap_at.y + 4 * t) = saved;
  threadgroup float sums[32], inv[1];
  lfm2_rms(v, w, eps, x, base, t, sg, lane, sums, inv);
}

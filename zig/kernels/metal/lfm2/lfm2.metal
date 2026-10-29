// LFM2-MoE's kernels beside the core row projections: every row's arithmetic is its own, whatever the rows beside it.
#include <metal_stdlib>
using namespace metal;

// The hidden width, heads, head width, experts and picks (the config's pinned geometry).
constant constexpr uint D = 2048;
constant constexpr uint QH = 32;
constant constexpr uint KVH = 8;
constant constexpr uint HD = 64;
constant constexpr uint E = 64;
constant constexpr uint TOPK = 4;
// Attention: key chunks a row splits into at most.
constant constexpr uint CHUNKS = 32;

// Table row ids[r] as bf16 scale * q + bias (MLX's dequantize), one packed unit (1 byte, 3 at 6 bits) a thread.
template <int BITS>
inline void lfm2_unit(const device uint8_t* table, const device bfloat* scales, const device bfloat* biases, long id, int unit,
                      device bfloat* out) {
  constexpr int PER = BITS == 6 ? 4 : 8 / BITS;
  const int first = unit * PER;
  const bfloat scale = scales[id * (D / 64) + first / 64];
  const bfloat bias = biases[id * (D / 64) + first / 64];
  const device uint8_t* w = table + id * (long(D) * BITS / 8) + long(unit) * (BITS == 6 ? 3 : 1);
  if (BITS == 6) {
    out[0] = (w[0] & 0x3f) * scale + bias;
    out[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
    out[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
    out[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
  } else {
    const uint byte = w[0];
    for (int i = 0; i < PER; ++i) {
      const uint8_t q = BITS == 8 ? uint8_t(byte) : uint8_t((byte >> (BITS * i)) & ((1u << BITS) - 1u));
      out[i] = scale * q + bias;
    }
  }
}

template <int BITS>
[[kernel]] void lfm2_embed(const device uint* ids [[buffer(0)]], const device uint8_t* table [[buffer(1)]],
                           const device bfloat* scales [[buffer(2)]], const device bfloat* biases [[buffer(3)]],
                           device bfloat* out [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  constexpr int PER = BITS == 6 ? 4 : 8 / BITS;
  if (at.x < D / PER) lfm2_unit<BITS>(table, scales, biases, long(ids[at.y]), int(at.x), out + size_t(at.y) * D + PER * at.x);
}
template [[host_name("lfm2_embed_b4")]] [[kernel]] decltype(lfm2_embed<4>) lfm2_embed<4>;
template [[host_name("lfm2_embed_b6")]] [[kernel]] decltype(lfm2_embed<6>) lfm2_embed<6>;
template [[host_name("lfm2_embed_b8")]] [[kernel]] decltype(lfm2_embed<8>) lfm2_embed<8>;

// x = w * bf16(h * rsqrt(mean(h^2) + eps)) over a row's 4 values a thread (v): MLX's one-row RMS order.
inline void lfm2_rms(const thread float* v, const device bfloat* w, float eps, device bfloat* x, size_t base, uint t, uint sg,
                     uint lane, threadgroup float* sums, threadgroup float* inv) {
  float sq = 0.0f;
  for (int j = 0; j < 4; ++j) sq += v[j] * v[j];
  sq = simd_sum(sq);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(sums[lane]);
    if (lane == 0) inv[0] = metal::precise::rsqrt(all / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = 0; j < 4; ++j) x[base + j] = w[4 * t + j] * static_cast<bfloat>(v[j] * inv[0]);
}

// h += r (bf16) when `add`, then x = norm(h), 4 values a thread.
[[kernel]] void lfm2_norm(device bfloat* h [[buffer(0)]], const device bfloat* r [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                          constant float& eps [[buffer(3)]], constant uint& add [[buffer(4)]], device bfloat* x [[buffer(5)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * D + 4 * t;
  float v[4];
  for (int j = 0; j < 4; ++j) {
    bfloat hv = h[base + j];
    if (add != 0) {
      hv = hv + r[base + j];
      h[base + j] = hv;
    }
    v[j] = hv;
  }
  threadgroup float sums[32], inv[1];
  lfm2_rms(v, w, eps, x, base, t, sg, lane, sums, inv);
}

// lfm2_combine_sorted's sum (Y in slot order) added to h as lfm2_norm's r, then x = norm(h): the same bits.
[[kernel]] void lfm2_combine_norm(const device bfloat* y [[buffer(0)]], const device bfloat* wts [[buffer(1)]], device bfloat* h [[buffer(2)]],
                                  const device bfloat* w [[buffer(3)]], constant float& eps [[buffer(4)]], device bfloat* x [[buffer(5)]],
                                  uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * D + 4 * t;
  float v[4];
  for (int j = 0; j < 4; ++j) {
    float acc = 0.0f;
    for (uint k = 0; k < TOPK; ++k) {
      const size_t slot = size_t(row) * TOPK + k;
      acc += float(static_cast<bfloat>(float(y[slot * D + 4 * t + j]) * float(wts[slot])));
    }
    const bfloat hv = h[base + j] + static_cast<bfloat>(acc);
    h[base + j] = hv;
    v[j] = hv;
  }
  threadgroup float sums[32], inv[1];
  lfm2_rms(v, w, eps, x, base, t, sg, lane, sums, inv);
}

// A segment's short conv: Bx = B * x, 3 taps over (state [2, D], Bx), y = C * conv; SNAP: rows' states from FIRST on.
inline void lfm2_conv_row(const device bfloat* bcx, const device bfloat* state, const device bfloat* w, device bfloat* y,
                          device bfloat* snap, uint first, uint ch, uint r) {
  bfloat bx[3];
  for (int tap = 0; tap < 3; ++tap) {
    const int src = int(r) + tap - 2;
    if (src >= 0) {
      const device bfloat* row = bcx + size_t(src) * 3 * D;
      bx[tap] = row[ch] * row[2 * D + ch];
    } else {
      bx[tap] = state[size_t(src + 2) * D + ch];
    }
  }
  float acc = 0.0f;
  for (int tap = 0; tap < 3; ++tap) acc += float(bx[tap]) * float(w[ch * 3 + tap]);
  y[size_t(r) * D + ch] = bcx[size_t(r) * 3 * D + D + ch] * bfloat(acc);
  if (r < first) return;
  snap[(size_t(r - first) * 2) * D + ch] = bx[1];
  snap[(size_t(r - first) * 2 + 1) * D + ch] = bx[2];
}
[[kernel]] void lfm2_conv(const device bfloat* bcx [[buffer(0)]], const device bfloat* state [[buffer(1)]],
                          const device bfloat* w [[buffer(2)]], device bfloat* y [[buffer(3)]], device bfloat* snap [[buffer(4)]],
                          constant uint& first [[buffer(5)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x < D) lfm2_conv_row(bcx, state, w, y, snap, first, at.x, at.y);
}

// One head of one row: RMS norm over its 64 values (lane l: l and l + 32), then RoPE on the pair at position `pos`.
inline void lfm2_head(const device bfloat* src, const device bfloat* gain, float eps, float log2_theta, uint pos, uint lane,
                      thread bfloat& a, thread bfloat& b) {
  const float x0 = src[lane], x1 = src[lane + 32];
  const float inv = metal::precise::rsqrt(simd_sum(x0 * x0 + x1 * x1) / float(HD) + eps);
  const float n0 = float(gain[lane] * static_cast<bfloat>(x0 * inv));
  const float n1 = float(gain[lane + 32] * static_cast<bfloat>(x1 * inv));
  const float theta = float(pos) * metal::exp2(-(float(lane) / float(HD / 2)) * log2_theta);
  const float c = metal::precise::cos(theta), s = metal::precise::sin(theta);
  a = static_cast<bfloat>(n0 * c - n1 * s);
  b = static_cast<bfloat>(n0 * s + n1 * c);
}

// A simdgroup a (head, row): queries normed and rotated, keys and values into the [KVH, cap, HD] cache at len + r.
struct lfm2_qk_args { uint len, cap; float eps, log2_theta; };
inline void lfm2_qk_row(const device bfloat* q, const device bfloat* k, const device bfloat* v, const device bfloat* qn,
                        const device bfloat* kn, lfm2_qk_args p, device bfloat* queries, device bfloat* keys,
                        device bfloat* values, uint head, uint r, uint lane) {
  const uint pos = p.len + r;
  bfloat a, b;
  if (head < QH) {
    lfm2_head(q + (size_t(r) * QH + head) * HD, qn, p.eps, p.log2_theta, pos, lane, a, b);
    device bfloat* out = queries + (size_t(r) * QH + head) * HD;
    out[lane] = a;
    out[lane + 32] = b;
    return;
  }
  const uint kh = head - QH;
  lfm2_head(k + (size_t(r) * KVH + kh) * HD, kn, p.eps, p.log2_theta, pos, lane, a, b);
  const size_t at = (size_t(kh) * p.cap + pos) * HD;
  keys[at + lane] = a;
  keys[at + lane + 32] = b;
  values[at + lane] = v[(size_t(r) * KVH + kh) * HD + lane];
  values[at + lane + 32] = v[(size_t(r) * KVH + kh) * HD + lane + 32];
}
[[kernel]] void lfm2_qk(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
                        const device bfloat* qn [[buffer(3)]], const device bfloat* kn [[buffer(4)]], constant lfm2_qk_args& p [[buffer(5)]],
                        device bfloat* queries [[buffer(6)]], device bfloat* keys [[buffer(7)]], device bfloat* values [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  lfm2_qk_row(q, k, v, qn, kn, p, queries, keys, values, tg.x, tg.y, lane);
}

// Keys a chunk: a row's keys split at fixed positions, so prompt tiles and decode rows cut them the same way.
constant constexpr uint PER = 4096;
// Keys a sub-span: a chunk's state is the in-order merge of its sub-spans' (tiles fold them, decode rows run one each).
constant constexpr uint SUB = 512;
constant constexpr uint SUBS = PER / SUB;
struct lfm2_attn_args { uint len, cap; float scale; uint subs; };

// (M, L, O) merged with a sub-span's (m, l, o): O's scale `alpha` and o's `beta` for the caller's elements.
inline void lfm2_merge(thread float& M, thread float& L, float m, float l, thread float& alpha, thread float& beta) {
  if (m == -INFINITY) {
    alpha = 1.0f;
    beta = 0.0f;
    return;
  }
  const float M2 = max(M, m);
  alpha = M == -INFINITY ? 0.0f : metal::fast::exp(M - M2);
  beta = metal::fast::exp(m - M2);
  L = fma(L, alpha, l * beta);
  M = M2;
}

// One 8x8x8 MMA on per-lane fragments (the lane's (row fm, columns fn, fn + 1) of each operand), fp32 accumulate.
inline float2 lfm2_mma(float2 c, vec<bfloat, 2> a, vec<bfloat, 2> b) {
  simdgroup_matrix<float, 8, 8> C;
  simdgroup_matrix<bfloat, 8, 8> A, B;
  reinterpret_cast<thread float2&>(C.thread_elements()) = c;
  reinterpret_cast<thread vec<bfloat, 2>&>(A.thread_elements()) = a;
  reinterpret_cast<thread vec<bfloat, 2>&>(B.thread_elements()) = b;
  simdgroup_multiply_accumulate(C, A, B, C);
  return reinterpret_cast<thread float2&>(C.thread_elements());
}

constant constexpr uint KB = 32;   // keys a staged block
constant constexpr uint KP = 72;   // bf16 a staged value row (64 and pad)
constant constexpr uint KT = 36;   // keys a staged transposed K row (32 and pad)
// A thread's share of a 32-key block: two 8-wide parts of V, the same part of two neighbouring keys of K.
inline void lfm2_fetch(const device bfloat* kb, const device bfloat* vb, uint j0, uint cap, uint tid, thread uint4* pre) {
  for (uint i = 0; i < 2; ++i) {
    const uint idx = tid + 128 * i, k = idx / 8, part = idx % 8;
    pre[i] = *(const device uint4*)(vb + size_t(min(j0 + k, cap - 1)) * HD + part * 8);
  }
  const uint kp = tid % (KB / 2), part = tid / (KB / 2);
  pre[2] = *(const device uint4*)(kb + size_t(min(j0 + 2 * kp, cap - 1)) * HD + part * 8);
  pre[3] = *(const device uint4*)(kb + size_t(min(j0 + 2 * kp + 1, cap - 1)) * HD + part * 8);
}

// The fetched share staged: V by rows, K transposed (the thread's 8 (d, two keys) pairs).
inline void lfm2_stage(const thread uint4* pre, threadgroup uint* ks, threadgroup bfloat* vs, uint tid) {
  for (uint i = 0; i < 2; ++i) {
    const uint idx = tid + 128 * i, k = idx / 8, part = idx % 8;
    *(threadgroup uint4*)(vs + k * KP + part * 8) = pre[i];
  }
  const uint kp = tid % (KB / 2), part = tid / (KB / 2);
  for (uint d = 0; d < 8; ++d) {
    const uint lo = (pre[2][d / 2] >> (16 * (d % 2))) & 0xffffu, hi = (pre[3][d / 2] >> (16 * (d % 2))) & 0xffffu;
    ks[(part * 8 + d) * (KT / 2) + kp] = lo | (hi << 16);
  }
}

// Every row's attention partials (one kernel for prompt and decode): a threadgroup a (chunk, kv head, 8 * RB rows).
template <int RB>  // a simdgroup's RB row blocks share each staged K and V fragment, each block's arithmetic its own
inline void lfm2_attn_rows(const device bfloat* queries, const device bfloat* keys, const device bfloat* values,
                           lfm2_attn_args p, device float* pm, device float* pl, device float* po, uint rows, uint3 tg,
                           uint tid, uint sg, uint lane, threadgroup uint (*ks)[HD * (KT / 2)],
                           threadgroup bfloat (*vs)[KB * KP]) {
  // Row tiles first in the grid: the threadgroups in flight walk the same keys together (one chunk, one kv head).
  const uint chunk = tg.z / p.subs, sub0 = p.subs == 1 ? 0 : tg.z % SUBS, sub1 = p.subs == 1 ? SUBS : sub0 + 1;
  const uint kh = tg.y, r0 = tg.x * 8 * RB, head = kh * (QH / KVH) + sg;
  const uint qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
  uint count[RB];  // this lane's rows' visible keys
  for (int b = 0; b < RB; ++b) count[b] = p.len + min(r0 + 8 * b + fm, rows - 1) + 1;
  const uint top = p.len + min(r0 + 8 * RB - 1, rows - 1) + 1;  // the tile's most
  if (chunk * PER + sub0 * SUB >= top) return;
  vec<bfloat, 2> q[RB][8];
  for (int b = 0; b < RB; ++b)
    for (int d = 0; d < 8; ++d) q[b][d] = *(const device vec<bfloat, 2>*)(queries + (size_t(r0 + 8 * b + fm) * QH + head) * HD + 8 * d + fn);
  const device bfloat* kb = keys + size_t(kh) * p.cap * HD;
  const device bfloat* vb = values + size_t(kh) * p.cap * HD;
  float M[RB], L[RB];
  float2 O[RB][8];
  for (int b = 0; b < RB; ++b) {
    M[b] = -INFINITY;
    L[b] = 0.0f;
    for (int d = 0; d < 8; ++d) O[b][d] = float2(0.0f);
  }
  for (uint sub = sub0; sub < sub1; ++sub) {
    const uint first = chunk * PER + sub * SUB, last = min(top, first + SUB);
    float2 o[RB][8];
    float m[RB], l[RB];
    for (int b = 0; b < RB; ++b) {
      m[b] = -INFINITY;
      l[b] = 0.0f;
      for (int d = 0; d < 8; ++d) o[b][d] = float2(0.0f);
    }
    // Two staging buffers: the next block's loads fly over this one's math, then land in the other buffer.
    uint4 pre[4];
    lfm2_fetch(kb, vb, first, p.cap, tid, pre);
    lfm2_stage(pre, ks[0], vs[0], tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint j0 = first; j0 < last; j0 += KB, buf ^= 1) {
      const bool more = j0 + KB < last;
      if (more) lfm2_fetch(kb, vb, j0 + KB, p.cap, tid, pre);
      const threadgroup uint* ksb = ks[buf];
      const threadgroup bfloat* vsb = vs[buf];
      float2 sm[RB][4];
      for (int b = 0; b < RB; ++b)
        for (uint t = 0; t < 4; ++t) sm[b][t] = float2(0.0f);
      for (uint t = 0; t < 4; ++t)
        for (int d = 0; d < 8; ++d) {
          const vec<bfloat, 2> kt = as_type<vec<bfloat, 2>>(ksb[(8 * d + fm) * (KT / 2) + (8 * t + fn) / 2]);
          for (int b = 0; b < RB; ++b) sm[b][t] = lfm2_mma(sm[b][t], q[b][d], kt);
        }
      vec<bfloat, 2> pb[RB][4];
      float alpha[RB];
      for (int b = 0; b < RB; ++b) {
        float sv[4][2];
        float bm = -INFINITY;
        for (uint t = 0; t < 4; ++t)
          for (uint e = 0; e < 2; ++e) {
            const float v = sm[b][t][e] * p.scale;
            sv[t][e] = j0 + 8 * t + fn + e < count[b] ? v : -INFINITY;
            bm = max(bm, sv[t][e]);
          }
        bm = max(bm, simd_shuffle_xor(bm, ushort(1)));
        bm = max(bm, simd_shuffle_xor(bm, ushort(8)));
        const bool live = bm != -INFINITY;  // per row: a block with no visible key leaves its state as it is
        const float m2 = live ? max(m[b], bm) : m[b];
        float bsum[4];
        for (uint t = 0; t < 4; ++t) {
          const float p0 = live ? metal::fast::exp(sv[t][0] - m2) : 0.0f, p1 = live ? metal::fast::exp(sv[t][1] - m2) : 0.0f;
          float bs = p0 + p1;
          bs = bs + simd_shuffle_xor(bs, ushort(1));
          bs = bs + simd_shuffle_xor(bs, ushort(8));
          bsum[t] = bs;
          pb[b][t] = vec<bfloat, 2>(bfloat(p0), bfloat(p1));
        }
        alpha[b] = live ? metal::fast::exp(m[b] - m2) : 1.0f;
        l[b] = fma(l[b], alpha[b], (bsum[0] + bsum[1]) + (bsum[2] + bsum[3]));
        m[b] = m2;
      }
      for (int d = 0; d < 8; ++d) {
        for (int b = 0; b < RB; ++b) o[b][d] *= alpha[b];
        for (uint t = 0; t < 4; ++t) {
          const vec<bfloat, 2> v = *(const threadgroup vec<bfloat, 2>*)(vsb + (8 * t + fm) * KP + 8 * d + fn);
          for (int b = 0; b < RB; ++b) o[b][d] = lfm2_mma(o[b][d], pb[b][t], v);
        }
      }
      if (more) lfm2_stage(pre, ks[buf ^ 1], vs[buf ^ 1], tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int b = 0; b < RB; ++b) {
      float alpha, beta;
      lfm2_merge(M[b], L[b], m[b], l[b], alpha, beta);
      for (int d = 0; d < 8; ++d) O[b][d] = fma(O[b][d], alpha, o[b][d] * beta);
    }
  }
  for (int b = 0; b < RB; ++b) {
    const uint row = r0 + 8 * b + fm;
    if (row >= rows) return;
    const size_t at = (size_t(row) * QH + head) * (CHUNKS * p.subs) + chunk * p.subs + sub0;
    if ((lane & 9) == 0) {
      pm[at] = M[b];
      pl[at] = L[b];
    }
    for (int d = 0; d < 8; ++d) *(device float2*)(po + at * HD + 8 * d + fn) = O[b][d];
  }
}
template <int RB>
[[kernel]] void lfm2_attn_mma(const device bfloat* queries [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                              const device bfloat* values [[buffer(2)]], constant lfm2_attn_args& p [[buffer(3)]],
                              device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], device float* po [[buffer(6)]],
                              constant uint& rows [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint ks[2][HD * (KT / 2)];
  threadgroup bfloat vs[2][KB * KP];
  lfm2_attn_rows<RB>(queries, keys, values, p, pm, pl, po, rows, tg, tid, sg, lane, ks, vs);
}
template [[host_name("lfm2_attn_mma")]] [[kernel]] decltype(lfm2_attn_mma<1>) lfm2_attn_mma<1>;
template [[host_name("lfm2_attn_mma_r2")]] [[kernel]] decltype(lfm2_attn_mma<2>) lfm2_attn_mma<2>;

// The chunks merged in chunk order for (head, row): Y [rows, QH*HD] bf16.
inline void lfm2_merge_row(const device float* pm, const device float* pl, const device float* po, lfm2_attn_args p,
                           device bfloat* y, uint head, uint r, uint lane) {
  const uint count = p.len + r + 1;
  const uint per = PER;
  const uint chunks = (count + per - 1) / per;
  const size_t at = (size_t(r) * QH + head) * (CHUNKS * p.subs);
  // Each chunk's state: its sub-spans' partials folded in order (one a chunk from the prompt tiles).
  float cm[CHUNKS], cl[CHUNKS], c0[CHUNKS], c1[CHUNKS];
  for (uint c = 0; c < chunks; ++c) {
    const uint subs = min(p.subs, (count - c * per + SUB - 1) / SUB);
    float M = -INFINITY, L = 0.0f, O0 = 0.0f, O1 = 0.0f;
    for (uint s = 0; s < subs; ++s) {
      const size_t i = at + c * p.subs + s;
      float alpha, beta;
      lfm2_merge(M, L, pm[i], pl[i], alpha, beta);
      O0 = fma(O0, alpha, po[i * HD + lane] * beta);
      O1 = fma(O1, alpha, po[i * HD + lane + 32] * beta);
    }
    cm[c] = M;
    cl[c] = L;
    c0[c] = O0;
    c1[c] = O1;
  }
  float m = -INFINITY;
  for (uint c = 0; c < chunks; ++c) m = max(m, cm[c]);
  float l = 0.0f, o0 = 0.0f, o1 = 0.0f;
  for (uint c = 0; c < chunks; ++c) {
    const float f = metal::fast::exp(cm[c] - m);
    l += cl[c] * f;
    o0 += c0[c] * f;
    o1 += c1[c] * f;
  }
  device bfloat* out = y + (size_t(r) * QH + head) * HD;
  out[lane] = static_cast<bfloat>(o0 / l);
  out[lane + 32] = static_cast<bfloat>(o1 / l);
}
[[kernel]] void lfm2_attn_merge(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                                const device float* po [[buffer(2)]], constant lfm2_attn_args& p [[buffer(3)]],
                                device bfloat* y [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  lfm2_merge_row(pm, pl, po, p, y, tg.x, tg.y, lane);
}

// Router logits [rows, E] in fp32: a simdgroup an expert, lane l over inputs 4l..4l+3 of each 128, then simd_sum.
[[kernel]] void lfm2_route_logits(const device bfloat* x [[buffer(0)]], const device float* router [[buffer(1)]],
                                  device float* logits [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint e = tg.x * 8 + sg, row = tg.y;
  const device bfloat4* xr = (const device bfloat4*)(x + size_t(row) * D) + lane;
  const device float4* wr = (const device float4*)(router + size_t(e) * D) + lane;
  float acc = 0.0f;
  for (uint i = 0; i < D / 128; ++i) {
    const float4 xv = float4(xr[i * 32]), wv = wr[i * 32];
    acc += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
  }
  acc = simd_sum(acc);
  if (lane == 0) logits[size_t(row) * E + e] = acc;
}

// A row's picks, a simdgroup a row: top-k of sigmoid + bias (lower id on ties), sigmoids / (sum + 1e-6) * scale.
[[kernel]] void lfm2_route_select(const device float* logits [[buffer(0)]], const device float* bias [[buffer(1)]],
                                  constant float& scale [[buffer(2)]], device uint* ids [[buffer(3)]], device bfloat* wts [[buffer(4)]],
                                  uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float gate[2], score[2];
  for (uint h = 0; h < 2; ++h) {
    const uint e = lane + 32 * h;
    gate[h] = 1.0f / (1.0f + metal::precise::exp(-logits[size_t(row) * E + e]));
    score[h] = gate[h] + bias[e];
  }
  uint pick[TOPK];
  float picked[TOPK];
  float total = 0.0f;
  for (uint k = 0; k < TOPK; ++k) {
    const uint mine = score[1] > score[0] ? 1 : 0;
    float best = score[mine];
    uint at = lane + 32 * mine;
    for (uint off = 16; off > 0; off >>= 1) {
      const float b = simd_shuffle_xor(best, off);
      const uint i = simd_shuffle_xor(at, off);
      if (b > best || (b == best && i < at)) {
        best = b;
        at = i;
      }
    }
    pick[k] = at;
    picked[k] = simd_shuffle(gate[at / 32], at % 32);
    total += picked[k];
    if (at % 32 == lane) score[at / 32] = -INFINITY;
  }
  if (lane < TOPK) {
    ids[row * TOPK + lane] = pick[lane];
    wts[row * TOPK + lane] = static_cast<bfloat>(picked[lane] / (total + 1e-6f) * scale);
  }
}

// act = silu(gate) * up, 8 a thread, bf16 at each step (mlx-lm's compiled swiglu, as the tiles' twin).
[[kernel]] void lfm2_swiglu(const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]],
                            device bfloat* act [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i * 8 >= count) return;
  const vec<bfloat, 8> g = ((const device vec<bfloat, 8>*)gate)[i], u = ((const device vec<bfloat, 8>*)up)[i];
  vec<bfloat, 8> out;
  for (int j = 0; j < 8; ++j) {
    const float gf = g[j];
    const bfloat s = static_cast<bfloat>(1.0f / (1.0f + metal::precise::exp(-gf)));
    const bfloat silu = static_cast<bfloat>(gf * float(s));
    out[j] = static_cast<bfloat>(float(silu) * float(u[j]));
  }
  ((device vec<bfloat, 8>*)act)[i] = out;
}

// Picks sorted by expert, in slot order within one: STARTS [E + 1], ORDER (sorted -> slot), POS (slot -> sorted).
constant constexpr uint SORT_CHUNKS = 16;  // thread (e, c) counts expert e's picks in chunk c, then places them
[[kernel]] void lfm2_sort(const device uint* ids [[buffer(0)]], constant uint& slots [[buffer(1)]], device uint* starts [[buffer(2)]],
                          device uint* order [[buffer(3)]], device uint* pos [[buffer(4)]], device uint* rows [[buffer(5)]],
                          uint t [[thread_position_in_threadgroup]]) {
  threadgroup uint counts[E * SORT_CHUNKS];
  threadgroup uint totals[E];
  // ROWS: sorted -> the slot's token row, the grouped tiles' input map.
  const uint e = t / SORT_CHUNKS, c = t % SORT_CHUNKS;
  const uint span = (slots + SORT_CHUNKS - 1) / SORT_CHUNKS, lo = c * span, hi = min(lo + span, slots);
  uint n = 0;
  for (uint j = lo; j < hi; ++j) n += ids[j] == e ? 1 : 0;
  counts[t] = n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (c == 0) {
    uint total = 0;
    for (uint i = 0; i < SORT_CHUNKS; ++i) total += counts[e * SORT_CHUNKS + i];
    totals[e] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint at = 0;
  for (uint i = 0; i < e; ++i) at += totals[i];
  if (c == 0) starts[e] = at;
  if (t == E * SORT_CHUNKS - 1) starts[E] = at + totals[e];
  for (uint i = 0; i < c; ++i) at += counts[e * SORT_CHUNKS + i];
  for (uint j = lo; j < hi; ++j)
    if (ids[j] == e) {
      order[at] = j;
      rows[at] = j / TOPK;
      pos[j] = at;
      ++at;
    }
}

// R [rows, D] = the fp32 sum of bf16(y * weight) over a row's picks in pick order, Y's row for slot s at POS[s].
[[kernel]] void lfm2_combine_sorted(const device bfloat* y [[buffer(0)]], const device bfloat* wts [[buffer(1)]],
                                    device bfloat* r [[buffer(2)]], const device uint* pos [[buffer(3)]],
                                    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  float acc = 0.0f;
  for (uint k = 0; k < TOPK; ++k) {
    const size_t slot = size_t(at.y) * TOPK + k;
    acc += float(static_cast<bfloat>(float(y[size_t(pos[slot]) * D + at.x]) * float(wts[slot])));
  }
  r[size_t(at.y) * D + at.x] = static_cast<bfloat>(acc);
}

// Row r's greedy pick over `vocab` bf16 logits: the lowest id of the largest, then 1 if any logit is not finite.
[[kernel]] void lfm2_argmax(const device ushort* logits [[buffer(0)]], constant uint& vocab [[buffer(1)]],
                            device uint* best [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tv[32];
  threadgroup uint ti[32], tn[32];
  const device ushort* x = logits + size_t(row) * vocab;
  float bv = -INFINITY;
  uint bi = 0xffffffffu, bad = 0;
  // Eight logits a load, ids rising along a thread: a strictly larger value moves the pick (vocab % 8 == 0).
  for (uint v = 8 * t; v < vocab; v += 8 * 1024) {
    const uint4 w = *(const device uint4*)(x + v);
    const uint words[4] = {w.x, w.y, w.z, w.w};
    for (uint j = 0; j < 8; ++j) {
      const float f = as_type<float>((j % 2) ? (words[j / 2] & 0xFFFF0000u) : (words[j / 2] << 16));
      if (!isfinite(f)) bad = 1;
      else if (f > bv) {
        bv = f;
        bi = v + j;
      }
    }
  }
  for (uint off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_xor(bv, off);
    const uint oi = simd_shuffle_xor(bi, off);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  bad = simd_max(bad);
  if (lane == 0) {
    tv[sg] = bv;
    ti[sg] = bi;
    tn[sg] = bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  bv = tv[lane];
  bi = ti[lane];
  bad = tn[lane];
  for (uint off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_xor(bv, off);
    const uint oi = simd_shuffle_xor(bi, off);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  bad = simd_max(bad);
  if (lane == 0) {
    best[2 * row] = bi;
    best[2 * row + 1] = bad;
  }
}

#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;

// sum_i w_i x_i over one 32-value group: w = scale * q + bias
inline float qgroup_dot(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(float((bits >> (4 * n)) & 0xFu), xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
}
// Elementwise ops as the checkpoint's training framework does them on bf16 tensors: fp32 math, one rounding.
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float bsilu(float x) { return float(bfloat(x / (1.0f + metal::exp(-x)))); }
inline float fsig(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }


// Rank-k expert of a row inside one simdgroup: lane l holds logits l, l + 32, ...; rounds of (largest logit,
// lowest id); returns the id picked in round k and, through ``picked``, the logits of rounds 0..k.
template <int NE>
inline int simd_topk(const device float* logits, int k, uint lane, thread float* picked) {
  float v[NE / 32];
  for (int j = 0; j < NE / 32; j++) v[j] = logits[j * 32 + int(lane)];
  int id = 0;
  for (int round = 0; round <= k; round++) {
    float best = -INFINITY;
    int bid = NE;
    for (int j = 0; j < NE / 32; j++) {
      const int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    id = bid;
    if (int(lane) == bid % 32) v[bid / 32] = -INFINITY;
  }
  return id;
}
// simd_topk's rounds 0 .. TOPK-1 in one pass: ids[k] and logits picked[k] of each round
template <int NE, int TOPK>
inline void simd_topk_all(const device float* logits, uint lane, thread int* ids, thread float* picked) {
  float v[NE / 32];
  for (int j = 0; j < NE / 32; j++) v[j] = logits[j * 32 + int(lane)];
  for (int round = 0; round < TOPK; round++) {
    float best = -INFINITY;
    int bid = NE;
    for (int j = 0; j < NE / 32; j++) {
      const int e = j * 32 + int(lane);
      if (v[j] > best || (v[j] == best && e < bid)) { best = v[j]; bid = e; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
    }
    picked[round] = best;
    ids[round] = bid;
    if (int(lane) == bid % 32) v[bid / 32] = -INFINITY;
  }
}
// MLX's 4-bit qmv inner loop (quantized.h): 16 inputs a lane, pre-divided by 1, 16, 256, 4096 so the masked
// nibbles need no shift; w = scale * q + bias gives scale * dot(q, x) + bias * sum(x). As in MLX, each run of 4
// inputs is summed in bf16 (its x[i] + x[i + 1] + ... on bfloat16_t) before the fp32 sum: with that, a row's
// result is bit for bit MLX's one-row quantized matmul.
inline float load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
// one lane's 16 inputs times its 8 bytes of one weight row (qdot16 with the weights already loaded)
inline float qdot16w(const thread uint16_t* ws, const thread float* xt, float scale, float bias, float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
inline float qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
// float(v) for v < 2^23 by the exponent trick (an or and a subtract): the same value as a convert
inline float nib(uint v) { return as_type<float>(0x4B000000u | v) - 8388608.0f; }
// qdot16 and qgroup_dot with nib: the same bits, cheaper than the convert from two rows before M5, dearer at one
inline float qdot16x(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * nib(ws[i] & 0x000fu) + xt[4 * i + 1] * nib(ws[i] & 0x00f0u) +
             xt[4 * i + 2] * nib(ws[i] & 0x0f00u) + xt[4 * i + 3] * nib(ws[i] & 0xf000u);
  return scale * accum + sum * bias;
}
inline float qgroup_dotx(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(nib((bits >> (4 * n)) & 0xFu), xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
}
// two 16-bit words' nibbles q (q < 1024 each half) as half with no convert: 1024 + q by the exponent trick, minus 1024
inline half2 nib2(uint q) { return as_type<half2>(0x64006400u | q) - half2(1024.0h); }
// load16 with the third and fourth inputs divided by 16 and 1: with qdot16h's nibbles, the same products as qdot16's
inline float load16h(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 16.0f; xt[i + 3] = float(d);
  }
  return sum;
}
// qdot16 over load16h's inputs with half nibbles (q, 16 q, 16 q, q): the same products in the same order
inline float qdot16h(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint32_t* ws = (const device uint32_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 2; i++) {
    const uint u = ws[i];
    const half2 a = nib2(u & 0x000F000Fu), b = nib2(u & 0x00F000F0u), c = nib2((u >> 4) & 0x00F000F0u),
                d = nib2((u >> 12) & 0x000F000Fu);
    accum += xt[8 * i] * float(a.x) + xt[8 * i + 1] * float(b.x) + xt[8 * i + 2] * float(c.x) +
             xt[8 * i + 3] * float(d.x);
    accum += xt[8 * i + 4] * float(a.y) + xt[8 * i + 5] * float(b.y) + xt[8 * i + 6] * float(c.y) +
             xt[8 * i + 7] * float(d.y);
  }
  return scale * accum + sum * bias;
}
// qgroup_dot with half nibbles: the same fmas in the same order
inline float qgroup_doth(const device uint32_t* w, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int word = 0; word < 4; word++) {
    const uint32_t bits = w[word];
    const half2 q04 = nib2(bits & 0x000F000Fu), q15 = nib2((bits >> 4) & 0x000F000Fu),
                q26 = nib2((bits >> 8) & 0x000F000Fu), q37 = nib2((bits >> 12) & 0x000F000Fu);
    const float q[8] = {float(q04.x), float(q15.x), float(q26.x), float(q37.x),
                        float(q04.y), float(q15.y), float(q26.y), float(q37.y)};
    for (int n = 0; n < 8; n++) {
      const float xv = x[word * 8 + n];
      dq = fma(q[n], xv, dq);
      dx += xv;
    }
  }
  return fma(scale, dq, bias * dx);
}

inline float stream_rinv(const device float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}

// the bf16 at index i (0..7) of 8 packed bf16, as fp32
inline float bfv(uint4 v, int i) {
  const uint w = v[i / 2];
  return as_type<float>((i % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
// 2^-4e: a nibble left in place times an input scaled by this is the nibble's value times the input, exactly
inline float pre4(int e) { return as_type<float>(uint(127 - 4 * e) << 23); }
// a group's input sums of rows fn / fn + 1: the lane's 8 left to right, then the group's 4 words by shuffles
inline void group_sums(thread const float* xa, thread const float* xc, thread float& v, thread float& u) {
  v = xa[0]; u = xc[0];
  for (int i = 1; i < 8; i++) { v += xa[i]; u += xc[i]; }
  v += simd_shuffle_xor(v, ushort(4)); u += simd_shuffle_xor(u, ushort(4));
  v += simd_shuffle_xor(v, ushort(16)); u += simd_shuffle_xor(u, ushort(16));
}
// one group: P = sum q x in 4 steps (nibbles in place), then acc = fma(bias, sum x, fma(scale, P, acc))
inline void mma_sums(uint word, thread const float* xa, thread const float* xc, float v, float u, int fm, float sc,
                     float bi, thread float& acc0, thread float& acc1) {
  simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
  for (int st = 0; st < 4; st++) {
    const int e = 2 * st + fm % 2;
    simdgroup_matrix<float, 8, 8> am, bm;
    am.thread_elements()[0] = float(word & (0xFu << (8 * st)));
    am.thread_elements()[1] = float(word & (0xFu << (8 * st + 4)));
    bm.thread_elements()[0] = xa[e] * pre4(e);
    bm.thread_elements()[1] = xc[e] * pre4(e);
    simdgroup_multiply_accumulate(P, am, bm, P);
  }
  acc0 = fma(bi, v, fma(sc, P.thread_elements()[0], acc0));
  acc1 = fma(bi, u, fma(sc, P.thread_elements()[1], acc1));
}
inline void mma_group(uint word, thread const float* xa, thread const float* xc, int fm, float sc, float bi,
                      thread float& acc0, thread float& acc1) {
  float v, u;
  group_sums(xa, xc, v, u);
  mma_sums(word, xa, xc, v, u, fm, sc, bi, acc0, acc1);
}
// lane l's place in a tile: output fm, rows fn and fn + 1
inline int tile_fm(int l) { return ((l / 4) & 4) + ((l / 2) % 4); }
inline int tile_fn(int l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }

// values 8m .. 8m + 7 of the 32-value chunk at ``chunk`` (a chunk starts a word, and 8 values span at most 2 words)
template <int BITS>
inline void codes8(const device uint* chunk, int m, thread float* q) {
  const int bit = 8 * m * BITS, word = bit >> 5, shift = bit & 31;
  const uint hi = shift + 8 * BITS > 32 ? chunk[word + 1] : 0u;
  const ulong v = ((ulong(hi) << 32) | ulong(chunk[word])) >> shift;
  for (int i = 0; i < 8; i++) q[i] = float(uint(v >> (BITS * i)) & ((1u << BITS) - 1u));
}
// value i of a row's bit stream starting at ``row`` (for lookups: one value a thread)
template <int BITS>
inline uint code_at(const device uint* row, int i) {
  const int bit = i * BITS, word = bit >> 5, shift = bit & 31;
  uint v = row[word] >> shift;
  if (shift + BITS > 32) v |= row[word + 1] << (32 - shift);
  return v & ((1u << BITS) - 1u);
}
// qgroup_dot for any width: one 32-value chunk, codes in order, then scale and bias
template <int BITS>
inline float qchunk_dot(const device uint* chunk, float scale, float bias, const thread float* x) {
  float dq = 0.0f, dx = 0.0f;
  for (int m = 0; m < 4; m++) {
    float q[8];
    codes8<BITS>(chunk, m, q);
    for (int n = 0; n < 8; n++) {
      dq = fma(q[n], x[8 * m + n], dq);
      dx += x[8 * m + n];
    }
  }
  return fma(scale, dq, bias * dx);
}

// mma_sums with the lane's 8 codes as values (no nibbles in place, so no input scaling)
inline void mma_sums_q(thread const float* q, thread const float* xa, thread const float* xc, float v, float u, int fm,
                       float sc, float bi, thread float& acc0, thread float& acc1) {
  simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
  for (int st = 0; st < 4; st++) {
    const int e = 2 * st + fm % 2;
    simdgroup_matrix<float, 8, 8> am, bm;
    am.thread_elements()[0] = q[2 * st];
    am.thread_elements()[1] = q[2 * st + 1];
    bm.thread_elements()[0] = xa[e];
    bm.thread_elements()[1] = xc[e];
    simdgroup_multiply_accumulate(P, am, bm, P);
  }
  acc0 = fma(bi, v, fma(sc, P.thread_elements()[0], acc0));
  acc1 = fma(bi, u, fma(sc, P.thread_elements()[1], acc1));
}
// mma_group for any width: the chunk's group sums, then the lane's 8 codes (values 8 (fn / 2) ..) through the MMA
template <int BITS>
inline void mma_chunk(const device uint* chunk, thread const float* xa, thread const float* xc, int fm, int fn,
                      float sc, float bi, thread float& acc0, thread float& acc1) {
  float v, u, q[8];
  group_sums(xa, xc, v, u);
  codes8<BITS>(chunk, fn / 2, q);
  mma_sums_q(q, xa, xc, v, u, fm, sc, bi, acc0, acc1);
}
// scalar_group for any width: the MMA path's product sum of one chunk, in its order
template <int BITS>
inline float scalar_chunk(const device uint* chunk, const threadgroup float* x) {
  float q[32];
  for (int m = 0; m < 4; m++) codes8<BITS>(chunk, m, q + 8 * m);
  float p = 0.0f;
  for (int st = 0; st < 4; st++)
    for (int k = 0; k < 8; k++) {
      const int i = 8 * (k / 2) + 2 * st + k % 2;
      p = fma(q[i], x[i], p);
    }
  return p;
}

// one group's product sum on the MMA path, scalar and in its order: steps st 0..3, each an fp32 FMA chain over
// k 0..7 (input 8 (k / 2) + 2 st + k % 2, nibble left in place, input scaled by 2^-4e)
inline float scalar_group(const device uint* w, const threadgroup float* x) {
  const uint4 q = *((const device uint4*)w);
  float p = 0.0f;
  for (int st = 0; st < 4; st++) {
    for (int k = 0; k < 8; k++) {
      const int e = 2 * st + k % 2;
      p = fma(float(q[k / 2] & (0xFu << (4 * e))), x[8 * (k / 2) + e] * pre4(e), p);
    }
  }
  return p;
}
// a group's input sum as the MMA path's group_sums takes it: each run of 8 left to right, then pairs of runs
inline float scalar_sum(const threadgroup float* x) {
  float s[4];
  for (int m = 0; m < 4; m++) {
    s[m] = x[8 * m];
    for (int i = 1; i < 8; i++) s[m] += x[8 * m + i];
  }
  return (s[0] + s[1]) + (s[2] + s[3]);
}

[[max_total_threads_per_threadgroup(320)]]
[[kernel]] void custom_kernel_qa_hc_up_row_bc634afbd2016a85_bfloat16_t_float_float_float_uint32_t_bfloat16_t_bfloat16_t_floatc_int32_tc_bfloat16_t_bfloat16_t(
  const device bfloat16_t* HN [[buffer(0)]],
  const device float* SSP [[buffer(1)]],
  const device float* NW [[buffer(2)]],
  const device float* PART [[buffer(3)]],
  const device uint32_t* QW [[buffer(4)]],
  const device bfloat16_t* QS [[buffer(5)]],
  const device bfloat16_t* QB [[buffer(6)]],
  const constant float* eps [[buffer(7)]],
  const constant int32_t* rows [[buffer(8)]],
  device bfloat16_t* MIXED [[buffer(9)]],
  device bfloat16_t* INJOUT [[buffer(10)]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int S = 4;
  constexpr int D = 2560;
  constexpr int LOW = 320;
  constexpr int ND = 320;
  constexpr int KS = 10;
  constexpr int BITS = 6;
  constexpr int GS = 32;

  // The up projection of one row, bit for bit the MMA path's: threadgroup (i, r) takes dims 8 i .. 8 i + 7 of the
  // S streams. The prologue is the MMA path's; thread (g, s, d) takes one group's product sum, then each output's
  // chain over the groups in order, the sigmoid times the normed stream, summed over the streams in order.
  const uint t = thread_position_in_threadgroup.x;
  const int R = rows[0];
  constexpr int W = S * D, GL = LOW / 32, NO = 8 * S;
  const int d0 = int(threadgroup_position_in_grid.x) * 8;
  const int r = int(threadgroup_position_in_grid.z);
  threadgroup float act[LOW];
  threadgroup float vs[GL];
  threadgroup float ps[GL][NO];
  threadgroup float prod[S][8];
  for (int cc = int(t); cc < ND; cc += GL * NO) {
    float v = 0.0f;
    for (int k = 0; k < KS; k++) v += PART[(size_t(k) * R + r) * ND + cc];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (cc < LOW) act[cc] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0) INJOUT[r * S + (cc - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < GL) vs[t] = scalar_sum(act + 32 * t);
  const int g = int(t) / NO, n = int(t) % NO;
  const int s = n / 8, o = s * D + d0 + n % 8;
  ps[g][n] = scalar_chunk<BITS>(QW + size_t(o) * (LOW * BITS / 32) + g * BITS, act + 32 * g);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < NO) {
    float acc = 0.0f;
    for (int gg = 0; gg < GL; gg++)
      acc = fma(float(QB[size_t(o) * (LOW / GS) + gg * 32 / GS]), vs[gg],
                fma(float(QS[size_t(o) * (LOW / GS) + gg * 32 / GS]), ps[gg][n], acc));
    const float normed = float(bfloat((float(HN[size_t(r) * W + o]) * stream_rinv(SSP, r, s, D / 256, S, D, eps[0]))
                                      * NW[o]));
    prod[s][n % 8] = float(bfloat(bsig(float(bfloat(acc))) * normed));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < 8) {
    float total = 0.0f;
    for (int k = 0; k < S; k++) total += prod[k][t];
    MIXED[size_t(r) * D + d0 + int(t)] = bfloat(total / float(S));
  }

}

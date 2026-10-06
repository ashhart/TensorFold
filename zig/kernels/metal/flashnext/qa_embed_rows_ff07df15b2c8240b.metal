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
[[kernel]] void custom_kernel_qa_embed_rows_ff07df15b2c8240b_uint32_t_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t(
  const device uint32_t* IDS [[buffer(0)]],
  const device uint32_t* W [[buffer(1)]],
  const device bfloat16_t* SC [[buffer(2)]],
  const device bfloat16_t* BI [[buffer(3)]],
  device bfloat16_t* OUT [[buffer(4)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  constexpr int DIMS = 2560;
  constexpr int TILE = 4;
  constexpr int BITS = 6;
  constexpr int GS = 32;

  // Thread (d, r): dim d of token row r (quantized embedding, mx.dequantize's bf16(bf16(scale * q) + bias)),
  // written to each of the TILE copies of the row (the residual streams start as copies of the embedding).
  const int d = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  const size_t row = size_t(IDS[r]);
  const bfloat q = bfloat(float(code_at<BITS>(W + row * (DIMS * BITS / 32), d)));
  const bfloat v = SC[row * (DIMS / GS) + d / GS] * q + BI[row * (DIMS / GS) + d / GS];
  for (int t = 0; t < TILE; t++) OUT[(r * TILE + t) * DIMS + d] = v;

}

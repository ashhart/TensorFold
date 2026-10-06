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

constexpr int lane_values(int bits) { return bits == 6 || bits == 8 ? 8 : 16; }
// the VPT codes from value v0 of a row (v0 a multiple of VPT, so they start on a 2-byte boundary)
template <int BITS, int VPT>
inline void lane_codes(const device uint8_t* row, int v0, thread float* q) {
  constexpr int NH = VPT * BITS / 16;
  const device ushort* h = (const device ushort*)(row + v0 * BITS / 8);
  ushort u[NH + 1];
  for (int j = 0; j < NH; j++) u[j] = h[j];
  u[NH] = 0;
  for (int i = 0; i < VPT; i++) {
    const int bit = BITS * i, j = bit / 16, s = bit % 16;
    uint v = uint(u[j]) >> s;
    if (s + BITS > 16) v |= uint(u[j + 1]) << (16 - s);
    q[i] = float(v & ((1u << BITS) - 1u));
  }
}

// RPS gate and up rows from ``first`` over K inputs: per lane step, a dot of VPT codes, then scale and bias
template <int BITS, int GSZ, int K, int RPS>
inline void gateup_rows(const device uint32_t* GWp, const device uint32_t* UWp, const device bfloat* GSp,
                        const device bfloat* GBp, const device bfloat* USp, const device bfloat* UBp, size_t first,
                        const device bfloat* x, uint lane, thread float* ag, thread float* au) {
  constexpr int VPT = lane_values(BITS), RB = K * BITS / 8, KG = K / GSZ;
  const device uint8_t* gw = (const device uint8_t*)GWp + first * RB;
  const device uint8_t* uw = (const device uint8_t*)UWp + first * RB;
  for (int v0 = int(lane) * VPT; v0 < K; v0 += 32 * VPT) {
    float xv[VPT], sum = 0.0f;
    for (int i = 0; i < VPT; i++) { xv[i] = float(x[v0 + i]); sum += xv[i]; }
    for (int row = 0; row < RPS; row++) {
      float qg[VPT], qu[VPT];
      lane_codes<BITS, VPT>(gw + row * RB, v0, qg);
      lane_codes<BITS, VPT>(uw + row * RB, v0, qu);
      float dg = 0.0f, du = 0.0f;
      for (int i = 0; i < VPT; i++) { dg = fma(qg[i], xv[i], dg); du = fma(qu[i], xv[i], du); }
      const size_t at = (first + row) * KG + v0 / GSZ;
      ag[row] += fma(float(GSp[at]), dg, float(GBp[at]) * sum);
      au[row] += fma(float(USp[at]), du, float(UBp[at]) * sum);
    }
  }
}
// 8 down rows from ``first`` over NI inputs, the lane's input chunks read once
template <int BITS, int GSZ, int NI>
inline void down_rows(const device uint32_t* W, const device bfloat* S, const device bfloat* B, size_t first,
                      const device bfloat* x, uint lane, thread float* out) {
  constexpr int VPT = lane_values(BITS), NC = (NI / VPT + 31) / 32, RB = NI * BITS / 8, KG = NI / GSZ;
  float xv[NC][VPT], sums[NC];
  for (int c = 0; c < NC; c++) {
    const int v0 = (c * 32 + int(lane)) * VPT;
    sums[c] = 0.0f;
    for (int i = 0; i < VPT; i++) { xv[c][i] = v0 < NI ? float(x[v0 + i]) : 0.0f; sums[c] += xv[c][i]; }
  }
  for (int row = 0; row < 8; row++) {
    const device uint8_t* w = (const device uint8_t*)W + (first + row) * RB;
    float acc = 0.0f;
    for (int c = 0; c < NC; c++) {
      const int v0 = (c * 32 + int(lane)) * VPT;
      if (v0 < NI) {
        float q[VPT];
        lane_codes<BITS, VPT>(w, v0, q);
        float d = 0.0f;
        for (int i = 0; i < VPT; i++) d = fma(q[i], xv[c][i], d);
        const size_t at = (first + row) * KG + v0 / GSZ;
        acc += fma(float(S[at]), d, float(B[at]) * sums[c]);
      }
    }
    out[row] = acc;
  }
}
[[kernel]] void custom_kernel_qa_expert_down_y_346a3e7c6008e1ac_bfloat16_t_uint32_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_int32_tc_bfloat16_t(
  const device bfloat16_t* ACT [[buffer(0)]],
  const device uint32_t* PICK [[buffer(1)]],
  const device uint32_t* DW [[buffer(2)]],
  const device bfloat16_t* DS [[buffer(3)]],
  const device bfloat16_t* DB [[buffer(4)]],
  const device uint32_t* SDW [[buffer(5)]],
  const device bfloat16_t* SDS [[buffer(6)]],
  const device bfloat16_t* SDB [[buffer(7)]],
  const constant int32_t* rows [[buffer(8)]],
  device bfloat16_t* Y [[buffer(9)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int NI = 640;
  constexpr int D = 2560;
  constexpr int TOPK = 10;
  constexpr int SG = 2;
  constexpr int WB = 6;
  constexpr int WG = 32;
  constexpr int SWB = 6;
  constexpr int SWG = 32;

  // expert_down_y for any width: simdgroup (row, slot) pair, dims 8 b .. 8 b + 7, one fp32 sum a dim
  const uint lane = thread_index_in_simdgroup;
  const int R = rows[0];
  const int pair = int(threadgroup_position_in_grid.z) * SG + int(simdgroup_index_in_threadgroup);
  constexpr int SLOTS = TOPK + 1;
  if (pair >= R * SLOTS) return;
  const int r = pair / SLOTS, k = pair % SLOTS;
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  const bool shared = k == TOPK;
  const device bfloat* x = ACT + (r * SLOTS + k) * NI;
  float out[8];
  if (shared) down_rows<SWB, SWG, NI>(SDW, SDS, SDB, size_t(d0), x, lane, out);
  else down_rows<WB, WG, NI>(DW, DS, DB, size_t(PICK[r * TOPK + k]) * D + d0, x, lane, out);
  for (int row = 0; row < 8; row++) {
    const float v = simd_sum(out[row]);
    if (lane == 0) Y[(r * SLOTS + k) * D + d0 + row] = bfloat(v);
  }

}

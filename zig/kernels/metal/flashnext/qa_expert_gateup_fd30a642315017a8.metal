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
[[kernel]] void custom_kernel_qa_expert_gateup_fd30a642315017a8_bfloat16_t_float_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_uint32_t_float(
  const device bfloat16_t* X [[buffer(0)]],
  const device float* LOGITS [[buffer(1)]],
  const device uint32_t* GW [[buffer(2)]],
  const device bfloat16_t* GS [[buffer(3)]],
  const device bfloat16_t* GB [[buffer(4)]],
  const device uint32_t* UW [[buffer(5)]],
  const device bfloat16_t* US [[buffer(6)]],
  const device bfloat16_t* UB [[buffer(7)]],
  const device uint32_t* SGW [[buffer(8)]],
  const device bfloat16_t* SGS [[buffer(9)]],
  const device bfloat16_t* SGB [[buffer(10)]],
  const device uint32_t* SUW [[buffer(11)]],
  const device bfloat16_t* SUS [[buffer(12)]],
  const device bfloat16_t* SUB [[buffer(13)]],
  device bfloat16_t* ACT [[buffer(14)]],
  device uint32_t* PICK [[buffer(15)]],
  device float* WTS [[buffer(16)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int K = 2560;
  constexpr int N = 640;
  constexpr int TOPK = 10;
  constexpr int SHARED = 1;
  constexpr int NE = 512;
  constexpr int NL = 513;
  constexpr int RPS = 4;
  constexpr int SG = 2;
  constexpr int WB = 6;
  constexpr int WG = 32;
  constexpr int SWB = 6;
  constexpr int SWG = 32;

  // expert_gateup's slots and picks for any width: routed experts WB-bit in groups of WG, the shared one SWB / SWG
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int p = int(threadgroup_position_in_grid.z);
  constexpr int SLOTS = TOPK + SHARED;
  const int r = p / SLOTS, slot = p % SLOTS;
  const bool shared = slot == TOPK;
  float picked[TOPK];
  const size_t e = shared ? 0 : size_t(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));
  if (!shared && threadgroup_position_in_grid.y == 0 && g == 0 && lane == 0) {
    PICK[r * TOPK + slot] = uint32_t(e);
    if (slot == TOPK - 1) {
      float total = 0.0f;
      float ex[TOPK];
      for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
      for (int kk = 0; kk < TOPK; kk++) WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
    }
  }
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  if (shared) gateup_rows<SWB, SWG, K, RPS>(SGW, SUW, SGS, SGB, SUS, SUB, size_t(row0), X + r * K, lane, ag, au);
  else gateup_rows<WB, WG, K, RPS>(GW, UW, GS, GB, US, UB, e * N + row0, X + r * K, lane, ag, au);
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }

}

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

[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void custom_kernel_q4_gdn_pipe_bf57ffcc8e75e07f_bfloat16_t_bfloat16_t_float_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_tc_bfloat16_t_bfloat16_t_float(
  const device bfloat16_t* P [[buffer(0)]],
  const device bfloat16_t* CS [[buffer(1)]],
  const device float* SIN [[buffer(2)]],
  const device bfloat16_t* CW [[buffer(3)]],
  const device bfloat16_t* ALOG [[buffer(4)]],
  const device bfloat16_t* DT [[buffer(5)]],
  const device bfloat16_t* NW [[buffer(6)]],
  const constant float* eps [[buffer(7)]],
  const constant int32_t* rows [[buffer(8)]],
  device bfloat16_t* OUT [[buffer(9)]],
  device bfloat16_t* CSO [[buffer(10)]],
  device float* SO [[buffer(11)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int NK = 16;
  constexpr int NV = 48;
  constexpr int DK = 128;
  constexpr int DV = 128;
  constexpr int TAPS = 4;
  constexpr int HAS_STATE = 1;
  constexpr int PIPE = 4;

  // gdn_step's arithmetic a row in three phases: every row's conv, norms and gates; the state updates; the outputs
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int hv = int(threadgroup_position_in_grid.x);
  const int hk = hv / (NV / NK);
  const int R = rows[0];
  constexpr int C = 2 * NK * DK + NV * DV;
  constexpr int PW = C + NV * DV + 2 * NV;
  constexpr int RPS = DV / 32;                              // state rows a simdgroup
  threadgroup float qs[PIPE][DK], ks[PIPE][DK], vs[PIPE][DV], ys[PIPE][DV];
  threadgroup float red[PIPE];
  threadgroup float gates[PIPE][2];
  int c = -1;
  if (int(t) < DK) c = hk * DK + int(t);
  else if (int(t) < 2 * DK) c = NK * DK + hk * DK + int(t) - DK;
  else if (int(t) < 2 * DK + DV) c = 2 * NK * DK + hv * DV + int(t) - 2 * DK;
  const bool writes_qk = (hv % (NV / NK)) == 0;
  for (int r = 0; r < R; r++) {
    if (c >= 0) {
      float conv = 0.0f;
      for (int tap = 0; tap < TAPS; tap++) {
        const int at = r + tap;
        const float xv = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[(at - (TAPS - 1)) * PW + c]);
        conv = fma(float(CW[c * TAPS + tap]), xv, conv);
      }
      const float act = bsilu(conv);
      if (int(t) < DK) qs[r][t] = act;
      else if (int(t) < 2 * DK) ks[r][int(t) - DK] = act;
      else vs[r][int(t) - 2 * DK] = act;
      if (c < 2 * NK * DK ? writes_qk : true) {
        for (int j = 0; j < TAPS - 1; j++) {
          const int at = r + 1 + j;
          CSO[(r * (TAPS - 1) + j) * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[(at - (TAPS - 1)) * PW + c];
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(sg) < 2 * R) {
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    threadgroup float* x = isq ? qs[r] : ks[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = x[lane * (DK / 32) + i];
      ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    const float inv = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
    for (int i = 0; i < DK / 32; i++) x[lane * (DK / 32) + i] *= inv;
  } else if (int(sg) < 3 * R && lane == 0) {
    const int r = int(sg) - 2 * R;
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float state[RPS][4];
  for (int j = 0; j < RPS; j++)
    for (int i = 0; i < 4; i++)
      state[j][i] = HAS_STATE ? SIN[(size_t(hv) * DV + sg * RPS + j) * DK + lane * 4 + i] : 0.0f;
  for (int r = 0; r < R; r++) {
    const float g = gates[r][0], beta = gates[r][1];
    float kk[4], qq[4];
    for (int i = 0; i < 4; i++) { kk[i] = ks[r][lane * 4 + i]; qq[i] = qs[r][lane * 4 + i]; }
    for (int j = 0; j < RPS; j++) {
      const int dv = int(sg) * RPS + j;
      float kv = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv += state[j][i] * kk[i];
      }
      kv = simd_sum(kv);
      const float delta = (vs[r][dv] - kv) * beta;
      float out = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out += state[j][i] * qq[i];
      }
      out = simd_sum(out);
      if (lane == 0) ys[r][dv] = float(bfloat(out));
      for (int i = 0; i < 4; i++) SO[((size_t(r) * NV + hv) * DV + dv) * DK + lane * 4 + i] = state[j][i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = ys[sg][lane * (DV / 32) + i]; ss = fma(v, v, ss); }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < DV) {
    for (int r = 0; r < R; r++) {
      const float y = float(bfloat(float(NW[t]) * float(bfloat(ys[r][t] * red[r]))));
      const float z = float(P[r * PW + C + hv * DV + int(t)]);
      OUT[r * NV * DV + hv * DV + int(t)] = bfloat(y * fsig(z));
    }
  }

}

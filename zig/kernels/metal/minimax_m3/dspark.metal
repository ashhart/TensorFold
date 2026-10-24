// MiniMax-M3.1-Flash's DSpark drafter and the target's taps for it, at ref/dspark.py's rounding points.
#include <metal_stdlib>
using namespace metal;

struct mm_tap_args { uint rows, width, taps, tap; };

// T[(r * taps + tap) * width + i] = bf16(R[r * width + i]): the residual after a tapped layer, in the drafter's order.
[[kernel]] void mm_tap(const device float* R [[buffer(0)]], constant mm_tap_args& a [[buffer(1)]],
    device bfloat* T [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  T[((size_t)p.y * a.taps + a.tap) * a.width + p.x] = bfloat(R[(size_t)p.y * a.width + p.x]);
}

struct mm_norm16_args { uint rows, width; float eps; };

// Gemma's rms norm of bf16 rows: bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)).
[[kernel]] void mm_norm16(const device bfloat* X [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant mm_norm16_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device bfloat* x = X + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += n) sum = fma(float(x[i]), float(x[i]), sum);
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += n) out[(size_t)row * a.width + i] = bfloat(float(x[i]) * inv * (1.0f + float(w[i])));
}

struct mm_dprep_args { uint rows, heads, kv_heads, pos0, rotary, stride, q_off, k_off, v_off, ring; float eps, theta; };

// q heads normed and rotated to Q; k heads normed and rotated into the ring at (pos0 + row) % ring, v copied beside.
[[kernel]] void mm_draft_prep(const device bfloat* X [[buffer(0)]], const device bfloat* qw [[buffer(1)]],
    const device bfloat* kw [[buffer(2)]], constant mm_dprep_args& a [[buffer(3)]], device bfloat* Q [[buffer(4)]],
    device bfloat* K [[buffer(5)]], device bfloat* V [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint unit = tg.x, row = tg.y;
  if (unit >= a.heads + a.kv_heads || row >= a.rows) return;
  const bool is_q = unit < a.heads;
  const uint head = is_q ? unit : unit - a.heads;
  const device bfloat* src = X + (size_t)row * a.stride + (is_q ? a.q_off : a.k_off) + (size_t)head * 128;
  const device bfloat* w = is_q ? qw : kw;
  const uint p = a.pos0 + row;
  const size_t slot = p % a.ring;
  device bfloat* y = is_q ? Q + ((size_t)row * a.heads + head) * 128 : K + (slot * a.kv_heads + head) * 128;
  float v[4];
  float sum = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    v[j] = float(src[lane + 32 * j]);
    sum = fma(v[j], v[j], sum);
  }
  const float inv = precise::rsqrt(simd_sum(sum) / 128.0f + a.eps);
  threadgroup float nrm[128];
  for (uint j = 0; j < 4; ++j) nrm[lane + 32 * j] = float(bfloat(v[j] * inv * (1.0f + float(w[lane + 32 * j]))));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_r = a.rotary / 2;
  for (uint j = 0; j < 4; ++j) {
    const uint d = lane + 32 * j;
    if (d >= a.rotary) { y[d] = bfloat(nrm[d]); continue; }
    const uint i = d % half_r;
    const float angle = float(p) * (1.0f / precise::pow(a.theta, float(2 * i) / float(a.rotary)));
    const float c = float(bfloat(precise::cos(angle))), s = float(bfloat(precise::sin(angle)));
    const float rot = d < half_r ? -nrm[d + half_r] : nrm[d - half_r];
    y[d] = bfloat(float(bfloat(nrm[d] * c)) + float(bfloat(rot * s)));
  }
  if (is_q) return;
  const device bfloat* vs = X + (size_t)row * a.stride + a.v_off + (size_t)head * 128;
  device bfloat* vd = V + (slot * a.kv_heads + head) * 128;
  for (uint j = 0; j < 4; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

#define TF_UNROLL_4 _Pragma("clang loop unroll(full)")

struct mm_dvec_args { uint rows, pos0, heads, groups, blocks, first, hi, window, ring; float scale; };

// The block's attention partials over 128-key blocks: row r sees from its window's start up to hi.
[[kernel]] void mm_draft_vec(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], constant mm_dvec_args& a [[buffer(3)]], device float* PM [[buffer(4)]],
    device float* PL [[buffer(5)]], device float* PO [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint kv = tg.x, row = tg.y, z = tg.z;
  const uint G = a.heads / a.groups, head = kv * G + g;
  const size_t slot = ((size_t)head * a.rows + row) * a.blocks + z;
  const uint p = a.pos0 + row;
  const uint lo = p + 1 > a.window ? p + 1 - a.window : 0;
  const uint block = a.first + z;
  const uint from = max(lo, block * 128), to = min(a.hi, block * 128 + 128);
  if (from >= to) {
    if (lane == 0) { PM[slot] = -INFINITY; PL[slot] = 0.0f; }
    return;
  }
  const size_t ld = (size_t)a.groups * 128;
  const float4 qv = a.scale * float4(*(const device bfloat4*)(Q + ((size_t)row * a.heads + head) * 128 + 4 * lane));
  float4 acc = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint i0 = from; i0 < to; i0 += 4) {
    float score[4];
    float4 vr[4];
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const bool live = i0 + u < to;
      const size_t at = (size_t)((live ? i0 + u : from) % a.ring) * ld + kv * 128 + 4 * lane;
      vr[u] = float4(*(const device bfloat4*)(V + at));
      const float sc = simd_sum(dot(qv, float4(*(const device bfloat4*)(K + at))));
      score[u] = live ? sc : -INFINITY;
    }
    const float peak = max(max(top, max(score[0], score[1])), max(score[2], score[3]));
    const float rescale = fast::exp(top - peak);
    mass *= rescale;
    acc *= rescale;
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const float w = fast::exp(score[u] - peak);
      mass += w;
      acc += w * vr[u];
    }
    top = peak;
  }
  if (lane == 0) { PM[slot] = top; PL[slot] = mass; }
  *(device float4*)(PO + slot * 128 + 4 * lane) = acc;
}

struct mm_markov_args { uint vocab, rank, step; };

// One draft's Markov step: logits[step, v] + w2[v] . w1[seq[step]] (w2 8-bit affine), the best ties to the lower id.
[[kernel]] void mm_markov_step(const device bfloat* L [[buffer(0)]], const device bfloat* W1 [[buffer(1)]],
    const device uint* W2 [[buffer(2)]], const device bfloat* S2 [[buffer(3)]], const device bfloat* B2 [[buffer(4)]],
    const device uint* seq [[buffer(5)]], constant mm_markov_args& a [[buffer(6)]], device float* best [[buffer(7)]],
    device uint* best_id [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float e[256];
  threadgroup float vals[8];
  threadgroup uint ids[8];
  const uint prev = seq[a.step];
  for (uint i = t; i < a.rank; i += 256) e[i] = float(W1[(size_t)prev * a.rank + i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint v = group * 256 + t;
  float score = -INFINITY;
  if (v < a.vocab) {
    const device uint* w = W2 + (size_t)v * (a.rank / 4);
    float sum = 0.0f;
    for (uint g = 0; g < a.rank / 64; ++g) {
      const float s = float(S2[(size_t)v * (a.rank / 64) + g]), b = float(B2[(size_t)v * (a.rank / 64) + g]);
      for (uint j = 0; j < 16; ++j) {
        const uint word = w[g * 16 + j];
        for (uint k = 0; k < 4; ++k) sum = fma(float(bfloat(s * float((word >> (8 * k)) & 0xffu) + b)), e[g * 64 + j * 4 + k], sum);
      }
    }
    score = float(L[(size_t)a.step * a.vocab + v]) + sum;
  }
  uint at = v;
  for (ushort d = 16; d > 0; d >>= 1) {
    const float o = simd_shuffle_xor(score, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (o > score || (o == score && oi < at)) { score = o; at = oi; }
  }
  if (lane == 0) { vals[sg] = score; ids[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float b = vals[0];
    uint bi = ids[0];
    for (uint k = 1; k < 8; ++k) if (vals[k] > b || (vals[k] == b && ids[k] < bi)) { b = vals[k]; bi = ids[k]; }
    best[group] = b;
    best_id[group] = bi;
  }
}

// The step's draft: the best of every threadgroup's best, into seq[step + 1].
[[kernel]] void mm_markov_pick(const device float* best [[buffer(0)]], const device uint* best_id [[buffer(1)]],
    constant uint2& a [[buffer(2)]], device uint* seq [[buffer(3)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float vals[32];
  threadgroup uint ids[32];
  float b = -INFINITY;
  uint bi = 0xffffffffu;
  for (uint i = t; i < a.x; i += 1024) if (best[i] > b || (best[i] == b && best_id[i] < bi)) { b = best[i]; bi = best_id[i]; }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float o = simd_shuffle_xor(b, d);
    const uint oi = simd_shuffle_xor(bi, d);
    if (o > b || (o == b && oi < bi)) { b = o; bi = oi; }
  }
  if (lane == 0) { vals[sg] = b; ids[sg] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  b = vals[lane];
  bi = ids[lane];
  for (ushort d = 16; d > 0; d >>= 1) {
    const float o = simd_shuffle_xor(b, d);
    const uint oi = simd_shuffle_xor(bi, d);
    if (o > b || (o == b && oi < bi)) { b = o; bi = oi; }
  }
  if (lane == 0) seq[a.y + 1] = bi;
}

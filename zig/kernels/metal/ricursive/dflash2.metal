// Ripple's DFlash2 drafter (z-lab's DFlash2DraftModel) and the target's taps: 16-row blocks over 5 tapped layers.
#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL_4 _Pragma("clang loop unroll(full)")

// The target's residual after a tapped layer, every row, into the features: row r's tap `at` of `taps`.
struct rc_tap_args { uint rows, width, taps, at; };

[[kernel]] void rc_tap(const device bfloat* R [[buffer(0)]], constant rc_tap_args& a [[buffer(1)]],
    device bfloat* F [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  F[((size_t)p.y * a.taps + a.at) * a.width + p.x] = R[(size_t)p.y * a.width + p.x];
}

struct rc_df_norm_args { uint rows, width; float eps; uint add; };

// RicursiveRMSNorm at rms_norm_eps, bf16(w * bf16(x * rsqrt(mean(x^2) + eps))); a.add first takes bf16(R + X).
inline void rc_df_norm_row(device float* x, const device bfloat* w, uint width, float eps, device bfloat* out,
    uint t, uint n, threadgroup float* part) {
  float sum = 0.0f;
  for (uint i = t; i < width; i += n) sum = fma(x[i], x[i], sum);
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(width) + eps);
  for (uint i = t; i < width; i += n) out[i] = bfloat(float(w[i]) * float(bfloat(x[i] * inv)));
}

[[kernel]] void rc_df_norm(device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], constant rc_df_norm_args& a [[buffer(3)]], device bfloat* out [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  device float* x = R + (size_t)row * a.width;
  if (a.add)
    for (uint i = t; i < a.width; i += n) x[i] = float(bfloat(x[i] + float(X[(size_t)row * a.width + i])));
  threadgroup_barrier(mem_flags::mem_device);
  rc_df_norm_row(x, w, a.width, a.eps, out + (size_t)row * a.width, t, n, part);
}

// The embedded block into the fp32 residual: R row r = E[ids[r]] (the target's table, unscaled, as z-lab embeds).
[[kernel]] void rc_df_embed(const device uint* ids [[buffer(0)]], const device bfloat* E [[buffer(1)]],
    constant rc_df_norm_args& a [[buffer(2)]], device float* R [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  R[(size_t)p.y * a.width + p.x] = float(E[(size_t)ids[p.y] * a.width + p.x]);
}

// The same norm of bf16 rows (fc's output into the context's hidden_norm).
[[kernel]] void rc_df_norm16(const device bfloat* X [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant rc_df_norm_args& a [[buffer(2)]], device bfloat* out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
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
  for (uint i = t; i < a.width; i += n) out[(size_t)row * a.width + i] = bfloat(float(w[i]) * float(bfloat(float(x[i]) * inv)));
}

struct rc_df_prep_args { uint rows, heads, kv_heads, pos0, stride, q_off, k_off, v_off, ring, rope; float eps, theta; };

// Per-head RicursiveRMSNorm, RicursiveRotaryEmbedding's bf16 tables unless a.rope is 0, z-lab's rotation points.
[[kernel]] void rc_df_prep(const device bfloat* X [[buffer(0)]], const device bfloat* qw [[buffer(1)]],
    const device bfloat* kw [[buffer(2)]], constant rc_df_prep_args& a [[buffer(3)]], device bfloat* Q [[buffer(4)]],
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
  for (uint j = 0; j < 4; ++j) nrm[lane + 32 * j] = float(bfloat(float(w[lane + 32 * j]) * float(bfloat(v[j] * inv))));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < 4; ++j) {
    const uint d = lane + 32 * j;
    if (a.rope == 0) {
      y[d] = bfloat(nrm[d]);
      continue;
    }
    const uint i = d % 64;
    const float angle = float(p) * (1.0f / precise::pow(a.theta, float(2 * i) / 128.0f));
    const float c = float(bfloat(precise::cos(angle))), s = float(bfloat(precise::sin(angle)));
    const float rot = d < 64 ? -nrm[d + 64] : nrm[d - 64];
    y[d] = bfloat(float(bfloat(nrm[d] * c)) + float(bfloat(rot * s)));
  }
  if (is_q) return;
  const device bfloat* vs = X + (size_t)row * a.stride + a.v_off + (size_t)head * 128;
  device bfloat* vd = V + (slot * a.kv_heads + head) * 128;
  for (uint j = 0; j < 4; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

struct rc_df_conv_args { uint rows, width, block, groups, side, taps; };

// z-lab's dynamic causal conv, per tap t in the block: y = bf16(y + bf16(base_t x[r-t])), y = bf16(y + dyn_t x[r-t]).
[[kernel]] void rc_df_conv(const device bfloat* X [[buffer(0)]], const device bfloat* D [[buffer(1)]],
    const device bfloat* base [[buffer(2)]], constant rc_df_conv_args& a [[buffer(3)]], device bfloat* Y [[buffer(4)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const uint ch = p.x, r = p.y, g = ch / (a.width / a.groups);
  // a row's dynamic kernel: [side][tap][group]
  const device bfloat* d = D + (size_t)r * 2 * a.taps * a.groups + a.side * a.taps * a.groups;
  const device bfloat* bs = base + (size_t)a.side * a.taps * a.width;
  float y = 0.0f;
  for (uint t = 0; t < a.taps; ++t) {
    if (r % a.block < t) break;
    const float x = float(X[(size_t)(r - t) * a.width + ch]);
    y = float(bfloat(y + float(bfloat(float(bs[(size_t)t * a.width + ch]) * x))));
    y = float(bfloat(fma(float(d[t * a.groups + g]), x, y)));
  }
  Y[(size_t)r * a.width + ch] = bfloat(y);
}

struct rc_df_act_args { uint rows, width; };

// RicursiveMLP's activation (silu, no clamp) on a stacked gate|up row: bf16(bf16(silu(g)) * u).
[[kernel]] void rc_df_silu(const device bfloat* GU [[buffer(0)]], constant rc_df_act_args& a [[buffer(1)]],
    device bfloat* A [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t at = (size_t)p.y * 2 * a.width + p.x;
  const float g = float(GU[at]);
  const float e = precise::exp(-fabs(g)), y = 1.0f / (1.0f + e);
  const float s = g < 0.0f ? e * y : y; // e / (1 + e) below 0, which 1 - y would cancel
  A[(size_t)p.y * a.width + p.x] = bfloat(float(bfloat(g * s)) * float(GU[at + a.width]));
}

// The first eight are llama's ll_vec_args, which ll_merge_vec reads; then the block's end, window, ring and floor.
struct rc_df_vec_args { uint rows, pos0, heads, groups, blocks, chunk, first; float scale; uint hi, window, ring, floor; };

// A block's attention, a (key head, row, 128-key block) a threadgroup: keys max(p + 1 - window, floor) to hi.
[[kernel]] void rc_df_vec(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], constant rc_df_vec_args& a [[buffer(3)]], device float* PM [[buffer(4)]],
    device float* PL [[buffer(5)]], device float* PO [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint kv = tg.x, row = tg.y, z = tg.z;
  const uint G = a.heads / a.groups, head = kv * G + g;
  const size_t slot = ((size_t)head * a.rows + row) * a.blocks + z;
  const uint p = a.pos0 + row;
  const uint lo = max(p + 1 > a.window ? p + 1 - a.window : 0, a.floor);
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

struct rc_df_topk_args { uint n, stride, chunks, from_ids; };

inline bool rc_df_before(float a, uint ia, float b, uint ib) { return a > b || (a == b && ia < ib); }

// The best 32 of a row's 2048-value chunk (value desc, id asc) by a bitonic sort, out at (r * chunks + c) * 32.
[[kernel]] void rc_df_topk(const device bfloat* L [[buffer(0)]], const device float* VIN [[buffer(1)]],
    const device uint* IIN [[buffer(2)]], constant rc_df_topk_args& a [[buffer(3)]], device float* VOUT [[buffer(4)]],
    device uint* IOUT [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float v[2048];
  threadgroup uint id[2048];
  const uint row = tg.y, chunk = tg.x;
  for (uint i = t; i < 2048; i += 1024) {
    const uint at = chunk * 2048 + i;
    if (at < a.n) {
      const size_t src = (size_t)row * a.stride + at;
      v[i] = a.from_ids ? VIN[src] : float(L[src]);
      id[i] = a.from_ids ? IIN[src] : at;
    } else {
      v[i] = -INFINITY;
      id[i] = 0xffffffffu;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= 2048; k <<= 1)
    for (uint j = k >> 1; j > 0; j >>= 1) {
      for (uint i = t; i < 2048; i += 1024) {
        const uint l = i ^ j;
        if (l > i) {
          const bool down = (i & k) == 0; // descending runs where the k-bit is clear
          const bool swap = down ? rc_df_before(v[l], id[l], v[i], id[i]) : rc_df_before(v[i], id[i], v[l], id[l]);
          if (swap) {
            const float tv = v[i]; v[i] = v[l]; v[l] = tv;
            const uint ti = id[i]; id[i] = id[l]; id[l] = ti;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  if (t < 32) {
    VOUT[((size_t)row * a.chunks + chunk) * 32 + t] = v[t];
    IOUT[((size_t)row * a.chunks + chunk) * 32 + t] = id[t];
  }
}

struct rc_df_walk_args { uint slots, rank, top, vocab; };

// z-lab's CandidateSelector, greedy: slot e scores c after p as bf16(unary + bf16(sum_r bf16(A[p][r]H[e][r]) B[c][r])).
[[kernel]] void rc_df_walk(const device float* UV [[buffer(0)]], const device uint* UI [[buffer(1)]],
    const device bfloat* H [[buffer(2)]], const device bfloat* A [[buffer(3)]], const device bfloat* B [[buffer(4)]],
    constant rc_df_walk_args& a [[buffer(5)]], device uint* seq [[buffer(6)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8][32];
  threadgroup uint prev;
  if (t == 0) prev = seq[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = 0; e < a.slots; ++e) {
    const float ah = float(bfloat(float(A[(size_t)prev * a.rank + t]) * float(H[(size_t)e * a.rank + t])));
    for (uint c = 0; c < a.top; ++c) {
      const uint cand = UI[e * a.top + c];
      const float s = simd_sum(ah * float(B[(size_t)min(cand, a.vocab - 1) * a.rank + t]));
      if (lane == 0) part[sg][c] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float best = -INFINITY;
      uint bi = 0;
      for (uint c = 0; c < a.top; ++c) {
        float pair = 0.0f;
        for (uint k = 0; k < a.rank / 32; ++k) pair += part[k][c];
        const float s = float(bfloat(UV[e * a.top + c] + float(bfloat(pair))));
        if (UI[e * a.top + c] < a.vocab && s > best) { best = s; bi = c; }
      }
      prev = UI[e * a.top + bi];
      seq[1 + e] = prev;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

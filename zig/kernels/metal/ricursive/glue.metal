// Ripple's glue at the release's rounding points (modeling_ricursive in bf16): bf16 residual, four norms a layer.
#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL_4 _Pragma("clang loop unroll(full)")

constant constexpr uint DH = RC_DH; // the head dim (128), defined by the loader

// Up to RC_NP of a thread's row elements (t, t + n, ...) held in registers.
constant constexpr uint RC_NP = 32;

struct rc_args { uint rows, width, flags; float eps, scale, blend; };
constant constexpr uint RC_ADD = 1;  // rc_add_norm: R += norm(Y) by w_post before the norm
constant constexpr uint RC_LERP = 2; // rc_add_norm: also VX = lerp(V0, X, blend), v's input

// R and V0 row r = bf16(E[ids[r]] * scale): the scaled embedding, also the value residual's source.
[[kernel]] void rc_embed(const device uint* ids [[buffer(0)]], const device bfloat* E [[buffer(1)]],
    constant rc_args& a [[buffer(2)]], device bfloat* R [[buffer(3)]], device bfloat* V0 [[buffer(4)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t at = (size_t)p.y * a.width + p.x;
  const bfloat v = bfloat(float(E[(size_t)ids[p.y] * a.width + p.x]) * a.scale);
  R[at] = v;
  V0[at] = v;
}

// A row's fp32 sum of squares over the threadgroup into 1 / rms: rsqrt(mean + eps), the parts summed in order.
inline float rc_inv(float sum, constant rc_args& a, uint t, uint n, threadgroup float* part) {
  sum = simd_sum(sum);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < n / 32; ++s) total += part[s];
  return precise::rsqrt(total / float(a.width) + a.eps);
}

// torch.lerp on bf16 (its vectorized form): fma(w, e - s, s) below one half, else fma(w - 1, e - s, e); rounded once.
inline bfloat rc_lerp(float s, float e, float w) {
  return bfloat(fabs(w) < 0.5f ? fma(w, e - s, s) : fma(w - 1.0f, e - s, e));
}

// A row a group: R = bf16(R + norm(Y, w_post)) if RC_ADD; X = norm(R, w_next); VX = lerp(V0, X, blend) if RC_LERP.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void rc_add_norm(device bfloat* R [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
    const device bfloat* w_post [[buffer(2)]], const device bfloat* w_next [[buffer(3)]],
    const device bfloat* V0 [[buffer(4)]], constant rc_args& a [[buffer(5)]], device bfloat* X [[buffer(6)]],
    device bfloat* VX [[buffer(7)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  threadgroup float part_y[32], part_r[32];
  const size_t base = (size_t)row * a.width;
  float r[RC_NP];
  TF_UNROLL_4
  for (uint p = 0; p < RC_NP; ++p) r[p] = t + p * n < a.width ? float(R[base + t + p * n]) : 0.0f;
  if (a.flags & RC_ADD) {
    float y[RC_NP], sum = 0.0f;
    TF_UNROLL_4
    for (uint p = 0; p < RC_NP; ++p) {
      y[p] = t + p * n < a.width ? float(Y[base + t + p * n]) : 0.0f;
      sum = fma(y[p], y[p], sum);
    }
    const float inv = rc_inv(sum, a, t, n, part_y);
    TF_UNROLL_4
    for (uint p = 0; p < RC_NP; ++p) {
      if (t + p * n >= a.width) continue;
      const float normed = float(bfloat(float(w_post[t + p * n]) * float(bfloat(y[p] * inv))));
      r[p] = float(bfloat(r[p] + normed));
      R[base + t + p * n] = bfloat(r[p]);
    }
  }
  float sum = 0.0f;
  TF_UNROLL_4
  for (uint p = 0; p < RC_NP; ++p) sum = fma(r[p], r[p], sum);
  const float inv = rc_inv(sum, a, t, n, part_r);
  TF_UNROLL_4
  for (uint p = 0; p < RC_NP; ++p) {
    const uint i = t + p * n;
    if (i >= a.width) continue;
    const bfloat x = bfloat(float(w_next[i]) * float(bfloat(r[p] * inv)));
    X[base + i] = x;
    if (a.flags & RC_LERP) VX[base + i] = rc_lerp(float(V0[base + i]), float(x), a.blend);
  }
}

struct rc_prep_args { uint rows, heads, kv_heads, pos, stride, ring, rope; float eps; };

// Per head: q_norm or k_norm, then on sliding layers fp32 RoPE (bf16 cos, sin) rounded once; k, v at (pos+row) % ring.
[[kernel]] void rc_prep(const device bfloat* X [[buffer(0)]], const device float* inv [[buffer(1)]],
    const device bfloat* qw [[buffer(2)]], const device bfloat* kw [[buffer(3)]], constant rc_prep_args& a [[buffer(4)]],
    device bfloat* QN [[buffer(5)]], device bfloat* KC [[buffer(6)]], device bfloat* VC [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint PER = DH / 32, HALF = DH / 2;
  static_assert(PER == 4, "a lane holds dims lane + 32j: a dim's RoPE partner is in the same lane");
  const uint unit = tg.x, row = tg.y;
  if (unit >= a.heads + a.kv_heads || row >= a.rows) return;
  const bool is_q = unit < a.heads;
  const uint head = is_q ? unit : unit - a.heads;
  const device bfloat* src = X + (size_t)row * a.stride + (is_q ? 0 : a.heads * DH) + (size_t)head * DH;
  const device bfloat* w = is_q ? qw : kw;
  float x[PER], sum = 0.0f;
  TF_UNROLL_4
  for (uint j = 0; j < PER; ++j) {
    x[j] = float(src[lane + 32 * j]);
    sum = fma(x[j], x[j], sum);
  }
  const float r = precise::rsqrt(simd_sum(sum) / float(DH) + a.eps);
  TF_UNROLL_4
  for (uint j = 0; j < PER; ++j) x[j] = float(bfloat(float(w[lane + 32 * j]) * float(bfloat(x[j] * r))));
  const size_t slot = (a.pos + row) % a.ring;
  const size_t kvw = (size_t)a.kv_heads * DH;
  device bfloat* y = is_q ? QN + ((size_t)row * a.heads + head) * DH : KC + slot * kvw + (size_t)head * DH;
  const float position = float(a.pos + row);
  TF_UNROLL_4
  for (uint j = 0; j < PER; ++j) {
    const uint d = lane + 32 * j;
    if (a.rope == 0) {
      y[d] = bfloat(x[j]);
      continue;
    }
    const float angle = position * inv[d % HALF];
    const float c = float(bfloat(precise::cos(angle))), s = float(bfloat(precise::sin(angle)));
    const float other = d < HALF ? -x[j + 2] : x[j - 2];
    y[d] = bfloat(x[j] * c + other * s);
  }
  if (is_q) return;
  const device bfloat* vs = X + (size_t)row * a.stride + (a.heads + a.kv_heads) * DH + (size_t)head * DH;
  device bfloat* vd = VC + slot * kvw + (size_t)head * DH;
  for (uint j = 0; j < PER; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

// The output gate: A = bf16(A * bf16(sigmoid(fp32 G))), G the bf16 g_proj rows (the release's gate_full_precision).
[[kernel]] void rc_gate(device bfloat* A [[buffer(0)]], const device bfloat* G [[buffer(1)]],
    constant rc_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t i = (size_t)p.y * a.width + p.x;
  const float g = float(G[i]);
  const float s = float(bfloat(1.0f / (1.0f + precise::exp(-g))));
  A[i] = bfloat(float(A[i]) * s);
}

// The release's FFN on a stacked gate|up row: gate at most scale, up within +-scale, then bf16(bf16(silu(g)) * u).
[[kernel]] void rc_silu_clamp(const device bfloat* GU [[buffer(0)]], constant rc_args& a [[buffer(1)]],
    device bfloat* A [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const size_t at = (size_t)p.y * 2 * a.width + p.x;
  // torch.clamp keeps a NaN where Metal's min and clamp return the other operand
  const float g0 = float(GU[at]), u0 = float(GU[at + a.width]);
  const float g = isnan(g0) ? g0 : min(g0, a.scale), u = isnan(u0) ? u0 : clamp(u0, -a.scale, a.scale);
  const float silu = float(bfloat(g / (1.0f + precise::exp(-g))));
  A[(size_t)p.y * a.width + p.x] = bfloat(silu * u);
}

struct rc_vec_args { uint rows, pos, heads, groups, blocks, chunk, first; float scale; uint window, ring; };

// Decode partials a (key head, row, block) a group: ll_decode_vec over the window at slot i % ring, blocks fixed.
[[kernel]] void rc_decode_vec(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], constant rc_vec_args& a [[buffer(3)]], device float* PM [[buffer(4)]],
    device float* PL [[buffer(5)]], device float* PO [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint PER = DH / 32;
  static_assert(PER == 4, "a lane holds one bfloat4 of a head");
  const uint kv = tg.x, row = tg.y, block = a.first + tg.z;
  const uint G = a.heads / a.groups, head = kv * G + g;
  const uint p = a.pos + row;
  const uint low = p + 1 > a.window ? p + 1 - a.window : 0;
  const uint from = max(block * a.chunk, low), to = min(p + 1, (block + 1) * a.chunk);
  const size_t ld = (size_t)a.groups * DH;
  const device bfloat* qp = Q + ((size_t)row * a.heads + head) * DH + PER * lane;
  const float4 q0 = a.scale * float4(*(const device bfloat4*)qp);
  float4 acc0 = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint i0 = from; i0 < to; i0 += 4) {
    float score[4];
    float4 v0[4];
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const bool live = i0 + u < to; // past the block's end: a masked key (read at the block's first, weight 0)
      const uint i = live ? i0 + u : from;
      const size_t at = (size_t)(i % a.ring) * ld + kv * DH + PER * lane;
      v0[u] = float4(*(const device bfloat4*)(V + at));
      const float sc = simd_sum(dot(q0, float4(*(const device bfloat4*)(K + at))));
      score[u] = live ? sc : -INFINITY;
    }
    const float peak = max(max(top, max(score[0], score[1])), max(score[2], score[3]));
    const float rescale = fast::exp(top - peak);
    mass *= rescale;
    acc0 *= rescale;
    TF_UNROLL_4
    for (uint u = 0; u < 4; ++u) {
      const float w = fast::exp(score[u] - peak);
      mass += w;
      acc0 += w * v0[u];
    }
    top = peak;
  }
  const size_t slot = ((size_t)head * a.rows + row) * a.blocks + tg.z;
  if (lane == 0) { PM[slot] = top; PL[slot] = mass; }
  *(device float4*)(PO + slot * DH + PER * lane) = acc0;
}

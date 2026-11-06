// Ripple's DFlash2 drafter glue at dflash2.metal's bf16 points (norm, norm16, prep, conv, silu), streams batched.
#pragma once

#include "ricursive.cuh"

namespace tf_rcd {

typedef __nv_bfloat16 bf16;
using tf_rc::bf;
using tf_rc::ld;
using tf_rc::MOST;
using tf_rc::NT;
using tf_rc::point;

// add_norm's rounding points, a bit each; a control clears one.
constexpr int RESID = 1;  // bf16(R + Y): the residual the norm reads
constexpr int X_ROW = 2;  // bf16(r * inv) before w
constexpr int NORM_ALL = RESID | X_ROW;

// A row a block, R in place: R = bf16(R + Y), then X = bf16(w bf16(R inv)); Y null (norm16): X = norm(R), R only read.
template <int K>
__device__ __forceinline__ void add_norm(bf16* __restrict__ R, const bf16* __restrict__ Y, const bf16* __restrict__ w,
                                         bf16* __restrict__ X, int rows, int width, float eps) {
  __shared__ float part[NT / 32];
  const int row = blockIdx.x, chunks = width / 8;
  if (row >= rows) return;
  const size_t base = (size_t)row * width;
  float r[MOST][8];
  tf_rc::load_row(R + base, chunks, r);
  if (Y != nullptr) {
    float y[MOST][8];
    tf_rc::load_row(Y + base, chunks, y);
    uint4* out = reinterpret_cast<uint4*>(R + base);
#pragma unroll
    for (int j = 0; j < MOST; ++j) {
      const int c = threadIdx.x + j * NT;
      if (c >= chunks) continue;
#pragma unroll
      for (int e = 0; e < 8; ++e) r[j][e] = point<K, RESID>(__fadd_rn(r[j][e], y[j][e]));
      out[c] = tf_common::pack8(r[j]);  // stored as bf16 either way: a control changes only what the norm reads
    }
  }
  const float inv = tf_rc::inv_rms(r, part, width, eps);
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
#pragma unroll
  for (int j = 0; j < MOST; ++j) {
    const int c = threadIdx.x + j * NT;
    if (c >= chunks) continue;
    float wv[8], x[8];
    tf_common::unpack8(w4[c], wv);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] = __fmul_rn(wv[e], point<K, X_ROW>(__fmul_rn(r[j][e], inv)));
    reinterpret_cast<uint4*>(X + base)[c] = tf_common::pack8(x);
  }
}

// Head dim 128, a lane's dims lane + 32 j; a warp a (row, head), 8 warps a block.
constexpr int HD = 128, HALF = 64, PER = HD / 32, WARPS = NT / 32;

// prep's rounding points, a bit each; a control clears one (OUT: the output truncated, not rounded).
constexpr int NORM_ROW = 1;  // bf16(v inv)
constexpr int NORM_OUT = 2;  // bf16(w that): the normed head
constexpr int COS = 4;       // bf16(cos)
constexpr int SIN = 8;       // bf16(sin)
constexpr int ROT_C = 16;    // bf16(nrm c)
constexpr int ROT_S = 32;    // bf16(rot s)
constexpr int OUT = 64;      // the stored output, to nearest even
constexpr int PREP_ALL = 127;
// Not rounding points: a control sets one.
constexpr int NOPE_ROPES = 128;  // the NoPE layer rotated too
constexpr int MASK = 256;        // a key's slot is p & (ring - 1): a power-of-two ring's wrap, in bounds but wrong here

// One stream's rows in a batched launch: its layer's K and V rings, its first row, and that row's position.
struct Segment {
  unsigned long long k, v;
  long long begin, pos;
};

// A warp a (row, unit): query head `unit` into Q [rows][heads][128], else key head unit - heads into ring row p % ring.
template <int K>
__device__ __forceinline__ void prep(const bf16* __restrict__ X, const bf16* __restrict__ qw, const bf16* __restrict__ kw,
                                     const float* __restrict__ inv, const Segment* __restrict__ segs, int nsegs,
                                     bf16* __restrict__ Q, int rows, int heads, int kv_heads, int stride, int q_off,
                                     int k_off, int v_off, int ring, int rope, float eps) {
  const int unit = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31, row = blockIdx.y;
  if (unit >= heads + kv_heads || row >= rows) return;  // whole warps: the shuffles stay converged
  // the row's segment: the last one beginning at or before it (segments are in row order)
  int s = 0;
  while (s + 1 < nsegs && segs[s + 1].begin <= row) ++s;
  const Segment seg = segs[s];
  const long long p = seg.pos + (row - seg.begin);
  const long long slot = (K & MASK) ? (p & (ring - 1)) : p % ring;
  const size_t kvw = (size_t)kv_heads * HD;
  const bool is_q = unit < heads;
  const int head = is_q ? unit : unit - heads;
  const bf16* src = X + (size_t)row * stride + (is_q ? q_off : k_off) + (size_t)head * HD;
  const bf16* w = is_q ? qw : kw;
  bf16* y = is_q ? Q + ((size_t)row * heads + head) * HD
                 : reinterpret_cast<bf16*>(seg.k) + (size_t)slot * kvw + (size_t)head * HD;
  // each head bf16(w bf16(v r)), r = 1 / sqrt(sum v^2 / 128 + eps)
  float v[PER];
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    v[j] = ld(src + lane + 32 * j);
    sum = __fmaf_rn(v[j], v[j], sum);
  }
  const float r = tf_common::rms_scale(tf_common::warp_sum(sum), HD, eps);
  float n[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) n[j] = point<K, NORM_OUT>(__fmul_rn(ld(w + lane + 32 * j), point<K, NORM_ROW>(__fmul_rn(v[j], r))));
  // rope, rotate-half at p: bf16(bf16(n c) + bf16(rot s)), c and s the bf16 cos and sin of p inv[d % 64]
  const bool rotate = rope != 0 || (K & NOPE_ROPES) != 0;
  const float position = (float)p;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int d = lane + 32 * j;
    float o = n[j];
    if (rotate) {
      const float angle = __fmul_rn(position, inv[d % HALF]);
      const float c = point<K, COS>(cosf(angle)), sn = point<K, SIN>(sinf(angle));
      const float rot = j < 2 ? -n[j + 2] : n[j - 2];  // d < 64 pairs with d + 64 (negated), else with d - 64
      o = __fadd_rn(point<K, ROT_C>(__fmul_rn(n[j], c)), point<K, ROT_S>(__fmul_rn(rot, sn)));
    }
    y[d] = (K & OUT) ? __float2bfloat16_rn(o) : __float2bfloat16_rz(o);
  }
  if (is_q) return;
  const bf16* vs = X + (size_t)row * stride + v_off + (size_t)head * HD;
  bf16* vd = reinterpret_cast<bf16*>(seg.v) + (size_t)slot * kvw + (size_t)head * HD;
#pragma unroll
  for (int j = 0; j < PER; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];  // the value as it is
}

// conv's rounding points, a bit each; a control clears one.
constexpr int C_BASE = 1;  // bf16(base x)
constexpr int C_ADD = 2;   // bf16(y + that)
constexpr int C_DYN = 4;   // bf16(fma(dyn, x, y))
constexpr int CONV_ALL = 7;
// Not rounding points: a control sets one.
constexpr int TWO_ROUND = 8;  // Qwen's form: bf16(y + bf16(dyn x)) for the fma
constexpr int NO_EDGE = 16;   // the block-edge guard dropped: taps reach the block before (never above row 0)
constexpr int SWAP = 32;      // base takes the fma and dyn the rounded product

// z-lab's causal conv, lag t ascending in the block, x = X[r - t]: y = bf16(y + bf16(base_t x)), bf16(fma(dyn_t, x, y))
template <int K>
__device__ __forceinline__ void conv(const bf16* __restrict__ X, const bf16* __restrict__ D, const bf16* __restrict__ base,
                                     bf16* __restrict__ Y, int rows, int width, int block, int groups, int side, int taps) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (ch >= width || r >= rows) return;
  const int g = ch / (width / groups);
  // D rows are [side][tap][group], base [side][tap][width]
  const bf16* d = D + (size_t)r * 2 * taps * groups + (size_t)side * taps * groups;
  const bf16* bs = base + (size_t)side * taps * width;
  float y = 0.0f;
  for (int t = 0; t < taps; ++t) {
    if ((K & NO_EDGE) ? r < t : r % block < t) break;
    const float x = ld(X + (size_t)(r - t) * width + ch);
    const float b = ld(bs + (size_t)t * width + ch), dv = ld(d + t * groups + g);
    const float lead = (K & SWAP) ? dv : b, fused = (K & SWAP) ? b : dv;
    y = point<K, C_ADD>(__fadd_rn(y, point<K, C_BASE>(__fmul_rn(lead, x))));
    y = point<K, C_DYN>((K & TWO_ROUND) ? __fadd_rn(y, bf(__fmul_rn(fused, x))) : __fmaf_rn(fused, x, y));
  }
  Y[(size_t)r * width + ch] = __float2bfloat16_rn(y);
}

constexpr int GS = 1;  // silu's rounding point: bf16(g s) before up's product

// Qwen3's MLP activation on a stacked gate | up row, no clamp: bf16(bf16(g s) u), s = sigmoid(g) from e = exp(-|g|).
template <int K>
__device__ __forceinline__ void silu(const bf16* __restrict__ GU, bf16* __restrict__ A, int rows, int width) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (x >= width || row >= rows) return;
  const size_t at = (size_t)row * 2 * width + x;
  const float g = ld(GU + at), u = ld(GU + at + width);
  const float e = expf(-fabsf(g)), y = __fdiv_rn(1.0f, __fadd_rn(1.0f, e));
  const float s = g < 0.0f ? __fmul_rn(e, y) : y;  // e / (1 + e) below 0, which 1 - y would cancel
  A[(size_t)row * width + x] = __float2bfloat16_rn(__fmul_rn(point<K, GS>(__fmul_rn(g, s)), u));
}

}  // namespace tf_rcd

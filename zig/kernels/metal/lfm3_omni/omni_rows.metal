// LFM3-Omni's few-row path, each kernel rounding where the separate ops it fuses round (om_head_norm, om_rope, ...).
#include "om_common.h"

struct RowsArgs {
  int K, N, T, ldx, ldy, has_y, has_yf, epi;
  long sw, sx, sy;
};

inline float4 bf4(uint2 w) {
  return float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xFFFF0000u), as_type<float>(w.y << 16), as_type<float>(w.y & 0xFFFF0000u));
}

// y[r, n] = x[r] . W[n] for T <= R rows, 4 weight rows a simdgroup; epi 0 bf16 Y and / or fp32 YF, epi 2 SwiGLU pairs.
template <int R>
inline void rows_body(device const bf* W, device const bf* X, device bf* Y, device float* YF, constant RowsArgs& a, uint3 tg,
                      uint sg, uint lane) {
  W += (long)tg.z * a.sw;
  X += (long)tg.z * a.sx;
  Y += (long)tg.z * a.sy;
  const int nr = min(R, a.T);
  const int K = a.K;
  const int n0 = (tg.x * 4 + sg) * 4;
  if (n0 >= a.N) return;
  device const uint4* w[4];
  for (int j = 0; j < 4; ++j) w[j] = (device const uint4*)(W + (long)min(n0 + j, a.N - 1) * K);
  float acc[R][4];
  for (int r = 0; r < R; ++r)
    for (int j = 0; j < 4; ++j) acc[r][j] = 0.0f;
  for (int c = lane; c < K / 8; c += 32) {
    uint4 wv[4];
    for (int j = 0; j < 4; ++j) wv[j] = w[j][c];
    for (int r = 0; r < R; ++r) {
      if (r >= nr) break;
      const uint4 xv = ((device const uint4*)(X + (long)r * a.ldx))[c];
      const float4 x0 = bf4(xv.xy), x1 = bf4(xv.zw);
      for (int j = 0; j < 4; ++j) {
        const float4 w0 = bf4(wv[j].xy), w1 = bf4(wv[j].zw);
        float s = acc[r][j];
        s = fma(w0.x, x0.x, s);
        s = fma(w0.y, x0.y, s);
        s = fma(w0.z, x0.z, s);
        s = fma(w0.w, x0.w, s);
        s = fma(w1.x, x1.x, s);
        s = fma(w1.y, x1.y, s);
        s = fma(w1.z, x1.z, s);
        s = fma(w1.w, x1.w, s);
        acc[r][j] = s;
      }
    }
  }
  for (int r = 0; r < R; ++r) {
    if (r >= nr) break;
    float v[4];
    for (int j = 0; j < 4; ++j) v[j] = simd_sum(acc[r][j]);
    if (lane != 0) continue;
    if (a.epi == 2) {
      for (int p = 0; p < 2; ++p) {
        const float x = rb(v[2 * p]), c = rb(v[2 * p + 1]);
        Y[(long)r * a.ldy + n0 / 2 + p] = b(rb(x / (1.0f + precise::exp(-x))) * c);
      }
      continue;
    }
    for (int j = 0; j < 4; ++j) {
      if (n0 + j >= a.N) break;
      if (a.has_y) Y[(long)r * a.ldy + n0 + j] = b(v[j]);
      if (a.has_yf) YF[(long)r * a.N + n0 + j] = v[j];
    }
  }
}

#define OM_ROWS(R)                                                                                                                  \
  [[max_total_threads_per_threadgroup(128)]] kernel void om_rows_##R(                                                              \
      device const bf* W [[buffer(0)]], device const bf* X [[buffer(1)]], device bf* Y [[buffer(2)]], device float* YF [[buffer(3)]], \
      constant RowsArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],     \
      uint lane [[thread_index_in_simdgroup]]) {                                                                                   \
    rows_body<R>(W, X, Y, YF, a, tg, sg, lane);                                                                                    \
  }
OM_ROWS(1)
OM_ROWS(4)
OM_ROWS(8)
OM_ROWS(12)
OM_ROWS(16)

struct QkArgs {
  int T, H, Hkv, hd, ld, pos0, mode, nk;
  float eps;
};

// q | k | v | gate rows: q and k heads normed and rotated as om_rope, k and v written to cache row pos0 + t.
kernel void om_qk_prep(device bf* P [[buffer(0)]], device const bf* QN [[buffer(1)]], device const bf* KN [[buffer(2)]],
                       device const float* INV [[buffer(3)]], device bf* KC [[buffer(4)]], device bf* VC [[buffer(5)]],
                       constant QkArgs& a [[buffer(6)]], uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float sv[128];
  const int heads = a.H + a.Hkv;
  const int t = g / heads, h = g % heads;
  const bool q = h < a.H;
  const int kh = h - a.H;
  device bf* x = P + (long)t * a.ld + (q ? h * a.hd : a.H * a.hd + kh * a.hd);
  device const bf* w = q ? QN : KN;
  float ss = 0.0f;
  for (int i = lane; i < a.hd; i += 32) ss += f(x[i]) * f(x[i]);
  ss = simd_sum(ss);
  const float inv = precise::rsqrt(ss / float(a.hd) + a.eps);
  for (int i = lane; i < a.hd; i += 32) sv[i] = rb(f(w[i]) * rb(f(x[i]) * inv));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const int half_ = a.hd >> 1;
  device bf* out = q ? x : KC + (long)(a.pos0 + t) * a.nk + kh * a.hd;
  for (int p = lane; p < half_; p += 32) {
    const float fr = float(a.pos0 + t) * INV[p];
    float c = precise::cos(fr), s = precise::sin(fr);
    if (a.mode != 0) {
      c = rb(c);
      s = rb(s);
    }
    const float xa = sv[p], z = sv[p + half_];
    if (a.mode == 2) {
      out[p] = b(rb(xa * c) + rb(-z * s));
      out[p + half_] = b(rb(z * c) + rb(xa * s));
    } else {
      out[p] = b(xa * c - z * s);
      out[p + half_] = b(z * c + xa * s);
    }
  }
  if (!q) {
    device const bf* v = P + (long)t * a.ld + a.H * a.hd + a.nk + kh * a.hd;
    for (int i = lane; i < a.hd; i += 32) VC[(long)(a.pos0 + t) * a.nk + kh * a.hd + i] = v[i];
  }
}

struct AttendArgs {
  int ldkv, H, Hkv, hd, pos, kv_len, window, causal, has_gate, gate_off;
  float scale;
};

#define OM_SPAN 4096

inline float dot8(uint4 k, threadgroup const float* q) {
  const float4 k0 = bf4(k.xy), k1 = bf4(k.zw);
  return k0.x * q[0] + k0.y * q[1] + k0.z * q[2] + k0.w * q[3] + k1.x * q[4] + k1.y * q[5] + k1.z * q[6] + k1.w * q[7];
}

// One query row over the cache, a group of 256 a head, then LFM3's sigmoid gate; hd % 8 == 0, hd <= 128.
[[max_total_threads_per_threadgroup(256)]] kernel void om_attend(device const bf* Q [[buffer(0)]], device const bf* Kc [[buffer(1)]],
                                                              device const bf* Vc [[buffer(2)]], device bf* O [[buffer(3)]],
                                                              constant AttendArgs& a [[buffer(4)]], uint h [[threadgroup_position_in_grid]],
                                                              uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sc[OM_SPAN];
  threadgroup float qs[128];
  threadgroup float part[2048];
  threadgroup float red[8];
  const int kh = h / (a.H / a.Hkv);
  const int hi = a.causal ? min(a.pos + 1, a.kv_len) : a.kv_len;
  const int lo = (a.causal && a.window > 0) ? max(0, a.pos + 1 - a.window) : 0;
  const int chunks = a.hd / 8;
  if (int(tid) < a.hd) qs[tid] = f(Q[h * a.hd + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (int k = lo + tid; k < hi; k += 256) {
    device const uint4* kr = (device const uint4*)(Kc + (long)k * a.ldkv + kh * a.hd);
    float s = 0.0f;
    for (int c = 0; c < chunks; ++c) s += dot8(kr[c], qs + 8 * c);
    s *= a.scale;
    sc[k - lo] = s;
    m = max(m, s);
  }
  m = group_max<256>(m, red, tid);
  float l = 0.0f;
  for (int k = lo + tid; k < hi; k += 256) {
    const float e = precise::exp(sc[k - lo] - m);
    sc[k - lo] = e;
    l += e;
  }
  l = group_sum<256>(l, red, tid);
  const int groups = 256 / chunks;
  const int dc = tid % chunks, kg = tid / chunks;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (kg < groups) {
    for (int k = lo + kg; k < hi; k += groups) {
      const float p = sc[k - lo];
      const uint4 v = ((device const uint4*)(Vc + (long)k * a.ldkv + kh * a.hd))[dc];
      const float4 v0 = bf4(v.xy), v1 = bf4(v.zw);
      acc[0] += p * v0.x;
      acc[1] += p * v0.y;
      acc[2] += p * v0.z;
      acc[3] += p * v0.w;
      acc[4] += p * v1.x;
      acc[5] += p * v1.y;
      acc[6] += p * v1.z;
      acc[7] += p * v1.w;
    }
    for (int j = 0; j < 8; ++j) part[kg * a.hd + dc * 8 + j] = acc[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(tid) < a.hd) {
    float o = 0.0f;
    for (int g = 0; g < groups; ++g) o += part[g * a.hd + tid];
    bf ob = b(o * (1.0f / l));
    if (a.has_gate) ob = b(f(ob) * rb(sigm(f(Q[a.gate_off + h]))));
    O[h * a.hd + tid] = ob;
  }
}

struct ArgmaxArgs2 {
  int N, ns, per;
};

// Stage 1 of the greedy pick: each group's best of `per` logits (ids in SUP excluded), as (value, index) pairs.
[[max_total_threads_per_threadgroup(256)]] kernel void om_argmax_part(device const bf* X [[buffer(0)]], device const int* SUP [[buffer(1)]],
                                                                   device float* PV [[buffer(2)]], device int* PI [[buffer(3)]],
                                                                   constant ArgmaxArgs2& a [[buffer(4)]], uint g [[threadgroup_position_in_grid]],
                                                                   uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float bv[8];
  threadgroup int bi[8];
  float best = -INFINITY;
  int at = 0x7fffffff;
  const int lo = g * a.per, hi = min(a.N, lo + a.per);
  for (int i = lo + tid; i < hi; i += 256) {
    bool skip = false;
    for (int j = 0; j < a.ns; ++j) skip |= SUP[j] == i;
    if (skip) continue;
    const float v = f(X[i]);
    if (v > best || (v == best && i < at)) {
      best = v;
      at = i;
    }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(best, o);
    const int oi = simd_shuffle_xor(at, o);
    if (ov > best || (ov == best && oi < at)) {
      best = ov;
      at = oi;
    }
  }
  if ((tid & 31) == 0) {
    bv[tid >> 5] = best;
    bi[tid >> 5] = at;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    for (int w = 1; w < 8; ++w)
      if (bv[w] > best || (bv[w] == best && bi[w] < at)) {
        best = bv[w];
        at = bi[w];
      }
    PV[g] = best;
    PI[g] = at;
  }
}

// Stage 2: the first index of the largest over the n parts.
kernel void om_argmax_final(device const float* PV [[buffer(0)]], device const int* PI [[buffer(1)]], device int* OUT [[buffer(2)]],
                            constant int& n [[buffer(3)]], uint lane [[thread_index_in_threadgroup]]) {
  float best = -INFINITY;
  int at = 0x7fffffff;
  for (int i = lane; i < n; i += 32) {
    if (PV[i] > best || (PV[i] == best && PI[i] < at)) {
      best = PV[i];
      at = PI[i];
    }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(best, o);
    const int oi = simd_shuffle_xor(at, o);
    if (ov > best || (ov == best && oi < at)) {
      best = ov;
      at = oi;
    }
  }
  if (lane == 0) OUT[0] = at;
}

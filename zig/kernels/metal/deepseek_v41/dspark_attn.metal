// DeepSeek V4.1 DSpark bidirectional block attention: block rows share 1x ring+block KV load in TG memory.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

#define TF_DA_D 512u
#define TF_DA_NEG (-1e30f)

struct TfDsparkAttnArgs {
  uint asks, n, heads, window;
  float scale;
  uint pad;
};

struct TfDsparkAttnAsk {
  uint base;
  uint first_mod;
  uint ctx_count;
  uint start_pos;
};

struct TfDaKey {
  uint4 a, b;
};

struct TfDaAcc {
  float m;
  float l;
  float acc[16];
};

inline float tf_da_lo(uint w) { return float(as_type<bfloat2>(w).x); }
inline float tf_da_hi(uint w) { return float(as_type<bfloat2>(w).y); }

inline void tf_da_x16(TfDaKey k, thread float* x) {
  for (uint j = 0; j < 4u; ++j) {
    x[2u * j] = tf_da_lo(k.a[j]);
    x[2u * j + 1u] = tf_da_hi(k.a[j]);
    x[8u + 2u * j] = tf_da_lo(k.b[j]);
    x[8u + 2u * j + 1u] = tf_da_hi(k.b[j]);
  }
}

inline float tf_da_step(float acc, float q, float k) {
  const float r = fma(q, k, acc);
  return TF_CTL == 3 ? float(half(r)) : r;
}

inline float tf_da_dot(thread const float* q, TfDaKey k) {
  float x[16];
  tf_da_x16(k, x);
  float v = 0.0f;
  for (uint i = 0; i < 16u; ++i) v = tf_da_step(v, q[i], x[i]);
  return v;
}

inline float tf_da_reduce2(float v0, float v1, uint lane) {
  const bool b0 = (lane & 1u) != 0u;
  float w = (b0 ? v1 : v0) + simd_shuffle_xor(b0 ? v0 : v1, 1);
  for (uint m = 2u; m < 32u; m <<= 1) w += simd_shuffle_xor(w, m);
  return w;
}

inline void tf_da_update(thread TfDaAcc& s, float sc, thread const float* x) {
  if (TF_CTL == 1 && sc != sc) return;
  float m = s.m;
  if (sc > m) {
    const float a = precise::exp(m - sc);
    s.l = s.l * a;
    for (uint i = 0; i < 16u; ++i) s.acc[i] = s.acc[i] * a;
    s.m = sc;
    m = sc;
  }
  const float p = precise::exp(sc - m);
  s.l = s.l + p;
  for (uint i = 0; i < 16u; ++i) s.acc[i] = fma(p, x[i], s.acc[i]);
}

inline void tf_da_batch(thread TfDaAcc& s, thread const float* q, thread const TfDaKey* k, uint valid, float scale,
                        uint lane) {
  const float w = tf_da_reduce2(tf_da_dot(q, k[0]), tf_da_dot(q, k[1]), lane) * scale;
  const bool mine = ((valid >> (lane & 1u)) & 1u) != 0u;
  if (!simd_any(mine && (w > s.m || (TF_CTL == 1 && w != w)))) {
    const float p = precise::exp(w - s.m);
    for (uint u = 0; u < 2u; ++u) {
      if ((valid >> u) & 1u) {
        float x[16];
        tf_da_x16(k[u], x);
        const float pu = simd_broadcast(p, u);
        s.l = s.l + pu;
        for (uint i = 0; i < 16u; ++i) s.acc[i] = fma(pu, x[i], s.acc[i]);
      }
    }
    return;
  }
  for (uint u = 0; u < 2u; ++u) {
    if ((valid >> u) & 1u) {
      float x[16];
      tf_da_x16(k[u], x);
      tf_da_update(s, simd_broadcast(w, u), x);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(512)]] void tf_dsv41_attn_dspark_bf16(
    const device uint4* Q [[buffer(0)]],
    const device uint4* RING [[buffer(1)]],
    const device TfDsparkAttnAsk* ASKS [[buffer(2)]],
    const device float* SK [[buffer(3)]],
    const device float* CS [[buffer(4)]],
    device bfloat* O [[buffer(5)]],
    constant TfDsparkAttnArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kbuf[4][64];
  const uint h = tg.x, j = tg.y;
  if (h >= a.heads || j >= a.asks || sg >= a.n) return;
  const TfDsparkAttnAsk ask = ASKS[j];
  const uint row = j * a.n + sg;
  const uint nthreads = 32u * a.n;
  const uint total_keys = ask.ctx_count + a.n;
  const uint eff_keys = (TF_CTL == 2 && total_keys > 1u) ? (total_keys - 1u) : total_keys;

  float q[16];
  const ulong qat = (ulong(row) * a.heads + h) * (TF_DA_D / 8u) + 2u * lane;
  tf_da_x16(TfDaKey{Q[qat], Q[qat + 1u]}, q);

  TfDaAcc s;
  s.m = TF_DA_NEG;
  s.l = 0.0f;
  for (uint i = 0; i < 16u; ++i) s.acc[i] = 0.0f;

  for (uint k0 = 0; k0 < eff_keys; k0 += 4u) {
    const uint kn = min(4u, eff_keys - k0);
    for (uint idx = tid; idx < kn * 64u; idx += nthreads) {
      const uint u = idx >> 6, w = idx & 63u;
      const uint kt = k0 + u;
      const uint krow = (kt < ask.ctx_count)
          ? (ask.base + ((ask.first_mod + kt) % a.window))
          : (ask.base + a.window + (kt - ask.ctx_count));
      kbuf[u][w] = RING[ulong(krow) * 64u + w];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = 0; u < kn; u += 2u) {
      TfDaKey kk[2];
      kk[0] = TfDaKey{kbuf[u][2u * lane], kbuf[u][2u * lane + 1u]};
      const bool has1 = (u + 1u) < kn;
      kk[1] = has1 ? TfDaKey{kbuf[u + 1u][2u * lane], kbuf[u + 1u][2u * lane + 1u]} : kk[0];
      tf_da_batch(s, q, kk, has1 ? 3u : 1u, a.scale, lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float sink = SK[h];
  const float top = sink > s.m ? sink : s.m;
  const float af = precise::exp(s.m - top);
  const float e = precise::exp(sink - top);
  const float den = fma(s.l, af, e);
  float x[16];
  for (uint i = 0; i < 16u; ++i) x[i] = (s.acc[i] * af) / den;
  if (lane >= 28u) {
    const device float* cs = CS + ulong(ask.start_pos + sg) * 64u;
    for (uint i = 0; i < 16u; i += 2u) {
      const uint pr = 8u * (lane - 28u) + i / 2u;
      const float c = cs[pr], sn = cs[32u + pr];
      const float ne = fma(x[i], c, x[i + 1u] * sn);
      const float no = fma(x[i + 1u], c, -(x[i] * sn));
      x[i] = ne;
      x[i + 1u] = no;
    }
  }
  device bfloat* out = O + (ulong(row) * a.heads + h) * TF_DA_D;
  for (uint i = 0; i < 16u; ++i) out[16u * lane + i] = bfloat(x[i]);
  if (TF_CTL == 4 && row == 0u && h == 0u && lane == 0u) O[ulong(a.asks) * a.n * a.heads * TF_DA_D] = bfloat(1.0f);
}

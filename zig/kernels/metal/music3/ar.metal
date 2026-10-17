// music3 AR rows: embeddings, RMS norm, the decode GEMV (bf16 / int8), int8 quantizing, body and depth attention.
#include "m3_common.h"

// ------------------------------------------------------------------------------------------------ rows

// out[r] = table[ids[r]] (bf16 rows of dim).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_embed(device const int* ids [[buffer(0)]], device const uint4* table [[buffer(1)]],
                     device uint4* out [[buffer(2)]], constant uint& dim [[buffer(3)]],
                     uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint n = dim / 8;
  for (uint i = t; i < n; i += 256) out[(ulong)r * n + i] = table[(ulong)ids[r] * n + i];
}

struct CodeArgs {
  int offset;
  uint rows, dim;
};

// out[b] = table[*code + offset] for rows b < rows (the same code for both CFG rows).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_embed_code(device const int* code [[buffer(0)]], device const uint4* table [[buffer(1)]],
                          device uint4* out [[buffer(2)]], constant CodeArgs& a [[buffer(3)]],
                          uint t [[thread_index_in_threadgroup]]) {
  const uint n = a.dim / 8;
  device const uint4* src = table + (ulong)(*code + a.offset) * n;
  for (uint r = 0; r < a.rows; ++r)
    for (uint i = t; i < n; i += 256) out[(ulong)r * n + i] = src[i];
}

struct FrameArgs {
  uint rows, dim;
  float scale;
};

// A frame fed back: (audio[c0] + sum extra[c_i + (i - 1) 1024]) / sqrt(8), rounded as torch rounds it, both rows.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_frame_embed(device const int* codes [[buffer(0)]], device const bf16* audio [[buffer(1)]],
                           device const bf16* extra [[buffer(2)]], device bf16* out [[buffer(3)]],
                           constant FrameArgs& a [[buffer(4)]], uint gid [[thread_position_in_grid]],
                           uint gn [[threads_per_grid]]) {
  for (uint d = gid; d < a.dim; d += gn) {
    float s = 0.0f;
    for (int i = 1; i < 8; ++i) s += b2f(extra[(ulong)(codes[i] + (i - 1) * 1024) * a.dim + d]);
    const float c0 = b2f(audio[(ulong)codes[0] * a.dim + d]);
    const bf16 v = f2b(rb(c0 + rb(s)) * a.scale);
    for (uint r = 0; r < a.rows; ++r) out[(ulong)r * a.dim + d] = v;
  }
}

struct NormArgs {
  uint xs, ys, dim;
  float eps;
};

// RMS norm of rows (x rows xs apart, y rows ys apart): y = x rsqrt(mean(x^2) + eps) w, fp32 inside, one rounding.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_rmsnorm(device const bf16* x [[buffer(0)]], device const bf16* w [[buffer(1)]],
                       device bf16* y [[buffer(2)]], constant NormArgs& a [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh[32];
  device const bf16* xr = x + (ulong)row * a.xs;
  float ss = 0.0f;
  for (uint i = t; i < a.dim; i += 256) {
    const float v = b2f(xr[i]);
    ss = fma(v, v, ss);
  }
  ss = group_sum(ss, sh, sg, lane, 8);
  const float rs = 1.0f / precise::sqrt(ss / float(a.dim) + a.eps);
  device bf16* yr = y + (ulong)row * a.ys;
  for (uint i = t; i < a.dim; i += 256) yr[i] = f2b(b2f(xr[i]) * rs * b2f(w[i]));
}

// x = round(x + b): prompt rows' residual adds after the GEMMs.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_add(device bf16* x [[buffer(0)]], device const bf16* b [[buffer(1)]],
                   constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) x[i] = f2b(b2f(x[i]) + b2f(b[i]));
}

struct SwigluArgs {
  uint width, rows;
};

// y[r, k] = round(round(silu(g)) u) of gate | up rows [rows, 2 width] (prompt rows).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_swiglu(device const bf16* gu [[buffer(0)]], device bf16* y [[buffer(1)]],
                      constant SwigluArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.width * a.rows) return;
  const uint r = i / a.width, k = i % a.width;
  const float g = b2f(gu[(ulong)r * 2 * a.width + k]), u = b2f(gu[(ulong)r * 2 * a.width + a.width + k]);
  y[i] = f2b(rb(m3_silu(g)) * u);
}

// ------------------------------------------------------------------------------------------------ int8

// One bf16 weight row a group to int8 with a symmetric scale absmax / 127 (fp32): a scale the row, or a group of 128.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_quant(device const bf16* w [[buffer(0)]], device char* q [[buffer(1)]],
                     device float* sc [[buffer(2)]], constant uint2& a [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh[32];
  const uint k = a.x, group = a.y == 0 ? k : a.y, groups = k / group;
  device const bf16* wr = w + (ulong)row * k;
  device char* qr = q + (ulong)row * k;
  for (uint gi = 0; gi < groups; ++gi) {
    float m = 0.0f;
    for (uint i = t; i < group; i += 256) m = max(m, fabs(b2f(wr[gi * group + i])));
    m = group_max(m, sh, sg, lane, 8);
    const float s = m > 0.0f ? m / 127.0f : 1.0f;
    const float inv = 1.0f / s;
    for (uint i = t; i < group; i += 256) qr[gi * group + i] = char(clamp(int(rint(b2f(wr[gi * group + i]) * inv)), -127, 127));
    if (t == 0) sc[(ulong)row * groups + gi] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// int8 rows back to bf16 (round(q s)) for the prompt rows' GEMMs; a.y is the group (0: a scale a row).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_dequant(device const char* q [[buffer(0)]], device const float* sc [[buffer(1)]],
                       device bf16* w [[buffer(2)]], constant uint2& a [[buffer(3)]],
                       uint2 g [[thread_position_in_grid]]) {
  const uint k = a.x, group = a.y == 0 ? k : a.y;
  if (g.x >= k) return;
  const ulong i = (ulong)g.y * k + g.x;
  w[i] = f2b(float(q[i]) * sc[(ulong)g.y * (k / group) + g.x / group]);
}

// ------------------------------------------------------------------------------------------------ the decode GEMV

#ifndef M3_GEMV_U
#define M3_GEMV_U 1
#endif
#ifndef M3_GEMV_SG
#define M3_GEMV_SG 2
#endif

struct GemvArgs {
  uint xs, K, N, ys, res_div, has_y, has_yf, has_res;
};

inline void wfloats(uint4 w, thread float* f, bool i8) {
  if (i8) {
    const char4 c[4] = {as_type<char4>(w.x), as_type<char4>(w.y), as_type<char4>(w.z), as_type<char4>(w.w)};
    for (int j = 0; j < 4; ++j) {
      const float4 v = float4(c[j]);
      f[4 * j] = v.x;
      f[4 * j + 1] = v.y;
      f[4 * j + 2] = v.z;
      f[4 * j + 3] = v.w;
    }
  } else {
    const float4 a = bf4(w.xy), b = bf4(w.zw);
    f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
  }
}

// y = W x for M rows; a simdgroup two weight rows (two outputs, or a SwiGLU pair n, N / 2 + n), U loads in flight.
template <int M, bool I8, bool PAIR, bool GRP>
inline void gemv(device const uchar* W, device const float* sc, device const bf16* x, device bf16* y, device float* yf,
                 device const bf16* res, constant GemvArgs& a, uint tg, uint sg, uint lane) {
  constexpr int E = I8 ? 16 : 8;
  constexpr int U = M3_GEMV_U;
  const uint outs = PAIR ? a.N / 2 : a.N;
  const uint o0 = PAIR ? tg * M3_GEMV_SG + sg : (tg * M3_GEMV_SG + sg) * 2;
  if (o0 >= outs) return;
  const uint row[2] = {o0, PAIR ? o0 + outs : min(o0 + 1, outs - 1)};
  const uint chunks = a.K / E;
  device const uint4* wr[2];
  for (int r = 0; r < 2; ++r) wr[r] = (device const uint4*)(W + (ulong)row[r] * a.K * (I8 ? 1 : 2));
  float acc[2][M];
  for (int r = 0; r < 2; ++r)
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
  for (uint c0 = lane; c0 < chunks; c0 += 32 * U) {
    uint4 wv[U][2];
    for (int u = 0; u < U; ++u)
      for (int r = 0; r < 2; ++r) wv[u][r] = c0 + 32 * u < chunks ? wr[r][c0 + 32 * u] : uint4(0);
    for (int u = 0; u < U; ++u) {
      const uint c = c0 + 32 * u;
      if (c >= chunks) break;
      float wf[2][E], gs[2] = {1.0f, 1.0f};
      for (int r = 0; r < 2; ++r) {
        wfloats(wv[u][r], wf[r], I8);
        if (GRP) gs[r] = sc[(ulong)row[r] * (a.K / 128) + c / 8];
      }
      for (int m = 0; m < M; ++m) {
        device const uint4* xp = (device const uint4*)(x + (ulong)m * a.xs + c * E);
        float xf[E];
        for (int h = 0; h < E / 8; ++h) {
          const uint4 v = xp[h];
          const float4 p = bf4(v.xy), q = bf4(v.zw);
          xf[8 * h] = p.x, xf[8 * h + 1] = p.y, xf[8 * h + 2] = p.z, xf[8 * h + 3] = p.w;
          xf[8 * h + 4] = q.x, xf[8 * h + 5] = q.y, xf[8 * h + 6] = q.z, xf[8 * h + 7] = q.w;
        }
        for (int r = 0; r < 2; ++r) {
          float part = 0.0f;
          for (int e = 0; e < E; ++e) part = fma(wf[r][e], xf[e], part);
          acc[r][m] = GRP ? fma(gs[r], part, acc[r][m]) : acc[r][m] + part;
        }
      }
    }
  }
  for (int m = 0; m < M; ++m) {
    float v[2];
    for (int r = 0; r < 2; ++r) v[r] = simd_sum(acc[r][m]) * (I8 && !GRP ? sc[row[r]] : 1.0f);
    if (lane != 0) continue;
    if (PAIR) {
      const float g = rb(v[0]), u = rb(v[1]);
      y[(ulong)m * a.ys + o0] = f2b(rb(m3_silu(g)) * u);
      continue;
    }
    for (int r = 0; r < 2; ++r) {
      const uint o = o0 + r;
      if (o >= outs) break;
      if (a.has_yf) yf[(ulong)m * a.N + o] = rb(v[r]);
      if (a.has_y) {
        float rr = rb(v[r]);
        if (a.has_res) rr = rb(b2f(res[(ulong)(m / a.res_div) * a.N + o]) + rr);
        y[(ulong)m * a.ys + o] = f2b(rr);
      }
    }
  }
}

#define GEMV(NAME, M, I8, PAIR, GRP)                                                                                 \
  [[max_total_threads_per_threadgroup(32 * M3_GEMV_SG)]] kernel void NAME(                                                     \
      device const uchar* w [[buffer(0)]], device const float* sc [[buffer(1)]], device const bf16* x [[buffer(2)]], \
      device bf16* y [[buffer(3)]], device float* yf [[buffer(4)]], device const bf16* res [[buffer(5)]],          \
      constant GemvArgs& a [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],                                \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                       \
    gemv<M, I8, PAIR, GRP>(w, sc, x, y, yf, res, a, tg, sg, lane);                                                 \
  }
GEMV(m3_gemv_bf16_2, 2, false, false, false)
GEMV(m3_gemv_bf16_4, 4, false, false, false)
GEMV(m3_gemv_i8_2, 2, true, false, false)
GEMV(m3_gemv_i8_4, 4, true, false, false)
GEMV(m3_gemv_g8_2, 2, true, false, true)
GEMV(m3_gemv_g8_4, 4, true, false, true)
GEMV(m3_glu_bf16_2, 2, false, true, false)
GEMV(m3_glu_bf16_4, 4, false, true, false)
GEMV(m3_glu_i8_2, 2, true, true, false)
GEMV(m3_glu_i8_4, 4, true, true, false)
GEMV(m3_glu_g8_2, 2, true, true, true)
GEMV(m3_glu_g8_4, 4, true, true, true)

// Drama 3's codec (Fish S2 modded DAC) in fp32: the GEMM and the glue between them, time-major [rows, channels].
#include "d3_common.h"

struct GemmArgs {
  uint rows, o, cin, ldx, ldy;
  float beta;
  uint taps, dil, base;  // X row t + base - (taps - 1 - j) dil feeds tap j (base: the stage's carried rows)
};

// Y [t, o] = sum_j X [t - (taps - 1 - j) dil] W_j^T + beta Y in fp32 fma chains: 256 threads a 64 x 64 tile.
[[max_total_threads_per_threadgroup(256)]] kernel void c_gemm(device const float* X [[buffer(0)]], device const float* W [[buffer(1)]],
                   device float* Y [[buffer(2)]], constant GemmArgs& a [[buffer(3)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint BM = 64, BN = 64, BK = 16;
  threadgroup float4 xs[BK][BM / 4], ws[BK][BN / 4];
  const uint t0 = tg.y * BM, o0 = tg.x * BN;
  const uint tx = tid % 16, ty = tid / 16;
  const uint K = a.cin * a.taps;
  const bool vec = (a.cin % 4 == 0) && (a.ldx % 4 == 0);
  float acc[4][4];
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j) acc[i][j] = 0.0f;
  // loader: thread tid loads row lr (0..63) at k offset lk (0, 4, 8, 12) of each operand
  const uint lr = tid / 4, lk = (tid % 4) * 4;
  for (uint k0 = 0; k0 < K; k0 += BK) {
    float xv[4], wv[4];
    if (vec) {
      const uint k = k0 + lk;
      float4 x4 = 0.0f, w4 = 0.0f;
      if (k < K) {
        const uint j = k / a.cin, ci = k - j * a.cin;
        const long xr = long(t0 + lr + a.base) - long((a.taps - 1 - j) * a.dil);
        if (t0 + lr < a.rows) x4 = *(device const float4*)(X + xr * long(a.ldx) + ci);
        if (o0 + lr < a.o) w4 = *(device const float4*)(W + ((ulong)j * a.o + o0 + lr) * a.cin + ci);
      }
      for (uint q = 0; q < 4; ++q) {
        xv[q] = x4[q];
        wv[q] = w4[q];
      }
    } else {
      for (uint q = 0; q < 4; ++q) {
        const uint k = k0 + lk + q;
        const uint j = k / a.cin, ci = k - j * a.cin;
        const long xr = long(t0 + lr + a.base) - long((a.taps - 1 - j) * a.dil);
        xv[q] = (t0 + lr < a.rows && k < K) ? X[xr * long(a.ldx) + ci] : 0.0f;
        wv[q] = (o0 + lr < a.o && k < K) ? W[((ulong)j * a.o + o0 + lr) * a.cin + ci] : 0.0f;
      }
    }
    for (uint q = 0; q < 4; ++q) {
      ((threadgroup float*)xs[lk + q])[lr] = xv[q];
      ((threadgroup float*)ws[lk + q])[lr] = wv[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; ++kk) {
      const float4 av = xs[kk][ty], bv = ws[kk][tx];
      for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 4; ++i) {
    const uint t = t0 + ty * 4 + i;
    if (t >= a.rows) break;
    for (uint j = 0; j < 4; ++j) {
      const uint o = o0 + tx * 4 + j;
      if (o >= a.o) break;
      device float* y = Y + (ulong)t * a.ldy + o;
      *y = a.beta != 0.0f ? fma(a.beta, *y, acc[i][j]) : acc[i][j];
    }
  }
}

struct GatherArgs {
  uint fs, D;
};

// out[t] = table[code 0] + sum of residual codebook rows of codes 1..9 (frames [t, fs] columns 1..10, clamped).
[[max_total_threads_per_threadgroup(256)]] kernel void c_vq_gather(device const int* frames [[buffer(0)]], device const float* table [[buffer(1)]],
                        device float* out [[buffer(2)]], constant GatherArgs& a [[buffer(3)]],
                        uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]]) {
  device const int* f = frames + (ulong)t * a.fs;
  int idx[10];
  idx[0] = clamp(f[1], 0, 4095);
  for (int c = 1; c < 10; ++c) idx[c] = 4096 + (c - 1) * 1024 + clamp(f[1 + c], 0, 1023);
  for (uint i = tid; i < a.D; i += nt) {
    float r = table[(ulong)idx[1] * a.D + i];
    for (int c = 2; c < 10; ++c) r += table[(ulong)idx[c] * a.D + i];
    out[(ulong)t * a.D + i] = table[(ulong)idx[0] * a.D + i] + r;
  }
}

// Encoder RVQ, a group a row: 10 stages of project, l2-normalize, nearest normalized code, subtract; codes [t, 10].
[[max_total_threads_per_threadgroup(256)]] kernel void c_vq_encode(device float* z [[buffer(0)]], device const float* in_w [[buffer(1)]],
                        device const float* in_b [[buffer(2)]], device const float* cbn [[buffer(3)]],
                        device const float* table [[buffer(4)]], device int* codes [[buffer(5)]],
                        constant uint& D [[buffer(6)]], uint t [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  threadgroup float bv[32];
  threadgroup int bi[32];
  threadgroup int chosen;
  device float* r = z + (ulong)t * D;
  for (int st = 0; st < 10; ++st) {
    const int N = st == 0 ? 4096 : 1024;
    const int base = st == 0 ? 0 : 4096 + (st - 1) * 1024;
    float e[8];
    for (int k = 0; k < 8; ++k) {
      float acc = 0.0f;
      for (uint i = tid; i < D; i += nt) acc += in_w[((ulong)st * 8 + k) * D + i] * r[i];
      e[k] = group_sum(acc, scratch, sg, lane, nt / 32) + in_b[st * 8 + k];
    }
    float nrm = 0.0f;
    for (int k = 0; k < 8; ++k) nrm += e[k] * e[k];
    const float inv = 1.0f / max(sqrt(nrm), 1e-12f);
    float en[8], esq = 0.0f;
    for (int k = 0; k < 8; ++k) {
      en[k] = e[k] * inv;
      esq += en[k] * en[k];
    }
    float best = -INFINITY;
    int besti = 0x7FFFFFFF;
    for (int n = int(tid); n < N; n += int(nt)) {
      device const float* c = cbn + (ulong)(base + n) * 8;
      float dot = 0.0f, csq = 0.0f;
      for (int k = 0; k < 8; ++k) {
        dot += en[k] * c[k];
        csq += c[k] * c[k];
      }
      const float s = -(esq - 2.0f * dot + csq);
      if (s > best || (s == best && n < besti)) {
        best = s;
        besti = n;
      }
    }
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_xor(best, o);
      const int oi = simd_shuffle_xor(besti, o);
      if (ov > best || (ov == best && oi < besti)) {
        best = ov;
        besti = oi;
      }
    }
    if (lane == 0) {
      bv[sg] = best;
      bi[sg] = besti;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float v0 = bv[0];
      int b0 = bi[0];
      for (uint w = 1; w < nt / 32; ++w)
        if (bv[w] > v0 || (bv[w] == v0 && bi[w] < b0)) {
          v0 = bv[w];
          b0 = bi[w];
        }
      chosen = b0;
      codes[t * 10 + st] = b0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* q = table + (ulong)(base + chosen) * D;
    for (uint i = tid; i < D; i += nt) r[i] -= q[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct CNormArgs {
  uint D;
  float eps;
};

// RMSNorm (the codec transformers'): y = x * rsqrt(mean(x^2) + eps) * w; in place allowed.
[[max_total_threads_per_threadgroup(256)]] kernel void c_rmsnorm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                      device float* y [[buffer(2)]], constant CNormArgs& a [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device const float* xr = x + (ulong)row * a.D;
  float ss = 0.0f;
  for (uint i = tid; i < a.D; i += nt) ss += xr[i] * xr[i];
  const float inv = rsqrt(group_sum(ss, scratch, sg, lane, nt / 32) / float(a.D) + a.eps);
  for (uint i = tid; i < a.D; i += nt) y[(ulong)row * a.D + i] = xr[i] * inv * w[i];
}

// LayerNorm (ConvNeXt, eps 1e-6): y = (x - mean) / sqrt(var + eps) * w + b; in place allowed.
[[max_total_threads_per_threadgroup(256)]] kernel void c_layernorm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                        device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                        constant CNormArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device const float* xr = x + (ulong)row * a.D;
  float s = 0.0f;
  for (uint i = tid; i < a.D; i += nt) s += xr[i];
  const float mean = group_sum(s, scratch, sg, lane, nt / 32) / float(a.D);
  float v = 0.0f;
  for (uint i = tid; i < a.D; i += nt) {
    const float d = xr[i] - mean;
    v += d * d;
  }
  const float inv = rsqrt(group_sum(v, scratch, sg, lane, nt / 32) / float(a.D) + a.eps);
  for (uint i = tid; i < a.D; i += nt) y[(ulong)row * a.D + i] = (xr[i] - mean) * inv * w[i] + b[i];
}

struct RopeKvArgs {
  uint pos0, ring;
};

// Codec rotary on q and k of qkv rows (16 heads of 64, pairs 2i, 2i + 1, bf16 table); k / v to ring rows; q in place.
[[max_total_threads_per_threadgroup(512)]] kernel void c_rope_kv(device float* qkv [[buffer(0)]], device float* kc [[buffer(1)]], device float* vc [[buffer(2)]],
                      constant RopeKvArgs& a [[buffer(3)]], uint t [[threadgroup_position_in_grid]],
                      uint i [[thread_position_in_threadgroup]]) {
  const uint pos = a.pos0 + t;
  const uint pr = i & 31;
  const float freq = 1.0f / precise::pow(10000.0f, float(2 * pr) / 64.0f);
  const float ang = float(pos) * freq;
  const float c = rb(precise::cos(ang)), s = rb(precise::sin(ang));
  device float* row = qkv + (ulong)t * 3072;
  const ulong cr = (ulong)(pos % a.ring) * 1024;
  const float q0 = row[2 * i], q1 = row[2 * i + 1];
  row[2 * i] = q0 * c - q1 * s;
  row[2 * i + 1] = q1 * c + q0 * s;
  const float k0 = row[1024 + 2 * i], k1 = row[1024 + 2 * i + 1];
  kc[cr + 2 * i] = k0 * c - k1 * s;
  kc[cr + 2 * i + 1] = k1 * c + k0 * s;
  vc[cr + 2 * i] = row[2048 + 2 * i];
  vc[cr + 2 * i + 1] = row[2048 + 2 * i + 1];
}

struct CAttnArgs {
  uint pos0, window, ring;
};

// Window-limited causal attention over the ring caches: a simdgroup a (row, head), 2 dims a lane; out [t, 1024].
[[max_total_threads_per_threadgroup(256)]] kernel void c_attn(device const float* qkv [[buffer(0)]], device const float* kc [[buffer(1)]],
                   device const float* vc [[buffer(2)]], device float* out [[buffer(3)]],
                   constant CAttnArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint t = tg.x, h = tg.y * 8 + sg;
  const int pos = int(a.pos0 + t);
  device const float* q = qkv + (ulong)t * 3072 + h * 64;
  const float q0 = q[2 * lane] * 0.125f, q1 = q[2 * lane + 1] * 0.125f;
  float m = -INFINITY, l = 0.0f, a0 = 0.0f, a1 = 0.0f;
  for (int p = max(0, pos - int(a.window) + 1); p <= pos; ++p) {
    const ulong r = (ulong)(uint(p) % a.ring) * 1024 + h * 64 + 2 * lane;
    const float sc = simd_sum(fma(q1, kc[r + 1], q0 * kc[r]));
    const float nm = max(m, sc);
    const float corr = precise::exp(m - nm), e = precise::exp(sc - nm);
    l = l * corr + e;
    a0 = a0 * corr + e * vc[r];
    a1 = a1 * corr + e * vc[r + 1];
    m = nm;
  }
  out[(ulong)t * 1024 + h * 64 + 2 * lane] = a0 / l;
  out[(ulong)t * 1024 + h * 64 + 2 * lane + 1] = a1 / l;
}

struct ElemArgs {
  uint total, n, flag;
};

// x [rows, D] += gamma * y (LayerScale and the residual add); flag 0: no gamma (a plain add).
[[max_total_threads_per_threadgroup(256)]] kernel void c_add_gamma(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                        device const float* gamma [[buffer(2)]], constant ElemArgs& a [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= a.total) return;
  x[i] = fma(a.flag ? gamma[i % a.n] : 1.0f, y[i], x[i]);
}

// SwiGLU of [w1 | w3] rows: y = silu(a) * b; grid (F / 256, rows).
[[max_total_threads_per_threadgroup(256)]] kernel void c_silu_mul(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                       constant uint& F [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x, t = gid.y;
  if (i >= F) return;
  const float av = x[(ulong)t * 2 * F + i], bv = x[(ulong)t * 2 * F + F + i];
  y[(ulong)t * F + i] = av / (1.0f + precise::exp(-av)) * bv;
}

// erfc(z) for z >= 0 by Numerical Recipes' erfcc Chebyshev fit (relative error below 1.2e-7 everywhere).
inline float erfc_pos(float z) {
  const float t = 1.0f / (1.0f + 0.5f * z);
  const float p = -z * z - 1.26551223f +
                  t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f +
                  t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
  return t * precise::exp(p);
}

// GELU (exact, erf) in place: 0.5 x (1 + erf(x / sqrt 2)), 1 + erf from erfc so the tails keep their precision.
[[max_total_threads_per_threadgroup(256)]] kernel void c_gelu(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float v = x[i], z = v * 0.70710678118654752f;
  const float one_plus = z >= 0.0f ? 2.0f - erfc_pos(z) : erfc_pos(-z);
  x[i] = 0.5f * v * one_plus;
}

// y [rows, n] += b[col] (bias of a GEMM's columns), tanh when flag.
[[max_total_threads_per_threadgroup(256)]] kernel void c_bias(device float* y [[buffer(0)]], device const float* b [[buffer(1)]],
                   constant ElemArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.total) return;
  const float v = y[i] + b[i % a.n];
  y[i] = a.flag ? precise::tanh(v) : v;
}

// Snake: y = x + 1 / (alpha + 1e-9) * sin(alpha x)^2 per channel (alpha [C]); flag 0: a copy.
[[max_total_threads_per_threadgroup(256)]] kernel void c_snake(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                    device const float* alpha [[buffer(2)]], constant ElemArgs& a [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
  if (i >= a.total) return;
  const float v = x[i];
  if (a.flag) {
    const float al = alpha[i % a.n];
    const float s = precise::sin(al * v);
    y[i] = v + (1.0f / (al + 1e-9f)) * (s * s);
  } else {
    y[i] = v;
  }
}

// ConvNeXt's causal depthwise conv, kernel 7: y[t, c] = b[c] + sum_j w[c, j] x[t - 6 + j, c], x with 6 margin rows.
[[max_total_threads_per_threadgroup(256)]] kernel void c_dwconv7(device const float* xm [[buffer(0)]], device const float* w [[buffer(1)]],
                      device const float* b [[buffer(2)]], device float* y [[buffer(3)]], constant uint& C [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= C) return;
  float acc = 0.0f;
  for (uint j = 0; j < 7; ++j) acc = fma(w[c * 7 + j], xm[(ulong)(t + j) * C + c], acc);
  y[(ulong)t * C + c] = acc + b[c];
}

// Move n 4-byte words (carried rows of a staging buffer, through a scratch copy when they overlap).
[[max_total_threads_per_threadgroup(256)]] kernel void c_copy(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]], constant uint& n [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}

// Exact nibble/exponent-bin MMA group dots, with compensated FP32 affine sums and direct BF16 rounding.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

struct TfBpArgs { int k, n, rows, group; };
inline float2 tf_bp_add(float2 a, float b) {
  if (!isfinite(a.x) || !isfinite(b)) return float2(a.x + b, 0);
  const float s = a.x + b, z = s - a.x;
  return float2(s, a.y + ((a.x - (s - z)) + (b - z)));
}
inline float2 tf_bp_pair(float2 a, float2 b) { return tf_bp_add(tf_bp_add(a, b.x), b.y); }
inline float2 tf_bp_normal(float2 a) {
  if (!isfinite(a.x)) return a;
  const float h = a.x + a.y, z = h - a.x;
  return float2(h, (a.x - (h - z)) + (a.y - z));
}
inline float2 tf_bp_mul(float2 a, float b) {
  const float p = a.x * b;
  const float2 product = tf_bp_normal(float2(p, fma(a.y, b, fma(a.x, b, -p))));
#ifndef TF_BP_BAD_NONFINITE
  if (!isfinite(product.x) || !isfinite(product.y)) {
    // Normalize only nonfinite/overflow inputs; cancellation can leave the whole sum in the low component.

    const float2 input = tf_bp_normal(a);
    const float high = input.x * b;
    if (!isfinite(high)) return float2(high, 0);
    const float low = fma(input.y, b, fma(input.x, b, -high));
    const float total = high + low;
    if (!isfinite(total)) return float2(total, 0);
    return tf_bp_normal(float2(high, low));
  }
#endif
  return product;
}
inline bfloat tf_bp_round(float2 a) {
  a = tf_bp_normal(a);
  if (!isfinite(a.x)) return bfloat(a.x);
  const uint word = as_type<uint>(a.x), mag = word & 0x7fffffffu;
  uint high = mag >> 16, low = mag & 0xffffu;
  const float residual = word >> 31 ? -a.y : a.y;
  if (low > 0x8000u || (low == 0x8000u && (residual > 0 || (residual == 0 && (high & 1u))))) ++high;
  return as_type<bfloat>(ushort((word >> 16 & 0x8000u) | high));
}
inline int tf_bp_row(int row, int rows) {
#ifdef TF_BP_BAD_ROWS
  if (rows > 4) return 0;
#endif
  return row;
}

[[kernel, max_total_threads_per_threadgroup(64)]]
void tf_binned_sums(const device bfloat* X [[buffer(0)]], constant TfBpArgs& a [[buffer(1)]],
                    device float2* sums [[buffer(2)]], device int2* meta [[buffer(3)]],
                    uint2 at [[thread_position_in_grid]]) {
  const int g = int(at.x), row = int(at.y), groups = a.k / a.group;
  if (g >= groups || row >= a.rows) return;
  float2 sum = float2(0);
  int high = 0, low = 255;
  const int input = tf_bp_row(row, a.rows);
  for (int i = 0; i < a.group; ++i) {
    const bfloat v = X[size_t(input) * a.k + g * a.group + i];
    sum = tf_bp_add(sum, float(v));
    const ushort word = as_type<ushort>(v);
    const int exponent = int((word >> 7) & 255);
    if (word & 0x7fff) { high = max(high, exponent); low = min(low, exponent); }
  }
  sums[size_t(row) * groups + g] = tf_bp_normal(sum);
  meta[size_t(row) * groups + g] = int2(high, max(1, (high - low) / (a.group == 64 ? 7 : 8) + 1));
}

template <int BITS>
inline float tf_bp_nonfinite(const device bfloat* X, const device uint* W, const device bfloat* S,
                              const device bfloat* B, int row, int col, int K, int GS) {
  float result = 0;
  for (int g = 0; g < K / GS; ++g) {
    float qx = 0, xs = 0;
    for (int i = 0; i < GS; ++i) {
      const int k = g * GS + i, bit = k * BITS, shift = bit % 32;
      uint code = W[size_t(col) * (K * BITS / 32) + bit / 32] >> shift;
      if (shift + BITS > 32) code |= W[size_t(col) * (K * BITS / 32) + bit / 32 + 1] << (32 - shift);
      const float x = float(X[size_t(row) * K + k]);
      qx += float(code & ((1u << BITS) - 1u)) * x;
      xs += x;
    }
    result += float(S[size_t(col) * (K / GS) + g]) * qx + float(B[size_t(col) * (K / GS) + g]) * xs;
  }
  return result;
}

#ifndef TF_SIMD_FRAGS
#define tf_bp_right frag_get_tg
#else
#define tf_bp_right frag_get_t
#endif

template <int BITS, int GS>
[[kernel, max_total_threads_per_threadgroup(64)]]
void tf_binned_projection(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
                           const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                           constant TfBpArgs& a [[buffer(4)]], device bfloat* Y [[buffer(5)]],
                           const device float2* sums [[buffer(6)]], const device int2* meta [[buffer(7)]],
                           uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint2 tile [[threadgroup_position_in_grid]]) {
  constexpr int BW = GS == 64 ? 7 : 8, PARTS = (BITS + 3) / 4, PITCH = 40;
  const int rb = int(tile.y) * 32, nb = int(tile.x) * 32, G = a.k / GS;
  const short2 home = frag_home(ushort(lane));
  threadgroup bfloat xb[32 * PITCH], wb[32 * PITCH];
  threadgroup float2 sb[32], xs[32];
  threadgroup int emax[32];
  threadgroup atomic_uint bins;
  float2 acc[2][8];
  bool special[8];
  TF_UNROLL
  for (short e = 0; e < 8; ++e) { special[e] = false; acc[0][e] = acc[1][e] = float2(0); }
  for (int g = 0; g < G; ++g) {
    if (tid == 0) atomic_store_explicit(&bins, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
      const int row = min(rb + int(tid), a.rows - 1), at = row * G + g;
      emax[tid] = meta[at].x;
      atomic_fetch_max_explicit(&bins, uint(meta[at].y), memory_order_relaxed);
      xs[tid] = sums[at];
      sb[tid] = float2(float(S[(nb + int(tid)) * G + g]), float(B[(nb + int(tid)) * G + g]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint count = atomic_load_explicit(&bins, memory_order_relaxed);
    float2 group[2][8];
    TF_UNROLL
    for (short e = 0; e < 8; ++e) {
      group[0][e] = group[1][e] = float2(0);
      special[e] |= emax[int(sg) * 16 + home.y + (e >> 2) * 8] == 255;
    }
    for (int part = 0; part < PARTS; ++part) for (uint bin = 0; bin < count; ++bin) {
      frag<float> lo = frag<float>(0), hi = frag<float>(0);
      for (int stage = 0; stage < GS / 32; ++stage) {
        TF_UNROLL
        for (short slot = 0; slot < 2; ++slot) {
          const int rr = int(tid) / 4 + int(slot) * 16;
          const int row = tf_bp_row(min(rb + rr, a.rows - 1), a.rows);
          const int k0 = g * GS + stage * 32 + int(tid) % 4 * 8;
          const uint4 iv = *((const device uint4*)(X + size_t(row) * a.k + k0));
          uint4 xv = uint4(0), qv = uint4(0);
          TF_UNROLL
          for (short i = 0; i < 8; ++i) {
            uint word = (iv[i / 2] >> (16 * (i % 2))) & 0xffffu;
            const int exponent = int((word >> 7) & 255u);
            if ((word & 0x7fffu) && uint((emax[rr] - exponent) / BW) != bin) word = 0;
            xv[i / 2] |= word << (16 * (i % 2));
          }
          const device ushort* wp = (const device ushort*)((const device uint8_t*)W +
              size_t(nb + rr) * (a.k * BITS / 8) + k0 * BITS / 8);
          const ulong packed = ulong(wp[0]) | (ulong(wp[1]) << 16) |
              (BITS > 4 ? ulong(wp[BITS > 4 ? 2 : 0]) << 32 : 0ul) |
              (BITS > 6 ? ulong(wp[BITS > 6 ? 3 : 0]) << 48 : 0ul);
          TF_UNROLL
          for (short i = 0; i < 8; ++i) {
            uint code = uint(packed >> (i * BITS)) & ((1u << BITS) - 1u);
            code = (code >> (4 * part)) & 15u;
            qv[i / 2] |= (as_type<uint>(float(code)) >> 16) << (16 * (i % 2));
          }
          *((threadgroup uint4*)(xb + rr * PITCH + int(tid) % 4 * 8)) = xv;
          *((threadgroup uint4*)(wb + rr * PITCH + int(tid) % 4 * 8)) = qv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        TF_UNROLL
        for (short kk = 0; kk < 32; kk += 16) {
          frag<bfloat> left, right0, right1;
          frag_get(left, xb, PITCH, int(sg) * 16, kk, home);
          tf_bp_right(right0, wb, PITCH, 0, kk, home);
          tf_bp_right(right1, wb, PITCH, 16, kk, home);
          mma_16x32<false, true>(lo, hi, left, right0, right1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      TF_UNROLL
      for (short e = 0; e < 8; ++e) {
        group[0][e] = tf_bp_add(group[0][e], lo[e] * (part == 0 ? 1.0f : 16.0f));
        group[1][e] = tf_bp_add(group[1][e], hi[e] * (part == 0 ? 1.0f : 16.0f));
      }
    }
    TF_UNROLL
    for (short j = 0; j < 2; ++j) {
      TF_UNROLL
      for (short e = 0; e < 8; ++e) {
        const float2 v = sb[16 * j + home.x + TF_COL(e)];
        acc[j][e] = tf_bp_pair(acc[j][e], tf_bp_pair(tf_bp_mul(group[j][e], v.x),
            tf_bp_mul(xs[int(sg) * 16 + home.y + (e >> 2) * 8], v.y)));
#ifdef TF_BP_BAD_ACC
        acc[j][e] = float2(float(tf_bp_round(acc[j][e])), 0);
#endif
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  TF_UNROLL
  for (short j = 0; j < 2; ++j) {
    TF_UNROLL
    for (short e = 0; e < 8; ++e) {
      const int row = rb + int(sg) * 16 + home.y + (e >> 2) * 8;
      const int col = nb + 16 * j + home.x + TF_COL(e);
      if (row < a.rows) Y[size_t(row) * a.n + col] = special[e]
          ? bfloat(tf_bp_nonfinite<BITS>(X, W, S, B, row, col, a.k, GS)) : tf_bp_round(acc[j][e]);
    }
  }
}

template <int BITS, int GS>
[[kernel, max_total_threads_per_threadgroup(64)]]
void tf_binned_twin(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
                     const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                     constant TfBpArgs& a [[buffer(4)]], device bfloat* Y [[buffer(5)]],
                     const device float2* sums [[buffer(6)]], const device int2* meta [[buffer(7)]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                     uint2 tile [[threadgroup_position_in_grid]]) {
  constexpr int RPS = 4, MB = 2, STEP = GS / 32, BW = GS == 64 ? 7 : 8, PARTS = (BITS + 3) / 4;
  const int col = (int(tile.x) * 2 + int(sg)) * RPS, first = int(tile.y) * MB, G = a.k / GS;
  if (col >= a.n || first >= a.rows) return;
  float2 acc[MB][RPS];
  bool special[MB];
  for (int b = 0; b < MB; ++b) {
    special[b] = false;
    for (int j = 0; j < RPS; ++j) acc[b][j] = float2(0);
  }
  for (int g = 0; g < G; ++g) {
    float x[MB][STEP];
    int2 m[MB];
    for (int b = 0; b < MB; ++b) {
      if (first + b >= a.rows) continue;
      const int row = tf_bp_row(min(first + b, a.rows - 1), a.rows);
      m[b] = meta[size_t(min(first + b, a.rows - 1)) * G + g];
      special[b] |= m[b].x == 255;
      for (int i = 0; i < STEP; ++i) x[b][i] = float(X[size_t(row) * a.k + g * GS + int(lane) * STEP + i]);
    }
    for (int j = 0; j < RPS; ++j) {
      uint q[STEP];
      for (int i = 0; i < STEP; ++i) {
        const int bit = (g * GS + int(lane) * STEP + i) * BITS, shift = bit % 32;
        uint code = W[size_t(col + j) * (a.k * BITS / 32) + bit / 32] >> shift;
        if (shift + BITS > 32) code |= W[size_t(col + j) * (a.k * BITS / 32) + bit / 32 + 1] << (32 - shift);
        q[i] = code & ((1u << BITS) - 1u);
      }
      const float scale = float(S[size_t(col + j) * G + g]), bias = float(B[size_t(col + j) * G + g]);
      for (int b = 0; b < MB; ++b) {
        if (first + b >= a.rows) continue;
        float2 group = float2(0);
        for (int part = 0; part < PARTS; ++part) for (int bin = 0; bin < m[b].y; ++bin) {
          float dot = 0;
          for (int i = 0; i < STEP; ++i) {
            const int exponent = int((as_type<uint>(x[b][i]) >> 23) & 255u);
            const float value = (m[b].x - exponent) / BW == bin ? x[b][i] : 0.0f;
            dot += float((q[i] >> (4 * part)) & 15u) * value;
          }
          group = tf_bp_add(group, simd_sum(dot) * (part == 0 ? 1.0f : 16.0f));
        }
        const float2 xs = sums[size_t(min(first + b, a.rows - 1)) * G + g];
        acc[b][j] = tf_bp_pair(acc[b][j], tf_bp_pair(tf_bp_mul(group, scale), tf_bp_mul(xs, bias)));
#ifdef TF_BP_BAD_ACC
        acc[b][j] = float2(float(tf_bp_round(acc[b][j])), 0);
#endif
      }
    }
  }
  if (lane == 0) for (int b = 0; b < MB; ++b) if (first + b < a.rows) {
    for (int j = 0; j < RPS; ++j) Y[size_t(first + b) * a.n + col + j] = special[b]
        ? bfloat(tf_bp_nonfinite<BITS>(X, W, S, B, first + b, col + j, a.k, GS)) : tf_bp_round(acc[b][j]);
  }
}

#define TF_BP_ENTRY(B, G) \
template [[host_name("tf_binned_q" #B "_g" #G)]] [[kernel]] \
decltype(tf_binned_projection<B, G>) tf_binned_projection<B, G>; \
template [[host_name("tf_binned_twin_q" #B "_g" #G)]] [[kernel]] \
decltype(tf_binned_twin<B, G>) tf_binned_twin<B, G>;
TF_BP_ENTRY(4, 32)
TF_BP_ENTRY(4, 64)
TF_BP_ENTRY(6, 32)
TF_BP_ENTRY(6, 64)
TF_BP_ENTRY(8, 32)
TF_BP_ENTRY(8, 64)

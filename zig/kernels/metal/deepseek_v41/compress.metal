// DeepSeek V4.1's compressor on Metal: the ratio-2 pool, the compressed KV's FP4 round trip and the index-key packing.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

// Control 1 of every kernel: a non-finite input is read as 0, so the output class changes.
inline float tf_c_in(float v) { return (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v; }

inline float tf_c_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }

// ceil(log2(v)) of a positive normal float from its bits, as the reference's fast_log2_ceil.
inline int tf_c_log2c(float v) {
  const uint b = as_type<uint>(v);
  return int((b >> 23) & 0xFFu) - 127 + (((b & 0x7FFFFFu) != 0u && TF_CTL != 2) ? 1 : 0);
}

// A magnitude of at most 448 as its nearest E4M3 code, ties to even (codes 0..8: subnormals and the first normal).
inline uint tf_c_e4m3(float a) {
  uint b = as_type<uint>(a);
  b += 0x7FFFFu + ((b >> 20) & 1u);
  return a < 0.015625f ? uint(rint(a * 512.0f)) : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
}

inline float tf_c_e4m3v(uint c) { return float(as_type<half>(ushort((c & 0x7Fu) << 7))) * 256.0f; }

// A magnitude of at most 6 as its nearest E2M1 code, ties to the even code (control 3: ties away from zero).
inline uint tf_c_e2m1(float m) {
  const bool t = TF_CTL == 3;
  if (m < 0.25f || (!t && m == 0.25f)) return 0u;
  if (m < 0.75f) return 1u;
  if (m < 1.25f || (!t && m == 1.25f)) return 2u;
  if (m < 1.75f) return 3u;
  if (m < 2.5f || (!t && m == 2.5f)) return 4u;
  if (m < 3.5f) return 5u;
  return (m < 5.0f || (!t && m == 5.0f)) ? 6u : 7u;
}

inline float tf_c_e2m1v(uint c) {
  const uint e = (c >> 1) & 3u, m = c & 1u;
  return e == 0u ? float(m) * 0.5f : ldexp(float(m) + 2.0f, int(e) - 2);
}

struct TfCPoolArgs {
  uint rows, dim;
};

// Compressor ratio 2 (model.py:429-487): a column a thread, rows in order; odd positions pool the two slots.
inline void tf_c_pool2_col(const device float* KV, const device float* SC, device float* ST, const device int* P,
                           device bfloat* Y, TfCPoolArgs a, uint d) {
  if (d >= a.dim) return;
  // the carry ST (kv[2][dim], score[2][dim]) keeps each slot's last value: any launch split gives the same bits
  float kv[2] = {ST[d], ST[a.dim + d]}, sc[2] = {ST[2 * a.dim + d], ST[3 * a.dim + d]};
  for (uint r = 0; r < a.rows; ++r) {
    const int pos = P[r];
    const int slot = TF_CTL == 3 ? int(r & 1u) : (pos & 1);
    kv[slot] = tf_c_in(KV[ulong(r) * a.dim + d]);
    sc[slot] = tf_c_in(SC[ulong(r) * a.dim + d]);
    if (slot == 1) {
      // pooled = kv0 * (e0 / s) + kv1 * (e1 / s) in fp32 with e = exp(score - max), rounded to bf16 once
      const float m = max(sc[0], sc[1]);
      const float e0 = precise::exp(sc[0] - m), e1 = precise::exp(sc[1] - m);
      const float sum = e0 + e1;
      const float pooled = kv[0] * precise::divide(e0, sum) + kv[1] * precise::divide(e1, sum);
      Y[ulong(r) * a.dim + d] = bfloat(pooled);
    }
  }
  if (TF_CTL == 4) Y[ulong(a.rows) * a.dim + d] = bfloat(1.0f);
  if (TF_CTL == 2) return;
  ST[d] = kv[0], ST[a.dim + d] = kv[1], ST[2 * a.dim + d] = sc[0], ST[3 * a.dim + d] = sc[1];
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_c_pool2(
    const device float* KV [[buffer(0)]], const device float* SC [[buffer(1)]], device float* ST [[buffer(2)]],
    const device int* P [[buffer(3)]], device bfloat* Y [[buffer(4)]], constant TfCPoolArgs& a [[buffer(5)]],
    uint d [[thread_position_in_grid]]) {
  tf_c_pool2_col(KV, SC, ST, P, Y, a, d);
}

struct TfCRtArgs {
  uint blocks, in_stride, out_row;
};

// The compressed KV's fp4_act_quant round trip (kernel.py:184), 16-value blocks, E4M3 scales, to row D[r] (< 0 skips).
inline void tf_c_fp4rt_block(const device bfloat* X, device bfloat* Y, const device int* D, constant TfCRtArgs& a,
                             uint2 g) {
  if (g.x >= a.blocks) return;
  const int at = D[g.y];
  if (at < 0) return;
  const device bfloat* x = X + ulong(g.y) * a.in_stride + 16u * g.x;
  float v[16], amax = 0.0f;
  bool bad = false;
  for (uint j = 0; j < 16; ++j) {
    v[j] = float(x[j]);
    bad |= TF_CTL != 1 && !isfinite(v[j]);
    amax = max(amax, fabs(tf_c_in(v[j])));
  }
  // the scale: E4M3 of max(amax, 6 2^-9) / 6; a block with a NaN or an infinity is all NaN
  const float sv = min(precise::divide(max(amax, 6.0f * tf_c_pow2(-9)), 6.0f), 448.0f);
  const float s = TF_CTL == 2 ? sv : tf_c_e4m3v(tf_c_e4m3(sv));
  for (uint j = 0; j < 16; ++j) {
    const float w = tf_c_in(v[j]);
    const float q = tf_c_e2m1v(tf_c_e2m1(min(fabs(precise::divide(w, s)), 6.0f))) * s;
    Y[ulong(at) * a.out_row + 16u * g.x + j] = bfloat(bad ? NAN : (signbit(w) ? -q : q));
  }
  if (TF_CTL == 4 && g.x == 0) Y[ulong(at) * a.out_row + a.out_row + 15u] = bfloat(1.0f);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_c_fp4rt(
    const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]], const device int* D [[buffer(2)]],
    constant TfCRtArgs& a [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  tf_c_fp4rt_block(X, Y, D, a, g);
}

struct TfCPackArgs {
  uint blocks, in_stride, code_row, scale_row, scale_off, use_dest;
};

// Index keys and queries, fp4_act_quant: 32-value blocks, E8M0 scales, E2M1 nibbles, to row D[r] (< 0 skips).
inline void tf_c_fp4pack_block(const device bfloat* X, device uchar* C, device uchar* S, const device int* D,
                               constant TfCPackArgs& a, uint2 g) {
  if (g.x >= a.blocks) return;
  const int at = a.use_dest != 0u ? D[g.y] : int(g.y);
  if (at < 0) return;
  const device bfloat* x = X + ulong(g.y) * a.in_stride + 32u * g.x;
  float v[32], amax = 0.0f;
  bool bad = false;
  for (uint j = 0; j < 32; ++j) {
    v[j] = float(x[j]);
    bad |= TF_CTL != 1 && !isfinite(v[j]);
    amax = max(amax, fabs(tf_c_in(v[j])));
  }
  // the scale: 2^ceil(log2(max(amax, 6 2^-126) / 6)); a NaN or infinity block packs zero codes under the byte 255
  const int e = tf_c_log2c(max(amax, 6.0f * tf_c_pow2(-126)) * (1.0f / 6.0f));
  device uchar* c = C + ulong(at) * a.code_row + 16u * g.x;
  for (uint j = 0; j < 16; ++j) {
    const float lo = tf_c_in(v[2 * j]), hi = tf_c_in(v[2 * j + 1]);
    const uint ql = tf_c_e2m1(min(fabs(lo) * tf_c_pow2(-e), 6.0f)) | (signbit(lo) ? 8u : 0u);
    const uint qh = tf_c_e2m1(min(fabs(hi) * tf_c_pow2(-e), 6.0f)) | (signbit(hi) ? 8u : 0u);
    c[j] = bad ? uchar(0) : uchar(ql | (qh << 4));
  }
  S[ulong(at) * a.scale_row + a.scale_off + g.x] = bad ? uchar(255) : uchar(e + 127);
  if (TF_CTL == 4 && g.x == 0) S[ulong(at) * a.scale_row + a.scale_row] = uchar(1);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_c_fp4pack(
    const device bfloat* X [[buffer(0)]], device uchar* C [[buffer(1)]], device uchar* S [[buffer(2)]],
    const device int* D [[buffer(3)]], constant TfCPackArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  tf_c_fp4pack_block(X, C, S, D, a, g);
}

// A slot's state at one layer of a launch over several slots (segs.zig's Seg), then its rows from `base`.
struct TfDsv41Seg {
  device uchar* window;
  device uchar* pool;
  device uchar* keys;
  device float* carry;
  device float* inputs;
  device uchar* comp;
  device uchar* ikeys;
  uint base, rows, nkeys, start;
};

// Control 6 of the slot kernels: every row takes the first slot's state.
inline uint tf_c_slot(const device int* RS, uint r) { return TF_CTL == 6 ? 0u : uint(RS[r]); }

struct TfCPoolSegArgs {
  uint keep, dim; // rows a slot's kept inputs hold (its score half starts keep rows on), columns
};

// tf_dsv41_c_pool2 over several slots, slot g.y over its rows and carry; its kv and score rows kept in its inputs.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_c_pool2_segs(
    const device float* KV [[buffer(0)]], const device float* SC [[buffer(1)]], const device int* P [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant TfCPoolSegArgs& a [[buffer(5)]], const device TfDsv41Seg* SG [[buffer(6)]],
    uint2 g [[thread_position_in_grid]]) {
  if (g.x >= a.dim) return;
  const TfDsv41Seg s = SG[g.y];
  const ulong b = s.base;
  const TfCPoolArgs one = {s.rows, a.dim};
  tf_c_pool2_col(KV + b * a.dim, SC + b * a.dim, SG[TF_CTL == 6 ? 0u : g.y].carry, P + b, Y + b * a.dim, one, g.x);
  if (s.inputs == nullptr) return;
  device uint* in = (device uint*)s.inputs;
  const device uint* kv = (const device uint*)KV + b * a.dim;
  const device uint* sc = (const device uint*)SC + b * a.dim;
  for (uint r = 0; r < s.rows; ++r) {
    in[ulong(r) * a.dim + g.x] = kv[ulong(r) * a.dim + g.x];
    in[ulong(a.keep + r) * a.dim + g.x] = sc[ulong(r) * a.dim + g.x];
  }
}

// tf_dsv41_c_fp4rt over several slots: row r into its slot's compressed rows (RS: row -> slot).
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_c_fp4rt_segs(
    const device bfloat* X [[buffer(0)]], const device int* D [[buffer(2)]], constant TfCRtArgs& a [[buffer(3)]],
    const device TfDsv41Seg* SG [[buffer(4)]], const device int* RS [[buffer(5)]], uint2 g [[thread_position_in_grid]]) {
  tf_c_fp4rt_block(X, (device bfloat*)SG[tf_c_slot(RS, g.y)].pool, D, a, g);
}

// tf_dsv41_c_fp4pack over several slots: row r into its slot's index keys, codes and scales (RS: row -> slot).
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_c_fp4pack_segs(
    const device bfloat* X [[buffer(0)]], const device int* D [[buffer(3)]], constant TfCPackArgs& a [[buffer(4)]],
    const device TfDsv41Seg* SG [[buffer(5)]], const device int* RS [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
  device uchar* k = SG[tf_c_slot(RS, g.y)].keys;
  tf_c_fp4pack_block(X, k, k, D, a, g);
}

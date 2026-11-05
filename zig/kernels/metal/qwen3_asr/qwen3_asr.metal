// Qwen3-ASR's own Metal ops (the core kernels do attention, GELU and unbiased GEMMs) at the CUDA binding's bf16 points.
#include <metal_stdlib>
using namespace metal;

inline float bf(float x) { return float(bfloat(x)); }

// One threadgroup's sum of every thread's `v`, returned to all of them (at most 1,024 threads).
inline float tg_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : 0.0f;
    s = simd_sum(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

struct Gemm {
  uint m, n, k;
  uint res;  // 1: add res [m, n]
};

// y [m, n] = bf16(bf16(x [m, k] . w [n, k]^T + b) + res): 32 x 32 tiles, four simdgroups of 16 x 16, fp32 sums.
constant constexpr uint BM = 32, BN = 32, BK = 32;
[[max_total_threads_per_threadgroup(128)]] kernel void q3a_gemm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    const device bfloat* res [[buffer(3)]], device bfloat* y [[buffer(4)]], constant Gemm& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float As[BM * BK];
  threadgroup float Bs[BN * BK];
  threadgroup float Cs[BM * BN];
  const uint m0 = tg.y * BM, n0 = tg.x * BN;
  const uint sm = (sg / 2) * 16, sn = (sg % 2) * 16;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < p.k; k0 += BK) {
    for (uint e = t; e < BM * BK; e += 128) {
      const uint r = e / BK, c = e % BK;
      const uint gm = m0 + r, gk = k0 + c, gn = n0 + r;
      As[e] = (gm < p.m && gk < p.k) ? float(x[ulong(gm) * p.k + gk]) : 0.0f;
      Bs[e] = (gn < p.n && gk < p.k) ? float(w[ulong(gn) * p.k + gk]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 a[2], bt[2];
      for (uint i = 0; i < 2; i++) simdgroup_load(a[i], As + (sm + 8 * i) * BK + kk, BK);
      for (uint j = 0; j < 2; j++) simdgroup_load(bt[j], Bs + (sn + 8 * j) * BK + kk, BK, ulong2(0, 0), true);
      for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], bt[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; i++)
    for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], Cs + (sm + 8 * i) * BN + sn + 8 * j, BN);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = t; e < BM * BN; e += 128) {
    const uint gm = m0 + e / BN, gn = n0 + e % BN;
    if (gm >= p.m || gn >= p.n) continue;
    const ulong at = ulong(gm) * p.n + gn;
    float v = bf(Cs[e] + float(b[gn]));
    if (p.res != 0) v = bf(v + float(res[at]));
    y[at] = bfloat(v);
  }
}

// The same for one row (decode): a simdgroup an output column, lanes strided over k, then simd_sum.
[[max_total_threads_per_threadgroup(256)]] kernel void q3a_gemv(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    const device bfloat* res [[buffer(3)]], device bfloat* y [[buffer(4)]], constant Gemm& p [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint col = tg * 8 + sg;
  if (col >= p.n) return;
  const device bfloat* wr = w + ulong(col) * p.k;
  float s = 0.0f;
  for (uint i = lane; i < p.k; i += 32) s = fma(float(x[i]), float(wr[i]), s);
  s = simd_sum(s);
  if (lane != 0) return;
  float v = bf(s + float(b[col]));
  if (p.res != 0) v = bf(v + float(res[col]));
  y[col] = bfloat(v);
}

struct Conv {
  uint n, fi, ti, c, fo, to, ld;
};

// A 3x3 stride-2 conv's taps, zero padded by 1: cols [n fo to, ld] from x [n, fi, ti, c], column ci 9 + kh 3 + kw.
[[max_total_threads_per_threadgroup(256)]] kernel void q3a_conv_cols(
    const device bfloat* x [[buffer(0)]], device bfloat* cols [[buffer(1)]], constant Conv& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint col = at.x, pos = at.y;
  if (col >= 9 * p.c || pos >= p.n * p.fo * p.to) return;
  const uint ci = col / 9, kh = (col % 9) / 3, kw = col % 3;
  const uint s = pos / (p.fo * p.to), f = (pos / p.to) % p.fo, t = pos % p.to;
  const int sf = int(2 * f + kh) - 1, st = int(2 * t + kw) - 1;
  bfloat v = bfloat(0.0f);
  if (sf >= 0 && st >= 0 && uint(sf) < p.fi && uint(st) < p.ti) v = x[((ulong(s) * p.fi + uint(sf)) * p.ti + uint(st)) * p.c + ci];
  cols[ulong(pos) * p.ld + col] = v;
}

// torch's [c, f] rows of a conv's channels-last output: y [n t, c f] from x [n, f, t, c].
[[max_total_threads_per_threadgroup(256)]] kernel void q3a_flatten(
    const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant Conv& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  const uint f = p.fi, tt = p.ti, c = p.c;
  if (i >= p.n * tt * c * f) return;
  const uint fi = i % f, ci = (i / f) % c, r = (i / (f * c)) % tt, s = i / (f * c * tt);
  y[i] = x[((ulong(s) * f + fi) * tt + r) * c + ci];
}

struct Rows {
  uint rows, width;
  float eps;
  uint has_res;
};

// LayerNorm with weight and bias, a threadgroup a row: bf16((x - mean) rsqrt(var + eps) w + b) of fp32 sums.
[[max_total_threads_per_threadgroup(256)]] kernel void q3a_layernorm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant Rows& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) s += float(xr[i]);
  const float mean = tg_sum(s, part, n, lane, sg) / float(p.width);
  float v = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float d = float(xr[i]) - mean;
    v += d * d;
  }
  const float inv = precise::rsqrt(tg_sum(v, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat((float(xr[i]) - mean) * inv * float(w[i]) + float(b[i]));
}

// h = bf16(x + res) (h unwritten without res: x is the stream), y = bf16(w bf16(h rsqrt(mean h^2 + eps))).
[[max_total_threads_per_threadgroup(256)]] kernel void q3a_add_rmsnorm(
    const device bfloat* x [[buffer(0)]], const device bfloat* res [[buffer(1)]], const device bfloat* w [[buffer(2)]],
    device bfloat* h [[buffer(3)]], device bfloat* y [[buffer(4)]], constant Rows& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const ulong base = ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    float v = float(x[base + i]);
    if (p.has_res != 0) {
      v = bf(v + float(res[base + i]));
      h[base + i] = bfloat(v);
    }
    s = fma(v, v, s);
  }
  const float inv = 1.0f / precise::sqrt(tg_sum(s, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) {
    const float v = p.has_res != 0 ? bf(float(x[base + i]) + float(res[base + i])) : float(x[base + i]);
    y[base + i] = bfloat(float(w[i]) * bf(v * inv));
  }
}

// out [rows, i] = bf16(silu(gate) up) of [gate | up] rows.
[[max_total_threads_per_threadgroup(256)]] kernel void q3a_swiglu(
    const device bfloat* gu [[buffer(0)]], device bfloat* out [[buffer(1)]], constant uint2& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  if (c >= p.y || r >= p.x) return;
  const float g = float(gu[ulong(r) * 2 * p.y + c]), u = float(gu[ulong(r) * 2 * p.y + p.y + c]);
  out[ulong(r) * p.y + c] = bfloat((g * (1.0f / (1.0f + precise::exp(-g)))) * u);
}

// out [rows, hidden] = table[ids[r]].
[[max_total_threads_per_threadgroup(256)]] kernel void q3a_embed(
    const device int* ids [[buffer(0)]], const device bfloat* table [[buffer(1)]], device bfloat* out [[buffer(2)]],
    constant uint2& p [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.y || at.y >= p.x) return;
  out[ulong(at.y) * p.y + at.x] = table[ulong(ids[at.y]) * p.y + at.x];
}

[[max_total_threads_per_threadgroup(256)]] kernel void q3a_copy(
    const device ushort* src [[buffer(0)]], device ushort* dst [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}

struct QkRope {
  uint h, hk, dim;
  float eps;
  uint cap;  // positions a layer's cache holds
};

// A group a (row, q|k|v head): q, k as bf16(w bf16(x rms^-1)), rotated in fp32, rounded once; k, v into the caches.
[[max_total_threads_per_threadgroup(512)]] kernel void q3a_qk_rope(
    const device bfloat* qkv [[buffer(0)]], const device bfloat* qn [[buffer(1)]], const device bfloat* kn [[buffer(2)]],
    const device float* cs [[buffer(3)]], const device float* sn [[buffer(4)]], const device int* pos [[buffer(5)]],
    device bfloat* q [[buffer(6)]], device bfloat* kc [[buffer(7)]], device bfloat* vc [[buffer(8)]], constant QkRope& p [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  threadgroup float row[512];
  const uint r = tg.x, hd = tg.y, dim = p.dim;
  const uint width = (p.h + 2 * p.hk) * dim;
  const float x = float(qkv[ulong(r) * width + hd * dim + d]);
  const uint at = uint(pos[r]);
  if (hd >= p.h + p.hk) {
    vc[(ulong(at) * p.hk + (hd - p.h - p.hk)) * dim + d] = bfloat(x);
    return;
  }
  const bool is_k = hd >= p.h;
  const float inv = 1.0f / precise::sqrt(tg_sum(x * x, part, dim, lane, sg) / float(dim) + p.eps);
  const float wv = float(is_k ? kn[d] : qn[d]);
  row[d] = bf(wv * bf(x * inv));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_ = dim / 2;
  const float y = row[d], yp = row[d < half_ ? d + half_ : d - half_];
  const float c = cs[ulong(at) * dim + d], s = sn[ulong(at) * dim + d];
  const bfloat out = bfloat(y * c + (d < half_ ? -yp : yp) * s);
  if (is_k) kc[(ulong(at) * p.hk + (hd - p.h)) * dim + d] = out;
  else q[(ulong(r) * p.h + hd) * dim + d] = out;
}

// A window's [q | k | v] rows to q, k and v, each [rows, heads x dim] (shared_attention's token-major layout).
[[max_total_threads_per_threadgroup(256)]] kernel void q3a_split_qkv(
    const device bfloat* qkv [[buffer(0)]], device bfloat* q [[buffer(1)]], device bfloat* k [[buffer(2)]],
    device bfloat* v [[buffer(3)]], constant uint2& p [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y, d = p.y;
  if (c >= d || r >= p.x) return;
  const ulong src = ulong(r) * 3 * d + c, dst = ulong(r) * d + c;
  q[dst] = qkv[src];
  k[dst] = qkv[src + d];
  v[dst] = qkv[src + 2 * d];
}

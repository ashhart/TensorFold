// DeepSeek-V4.1-Flash's kernels in the reference's arithmetic (inference/model.py); quantized projections are core's.
#include <metal_stdlib>
using namespace metal;

constant constexpr int D = 5120;
constant constexpr int S = 4;
constant constexpr int F = S * D;
constant constexpr int MIX = (2 + S) * S;
constant constexpr int HD = 512;  // the attention head (one shared key/value latent)
constant constexpr int IH = 32;   // indexer heads
constant constexpr int ID = 128;  // indexer head dim
constant constexpr int IK = 512;  // indexer picks a row

#pragma clang fp contract(off)

inline float bf_round(float v) { return float(bfloat(v)); }

// fp8 e4m3fn, round to nearest even, |v| <= 448 (callers clamp first).
inline float e4m3_round(float v) {
  const float a = fabs(v);
  if (a == 0.0f) return v;
  float q;
  if (a < 0.015625f) {  // subnormal: steps of 2^-9
    q = rint(a * 512.0f) / 512.0f;
  } else {
    const int e = ilogb(a);
    const float step = ldexp(1.0f, e - 3);
    q = rint(a / step) * step;
  }
  return copysign(q, v);
}

// fp4 e2m1, round to nearest even, |v| <= 6.
inline float e2m1_round(float v) {
  const float a = fabs(v);
  float q;
  if (a <= 0.25f) q = 0.0f;
  else if (a < 0.75f) q = 0.5f;
  else if (a <= 1.25f) q = 1.0f;
  else if (a < 1.75f) q = 1.5f;
  else if (a <= 2.5f) q = 2.0f;
  else if (a < 3.5f) q = 3.0f;
  else if (a <= 5.0f) q = 4.0f;
  else q = 6.0f;
  return copysign(q, v);
}

// 2^ceil(log2(x)) for x > 0, from the bits as the reference's fast_round_scale.
inline float pow2_ceil(float x) {
  const uint b = as_type<uint>(x);
  const int e = int((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0 ? 1 : 0);
  return as_type<float>(uint(e + 127) << 23);
}

// ---------------------------------------------------------------- embedding and hyper-connections

// X [rows, 4, D]: every copy of the stream is the token's embedding.
[[kernel]] void ds_embed(const device uint* IDS [[buffer(0)]], const device bfloat* E [[buffer(1)]],
                         device bfloat* X [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const uint d = g.x, r = g.y;
  const bfloat v = E[size_t(IDS[r]) * D + d];
  for (int s = 0; s < S; s++) X[(size_t(r) * S + s) * D + d] = v;
}

// The MoE branch: the routed experts' rows (in slot order) then the shared expert's, summed in f32.
static float combined(const device bfloat* Y, const device bfloat* SH, int topk, int r, int d) {
  float acc = 0.0f;
  for (int k = 0; k < topk; k++) acc += float(Y[(size_t(r) * topk + k) * D + d]);
  acc += float(SH[size_t(r) * D + d]);
  return float(bfloat(acc));
}

// The pending branch into the streams: x'[k] = post[k] * branch + sum_j comb[j][k] x[j] (topk > 0: the MoE's).
[[kernel]] void ds_hc_expand(device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                             const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                             const device bfloat* Y [[buffer(4)]], const device bfloat* SH [[buffer(5)]],
                             constant int& topk [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
  const uint d = g.x, r = g.y;
  device bfloat* x = X + size_t(r) * F;
  const float b = topk > 0 ? combined(Y, SH, topk, int(r), int(d)) : float(BR[size_t(r) * D + d]);
  float xs[S];
  for (int j = 0; j < S; j++) xs[j] = float(x[j * D + d]);
  const device float* c = COMB + r * S * S;
  for (int k = 0; k < S; k++) {
    float mm = 0.0f;
    for (int j = 0; j < S; j++) mm += c[j * S + k] * xs[j];
    x[k * D + d] = bfloat(POST[r * S + k] * b + mm);
  }
}

struct HcArgs {
  float eps;      // the mixes' and the norm's (norm_eps)
  float hc_eps;   // the split's
  int iters;
};

constant constexpr int SLICES = F / 1024;

// The mixes' partial sums: a threadgroup a (1024-column slice, row), 24 products and a sum of squares to PART.
[[kernel]] void ds_hc_mix_part(const device bfloat* X [[buffer(0)]], const device float* FN [[buffer(1)]],
                               device float* PART [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8][MIX + 1];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const device bfloat* x = X + size_t(r) * F + 1024 * j + 4 * int(t);
  float v[4];
  for (int i = 0; i < 4; i++) v[i] = float(x[i]);
  float acc[MIX + 1];
  acc[MIX] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int m = 0; m < MIX; m++) {
    const device float* f = FN + size_t(m) * F + 1024 * j + 4 * int(t);
    acc[m] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
  }
  for (int m = 0; m <= MIX; m++) {
    const float s = simd_sum(acc[m]);
    if (lane == 0) red[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) <= MIX) {
    float s = 0.0f;
    for (int k = 0; k < 8; k++) s += red[k][t];
    PART[(size_t(r) * SLICES + j) * (MIX + 1) + t] = s;
  }
}

// ds_hc_expand then ds_hc_mix_part: X's streams expanded into XO, the slice's mix sums from the bf16 written.
[[kernel]] void ds_hc_expand_mix(const device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                                 const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                                 const device float* FN [[buffer(4)]], device float* PART [[buffer(5)]],
                                 device bfloat* XO [[buffer(6)]], const device bfloat* Y [[buffer(7)]],
                                 const device bfloat* SH [[buffer(8)]], constant int2& topk_rows [[buffer(9)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8][MIX + 1];
  const int j = int(tg.x), r = int(tg.y), topk = topk_rows.x;
  const uint t = tpos.x;
  const int col = 1024 * j + 4 * int(t);
  const int k = col / D, d0 = col % D;
  const device bfloat* x = X + size_t(r) * F;
  const device float* cm = COMB + r * S * S;
  float v[4];
  for (int i = 0; i < 4; i++) {
    const int d = d0 + i;
    const float b = topk > 0 ? combined(Y, SH, topk, r, d) : float(BR[size_t(r) * D + d]);
    float xs[S];
    for (int jj = 0; jj < S; jj++) xs[jj] = float(x[jj * D + d]);
    float mm = 0.0f;
    for (int jj = 0; jj < S; jj++) mm += cm[jj * S + k] * xs[jj];
    const bfloat o = bfloat(POST[r * S + k] * b + mm);
    XO[size_t(r) * F + col + i] = o;
    v[i] = float(o);
  }
  float acc[MIX + 1];
  acc[MIX] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int m = 0; m < MIX; m++) {
    const device float* f = FN + size_t(m) * F + col;
    acc[m] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
  }
  for (int m = 0; m <= MIX; m++) {
    const float s = simd_sum(acc[m]);
    if (lane == 0) red[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) <= MIX) {
    float s = 0.0f;
    for (int q = 0; q < 8; q++) s += red[q][t];
    PART[(size_t(r) * SLICES + j) * (MIX + 1) + t] = s;
  }
}

// ds_hc_expand_mix's sums for a chunk: XR rows of a 1024-wide d slice (all streams) a threadgroup, FN read once.
constant constexpr int XR = 4;
constant constexpr int DSLICES = D / 1024;

[[kernel]] void ds_hc_expand_mix_rows(const device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                                      const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                                      const device float* FN [[buffer(4)]], device float* PART [[buffer(5)]],
                                      device bfloat* XO [[buffer(6)]], const device bfloat* Y [[buffer(7)]],
                                      const device bfloat* SH [[buffer(8)]], constant int2& topk_rows [[buffer(9)]],
                                      uint2 tg [[threadgroup_position_in_grid]],
                                      uint2 tpos [[thread_position_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8][XR][S][MIX + 1];
  const int q = int(tg.x), r0 = int(tg.y) * XR, topk = topk_rows.x;
  const int live = min(XR, topk_rows.y - r0);
  const uint t = tpos.x;
  const int d0 = 1024 * q + 4 * int(t);
  float v[XR][S][4];
  for (int rr = 0; rr < XR; rr++) {
    if (rr >= live) {
      for (int k = 0; k < S; k++)
        for (int i = 0; i < 4; i++) v[rr][k][i] = 0.0f;
      continue;
    }
    const int r = r0 + rr;
    const device bfloat* x = X + size_t(r) * F;
    const device float* cm = COMB + r * S * S;
    for (int i = 0; i < 4; i++) {
      const int d = d0 + i;
      const float b = topk > 0 ? combined(Y, SH, topk, r, d) : float(BR[size_t(r) * D + d]);
      float xs[S];
      for (int jj = 0; jj < S; jj++) xs[jj] = float(x[jj * D + d]);
      for (int k = 0; k < S; k++) {
        float mm = 0.0f;
        for (int jj = 0; jj < S; jj++) mm += cm[jj * S + k] * xs[jj];
        const bfloat o = bfloat(POST[r * S + k] * b + mm);
        XO[size_t(r) * F + k * D + d] = o;
        v[rr][k][i] = float(o);
      }
    }
  }
  for (int k = 0; k < S; k++) {
    for (int m = 0; m < MIX; m++) {
      const device float* f = FN + size_t(m) * F + k * D + d0;
      const float f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3];
      for (int rr = 0; rr < XR; rr++) {
        const float s = simd_sum(f0 * v[rr][k][0] + f1 * v[rr][k][1] + f2 * v[rr][k][2] + f3 * v[rr][k][3]);
        if (lane == 0) red[sg][rr][k][m] = s;
      }
    }
    for (int rr = 0; rr < XR; rr++) {
      const float s = simd_sum(v[rr][k][0] * v[rr][k][0] + v[rr][k][1] * v[rr][k][1] + v[rr][k][2] * v[rr][k][2] +
                               v[rr][k][3] * v[rr][k][3]);
      if (lane == 0) red[sg][rr][k][MIX] = s;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = int(t); e < XR * S * (MIX + 1); e += 256) {
    const int m = e % (MIX + 1), k = (e / (MIX + 1)) % S, rr = e / ((MIX + 1) * S);
    if (rr >= live) continue;
    float s = 0.0f;
    for (int w = 0; w < 8; w++) s += red[w][rr][k][m];
    PART[(size_t(r0 + rr) * SLICES + k * DSLICES + q) * (MIX + 1) + m] = s;
  }
}

// A threadgroup of 1024 a row: (MIX) the boundary's pre, post and Sinkhorn comb; the streams collapsed and normed.
template <bool MIXES>
[[kernel]] void ds_hc_boundary(const device bfloat* X [[buffer(0)]], const device float* PART [[buffer(1)]],
                               const device float* BASE [[buffer(2)]], const device float* SCALE [[buffer(3)]],
                               const device float* PRE_IN [[buffer(4)]], const device bfloat* NORMW [[buffer(5)]],
                               constant HcArgs& a [[buffer(6)]], device float* PRE_OUT [[buffer(7)]],
                               device float* POST_OUT [[buffer(8)]], device float* COMB_OUT [[buffer(9)]],
                               device bfloat* NORMED [[buffer(10)]], uint r [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[32][MIX + 1];
  threadgroup float tot[MIX + 1];
  threadgroup float pre_in[S];
  const device bfloat* x = X + size_t(r) * F;
  // the mixes' totals beside the previous boundary's lagged collapse, which overlaps the split's rms reduction
  if (MIXES && int(t) <= MIX) {
    float s = 0.0f;
    for (int j = 0; j < SLICES; j++) s += PART[(size_t(r) * SLICES + j) * (MIX + 1) + t];
    tot[t] = s;
  }
  if (t < uint(S)) pre_in[t] = PRE_IN[r * S + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ys[5];
  float ss = 0.0f;
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    float y = 0.0f;
    for (int s = 0; s < S; s++) y += pre_in[s] * float(x[s * D + d]);
    ys[i] = bf_round(y);
    ss += ys[i] * ys[i];
  }
  const float s1 = simd_sum(ss);
  if (lane == 0) red[sg][0] = s1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the split on simdgroup 0: a pre and post a lane (0-3), a comb cell a lane (0-15), row and column sums by xor
  if (MIXES && sg == 0) {
    const float rs = precise::rsqrt(tot[MIX] / float(F) + a.eps);
    if (lane < uint(S)) {
      PRE_OUT[r * S + lane] = 1.0f / (1.0f + precise::exp(-(tot[lane] * rs * SCALE[0] + BASE[lane]))) + a.hc_eps;
      POST_OUT[r * S + lane] = 2.0f / (1.0f + precise::exp(-(tot[S + lane] * rs * SCALE[1] + BASE[S + lane])));
    }
    const bool act = lane < uint(S * S);
    const uint cell = act ? lane : 0;
    const float v = tot[2 * S + cell] * rs * SCALE[2] + BASE[2 * S + cell];
    float mx = max(v, simd_shuffle_xor(v, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
    const float e = precise::exp(v - mx);
    float row = e + simd_shuffle_xor(e, ushort(1));
    row = row + simd_shuffle_xor(row, ushort(2));
    float cm = act ? e / row + a.hc_eps : 0.0f;
    for (int it = 0; it < a.iters; it++) {
      if (it > 0) {
        float rsum = cm + simd_shuffle_xor(cm, ushort(1));
        rsum = rsum + simd_shuffle_xor(rsum, ushort(2));
        cm = act ? cm / (rsum + a.hc_eps) : 0.0f;
      }
      float csum = cm + simd_shuffle_xor(cm, ushort(4));
      csum = csum + simd_shuffle_xor(csum, ushort(8));
      cm = act ? cm / (csum + a.hc_eps) : 0.0f;
    }
    if (act) COMB_OUT[r * S * S + lane] = cm;
  }
  if (t == 32) {  // simdgroup 1, while simdgroup 0 runs the split
    float s = 0.0f;
    for (int k = 0; k < 32; k++) s += red[k][0];
    pre_in[0] = precise::rsqrt(s / float(D) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = pre_in[0];
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    NORMED[size_t(r) * D + d] = bfloat(float(NORMW[d]) * (ys[i] * inv));
  }
}

template [[host_name("ds_hc_boundary")]] [[kernel]] decltype(ds_hc_boundary<true>) ds_hc_boundary<true>;
template [[host_name("ds_hc_collapse")]] [[kernel]] decltype(ds_hc_boundary<false>) ds_hc_boundary<false>;

// ---------------------------------------------------------------- norms, projections, rope, quantize steps

struct RowsArgs {
  int n;         // row width
  int x_stride;  // values between rows
  int y_stride;
  float eps;
};

// y = norm(x) * w, rms over n (in place allowed); a threadgroup of 256 a row.
[[kernel]] void ds_rms(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                       device bfloat* Y [[buffer(2)]], constant RowsArgs& a [[buffer(3)]],
                       uint r [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8];
  const device bfloat* x = X + size_t(r) * a.x_stride;
  float ss = 0.0f;
  for (int i = int(t); i < a.n; i += 256) {
    const float v = float(x[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int k = 0; k < 8; k++) tot += red[k];
  const float inv = precise::rsqrt(tot / float(a.n) + a.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* y = Y + size_t(r) * a.y_stride;
  for (int i = int(t); i < a.n; i += 256) y[i] = bfloat(float(W[i]) * (float(x[i]) * inv));
}

struct GemvArgs {
  int rows;
  int n;
  int k;
  int group_n;   // outputs that share one x slice (n for an ordinary matrix; o_rank for wo_a's groups)
  int x_stride;  // values between x rows
  int y_stride;
};

// y[r, j] = sum_k x[r, (j / group_n) * k + kk] W[j, kk]: bf16 weights, fp32 sums; a column a simdgroup, 8 rows.
template <typename XT, typename YT>
[[kernel]] void ds_gemv(const device XT* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                        device YT* Y [[buffer(2)]], constant GemvArgs& a [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  const int j = int(tg.x) * int(nsg) + int(sg);
  if (j >= a.n) return;
  const int r0 = int(tg.y) * 8;
  const int nr = min(8, a.rows - r0);
  const device bfloat* w = W + size_t(j) * a.k;
  const int xo = (j / a.group_n) * a.k;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  // 8 consecutive inputs a lane a step (k is a multiple of 256 at every shape this family multiplies)
  for (int kk = int(lane) * 8; kk < a.k; kk += 256) {
    float wv[8];
    for (int u = 0; u < 8; u++) wv[u] = float(w[kk + u]);
    for (int i = 0; i < nr; i++) {
      const device XT* xr = X + size_t(r0 + i) * a.x_stride + xo + kk;
      float part = 0.0f;
      for (int u = 0; u < 8; u++) part += float(xr[u]) * wv[u];
      acc[i] += part;
    }
  }
  for (int i = 0; i < nr; i++) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) Y[size_t(r0 + i) * a.y_stride + j] = YT(s);
  }
}

// ds_gemv's sums for a chunk over one x slice (group_n = n): 4 columns a simdgroup, each x value read once for them.
template <typename XT, typename YT>
[[kernel]] void ds_gemv_cols(const device XT* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                             device YT* Y [[buffer(2)]], constant GemvArgs& a [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int NC = 4;
  const int j0 = (int(tg.x) * 8 + int(sg)) * NC;
  if (j0 >= a.n) return;
  const int nc = min(NC, a.n - j0);
  const int r0 = int(tg.y) * 8;
  const int nr = min(8, a.rows - r0);
  float acc[NC][8];
  for (int c = 0; c < NC; c++)
    for (int i = 0; i < 8; i++) acc[c][i] = 0.0f;
  for (int kk = int(lane) * 8; kk < a.k; kk += 256) {
    float wv[NC][8];
    for (int c = 0; c < NC; c++) {
      const device bfloat* w = W + size_t(min(j0 + c, a.n - 1)) * a.k + kk;
      for (int u = 0; u < 8; u++) wv[c][u] = float(w[u]);
    }
    for (int i = 0; i < nr; i++) {
      const device XT* xr = X + size_t(r0 + i) * a.x_stride + kk;
      float xv[8];
      for (int u = 0; u < 8; u++) xv[u] = float(xr[u]);
      for (int c = 0; c < NC; c++) {
        float part = 0.0f;
        for (int u = 0; u < 8; u++) part += xv[u] * wv[c][u];
        acc[c][i] += part;
      }
    }
  }
  for (int c = 0; c < nc; c++)
    for (int i = 0; i < nr; i++) {
      const float s = simd_sum(acc[c][i]);
      if (lane == 0) Y[size_t(r0 + i) * a.y_stride + j0 + c] = YT(s);
    }
}

template [[host_name("ds_gemv_cols_bf_f32")]] [[kernel]]
decltype(ds_gemv_cols<bfloat, float>) ds_gemv_cols<bfloat, float>;
template [[host_name("ds_gemv_bf_bf")]] [[kernel]] decltype(ds_gemv<bfloat, bfloat>) ds_gemv<bfloat, bfloat>;
template [[host_name("ds_gemv_bf_f32")]] [[kernel]] decltype(ds_gemv<bfloat, float>) ds_gemv<bfloat, float>;

// x = bf16(x * c) in place: a bf16 tensor times a Python float, as the reference scales the indexer's weights.
[[kernel]] void ds_scale(device bfloat* X [[buffer(0)]], constant float& c [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  X[i] = bfloat(float(X[i]) * c);
}

struct RopeArgs {
  int heads;
  int stride;   // values between heads (rows are heads * stride apart)
  int offset;   // the rotated dims' first, inside a head
  int inverse;
};

// Adjacent pairs of a head's last 64 dims rotated by position * freq (torch.polar's complex product), bf16 out.

// TAB: the (cos, sin) from the call's table (ds_rope_table) at FREQ's variant, the same floats as computed here.
template <bool TAB>
[[kernel]] void ds_rope(device bfloat* X [[buffer(0)]], const device int* POS [[buffer(1)]],
                        const device float* FREQ [[buffer(2)]], constant RopeArgs& a [[buffer(3)]],
                        uint3 g [[thread_position_in_grid]]) {
  const int i = int(g.x), h = int(g.y), r = int(g.z);
  device bfloat* x = X + (size_t(r) * a.heads + h) * a.stride + a.offset + 2 * i;
  float cs, sn;
  if (TAB) {
    const float2 t = ((const device float2*)FREQ)[r * 32 + i];
    cs = t.x;
    sn = a.inverse ? -t.y : t.y;
  } else {
    const float ang = float(POS[r]) * FREQ[i];
    cs = precise::cos(ang);
    sn = a.inverse ? -precise::sin(ang) : precise::sin(ang);
  }
  const float re = float(x[0]), im = float(x[1]);
  x[0] = bfloat(re * cs - im * sn);
  x[1] = bfloat(re * sn + im * cs);
}

template [[host_name("ds_rope")]] [[kernel]] decltype(ds_rope<false>) ds_rope<false>;
template [[host_name("ds_rope_t")]] [[kernel]] decltype(ds_rope<true>) ds_rope<true>;

// The call's RoPE table: (cos, sin) of POS[r] * freq[i] for both frequency variants, [variant][rows][32], once a

// call, so the kernels that rotate by those positions read it instead of each computing it again.
[[kernel]] void ds_rope_table(const device int* POS [[buffer(0)]], const device float* PLAIN [[buffer(1)]],
                              const device float* COMPRESSED [[buffer(2)]], device float2* TAB [[buffer(3)]],
                              constant int& rows [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
  const int i = int(g.x), r = int(g.y), v = int(g.z);
  const float ang = float(POS[r]) * (v ? COMPRESSED : PLAIN)[i];
  TAB[(size_t(v) * rows + r) * 32 + i] = float2(precise::cos(ang), precise::sin(ang));
}

// act_quant's fused quantize-dequantize: fp8 e4m3 with a power-of-two scale a block of 32, in place.
[[kernel]] void ds_fp8_qdq(device bfloat* X [[buffer(0)]], uint b [[thread_position_in_grid]]) {
  device bfloat* x = X + size_t(b) * 32;
  float amax = 0.0f;
  for (int i = 0; i < 32; i++) amax = max(amax, fabs(float(x[i])));
  const float s = pow2_ceil(max(amax, 1e-4f) / 448.0f);
  for (int i = 0; i < 32; i++) x[i] = bfloat(e4m3_round(clamp(float(x[i]) / s, -448.0f, 448.0f)) * s);
}

// fp4_act_quant's: e2m1 with a block of `block` (32 with an e8m0 scale, 16 with an e4m3 one), in place.
[[kernel]] void ds_fp4_qdq(device bfloat* X [[buffer(0)]], constant int& e4m3_scale [[buffer(1)]],
                           uint b [[thread_position_in_grid]]) {
  const int block = e4m3_scale ? 16 : 32;
  device bfloat* x = X + size_t(b) * block;
  float amax = 0.0f;
  for (int i = 0; i < block; i++) amax = max(amax, fabs(float(x[i])));
  const float s = e4m3_scale ? e4m3_round(max(amax, 6.0f * 0.001953125f) / 6.0f) : pow2_ceil(max(amax, 6.0f * 1.1754944e-38f) / 6.0f);
  for (int i = 0; i < block; i++) x[i] = bfloat(e2m1_round(clamp(float(x[i]) / s, -6.0f, 6.0f)) * s);
}

struct CopyArgs {
  int width;  // values a row
  int ring;   // destination rows (index = pos % ring); 0: index = pos
};

// Rows into a cache at their positions (a ring when `ring` > 0).
[[kernel]] void ds_put_rows(const device bfloat* SRC [[buffer(0)]], const device int* POS [[buffer(1)]],
                            device bfloat* DST [[buffer(2)]], constant CopyArgs& a [[buffer(3)]],
                            uint2 g [[thread_position_in_grid]]) {
  const int p = POS[g.y];
  const int at = a.ring > 0 ? p % a.ring : p;
  DST[size_t(at) * a.width + g.x] = SRC[size_t(g.y) * a.width + g.x];
}

// f32 rows into a ring at their positions.
[[kernel]] void ds_put_rows_f32(const device float* SRC [[buffer(0)]], const device int* POS [[buffer(1)]],
                                device float* DST [[buffer(2)]], constant CopyArgs& a [[buffer(3)]],
                                uint2 g [[thread_position_in_grid]]) {
  const int p = POS[g.y];
  DST[size_t(p % a.ring) * a.width + g.x] = SRC[size_t(g.y) * a.width + g.x];
}

// ---------------------------------------------------------------- compressor

struct PoolArgs {
  int first;  // the first group to pool
  int ring;   // the projection ring's rows
};

// Groups of 2 positions pooled by their learned softmax gate (fp32), bf16 out: OUT[g - first] for each group.
[[kernel]] void ds_pool2(const device float* KV [[buffer(0)]], const device float* SC [[buffer(1)]],
                         device bfloat* OUT [[buffer(2)]], constant PoolArgs& a [[buffer(3)]],
                         uint2 g [[thread_position_in_grid]]) {
  const int d = int(g.x), grp = a.first + int(g.y);
  const int p0 = (2 * grp) % a.ring, p1 = (2 * grp + 1) % a.ring;
  const float s0 = SC[size_t(p0) * HD + d], s1 = SC[size_t(p1) * HD + d];
  const float m = max(s0, s1);
  const float e0 = precise::exp(s0 - m), e1 = precise::exp(s1 - m);
  const float sum = e0 + e1;
  const float v = KV[size_t(p0) * HD + d] * (e0 / sum) + KV[size_t(p1) * HD + d] * (e1 / sum);
  OUT[size_t(g.y) * HD + d] = bfloat(v);
}

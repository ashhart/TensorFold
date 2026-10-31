// Idol's background remover (BiRefNet: Swin-L, ASPP-deformable decoder) in fp32, each tensor [rows, channels].

#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL_BR _Pragma("clang loop unroll(full)")

// erf in fp32 to within an ulp (musl's erff), for nn.GELU's exact form.
inline float br_erf(float x) {
  const uint ix = as_type<uint>(x) & 0x7fffffffu;
  const bool neg = as_type<uint>(x) >> 31;
  if (ix >= 0x7f800000u) return isnan(x) ? x : (neg ? -1.0f : 1.0f);
  if (ix < 0x3f580000u) {
    if (ix < 0x31800000u) return 0.125f * (8.0f * x + 1.0270333290e+00f * x);
    const float z = x * x;
    const float r = 1.2837916613e-01f + z * (-3.2504209876e-01f + z * (-2.8481749818e-02f + z * (-5.7702702470e-03f +
                    z * -2.3763017452e-05f)));
    const float s = 1.0f + z * (3.9791721106e-01f + z * (6.5022252500e-02f + z * (5.0813062117e-03f +
                    z * (1.3249473704e-04f + z * -3.9602282413e-06f))));
    return x + x * (r / s);
  }
  float y;
  if (ix < 0x40c00000u) {
    const float ax = fabs(x);
    if (ix < 0x3fa00000u) {
      const float s = ax - 1.0f;
      const float P = -2.3621185683e-03f + s * (4.1485610604e-01f + s * (-3.7220788002e-01f + s * (3.1834661961e-01f +
                      s * (-1.1089469492e-01f + s * (3.5478305072e-02f + s * -2.1663755178e-03f)))));
      const float Q = 1.0f + s * (1.0642088205e-01f + s * (5.4039794207e-01f + s * (7.1828655899e-02f +
                      s * (1.2617121637e-01f + s * (1.3637083583e-02f + s * 1.1984500103e-02f)))));
      y = 8.4506291151e-01f + P / Q;
    } else {
      const float s = 1.0f / (ax * ax);
      float R, S;
      if (ix < 0x4036db6du) {
        R = -9.8649440333e-03f + s * (-6.9385856390e-01f + s * (-1.0558626175e+01f + s * (-6.2375331879e+01f +
            s * (-1.6239666748e+02f + s * (-1.8460508728e+02f + s * (-8.1287437439e+01f + s * -9.8143291473e+00f))))));
        S = 1.0f + s * (1.9651271820e+01f + s * (1.3765776062e+02f + s * (4.3456588745e+02f + s * (6.4538726807e+02f +
            s * (4.2900814819e+02f + s * (1.0863500214e+02f + s * (6.5702495575e+00f + s * -6.0424413532e-02f)))))));
      } else {
        R = -9.8649431020e-03f + s * (-7.9928326607e-01f + s * (-1.7757955551e+01f + s * (-1.6063638306e+02f +
            s * (-6.3756646729e+02f + s * (-1.0250950928e+03f + s * -4.8351919556e+02f)))));
        S = 1.0f + s * (3.0338060379e+01f + s * (3.2579251099e+02f + s * (1.5367296143e+03f + s * (3.1998581543e+03f +
            s * (2.5530502930e+03f + s * (4.7452853394e+02f + s * -2.2440952301e+01f))))));
      }
      const float z = as_type<float>(as_type<uint>(ax) & 0xffffe000u);
      y = 1.0f - precise::exp(-z * z - 0.5625f) * precise::exp((z - ax) * (z + ax) + R / S) / ax;
    }
  } else {
    y = 1.0f - 0x1p-120f;
  }
  return neg ? -y : y;
}

inline float br_sigmoid(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

struct BrConv {
  uint rows;
  uint cols;
  uint cin;
  uint ldx;
  uint cout;
  uint ldw;
  uint k;
  uint ldy;
  uint yoff;
  uint flags;
  uint row0;
};

// y = conv_k(x) + b; flags: 1 BN (post), 2 ReLU, 4 GELU, 8 + r, 16 rows by map; w [k * k * cin, ldw] by tap
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void br_conv(
    const device float* x [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    const device float* post [[buffer(3)]], const device float* r [[buffer(4)]], const device int* map [[buffer(5)]],
    device float* y [[buffer(6)]], constant BrConv& p [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint BK = 32;
  constexpr uint lda = BK + 4;
  // an implicit GEMM, a tap at a time: a 64 x 64 tile a threadgroup, fp32 sums
  threadgroup float As[64 * lda];
  threadgroup float Cs[64 * 68];
  const uint pixels = p.rows * p.cols;
  const uint m0 = tg.y * 64;
  const uint n0 = tg.x * 64;
  const uint arow = t / 2;
  const uint acol = (t % 2) * 16;
  const uint pix = m0 + arow;
  const int oy = int(pix / p.cols);
  const int ox = int(pix % p.cols);
  const int pad = int(p.k / 2);
  const uint sr = (sg / 2) * 32;
  const uint sc = (sg % 2) * 32;
  // sums in blocks of br_block terms added into the total: one running sum over 50k terms drifts twice torch's
  constexpr uint br_block = 256;
  simdgroup_float8x8 acc[4][4];
  simdgroup_float8x8 part[4][4];
  TF_UNROLL_BR
  for (uint i = 0; i < 4; i++) {
    TF_UNROLL_BR
    for (uint j = 0; j < 4; j++) {
      acc[i][j] = simdgroup_float8x8(0.0f);
      part[i][j] = simdgroup_float8x8(0.0f);
    }
  }
  const uint K = p.k * p.k * p.cin;
  for (uint k0 = 0; k0 < K; k0 += BK) {
    const uint tap = k0 / p.cin;
    const int iy = oy + int(tap / p.k) - pad;
    const int ix = ox + int(tap % p.k) - pad;
    threadgroup float4* dst = (threadgroup float4*)(As + arow * lda + acol);
    if (pix < pixels && iy >= 0 && ix >= 0 && iy < int(p.rows) && ix < int(p.cols)) {
      const device float4* src =
          (const device float4*)(x + ulong(uint(iy) * p.cols + uint(ix)) * p.ldx + k0 % p.cin + acol);
      TF_UNROLL_BR
      for (uint i = 0; i < 4; i++) dst[i] = src[i];
    } else {
      TF_UNROLL_BR
      for (uint i = 0; i < 4; i++) dst[i] = float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device float* wk = w + ulong(k0) * p.ldw + n0 + sc;
    TF_UNROLL_BR
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 am[4];
      simdgroup_float8x8 bm[4];
      TF_UNROLL_BR
      for (uint i = 0; i < 4; i++) simdgroup_load(am[i], As + (sr + i * 8) * lda + kk, lda);
      TF_UNROLL_BR
      for (uint j = 0; j < 4; j++) simdgroup_load(bm[j], wk + ulong(kk) * p.ldw + j * 8, p.ldw);
      TF_UNROLL_BR
      for (uint i = 0; i < 4; i++) {
        TF_UNROLL_BR
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(part[i][j], am[i], bm[j], part[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((k0 + BK) % br_block == 0 || k0 + BK >= K) {
      TF_UNROLL_BR
      for (uint i = 0; i < 4; i++) {
        TF_UNROLL_BR
        for (uint j = 0; j < 4; j++) {
          acc[i][j].thread_elements()[0] += part[i][j].thread_elements()[0];
          acc[i][j].thread_elements()[1] += part[i][j].thread_elements()[1];
          part[i][j] = simdgroup_float8x8(0.0f);
        }
      }
    }
  }
  TF_UNROLL_BR
  for (uint i = 0; i < 4; i++) {
    TF_UNROLL_BR
    for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], Cs + (sr + i * 8) * 68 + sc + j * 8, 68);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = t; e < 64 * 64; e += 128) {
    const uint row = m0 + e / 64;
    const uint col = n0 + e % 64;
    if (row >= pixels || col >= p.cout) continue;
    int out = int(p.row0 + row);
    if (p.flags & 16) out = map[row];
    if (out < 0) continue;
    const ulong at = ulong(out) * p.ldy + p.yoff + col;
    float v = Cs[(e / 64) * 68 + e % 64] + b[col];
    if (p.flags & 1) v = (v - post[col]) * post[p.ldw + col] + post[2 * p.ldw + col];
    if (p.flags & 2) v = v < 0.0f ? 0.0f : v;
    if (p.flags & 4) v = (v * 0.5f) * (1.0f + br_erf(v * 0.70710677f));
    if (p.flags & 8) v = r[at] + v;
    y[at] = v;
  }
}

struct BrNorm {
  uint rows;
  uint width;
  uint mode;
  uint src_w;
  uint ldy;
  uint yoff;
  float eps;
};

// Row r at channel c: mode 0 row r, 1 row map[r], 2 PatchMerging's [x0 | x1 | x2 | x3] at r's 2 x 2 patch
inline float br_norm_at(const device float* x, const device int* map, constant BrNorm& p, uint r, uint c) {
  if (p.mode == 0) return x[ulong(r) * p.width + c];
  if (p.mode == 1) return x[ulong(map[r]) * p.width + c];
  const uint q = p.width / 4;
  const uint part = c / q;
  const uint half_w = p.src_w / 2;
  const uint sy = (r / half_w) * 2 + (part & 1);
  const uint sx = (r % half_w) * 2 + (part >> 1);
  return x[ulong(sy * p.src_w + sx) * q + c % q];
}

// y = LayerNorm(row) * g + beta, two fp32 passes; a map's negative rows are zeros (torch pads after the norm)
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void br_norm(
    const device float* x [[buffer(0)]], const device int* map [[buffer(1)]], const device float* g [[buffer(2)]],
    const device float* beta [[buffer(3)]], device float* y [[buffer(4)]], constant BrNorm& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8];
  device float* out = y + ulong(row) * p.ldy + p.yoff;
  if (p.mode == 1 && map[row] < 0) {
    for (uint c = t; c < p.width; c += 256) out[c] = 0.0f;
    return;
  }
  float s = 0.0f;
  for (uint c = t; c < p.width; c += 256) s += br_norm_at(x, map, p, row, c);
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float all = 0.0f;
  TF_UNROLL_BR
  for (uint i = 0; i < 8; i++) all += part[i];
  const float mean = all / float(p.width);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float d2 = 0.0f;
  for (uint c = t; c < p.width; c += 256) {
    const float d = br_norm_at(x, map, p, row, c) - mean;
    d2 += d * d;
  }
  d2 = simd_sum(d2);
  if (lane == 0) part[sg] = d2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  all = 0.0f;
  TF_UNROLL_BR
  for (uint i = 0; i < 8; i++) all += part[i];
  const float rstd = 1.0f / precise::sqrt(all / float(p.width) + p.eps);
  for (uint c = t; c < p.width; c += 256) out[c] = (br_norm_at(x, map, p, row, c) - mean) * rstd * g[c] + beta[c];
}

struct BrWindow {
  uint width;
  uint heads;
  uint rows;
  uint cols;
  uint shift;
  float scale;
};

// Token i's region (0 to 8) in the shifted grid: rows and columns split at size - 12 and size - shift
inline uint br_region(uint i, uint wy, uint wx, constant BrWindow& p) {
  const uint y = wy * 12 + i / 12;
  const uint x = wx * 12 + i % 12;
  const uint ry = y < p.rows - 12 ? 0u : (y < p.rows - p.shift ? 1u : 2u);
  const uint rx = x < p.cols - 12 ? 0u : (x < p.cols - p.shift ? 1u : 2u);
  return ry * 3 + rx;
}

// A 12 x 12 window, one head, 48 queries: softmax(q scale k^T + bias (+ -100 across regions)) v
[[max_total_threads_per_threadgroup(192)]] [[kernel]] void br_window(
    const device float* qkv [[buffer(0)]], const device float* table [[buffer(1)]], device float* o [[buffer(2)]],
    constant BrWindow& p [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint N = 144;
  constexpr uint QN = 48;
  constexpr uint LS = 148;
  threadgroup float S[QN * LS];
  const uint h = tg.y;
  const uint win = tg.z;
  const uint ld = 3 * p.width;
  const uint base = win * N;
  const uint first = tg.x * QN;
  const uint qcol = h * 32;
  for (uint tile = sg; tile < (QN / 8) * (N / 8); tile += 6) {
    const uint ti = tile / (N / 8);
    const uint tj = tile % (N / 8);
    simdgroup_float8x8 s = simdgroup_float8x8(0.0f);
    TF_UNROLL_BR
    for (uint d = 0; d < 32; d += 8) {
      simdgroup_float8x8 qm;
      simdgroup_float8x8 km;
      simdgroup_load(qm, qkv + ulong(base + first + ti * 8) * ld + qcol + d, ld);
      qm.thread_elements()[0] *= p.scale;
      qm.thread_elements()[1] *= p.scale;
      simdgroup_load(km, qkv + ulong(base + tj * 8) * ld + p.width + qcol + d, ld, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s, qm, km, s);
    }
    simdgroup_store(s, S + ti * 8 * LS + tj * 8, LS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // four threads a query row
  const uint row = t / 4;
  const uint sub = t % 4;
  const uint qi = first + row;
  const uint wy = win / (p.cols / 12);
  const uint wx = win % (p.cols / 12);
  threadgroup float* sr = S + row * LS;
  const uint mine = p.shift ? br_region(qi, wy, wx, p) : 0u;
  float most = -INFINITY;
  for (uint j = sub; j < N; j += 4) {
    const uint at = ((qi / 12 + 11 - j / 12) * 23 + (qi % 12 + 11 - j % 12)) * p.heads + h;
    float v = sr[j] + table[at];
    if (p.shift) v = v + (br_region(j, wy, wx, p) == mine ? 0.0f : -100.0f);
    sr[j] = v;
    most = max(most, v);
  }
  most = max(most, simd_shuffle_xor(most, 1));
  most = max(most, simd_shuffle_xor(most, 2));
  float total = 0.0f;
  for (uint j = sub; j < N; j += 4) {
    const float e = precise::exp(sr[j] - most);
    sr[j] = e;
    total += e;
  }
  total += simd_shuffle_xor(total, 1);
  total += simd_shuffle_xor(total, 2);
  for (uint j = sub; j < N; j += 4) sr[j] = sr[j] / total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint tile = sg; tile < (QN / 8) * 4; tile += 6) {
    const uint ti = tile / 4;
    const uint tj = tile % 4;
    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
    for (uint kk = 0; kk < N; kk += 8) {
      simdgroup_float8x8 pm;
      simdgroup_float8x8 vm;
      simdgroup_load(pm, S + ti * 8 * LS + kk, LS);
      simdgroup_load(vm, qkv + ulong(base + kk) * ld + 2 * p.width + qcol + tj * 8, ld);
      simdgroup_multiply_accumulate(acc, pm, vm, acc);
    }
    simdgroup_store(acc, o + ulong(base + first + ti * 8) * p.width + qcol + tj * 8, p.width);
  }
}

struct BrResize {
  uint in_h;
  uint in_w;
  uint out_h;
  uint out_w;
  uint channels;
  uint ldx;
  uint xoff;
  uint ldy;
  uint yoff;
};

// y = F.interpolate(x, bilinear, align_corners=True): x [in_h * in_w, ldx] from xoff, y [out_h * out_w, ldy] from yoff.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void br_resize(
    const device float* x [[buffer(0)]], device float* y [[buffer(1)]], constant BrResize& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.channels || at.y >= p.out_h * p.out_w) return;
  const uint oy = at.y / p.out_w;
  const uint ox = at.y % p.out_w;
  const float sh = p.out_h > 1 ? float(p.in_h - 1) / float(p.out_h - 1) : 0.0f;
  const float sw = p.out_w > 1 ? float(p.in_w - 1) / float(p.out_w - 1) : 0.0f;
  const float fy = sh * float(oy);
  const float fx = sw * float(ox);
  const uint y0 = min(uint(fy), p.in_h - 1);
  const uint x0 = min(uint(fx), p.in_w - 1);
  const uint y1 = min(y0 + 1, p.in_h - 1);
  const uint x1 = min(x0 + 1, p.in_w - 1);
  const float ly = min(max(fy - float(y0), 0.0f), 1.0f);
  const float lx = min(max(fx - float(x0), 0.0f), 1.0f);
  const device float* src = x + p.xoff + at.x;
  const float v00 = src[ulong(y0 * p.in_w + x0) * p.ldx];
  const float v01 = src[ulong(y0 * p.in_w + x1) * p.ldx];
  const float v10 = src[ulong(y1 * p.in_w + x0) * p.ldx];
  const float v11 = src[ulong(y1 * p.in_w + x1) * p.ldx];
  const float top = (1.0f - lx) * v00 + lx * v01;
  const float bottom = (1.0f - lx) * v10 + lx * v11;
  y[ulong(at.y) * p.ldy + p.yoff + at.x] = (1.0f - ly) * top + ly * bottom;
}

struct BrTiles {
  uint out_h;
  uint out_w;
  uint src_w;
  uint channels;
  uint a;
  uint b;
  uint mode;
  uint cpad;
};

// Channel ch * a * b + i * b + j (0 past channels * a * b): mode 0 a x b patches, mode 1 image2patches' tiles
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void br_tiles(
    const device float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant BrTiles& p [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.cpad || at.y >= p.out_h * p.out_w) return;
  const ulong o = ulong(at.y) * p.cpad + at.x;
  const uint ab = p.a * p.b;
  if (at.x >= p.channels * ab) {
    dst[o] = 0.0f;
    return;
  }
  const uint ch = at.x / ab;
  const uint i = (at.x % ab) / p.b;
  const uint j = at.x % p.b;
  const uint oy = at.y / p.out_w;
  const uint ox = at.y % p.out_w;
  const uint sy = p.mode == 0 ? oy * p.a + i : i * p.out_h + oy;
  const uint sx = p.mode == 0 ? ox * p.b + j : j * p.out_w + ox;
  dst[o] = src[ulong(sy * p.src_w + sx) * p.channels + ch];
}

struct BrConv3 {
  uint rows;
  uint cols;
  uint cout;
};

// y [pixels, cout] = conv3x3(x) + b of a 3-channel picture x [pixels, 3], zero padded; w [27, cout] by (tap, channel).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void br_conv3(
    const device float* x [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    device float* y [[buffer(3)]], constant BrConv3& p [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.cout || at.y >= p.rows * p.cols) return;
  const int py = int(at.y / p.cols);
  const int px = int(at.y % p.cols);
  float s = 0.0f;
  TF_UNROLL_BR
  for (int ky = 0; ky < 3; ky++) {
    TF_UNROLL_BR
    for (int kx = 0; kx < 3; kx++) {
      const int iy = py + ky - 1;
      const int ix = px + kx - 1;
      if (iy < 0 || ix < 0 || iy >= int(p.rows) || ix >= int(p.cols)) continue;
      const device float* v = x + ulong(uint(iy) * p.cols + uint(ix)) * 3;
      TF_UNROLL_BR
      for (int c = 0; c < 3; c++) s = fma(v[c], w[((ky * 3 + kx) * 3 + c) * p.cout + at.x], s);
    }
  }
  y[ulong(at.y) * p.cout + at.x] = s + b[at.x];
}

struct BrDeform {
  uint rows;
  uint cols;
  uint k;
  uint first;
  uint count;
  uint ldo;
};

// torchvision's modulated deform_conv2d columns [count, k * k * 64]: 2 sigmoid(m) x the moved bilinear sample
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void br_deform(
    const device float* x [[buffer(0)]], const device float* offs [[buffer(1)]], device float* col [[buffer(2)]],
    constant BrDeform& p [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const uint taps = p.k * p.k;
  if (at.x >= taps || at.y >= p.count) return;
  const uint pix = p.first + at.y;
  const int oy = int(pix / p.cols);
  const int ox = int(pix % p.cols);
  const int pad = int(p.k / 2);
  const device float* off = offs + ulong(pix) * p.ldo;
  const float mask = 2.0f * br_sigmoid(off[2 * taps + at.x]);
  const float h = float(oy - pad + int(at.x / p.k)) + off[2 * at.x];
  const float w = float(ox - pad + int(at.x % p.k)) + off[2 * at.x + 1];
  device float4* out = (device float4*)(col + ulong(at.y) * taps * 64 + at.x * 64);
  const int H = int(p.rows);
  const int W = int(p.cols);
  if (!(h > -1.0f && h < float(H) && w > -1.0f && w < float(W))) {
    TF_UNROLL_BR
    for (uint c = 0; c < 16; c++) out[c] = mask * float4(0.0f);
    return;
  }
  const int hl = int(floor(h));
  const int wl = int(floor(w));
  const float lh = h - float(hl);
  const float lw = w - float(wl);
  const float hh = 1.0f - lh;
  const float hw = 1.0f - lw;
  const float w1 = hh * hw;
  const float w2 = hh * lw;
  const float w3 = lh * hw;
  const float w4 = lh * lw;
  const bool top = hl >= 0;
  const bool bottom = hl + 1 <= H - 1;
  const bool left = wl >= 0;
  const bool right = wl + 1 <= W - 1;
  const device float4* r1 = (const device float4*)(x + ulong(max(hl, 0) * W + max(wl, 0)) * 64);
  const device float4* r2 = (const device float4*)(x + ulong(max(hl, 0) * W + min(wl + 1, W - 1)) * 64);
  const device float4* r3 = (const device float4*)(x + ulong(min(hl + 1, H - 1) * W + max(wl, 0)) * 64);
  const device float4* r4 = (const device float4*)(x + ulong(min(hl + 1, H - 1) * W + min(wl + 1, W - 1)) * 64);
  TF_UNROLL_BR
  for (uint c = 0; c < 16; c++) {
    const float4 v1 = top && left ? r1[c] : float4(0.0f);
    const float4 v2 = top && right ? r2[c] : float4(0.0f);
    const float4 v3 = bottom && left ? r3[c] : float4(0.0f);
    const float4 v4 = bottom && right ? r4[c] : float4(0.0f);
    out[c] = mask * (((w1 * v1 + w2 * v2) + w3 * v3) + w4 * v4);
  }
}

struct BrPool {
  uint pixels;
  uint channels;
};

// mean [channels] of x [pixels, channels] over its pixels: a threadgroup a channel.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void br_pool(
    const device float* x [[buffer(0)]], device float* mean [[buffer(1)]], constant BrPool& p [[buffer(2)]],
    uint ch [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8];
  float s = 0.0f;
  for (uint i = t; i < p.pixels; i += 256) s += x[ulong(i) * p.channels + ch];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  float all = 0.0f;
  TF_UNROLL_BR
  for (uint i = 0; i < 8; i++) all += part[i];
  mean[ch] = all / float(p.pixels);
}

struct BrGate {
  uint pixels;
  uint channels;
};

// x [pixels, channels] *= sigmoid(g . w + b), g [pixels, 16] (the gradient attention's 1x1 to one channel).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void br_gate(
    const device float* g [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    device float* x [[buffer(3)]], constant BrGate& p [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.channels || at.y >= p.pixels) return;
  float s = 0.0f;
  TF_UNROLL_BR
  for (uint i = 0; i < 16; i++) s = fma(g[ulong(at.y) * 16 + i], w[i], s);
  const ulong o = ulong(at.y) * p.channels + at.x;
  x[o] = x[o] * br_sigmoid(s + b[0]);
}

struct BrHead {
  uint in_h;
  uint in_w;
  uint out_h;
  uint out_w;
  uint cp;
  uint cq;
};

// logits = conv_out1 of [bilinear (align_corners) of a [in, cp] | q [pixels, cq]], w [cp + cq]
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void br_head(
    const device float* a [[buffer(0)]], const device float* q [[buffer(1)]], const device float* w [[buffer(2)]],
    const device float* b [[buffer(3)]], device float* y [[buffer(4)]], constant BrHead& p [[buffer(5)]],
    uint at [[thread_position_in_grid]]) {
  if (at >= p.out_h * p.out_w) return;
  const uint oy = at / p.out_w;
  const uint ox = at % p.out_w;
  const float fy = (float(p.in_h - 1) / float(p.out_h - 1)) * float(oy);
  const float fx = (float(p.in_w - 1) / float(p.out_w - 1)) * float(ox);
  const uint y0 = min(uint(fy), p.in_h - 1);
  const uint x0 = min(uint(fx), p.in_w - 1);
  const uint y1 = min(y0 + 1, p.in_h - 1);
  const uint x1 = min(x0 + 1, p.in_w - 1);
  const float ly = min(max(fy - float(y0), 0.0f), 1.0f);
  const float lx = min(max(fx - float(x0), 0.0f), 1.0f);
  const device float* s00 = a + ulong(y0 * p.in_w + x0) * p.cp;
  const device float* s01 = a + ulong(y0 * p.in_w + x1) * p.cp;
  const device float* s10 = a + ulong(y1 * p.in_w + x0) * p.cp;
  const device float* s11 = a + ulong(y1 * p.in_w + x1) * p.cp;
  float s = 0.0f;
  for (uint c = 0; c < p.cp; c++) {
    const float top = (1.0f - lx) * s00[c] + lx * s01[c];
    const float bottom = (1.0f - lx) * s10[c] + lx * s11[c];
    s = fma((1.0f - ly) * top + ly * bottom, w[c], s);
  }
  const device float* qr = q + ulong(at) * p.cq;
  for (uint c = 0; c < p.cq; c++) s = fma(qr[c], w[p.cp + c], s);
  y[at] = s + b[0];
}

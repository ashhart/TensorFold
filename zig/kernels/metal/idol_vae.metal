// Idol's image encoder (FLUX.2's VAE encoder) in fp32, each picture tensor [rows, columns, channels].

#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL_IV _Pragma("clang loop unroll(full)")

struct IvConvIn {
  uint rows;
  uint cols;
};

// y [pixels, 128] = conv3x3(2 x - 1) + b of the picture x [pixels, 3] in [0, 1], zero padded; w [27, 128] by tap.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void iv_conv_in(
    const device float* x [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    device float* y [[buffer(3)]], constant IvConvIn& p [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  if (at.y >= p.rows * p.cols) return;
  const int py = int(at.y / p.cols);
  const int px = int(at.y % p.cols);
  float s = 0.0f;
  TF_UNROLL_IV
  for (int ky = 0; ky < 3; ky++) {
    TF_UNROLL_IV
    for (int kx = 0; kx < 3; kx++) {
      const int iy = py + ky - 1;
      const int ix = px + kx - 1;
      if (iy < 0 || ix < 0 || iy >= int(p.rows) || ix >= int(p.cols)) continue;
      const device float* v = x + ulong(uint(iy) * p.cols + uint(ix)) * 3;
      TF_UNROLL_IV
      for (int c = 0; c < 3; c++) s = fma(v[c] * 2.0f - 1.0f, w[((ky * 3 + kx) * 3 + c) * 128 + at.x], s);
    }
  }
  y[ulong(at.y) * 128 + at.x] = s + b[at.x];
}

struct IvConv {
  uint rows;
  uint cols;
  uint in_rows;
  uint in_cols;
  uint cin;
  uint cout;
  uint ldw;
  uint taps;
  uint add;
};

// y = conv(x) + b (+ r): a 1x1 (taps 1) or a 3x3 of stride 2 padded right and below (taps 9), w [taps * cin, ldw].
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void iv_conv(
    const device float* x [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    const device float* r [[buffer(3)]], device float* y [[buffer(4)]], constant IvConv& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint BK = 32;
  constexpr uint lda = BK + 4;
  // an implicit GEMM, x [in_rows, in_cols, cin] read a tap at a time: a 64 x 64 tile a threadgroup, fp32 sums
  threadgroup float As[64 * lda];
  threadgroup float Cs[64 * 68];
  const uint pixels = p.rows * p.cols;
  const uint m0 = tg.y * 64;
  const uint n0 = tg.x * 64;
  const uint arow = t / 2;
  const uint acol = (t % 2) * 16;
  const uint pix = m0 + arow;
  const uint oy = pix / p.cols;
  const uint ox = pix % p.cols;
  const uint stride = p.taps == 9 ? 2 : 1;
  const uint sr = (sg / 2) * 32;
  const uint sc = (sg % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  TF_UNROLL_IV
  for (uint i = 0; i < 4; i++) {
    TF_UNROLL_IV
    for (uint j = 0; j < 4; j++) acc[i][j] = simdgroup_float8x8(0.0f);
  }
  const uint K = p.taps * p.cin;
  for (uint k0 = 0; k0 < K; k0 += BK) {
    const uint tap = k0 / p.cin;
    const uint iy = oy * stride + tap / 3;
    const uint ix = ox * stride + tap % 3;
    threadgroup float4* dst = (threadgroup float4*)(As + arow * lda + acol);
    if (pix < pixels && iy < p.in_rows && ix < p.in_cols) {
      const device float4* src = (const device float4*)(x + ulong(iy * p.in_cols + ix) * p.cin + k0 % p.cin + acol);
      TF_UNROLL_IV
      for (uint i = 0; i < 4; i++) dst[i] = src[i];
    } else {
      TF_UNROLL_IV
      for (uint i = 0; i < 4; i++) dst[i] = float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device float* wk = w + ulong(k0) * p.ldw + n0 + sc;
    TF_UNROLL_IV
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 am[4];
      simdgroup_float8x8 bm[4];
      TF_UNROLL_IV
      for (uint i = 0; i < 4; i++) simdgroup_load(am[i], As + (sr + i * 8) * lda + kk, lda);
      TF_UNROLL_IV
      for (uint j = 0; j < 4; j++) simdgroup_load(bm[j], wk + ulong(kk) * p.ldw + j * 8, p.ldw);
      TF_UNROLL_IV
      for (uint i = 0; i < 4; i++) {
        TF_UNROLL_IV
        for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  TF_UNROLL_IV
  for (uint i = 0; i < 4; i++) {
    TF_UNROLL_IV
    for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], Cs + (sr + i * 8) * 68 + sc + j * 8, 68);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = t; e < 64 * 64; e += 128) {
    const uint row = m0 + e / 64;
    const uint col = n0 + e % 64;
    if (row >= pixels || col >= p.cout) continue;
    const ulong at = ulong(row) * p.cout + col;
    float v = Cs[(e / 64) * 68 + e % 64] + b[col];
    if (p.add) v = v + r[at];
    y[at] = v;
  }
}

// The patch rows (or columns) a, b and b's sign that B^T's row i combines: (0, 2, -), (1, 2, +), (2, 1, -), (1, 3, -).
inline uint iv_wa(uint i) { return i == 2 ? 2 : (i == 0 ? 0 : 1); }
inline uint iv_wb(uint i) { return i == 0 ? 2 : (i == 3 ? 3 : (i == 1 ? 2 : 1)); }
inline float iv_ws(uint i) { return i == 1 ? 1.0f : -1.0f; }

// y = conv3x3(x) + b (+ r) by Winograd F(2x2, 3x3), u = G g G^T [16, cin, ldw]; 16 tiles by 64 channels a group.
[[max_total_threads_per_threadgroup(512)]] [[kernel]] void iv_wino(
    const device float* x [[buffer(0)]], const device float* u [[buffer(1)]], const device float* b [[buffer(2)]],
    const device float* r [[buffer(3)]], device float* y [[buffer(4)]], constant IvConv& p [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint NF = 8;
  constexpr uint BK = 16;
  constexpr uint TILES = 16;
  constexpr uint PR = 6;
  constexpr uint PC = 18;
  // the input patch [6, 18, 16]; the epilogue's [16 points, 16 tiles, 32 channels] after
  threadgroup float buf[16 * 16 * 32];
  const uint n0 = tg.x * NF * 8;
  const int y0 = int(tg.y) * 4;
  const int x0 = int(tg.z) * 16;
  // this lane's two elements of every 8 x 8 fragment: row fm, columns fn and fn + 1
  const uint fm = ((lane / 4) & 4) + (lane / 2) % 4;
  const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
  const uint pi = sg / 4;
  const uint pj = sg % 4;
  simdgroup_float8x8 acc[2][NF];
  TF_UNROLL_IV
  for (uint i = 0; i < 2; i++) {
    TF_UNROLL_IV
    for (uint j = 0; j < NF; j++) acc[i][j] = simdgroup_float8x8(0.0f);
  }
  for (uint c0 = 0; c0 < p.cin; c0 += BK) {
    for (uint e = t; e < PR * PC * (BK / 4); e += 512) {
      const uint px = e / (BK / 4);
      const uint q = e % (BK / 4);
      const int iy = y0 - 1 + int(px / PC);
      const int ix = x0 - 1 + int(px % PC);
      float4 v = float4(0.0f);
      if (iy >= 0 && ix >= 0 && iy < int(p.rows) && ix < int(p.cols))
        v = *(const device float4*)(x + ulong(uint(iy) * p.cols + uint(ix)) * p.cin + c0 + q * 4);
      *(threadgroup float4*)(buf + px * BK + q * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL_IV
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 vm[2];
      TF_UNROLL_IV
      for (uint i = 0; i < 2; i++) {
        // B^T d B at this point, for tile i * 8 + fm and channels fn, fn + 1
        const uint tile = i * 8 + fm;
        const threadgroup float* d = buf + (((tile / 8) * 2) * PC + (tile % 8) * 2) * BK + kk + fn;
        TF_UNROLL_IV
        for (uint c = 0; c < 2; c++) {
          const threadgroup float* q = d + c;
          const float wa = q[(iv_wa(pi) * PC + iv_wa(pj)) * BK] + iv_ws(pi) * q[(iv_wb(pi) * PC + iv_wa(pj)) * BK];
          const float wb = q[(iv_wa(pi) * PC + iv_wb(pj)) * BK] + iv_ws(pi) * q[(iv_wb(pi) * PC + iv_wb(pj)) * BK];
          vm[i].thread_elements()[c] = wa + iv_ws(pj) * wb;
        }
      }
      const device float* up_ = u + (ulong(sg) * p.cin + c0 + kk) * p.ldw + n0;
      TF_UNROLL_IV
      for (uint j = 0; j < NF; j++) {
        simdgroup_float8x8 um;
        simdgroup_load(um, up_ + j * 8, p.ldw);
        TF_UNROLL_IV
        for (uint i = 0; i < 2; i++) simdgroup_multiply_accumulate(acc[i][j], vm[i], um, acc[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // the output transform A^T M A, 32 channels at a time
  for (uint h = 0; h < NF / 4; h++) {
    TF_UNROLL_IV
    for (uint i = 0; i < 2; i++) {
      TF_UNROLL_IV
      for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][h * 4 + j], buf + (sg * TILES + i * 8) * 32 + j * 8, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      // one tile and channel a thread: its 16 points, then the four outputs A^T M A, rows then columns
      const uint co = t % 32;
      const uint tile = t / 32;
      const uint col = n0 + h * 32 + co;
      const threadgroup float* m = buf + tile * 32 + co;
      float s[4][2];
      TF_UNROLL_IV
      for (uint j = 0; j < 4; j++) {
        const float m0 = m[(0 * 4 + j) * TILES * 32];
        const float m1 = m[(1 * 4 + j) * TILES * 32];
        const float m2 = m[(2 * 4 + j) * TILES * 32];
        const float m3 = m[(3 * 4 + j) * TILES * 32];
        s[j][0] = (m0 + m1) + m2;
        s[j][1] = (m1 - m2) - m3;
      }
      const int ty = y0 + int(tile / 8) * 2;
      const int tx = x0 + int(tile % 8) * 2;
      TF_UNROLL_IV
      for (uint rr = 0; rr < 2; rr++) {
        const float out[2] = {(s[0][rr] + s[1][rr]) + s[2][rr], (s[1][rr] - s[2][rr]) - s[3][rr]};
        TF_UNROLL_IV
        for (uint cc = 0; cc < 2; cc++) {
          const int oy = ty + int(rr);
          const int ox = tx + int(cc);
          if (col >= p.cout || oy >= int(p.rows) || ox >= int(p.cols)) continue;
          const ulong at = ulong(uint(oy) * p.cols + uint(ox)) * p.cout + col;
          float v = out[cc] + b[col];
          if (p.add) v = v + r[at];
          y[at] = v;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct IvNorm {
  uint pixels;
  uint channels;
  uint silu;
};

// y = GroupNorm(x) by stats (mean, rstd per group) with g, beta, then SiLU when silu; x, y [pixels, channels].
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void iv_norm(
    const device float* x [[buffer(0)]], const device float* stats [[buffer(1)]], const device float* g [[buffer(2)]],
    const device float* beta [[buffer(3)]], device float* y [[buffer(4)]], constant IvNorm& p [[buffer(5)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.y >= p.pixels || at.x >= p.channels) return;
  const uint grp = at.x / (p.channels / 32);
  const ulong i = ulong(at.y) * p.channels + at.x;
  float v = (x[i] - stats[grp]) * stats[32 + grp];
  v = v * g[at.x] + beta[at.x];
  y[i] = p.silu ? v / (1.0f + precise::exp(-v)) : v;
}

struct IvGroups {
  uint pixels;
  uint channels;
  uint chunk;
  uint chunks;
  uint centered;
  float eps;
};

// One chunk of pixels' sums of x a group, or of (x - mean)^2 when centered, into part [32, chunks]: a thread a channel.
[[max_total_threads_per_threadgroup(512)]] [[kernel]] void iv_group_part(
    const device float* x [[buffer(0)]], const device float* stats [[buffer(1)]], device float* part [[buffer(2)]],
    constant IvGroups& p [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]]) {
  threadgroup float sums[512];
  const uint size = p.channels / 32;
  const uint first = tg * p.chunk;
  const uint last = min(first + p.chunk, p.pixels);
  const float mean = p.centered ? stats[t / size] : 0.0f;
  float s = 0.0f;
  for (uint px = first; px < last; px++) {
    const float v = x[ulong(px) * p.channels + t];
    const float d = v - mean;
    s += p.centered ? d * d : v;
  }
  sums[t] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 32) {
    float all = 0.0f;
    for (uint c = 0; c < size; c++) all += sums[t * size + c];
    part[t * p.chunks + tg] = all;
  }
}

// stats from part's chunk sums, a simdgroup a group: mean[32] when not centered, else rstd[32] = rsqrt(var + eps).
[[max_total_threads_per_threadgroup(1024)]] [[kernel]] void iv_group_finish(
    const device float* part [[buffer(0)]], device float* stats [[buffer(1)]], constant IvGroups& p [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  float s = 0.0f;
  for (uint i = lane; i < p.chunks; i += 32) s += part[sg * p.chunks + i];
  s = simd_sum(s);
  if (lane != 0) return;
  const float n = float(p.pixels) * float(p.channels / 32);
  if (p.centered)
    stats[32 + sg] = metal::precise::rsqrt(s / n + p.eps);
  else
    stats[sg] = s / n;
}

struct IvAttention {
  uint rows;
  float scale;
};

// o [rows, 512] = softmax(q k^T * scale) v of qkv rows [rows, 1536] (both padded to 32 rows): 8 queries a group.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void iv_attention(
    const device float* qkv [[buffer(0)]], device float* o [[buffer(1)]], constant IvAttention& p [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint D = 512;
  constexpr uint Q = D / 4;
  constexpr uint ld = 3 * D;
  constexpr uint BK = 32;
  threadgroup float Sp[4][8 * BK];
  threadgroup float Ps[8 * BK];
  threadgroup float As[8];
  threadgroup uint rescale;
  // the rows of this lane's two elements of every 8 x 8 fragment
  const uint fm = ((lane / 4) & 4) + (lane / 2) % 4;
  const uint q0 = tg * 8;
  const uint d0 = sg * Q;
  simdgroup_float8x8 qm[Q / 8];
  simdgroup_float8x8 acc[Q / 8];
  TF_UNROLL_IV
  for (uint d = 0; d < Q / 8; d++) {
    simdgroup_load(qm[d], qkv + ulong(q0) * ld + d0 + d * 8, ld);
    acc[d] = simdgroup_float8x8(0.0f);
  }
  const uint row = t / 16;
  const uint col = (t % 16) * 2;
  float best = -INFINITY;
  float total = 0.0f;
  for (uint k0 = 0; k0 < p.rows; k0 += BK) {
    const device float* kp = qkv + ulong(k0) * ld + D + d0;
    TF_UNROLL_IV
    for (uint j = 0; j < BK / 8; j++) {
      simdgroup_float8x8 s = simdgroup_float8x8(0.0f);
      TF_UNROLL_IV
      for (uint d = 0; d < Q / 8; d++) {
        simdgroup_float8x8 km;
        simdgroup_load(km, kp + ulong(j * 8) * ld + d * 8, ld, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s, qm[d], km, s);
      }
      simdgroup_store(s, Sp[sg] + j * 8, BK);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v[2];
    float most = best;
    TF_UNROLL_IV
    for (uint c = 0; c < 2; c++) {
      const uint i = row * BK + col + c;
      const float sum = (Sp[0][i] + Sp[1][i]) + (Sp[2][i] + Sp[3][i]);
      v[c] = k0 + col + c < p.rows ? sum * p.scale : -INFINITY;
      most = max(most, v[c]);
    }
    TF_UNROLL_IV
    for (ushort m = 1; m < 16; m <<= 1) most = max(most, simd_shuffle_xor(most, m));
    const float alpha = precise::exp(best - most);
    float add = 0.0f;
    TF_UNROLL_IV
    for (uint c = 0; c < 2; c++) {
      const float e = precise::exp(v[c] - most);
      Ps[row * BK + col + c] = e;
      add += e;
    }
    TF_UNROLL_IV
    for (ushort m = 1; m < 16; m <<= 1) add += simd_shuffle_xor(add, m);
    total = total * alpha + add;
    best = most;
    if (t % 16 == 0) As[row] = alpha;
    if (t == 0) rescale = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t % 16 == 0 && alpha != 1.0f) rescale = 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rescale) {
      const float f = As[fm];
      TF_UNROLL_IV
      for (uint d = 0; d < Q / 8; d++) {
        acc[d].thread_elements()[0] *= f;
        acc[d].thread_elements()[1] *= f;
      }
    }
    const device float* vp = qkv + ulong(k0) * ld + 2 * D + d0;
    TF_UNROLL_IV
    for (uint j = 0; j < BK / 8; j++) {
      simdgroup_float8x8 pm;
      simdgroup_load(pm, Ps + j * 8, BK);
      TF_UNROLL_IV
      for (uint d = 0; d < Q / 8; d++) {
        simdgroup_float8x8 vm;
        simdgroup_load(vm, vp + ulong(j * 8) * ld + d * 8, ld);
        simdgroup_multiply_accumulate(acc[d], pm, vm, acc[d]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t % 16 == 0) As[row] = 1.0f / total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = As[fm];
  TF_UNROLL_IV
  for (uint d = 0; d < Q / 8; d++) {
    acc[d].thread_elements()[0] *= inv;
    acc[d].thread_elements()[1] *= inv;
    simdgroup_store(acc[d], o + ulong(q0) * D + d0 + d * 8, D);
  }
}

struct IvTokens {
  uint rows;
  uint cols;
  uint pad;
};

// Rows [pad + patches, 128]: pad zeros, then each 2 x 2 patch of z = mean + std * noise as (z - m) / s.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void iv_tokens(
    const device float* moments [[buffer(0)]], const device float* noise [[buffer(1)]],
    const device float* m [[buffer(2)]], const device float* s [[buffer(3)]], device float* out [[buffer(4)]],
    constant IvTokens& p [[buffer(5)]], uint2 at [[thread_position_in_grid]]) {
  const uint w2 = p.cols / 2;
  if (at.x >= 128 || at.y >= p.pad + (p.rows / 2) * w2) return;
  const ulong o = ulong(at.y) * 128 + at.x;
  if (at.y < p.pad) {
    out[o] = 0.0f;
    return;
  }
  // moments [rows, cols, 64] as mean | logvar, noise [rows, cols, 32]; a patch's channel c * 4 + dy * 2 + dx
  const uint patch = at.y - p.pad;
  const uint c = at.x / 4;
  const uint py = (patch / w2) * 2 + (at.x % 4) / 2;
  const uint px = (patch % w2) * 2 + at.x % 2;
  const uint pixel = py * p.cols + px;
  // torch.clamp keeps a NaN, where Metal's clamp (fmin, fmax) would drop it
  const float raw = moments[ulong(pixel) * 64 + 32 + c];
  const float logvar = isnan(raw) ? raw : clamp(raw, -30.0f, 20.0f);
  const float z = moments[ulong(pixel) * 64 + c] + precise::exp(0.5f * logvar) * noise[ulong(pixel) * 32 + c];
  out[o] = (z - m[at.x]) / s[at.x];
}

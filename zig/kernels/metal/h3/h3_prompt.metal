// MiniMax-H3-Image's prompt path on any Apple GPU: bf16 products with fp32 sums in simdgroup matrices, then glue.
#include <metal_stdlib>
using namespace metal;

struct HpGemm {
  uint M;
  uint N;
  uint K;
  uint bias;
};

// C [M, N] = bf16(A [M, K] B [N, K]^T + bias) with fp32 sums: a 64 x 64 tile on four simdgroups of 32 x 32.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void hp_gemm(
    const device bfloat* A [[buffer(0)]], const device bfloat* B [[buffer(1)]], const device bfloat* bias [[buffer(2)]],
    device bfloat* C [[buffer(3)]], constant HpGemm& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BM = 64, BN = 64, BK = 32, LD = BK + 8;
  threadgroup bfloat as[BM * LD];
  threadgroup bfloat bs[BN * LD];
  const int M = int(p.M), N = int(p.N), K = int(p.K);
  const int m0 = int(tg.y) * BM, n0 = int(tg.x) * BN;
  const int sm = int(sg) / 2 * 32, sn = int(sg) % 2 * 32;
  simdgroup_matrix<float, 8, 8> acc[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
  // each thread stages 16 values of a row of A and of B; rows past M read row M - 1 and are never stored
  const int lr = int(tid) / 2, lk = int(tid) % 2 * 16;
  const device bfloat* ap = A + long(min(m0 + lr, M - 1)) * K + lk;
  const device bfloat* bp = B + long(n0 + lr) * K + lk;
  threadgroup bfloat* at = as + lr * LD + lk;
  threadgroup bfloat* bt = bs + lr * LD + lk;
  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; i++) {
      *(threadgroup ushort4*)(at + 4 * i) = *(const device ushort4*)(ap + k0 + 4 * i);
      *(threadgroup ushort4*)(bt + 4 * i) = *(const device ushort4*)(bp + k0 + 4 * i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<bfloat, 8, 8> a[4], b[4];
      for (int i = 0; i < 4; i++) simdgroup_load(a[i], as + (sm + 8 * i) * LD + kk, LD);
      for (int j = 0; j < 4; j++) simdgroup_load(b[j], bs + (sn + 8 * j) * LD + kk, LD, ulong2(0, 0), true);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }
  // each lane's two elements of an 8 x 8 block: row fm, columns fn and fn + 1
  const int q = int(lane) / 4, fm = (q & 4) + (int(lane) / 2) % 4, fn = (q & 2) * 2 + int(lane) % 2 * 2;
  for (int i = 0; i < 4; i++) {
    const int r = m0 + sm + 8 * i + fm;
    if (r >= M) continue;
    for (int j = 0; j < 4; j++) {
      const int c = n0 + sn + 8 * j + fn;
      float2 v = float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
      if (p.bias != 0) v += float2(float(bias[c]), float(bias[c + 1]));
      C[long(r) * N + c] = bfloat(v.x);
      C[long(r) * N + c + 1] = bfloat(v.y);
    }
  }
}

struct HpNorm {
  uint width;
  uint mode;
  float eps;
};

// One RMSNorm value: mode 0 bf16(w bf16(x r)) as the Qwen text model rounds, mode 1 bf16(x r w) as H3's blocks do.
inline float hp_normed(float x, float r, float w, uint mode) {
  return mode == 0 ? float(bfloat(w * float(bfloat(x * r)))) : float(bfloat(x * r * w));
}

// y = RMSNorm(x) by w with fp32 statistics, a threadgroup of 256 a row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void hp_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant HpNorm& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += 256) s += float(xr[i]) * float(xr[i]);
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float all = 0.0f;
  for (uint i = 0; i < 8; i++) all += part[i];
  const float r = metal::precise::rsqrt(all / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += 256) yr[i] = bfloat(hp_normed(float(xr[i]), r, float(w[i]), p.mode));
}

struct HpHeads {
  uint heads;
  uint kv_heads;
  uint q_row;
  uint k_row;
  uint head;
  uint mode;
  uint rotary;
  float eps;
};

// q then k heads of 128 normed in place (mode as hp_norm), turned by cos, sin [rows, 64] when rotary; a simdgroup each.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void hp_heads(
    device bfloat* Q [[buffer(0)]], device bfloat* K [[buffer(1)]], const device bfloat* wq [[buffer(2)]],
    const device bfloat* wk [[buffer(3)]], const device float* cs [[buffer(4)]], const device float* sn [[buffer(5)]],
    constant HpHeads& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint row = tg.x, h = tg.y * 8 + sg;
  if (h >= p.heads + p.kv_heads) return;
  const bool is_q = h < p.heads;
  device bfloat* x = is_q ? Q + ulong(row) * p.q_row + h * p.head : K + ulong(row) * p.k_row + (h - p.heads) * p.head;
  const device bfloat* w = is_q ? wq : wk;
  float v[4];
  float ss = 0.0f;
  for (uint i = 0; i < 4; i++) {
    v[i] = float(x[lane * 4 + i]);
    ss += v[i] * v[i];
  }
  const float r = metal::precise::rsqrt(simd_sum(ss) / 128.0f + p.eps);
  for (uint i = 0; i < 4; i++) v[i] = hp_normed(v[i], r, float(w[lane * 4 + i]), p.mode);
  if (p.rotary != 0) {
    // channel c pairs with c + 64, 16 lanes away: rotate-half reads -x[c + 64] below the half and x[c - 64] above
    const float sign = lane < 16 ? -1.0f : 1.0f;
    for (uint i = 0; i < 4; i++) {
      const float o = simd_shuffle_xor(v[i], ushort(16));
      const ulong f = ulong(row) * 64 + (lane % 16) * 4 + i;
      v[i] = float(bfloat(v[i] * cs[f])) + float(bfloat(sign * o * sn[f]));
    }
  }
  for (uint i = 0; i < 4; i++) x[lane * 4 + i] = bfloat(v[i]);
}

struct HpAttn {
  uint rows;
  uint heads;
  uint kv_heads;
  uint q_row;
  uint k_row;
  uint v_row;
  uint head;
  uint causal;
  float scale;
};

// o [rows, heads * 128] = softmax(q k^T scale) v in fp32 over earlier rows (causal) or all; a simdgroup a row's head.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void hp_attention(
    const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]], const device bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]], constant HpAttn& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint row = tg.x, h = tg.y * 8 + sg;
  if (h >= p.heads) return;
  const uint kv = h / (p.heads / p.kv_heads);
  const float4 q = float4(*(const device bfloat4*)(Q + ulong(row) * p.q_row + h * p.head + lane * 4));
  const device bfloat* kp = K + kv * p.head + lane * 4;
  const device bfloat* vp = V + kv * p.head + lane * 4;
  const uint end = p.causal != 0 ? row + 1 : p.rows;
  float m = -INFINITY, l = 0.0f;
  float4 acc = 0.0f;
  for (uint j0 = 0; j0 < end; j0 += 4) {
    float s[4];
    uint at[4];
    for (uint u = 0; u < 4; u++) {
      at[u] = min(j0 + u, end - 1);
      s[u] = simd_sum(dot(q, float4(*(const device bfloat4*)(kp + ulong(at[u]) * p.k_row)))) * p.scale;
      if (j0 + u >= end) s[u] = -INFINITY;
    }
    const float top = max(max(m, max(s[0], s[1])), max(s[2], s[3]));
    const float old = metal::precise::exp(m - top);
    l *= old;
    acc *= old;
    for (uint u = 0; u < 4; u++) {
      const float e = metal::precise::exp(s[u] - top);
      l += e;
      acc += e * float4(*(const device bfloat4*)(vp + ulong(at[u]) * p.v_row));
    }
    m = top;
  }
  *(device bfloat4*)(O + ulong(row) * p.heads * 128 + h * 128 + lane * 4) = bfloat4(acc / l);
}

struct HpGlu {
  uint width;
  uint g_row;
  uint v_row;
  uint mode;
};

// o [rows, width] = silu(g) v: mode 0 bf16(bf16(silu g) v) (Qwen), 1 bf16(bf16(g bf16(sigmoid g)) v) (H3's blocks).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void hp_glu(
    const device bfloat* G [[buffer(0)]], const device bfloat* V [[buffer(1)]], device bfloat* O [[buffer(2)]],
    constant HpGlu& p [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.width) return;
  const float g = float(G[ulong(at.y) * p.g_row + at.x]), v = float(V[ulong(at.y) * p.v_row + at.x]);
  const float e = 1.0f + metal::precise::exp(-g);
  const float s = p.mode == 0 ? float(bfloat(g / e)) : float(bfloat(g * float(bfloat(1.0f / e))));
  O[ulong(at.y) * p.width + at.x] = bfloat(s * v);
}

// x = bf16(x + y), the residual add, over n values.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void hp_add(
    device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]], constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i < n) x[i] = bfloat(float(x[i]) + float(y[i]));
}

// x rows idx[i] = bf16(x + y [i]): DeepStack's features added at the image pads.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void hp_scatter_add(
    device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]], const device uint* idx [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= width) return;
  const ulong to = ulong(idx[at.y]) * width + at.x;
  x[to] = bfloat(float(x[to]) + float(y[ulong(at.y) * width + at.x]));
}

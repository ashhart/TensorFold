// music3 DiT glue in fp16: the condition, the windows, LayerNorm, rope, GLU, the guided velocity and the Euler step.
#include "m3_common.h"

// The condition mix of frame states [T, 8, 4096] (bf16) in fp16: sum_l w_l h_l (fp32 sum, fp16 out) times scale (fp16).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_cond_mix(device const bf16* states [[buffer(0)]], device const float* w8 [[buffer(1)]],
                        device half* out [[buffer(2)]], constant float2& a [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
  const uint n = uint(a.y) * 4096;
  if (i >= n) return;
  const uint t = i / 4096, d = i % 4096;
  float s = 0.0f;
  for (int l = 0; l < 8; ++l) s = fma(rh(b2f(states[((ulong)t * 8 + l) * 4096 + d])), w8[l], s);
  out[i] = half(rh(rh(s) * a.x));
}

// im2col for the k=3, pad 1 condition conv: col [T, 3, 4096] from x [T, 4096] (fp16; zero outside).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_im2col3(device const half* x [[buffer(0)]], device half* col [[buffer(1)]],
                       constant uint& T [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= T * 3 * 4096) return;
  const uint t = i / (3 * 4096), rem = i % (3 * 4096);
  const int src = int(t) + int(rem / 4096) - 1;
  col[i] = (src >= 0 && src < int(T)) ? x[(ulong)src * 4096 + rem % 4096] : half(0.0f);
}

// Window inputs [S, Lw, 2304] = [x, 0, condition] (branch 1 reads the null condition, the bias); short rows 0.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_dit_input(device const float* x [[buffer(0)]], device const half* cond [[buffer(1)]],
                         device const half* null_row [[buffer(2)]], device const int* wins [[buffer(3)]],
                         device half* full [[buffer(4)]], constant uint2& a [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint nwin = a.x, Lw = a.y, s = tg.y, j = tg.x;
  const uint w = s % nwin, branch = s / nwin;
  const int start = wins[2 * w], len = wins[2 * w + 1];
  device half* row = full + ((ulong)s * Lw + j) * 2304;
  for (uint c = t; c < 2304; c += 256) {
    float v = 0.0f;
    if (int(j) < len) {
      const ulong tt = ulong(start) + j;
      if (c < 128) v = x[tt * 128 + c];
      else if (c >= 256) v = branch == 0 ? float(cond[tt * 2048 + c - 256]) : float(null_row[c - 256]);
    }
    row[c] = half(v);
  }
}

// a = round(a + b) in fp16.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_add_h(device half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) a[i] = half(float(a[i]) + float(b[i]));
}

// The timestep embedding input: [cos(2 pi t f), sin(2 pi t f)] (fp32, then fp16), f = features [128].
[[max_total_threads_per_threadgroup(128)]] kernel void m3_time_features(device const half* f [[buffer(0)]], device half* out [[buffer(1)]],
                             constant float& t [[buffer(2)]], uint i [[thread_index_in_threadgroup]]) {
  const float a = 2.0f * 3.14159265358979323846f * t * float(f[i]);
  out[i] = half(precise::cos(a));
  out[128 + i] = half(precise::sin(a));
}

// y = silu(x) (fp16) for the timestep MLP.
[[max_total_threads_per_threadgroup(1024)]] kernel void m3_silu_h(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                      constant uint& n [[buffer(2)]], uint t [[thread_index_in_threadgroup]]) {
  for (uint i = t; i < n; i += 1024) y[i] = half(m3_silu(float(x[i])));
}

struct SmallArgs {
  uint K, N;
};

// One fp16 row through a small Linear: y [N] = round(W [N, K] x + b) (fp32 sum), a simdgroup an output.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_linear_h1(device const half* w [[buffer(0)]], device const half* b [[buffer(1)]],
                         device const half* x [[buffer(2)]], device half* y [[buffer(3)]], constant SmallArgs& a [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg * 8 + sg;
  if (n >= a.N) return;
  float s = 0.0f;
  for (uint k = lane; k < a.K; k += 32) s = fma(float(w[(ulong)n * a.K + k]), float(x[k]), s);
  s = simd_sum(s);
  if (lane == 0) y[n] = half(s + float(b[n]));
}

// dst [n, 2048] = src [idx[i], 2048] (fp16 rows: the condition's nearest-neighbour stretch to the latent timeline).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_gather_rows_h(device const half* src [[buffer(0)]], device const int* idx [[buffer(1)]],
                             device half* dst [[buffer(2)]], uint i [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]]) {
  for (uint c = t; c < 2048; c += 256) dst[(ulong)i * 2048 + c] = src[(ulong)idx[i] * 2048 + c];
}

// Row 0 of each sequence of h [S, Lseq, 2048] = the timestep embedding.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_put_time(device const half* te [[buffer(0)]], device half* h [[buffer(1)]],
                        constant uint& Lseq [[buffer(2)]], uint s [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]]) {
  for (uint c = t; c < 2048; c += 256) h[(ulong)s * Lseq * 2048 + c] = te[c];
}

// LayerNorm (eps 1e-5) of fp16 rows of 2048 in fp32; `has_bias` first adds the previous block's ff_out bias into x.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_layernorm_h(device half* x [[buffer(0)]], device const half* bias [[buffer(1)]],
                           device const half* g [[buffer(2)]], device const half* b [[buffer(3)]], device half* y [[buffer(4)]],
                           constant uint& has_bias [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh[32];
  device half* xr = x + (ulong)row * 2048;
  float v[8];
  float s = 0.0f;
  for (int j = 0; j < 8; ++j) {
    const uint c = t + 256 * j;
    v[j] = float(xr[c]);
    if (has_bias) {
      v[j] = rh(v[j] + float(bias[c]));
      xr[c] = half(v[j]);
    }
    s += v[j];
  }
  const float mean = group_sum(s, sh, sg, lane, 8) / 2048.0f;
  float q = 0.0f;
  for (int j = 0; j < 8; ++j) q = fma(v[j] - mean, v[j] - mean, q);
  const float var = group_sum(q, sh, sg, lane, 8) / 2048.0f;
  const float rs = 1.0f / precise::sqrt(var + 1e-5f);
  device half* yr = y + (ulong)row * 2048;
  for (int j = 0; j < 8; ++j) {
    const uint c = t + 256 * j;
    yr[c] = half((v[j] - mean) * rs * float(g[c]) + float(b[c]));
  }
}

// h [rows, 2048] = round(h + bias) (the last block's ff_out bias).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_bias_h(device half* h [[buffer(0)]], device const half* bias [[buffer(1)]],
                      constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) h[i] = half(float(h[i]) + float(bias[i % 2048]));
}

struct GatherArgs {
  uint nwin, Lw, L;
  float cfg;
};

// A batch of windows' guided velocity -(o + post) summed into vsum / vcnt, windows in order (no atomics).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_dit_gather(device const half* o [[buffer(0)]], device const half* post [[buffer(1)]],
                          device const int* wins [[buffer(2)]], device float* vsum [[buffer(3)]], device float* vcnt [[buffer(4)]],
                          constant GatherArgs& a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.L * 128) return;
  const int t = int(i / 128), c = int(i % 128);
  float sum = vsum[i], cnt = vcnt[i];
  for (uint w = 0; w < a.nwin; ++w) {
    const int start = wins[2 * w], len = wins[2 * w + 1];
    if (t < start || t >= start + len) continue;
    const ulong rc = ((ulong)w * a.Lw + (t - start)) * 128 + c, ru = ((ulong)(a.nwin + w) * a.Lw + (t - start)) * 128 + c;
    const float vc = -rh(float(o[rc]) + float(post[rc]));
    const float vu = -rh(float(o[ru]) + float(post[ru]));
    sum += vu + (vc - vu) * a.cfg;
    cnt += 1.0f;
  }
  vsum[i] = sum;
  vcnt[i] = cnt;
}

// Euler: x += (vsum / vcnt) (sigma_next - sigma); the sums cleared for the next step.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_euler(device float* x [[buffer(0)]], device float* vsum [[buffer(1)]],
                     device float* vcnt [[buffer(2)]], constant float2& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(a.y)) return;
  x[i] = fma(vsum[i] / vcnt[i], a.x, x[i]);
  vsum[i] = 0.0f;
  vcnt[i] = 0.0f;
}

// Gaussian noise [n] from (seed, i): Philox normals, two a counter.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_noise(device float* x [[buffer(0)]], constant ulong2& a [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
  if (2 * ulong(i) >= a.y) return;
  const float2 g = philox_normal2(a.x, i, 0x6e6f697365ul);
  x[2 * i] = g.x;
  if (2 * ulong(i) + 1 < a.y) x[2 * i + 1] = g.y;
}

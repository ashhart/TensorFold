// MiniMax-H3-Image's edit encoder (the VAE's CNN on one frame) in fp32; each picture tensor is [pixels, channels].

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define TF_UNROLL_HIE _Pragma("clang loop unroll(full)")

// A convolution: input and output sides, channels, the weight row length, the kernel side, stride, padding, add.
struct HieConv {
  uint h;
  uint w;
  uint oh;
  uint ow;
  uint cin;
  uint cout;
  uint ldw;
  uint k;
  uint stride;
  uint pad;
  uint add;
};

// Index i of an axis of n read with reflect padding (-1 reads 1, n reads n - 2).
inline int hie_reflect(int i, int n) {
  return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i);
}

// y = conv(x) + b (+ r), x [h * w, cin], w [k * k, cin, ldw]: 64 pixels by 64 channels a group, 16 by 32 a simdgroup.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void hie_conv(
    const device float* x [[buffer(0)]], const device float* wt [[buffer(1)]], const device float* b [[buffer(2)]],
    const device float* r [[buffer(3)]], device float* y [[buffer(4)]], constant HieConv& s [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float smem[64 * 64];
  threadgroup float* at = smem;
  threadgroup float* bt = smem + 64 * 32;
  const uint p0 = tg.x * 64;
  const uint n0 = tg.y * 64;
  const uint pixels = s.oh * s.ow;
  // this thread's loads: pixel row t / 4, channels (t % 4) * 8 + 8; weight row t / 8, columns (t % 8) * 8 + 8
  const uint ar = t / 4;
  const uint ak = (t % 4) * 8;
  const uint br = t / 8;
  const uint bc = (t % 8) * 8;
  const bool live = p0 + ar < pixels;
  const int oy = live ? int((p0 + ar) / s.ow) : 0;
  const int ox = live ? int((p0 + ar) % s.ow) : 0;
  const uint sm = (sg % 4) * 16;
  const uint sn = (sg / 4) * 32;
  simdgroup_float8x8 acc[2][4];
  TF_UNROLL_HIE for (uint i = 0; i < 2; i++) TF_UNROLL_HIE for (uint j = 0; j < 4; j++)
    acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint tap = 0; tap < s.k * s.k; tap++) {
    const int iy = hie_reflect(oy * int(s.stride) + int(tap / s.k) - int(s.pad), int(s.h));
    const int ix = hie_reflect(ox * int(s.stride) + int(tap % s.k) - int(s.pad), int(s.w));
    const device float* src = x + (uint(iy) * s.w + uint(ix)) * s.cin;
    const device float* ws = wt + tap * s.cin * s.ldw + n0 + bc;
    for (uint c0 = 0; c0 < s.cin; c0 += 32) {
      TF_UNROLL_HIE for (uint j = 0; j < 8; j++) {
        const uint c = c0 + ak + j;
        at[ar * 32 + ak + j] = (live && c < s.cin) ? src[c] : 0.0f;
      }
      const uint kr = c0 + br;
      TF_UNROLL_HIE for (uint j = 0; j < 8; j++) bt[br * 64 + bc + j] = kr < s.cin ? ws[kr * s.ldw + j] : 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      TF_UNROLL_HIE for (uint kk = 0; kk < 32; kk += 8) {
        simdgroup_float8x8 am[2];
        simdgroup_float8x8 bm[4];
        TF_UNROLL_HIE for (uint i = 0; i < 2; i++) simdgroup_load(am[i], at + (sm + i * 8) * 32 + kk, 32);
        TF_UNROLL_HIE for (uint j = 0; j < 4; j++) simdgroup_load(bm[j], bt + kk * 64 + sn + j * 8, 64);
        TF_UNROLL_HIE for (uint i = 0; i < 2; i++) TF_UNROLL_HIE for (uint j = 0; j < 4; j++)
          simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  TF_UNROLL_HIE for (uint i = 0; i < 2; i++) TF_UNROLL_HIE for (uint j = 0; j < 4; j++)
    simdgroup_store(acc[i][j], smem + (sm + i * 8) * 64 + sn + j * 8, 64);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = t; e < 64 * 64; e += 256) {
    const uint p = p0 + e / 64;
    const uint n = n0 + e % 64;
    if (p >= pixels || n >= s.cout) continue;
    float v = smem[e] + b[n];
    if (s.add != 0) v += r[p * s.cout + n];
    y[p * s.cout + n] = v;
  }
}

// GroupNorm's 32 groups over `pixels` of x [pixels, channels], summed in chunks of `chunk` pixels.
struct HieGroup {
  uint pixels;
  uint channels;
  uint chunk;
  uint chunks;
  uint centered;
  float eps;
};

// part [chunks, channels]: each channel's sum over one chunk of x, or (centered) of (x - its group's mean)^2.
[[max_total_threads_per_threadgroup(1024)]] [[kernel]] void hie_group_part(
    const device float* x [[buffer(0)]], const device float* stats [[buffer(1)]], device float* part [[buffer(2)]],
    constant HieGroup& g [[buffer(3)]], uint chunk [[threadgroup_position_in_grid]],
    uint c [[thread_position_in_threadgroup]]) {
  if (c >= g.channels) return;
  const float mean = g.centered != 0 ? stats[c / (g.channels / 32)] : 0.0f;
  const uint last = min((chunk + 1) * g.chunk, g.pixels);
  float sum = 0.0f;
  for (uint p = chunk * g.chunk; p < last; p++) {
    const float v = x[p * g.channels + c] - mean;
    sum += g.centered != 0 ? v * v : v;
  }
  part[chunk * g.channels + c] = sum;
}

// stats: a group's mean at its index, or (centered) 1 / sqrt(variance + eps) at 32 + its index; a simdgroup a group.
[[max_total_threads_per_threadgroup(1024)]] [[kernel]] void hie_group_finish(
    const device float* part [[buffer(0)]], device float* stats [[buffer(1)]], constant HieGroup& g [[buffer(2)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint per = g.channels / 32;
  float sum = 0.0f;
  for (uint i = lane; i < g.chunks * per; i += 32) sum += part[(i / per) * g.channels + sg * per + i % per];
  sum = simd_sum(sum);
  if (lane != 0) return;
  const float v = sum / float(g.pixels * per);
  if (g.centered != 0) {
    stats[32 + sg] = 1.0f / sqrt(v + g.eps);
  } else {
    stats[sg] = v;
  }
}

struct HieNorm {
  uint pixels;
  uint channels;
};

// y = SiLU((x - mean) * rstd * gamma + beta) with x's group statistics.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void hie_norm_silu(
    const device float* x [[buffer(0)]], const device float* stats [[buffer(1)]],
    const device float* gamma [[buffer(2)]], const device float* beta [[buffer(3)]], device float* y [[buffer(4)]],
    constant HieNorm& n [[buffer(5)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= n.channels || i.y >= n.pixels) return;
  const uint g = i.x / (n.channels / 32);
  const uint at = i.y * n.channels + i.x;
  const float v = (x[at] - stats[g]) * stats[32 + g] * gamma[i.x] + beta[i.x];
  y[at] = v / (1.0f + exp(-v));
}

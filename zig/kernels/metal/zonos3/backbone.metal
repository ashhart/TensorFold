// ours: ZONOS3 Mini's backbone on Metal, fp32 rows over bf16 weights: zonos3.cu's kernels, same threads and sum order.
#include "z3_common.h"

// Eight bf16 weights against eight fp32 inputs, in index order.
inline float dot8(uint4 w, threadgroup const float* x, float acc) {
  const uint p[4] = {w.x, w.y, w.z, w.w};
  for (int i = 0; i < 4; ++i) {
    acc = fma(as_type<float>(p[i] << 16), x[2 * i], acc);
    acc = fma(as_type<float>(p[i] & 0xFFFF0000u), x[2 * i + 1], acc);
  }
  return acc;
}

// Frame rows [T, 11] to the sum of their eleven embeddings, column 0 first: tables bf16 [rows, D], ten audio then text.
struct EmbedArgs { int audio_rows; int D; };
kernel void z3_embed(device const int* rows [[buffer(0)]], device const bf16* tables [[buffer(1)]],
                     device float* out [[buffer(2)]], constant EmbedArgs& a [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
  device const int* r = rows + (ulong)row * 11;
  device float* o = out + (ulong)row * a.D;
  for (int i = tid; i < a.D; i += nt) {
    float s = 0.f;
    for (int c = 0; c < 11; ++c) s += b2f(tables[((ulong)c * a.audio_rows + r[c]) * a.D + i]);
    o[i] = s;
  }
}

// y = x * rsqrt(mean(x^2) + eps) * w (w bf16, or none for a weightless norm); a group a row, D a multiple of 32.
struct NormArgs { int xs; int ys; int D; int has_w; float eps; };
kernel void z3_rmsnorm(device const float* x [[buffer(0)]], device const bf16* w [[buffer(1)]], device float* y [[buffer(2)]],
                       constant NormArgs& a [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  device const float* xr = x + (ulong)row * a.xs;
  device float* yr = y + (ulong)row * a.ys;
  float ss = 0.f;
  for (int i = tid; i < a.D; i += nt) ss = fma(xr[i], xr[i], ss);
  const float inv = rms_scale(block_sum_tree(ss, part, nt >> 5, lane, sg), a.D, a.eps);
  for (int i = tid; i < a.D; i += nt) {
    const float v = xr[i] * inv;
    yr[i] = a.has_w ? v * b2f(w[i]) : v;
  }
}

// Full AttnRes: a row's S sources (bank [S][cap][D]) mixed by softmax over s of sum(rms_norm(source) * query).
struct AttnResArgs { int S; int cap; int D; float eps; };
kernel void z3_attnres(device const float* bank [[buffer(0)]], device const bf16* query [[buffer(1)]],
                       device float* out [[buffer(2)]], constant AttnResArgs& a [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint nt [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  threadgroup float wgt[64];
  const uint groups = nt >> 5;
  for (int s = 0; s < a.S; ++s) {
    device const float* x = bank + ((ulong)s * a.cap + row) * a.D;
    float ss = 0.f;
    for (int i = tid; i < a.D; i += nt) ss = fma(x[i], x[i], ss);
    const float inv = rms_scale(block_sum_tree(ss, part, groups, lane, sg), a.D, a.eps);
    float d = 0.f;
    for (int i = tid; i < a.D; i += nt) d = fma(x[i] * inv, b2f(query[i]), d);
    d = block_sum_tree(d, part, groups, lane, sg);
    if (tid == 0) wgt[s] = d;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float m = wgt[0];
    for (int s = 1; s < a.S; ++s) m = fmax(m, wgt[s]);
    float total = 0.f;
    for (int s = 0; s < a.S; ++s) {
      wgt[s] = z3_exp(wgt[s] - m);
      total += wgt[s];
    }
    for (int s = 0; s < a.S; ++s) wgt[s] = precise::divide(wgt[s], total);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* o = out + (ulong)row * a.D;
  for (int i = tid; i < a.D; i += nt) {
    float acc = 0.f;
    for (int s = 0; s < a.S; ++s) acc = fma(bank[((ulong)s * a.cap + row) * a.D + i], wgt[s], acc);
    o[i] = acc;
  }
}

// y[r, n] = sum_k W[n, k] x[r / xrep, k] (+ bias[n]): bf16 W [N, K], K % 8 == 0, K <= 8192 (x in threadgroup memory).
struct LinearArgs { int xs; int xrep; int K; int N; int ys; int has_bias; int has_picks; long pick_stride; };
kernel void z3_linear(device const bf16* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                      device const bf16* bias [[buffer(3)]], device const int* picks [[buffer(4)]],
                      constant LinearArgs& a [[buffer(5)]], threadgroup float* xs [[threadgroup(0)]],
                      uint2 gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint nt = nsg * 32;
  const int r = gid.y;
  device const float* xr = x + (ulong)(r / a.xrep) * a.xs;
  for (int i = tid; i < a.K; i += nt) xs[i] = xr[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 8 simdgroups make 8 outputs of row r; picks move its W by picks[r] * pick_stride.
  const int n = gid.x * 8 + sg;
  device const bf16* base = a.has_picks ? W + (ulong)picks[r] * (ulong)a.pick_stride : W;
  device const uint4* wr = (device const uint4*)(base + (ulong)min(n, a.N - 1) * a.K);
  const int chunks = a.K >> 3;
  float acc = 0.f;
  for (int c = lane; c < chunks; c += 32) acc = dot8(wr[c], xs + c * 8, acc);
  acc = warp_sum(acc);
  if (lane == 0 && n < a.N) y[(ulong)r * a.ys + n] = a.has_bias ? acc + b2f(bias[n]) : acc;
}

// GDN depthwise causal conv (kernel 4, the newest input last) + SiLU over C channels; state keeps the last 3 inputs.
struct ConvArgs { int ps; int T; int C; };
kernel void z3_conv_silu(device const float* proj [[buffer(0)]], device const bf16* w [[buffer(1)]],
                         device float* state [[buffer(2)]], device float* out [[buffer(3)]],
                         constant ConvArgs& a [[buffer(4)]], uint c [[thread_position_in_grid]]) {
  if ((int)c >= a.C) return;
  const float w0 = b2f(w[c * 4 + 0]), w1 = b2f(w[c * 4 + 1]), w2 = b2f(w[c * 4 + 2]), w3 = b2f(w[c * 4 + 3]);
  float s0 = state[c * 3 + 0], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
  for (int t = 0; t < a.T; ++t) {
    const float xv = proj[(ulong)t * a.ps + c];
    float acc = s0 * w0;
    acc = fma(s1, w1, acc);
    acc = fma(s2, w2, acc);
    acc = fma(xv, w3, acc);
    out[(ulong)t * a.C + c] = z3_silu(acc);
    s0 = s1;
    s1 = s2;
    s2 = xv;
  }
  state[c * 3 + 0] = s0;
  state[c * 3 + 1] = s1;
  state[c * 3 + 2] = s2;
}

// GDN inputs of (row, head): q (l2-normed, times 128^-1/2) and k in place in conv [T][q | k | v]; decay and beta.
struct GdnPrepArgs { int ps; int a_off; };
kernel void z3_gdn_prep(device float* conv [[buffer(0)]], device const float* proj [[buffer(1)]],
                        device const bf16* A_log [[buffer(2)]], device const bf16* dt_bias [[buffer(3)]],
                        device float* gates [[buffer(4)]], constant GdnPrepArgs& a [[buffer(5)]],
                        uint2 gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  const int t = gid.x, h = gid.y;
  device float* q = conv + (ulong)t * 1536 + h * 128;
  device float* k = q + 512;
  const float qv = q[tid], kv = k[tid];
  const float qs = block_sum_tree(qv * qv, part, 4, lane, sg);
  const float ks = block_sum_tree(kv * kv, part, 4, lane, sg);
  q[tid] = qv * precise::rsqrt(qs + 1e-6f) * 0.088388347648318447f;
  k[tid] = kv * precise::rsqrt(ks + 1e-6f);
  if (tid == 0) {
    device const float* p = proj + (ulong)t * a.ps + a.a_off;
    gates[(t * 4 + h) * 2 + 0] = z3_exp(-z3_exp(b2f(A_log[h])) * z3_softplus(p[h] + b2f(dt_bias[h])));
    gates[(t * 4 + h) * 2 + 1] = z3_sigmoid(p[4 + h]);
  }
}

// GDN recurrence, token by token in fp32: a thread a value row of head h's state S [4][128][128].
struct GdnArgs { int T; };
kernel void z3_gdn(device const float* conv [[buffer(0)]], device const float* gates [[buffer(1)]],
                   device float* S [[buffer(2)]], device float* o [[buffer(3)]], constant GdnArgs& a [[buffer(4)]],
                   uint h [[threadgroup_position_in_grid]], uint v [[thread_index_in_threadgroup]]) {
  device float* srow = S + ((ulong)h * 128 + v) * 128;
  float row[128];
  for (int j = 0; j < 128; ++j) row[j] = srow[j];
  for (int t = 0; t < a.T; ++t) {
    device const float* q = conv + (ulong)t * 1536 + h * 128;
    device const float* k = q + 512;
    const float val = q[1024 + v];
    const float decay = gates[(t * 4 + h) * 2 + 0], beta = gates[(t * 4 + h) * 2 + 1];
    float pred = 0.f;
    for (int j = 0; j < 128; ++j) {
      row[j] *= decay;
      pred = fma(row[j], k[j], pred);
    }
    const float delta = beta * (val - pred);
    float out = 0.f;
    for (int j = 0; j < 128; ++j) {
      row[j] = fma(delta, k[j], row[j]);
      out = fma(row[j], q[j], out);
    }
    o[(ulong)t * 512 + h * 128 + v] = out;
  }
  for (int j = 0; j < 128; ++j) srow[j] = row[j];
}

// y = rms_norm(o, w, 1e-5) * silu(z) per head of 128: o [T][512], z in proj rows at z_off.
struct GatedNormArgs { int ps; int z_off; };
kernel void z3_gated_norm(device const float* o [[buffer(0)]], device const float* proj [[buffer(1)]],
                          device const bf16* w [[buffer(2)]], device float* y [[buffer(3)]],
                          constant GatedNormArgs& a [[buffer(4)]], uint2 gid [[threadgroup_position_in_grid]],
                          uint i [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  const int t = gid.x, h = gid.y;
  const float xv = o[(ulong)t * 512 + h * 128 + i];
  const float inv = rms_scale(block_sum_tree(xv * xv, part, 4, lane, sg), 128, 1e-5f);
  y[(ulong)t * 512 + h * 128 + i] = xv * inv * b2f(w[i]) * z3_silu(proj[(ulong)t * a.ps + a.z_off + h * 128 + i]);
}

// Attention inputs at pos0 + row: q rms-normed, times |temp|, rotated; k, v to the caches; simdgroup: head, lane: dim.
struct AttnPrepArgs { int pos0; float theta; };
kernel void z3_attn_prep(device float* q [[buffer(0)]], device const float* kv [[buffer(1)]],
                         device const bf16* temp [[buffer(2)]], device float* kc [[buffer(3)]],
                         device float* vc [[buffer(4)]], constant AttnPrepArgs& a [[buffer(5)]],
                         uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint h [[simdgroup_index_in_threadgroup]]) {
  const float pos = float(a.pos0 + (int)t);
  const float freq = precise::pow(a.theta, -float(lane & ~1u) / 32.f);
  const float ang = pos * freq;
  const float c = precise::cos(ang), s = precise::sin(ang);
  device float* qr = q + (ulong)t * 512 + tid;
  float xv = *qr;
  xv = xv * rms_scale(warp_sum(xv * xv), 32, 1e-6f) * fabs(b2f(temp[h]));
  float other = simd_shuffle_xor(xv, (ushort)1);
  *qr = (lane & 1) ? other * s + xv * c : xv * c - other * s;
  if (h < 4) {
    device const float* kr = kv + (ulong)t * 256;
    float kx = kr[tid];
    kx = kx * rms_scale(warp_sum(kx * kx), 32, 1e-6f);
    other = simd_shuffle_xor(kx, (ushort)1);
    const ulong at = (ulong)(a.pos0 + (int)t) * 128 + tid;
    kc[at] = (lane & 1) ? other * s + kx * c : kx * c - other * s;
    vc[at] = kr[128 + tid];
  }
}

// Causal attention of (row, head) over cache rows 0 .. pos0 + row, gated; keys by index mod 256, a fixed fold.
struct AttentionArgs { int pos0; };
[[max_total_threads_per_threadgroup(256)]] kernel void z3_attention(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]],
                         device const float* vc [[buffer(2)]], device const float* gate [[buffer(3)]],
                         device float* out [[buffer(4)]], constant AttentionArgs& a [[buffer(5)]],
                         uint2 gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  threadgroup float acc[256][17];
  threadgroup float total[33];
  const int t = gid.x, h = gid.y, kh = h >> 2;
  const int L = a.pos0 + t + 1;
  float qv[32];
  device const float* qr = q + (ulong)t * 512 + h * 32;
  for (int i = 0; i < 32; ++i) qv[i] = qr[i];
  const float scale = 0.17677669529663688f;
  float m = -INFINITY;
  for (int j = tid; j < L; j += 256) {
    device const float* k = kc + (ulong)j * 128 + kh * 32;
    float d = 0.f;
    for (int i = 0; i < 32; ++i) d = fma(qv[i], k[i], d);
    m = fmax(m, d * scale);
  }
  m = block_max_tree(m, part, 8, lane, sg);
  float s_[33];
  for (int i = 0; i < 33; ++i) s_[i] = 0.f;
  for (int j = tid; j < L; j += 256) {
    device const float* k = kc + (ulong)j * 128 + kh * 32;
    device const float* v = vc + (ulong)j * 128 + kh * 32;
    float d = 0.f;
    for (int i = 0; i < 32; ++i) d = fma(qv[i], k[i], d);
    const float e = z3_exp(d * scale - m);
    s_[32] += e;
    for (int i = 0; i < 32; ++i) s_[i] = fma(e, v[i], s_[i]);
  }
  // a thread's 33 sums folded in two halves (17 a thread fit), each in zonos3.cu's order: thread 0 to 255
  for (int half_ = 0; half_ < 2; ++half_) {
    const int first = half_ * 17, count = half_ == 0 ? 17 : 16;
    for (int i = 0; i < count; ++i) acc[tid][i] = s_[first + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)tid < count) {
      float s = 0.f;
      for (int w = 0; w < 256; ++w) s += acc[w][tid];
      total[first + tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid < 32) {
    const float g = z3_sigmoid(gate[(ulong)t * 16 + h]);
    out[(ulong)t * 512 + h * 32 + tid] = precise::divide(total[tid], total[32]) * g;
  }
}

// out[r, i] = in[r, a_off + i] * silu(in[r, b_off + i]) for i < n.
struct MulSiluArgs { int stride; int a_off; int b_off; int n; };
kernel void z3_mul_silu(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                        constant MulSiluArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const int i = gid.x;
  if (i >= a.n) return;
  device const float* r = in + (ulong)gid.y * a.stride;
  out[(ulong)gid.y * a.n + i] = r[a.a_off + i] * z3_silu(r[a.b_off + i]);
}

// A row's two experts: the top two of logits - bias (ties to the lower index), weights softmax of their raw logits.
struct RouteArgs { int E; };
kernel void z3_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                     device int* picks [[buffer(2)]], device float* w [[buffer(3)]], constant RouteArgs& a [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0) return;
  device const float* l = logits + (ulong)row * a.E;
  int b0 = -1, b1 = -1;
  float v0 = -INFINITY, v1 = -INFINITY;
  for (int e = 0; e < a.E; ++e) {
    const float v = l[e] - bias[e];
    if (b0 < 0 || v > v0) {
      b1 = b0;
      v1 = v0;
      b0 = e;
      v0 = v;
    } else if (b1 < 0 || v > v1) {
      b1 = e;
      v1 = v;
    }
  }
  const float m = fmax(l[b0], l[b1]);
  const float e0 = z3_exp(l[b0] - m), e1 = z3_exp(l[b1] - m);
  picks[row * 2] = b0;
  picks[row * 2 + 1] = b1;
  w[row * 2] = precise::divide(e0, e0 + e1);
  w[row * 2 + 1] = precise::divide(e1, e0 + e1);
}

// out[r] = down[2r] * w[2r] + down[2r + 1] * w[2r + 1], the first pick first.
struct CombineArgs { int n; };
kernel void z3_combine(device const float* down [[buffer(0)]], device const float* w [[buffer(1)]],
                       device float* out [[buffer(2)]], constant CombineArgs& a [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const int i = gid.x;
  if (i >= a.n) return;
  const ulong r = gid.y;
  float s = down[(2 * r) * a.n + i] * w[2 * r];
  s += down[(2 * r + 1) * a.n + i] * w[2 * r + 1];
  out[r * a.n + i] = s;
}

// Logits in place: cap * tanh(x / cap).
struct SoftcapArgs { uint n; float cap; };
kernel void z3_softcap(device float* x [[buffer(0)]], constant SoftcapArgs& a [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
  if (i < a.n) x[i] = a.cap * precise::tanh(precise::divide(x[i], a.cap));
}

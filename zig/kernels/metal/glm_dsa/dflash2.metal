// TP-sharded GLM-5.3-DFlash2 block-diffusion layer kernels (756 GB/s half4 BF16 GEMV + BF16-MMA prompt GEMM).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float df2_sum32(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

inline uint df2_bf16x2(float a, float b) {
  return uint(as_type<ushort>(bfloat(a))) | (uint(as_type<ushort>(bfloat(b))) << 16);
}

inline float4 df2_u2_f4(uint2 w) {
  return float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xFFFF0000u),
                as_type<float>(w.y << 16), as_type<float>(w.y & 0xFFFF0000u));
}

inline half4 df2_u2_h4(uint2 w) {
  return half4(half(as_type<float>(w.x << 16)), half(as_type<float>(w.x & 0xFFFF0000u)),
               half(as_type<float>(w.y << 16)), half(as_type<float>(w.y & 0xFFFF0000u)));
}

struct Df2TapArgs {
  uint rows, hidden, k_lo, k_loc;
  uint tap_idx, pad0, pad1, pad2;
};

// Capture this TP rank's K-slice of target layer residual s.h [rows, hidden] into taps_loc [rows, k_loc] (bf16).
kernel void df2_tap_capture(const device float* src [[buffer(0)]], device bfloat* dst [[buffer(1)]],
                            constant Df2TapArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint tap_lo = a.tap_idx * a.hidden, tap_hi = tap_lo + a.hidden;
  const uint ov_lo = max(tap_lo, a.k_lo), ov_hi = min(tap_hi, a.k_lo + a.k_loc);
  if (ov_lo >= ov_hi) return;
  const uint n = ov_hi - ov_lo;
  if (gid.x >= n || gid.y >= a.rows) return;
  const uint src_c = (ov_lo - tap_lo) + gid.x;
  const uint dst_c = (ov_lo - a.k_lo) + gid.x;
  dst[ulong(gid.y) * a.k_loc + dst_c] = bfloat(src[ulong(gid.y) * a.hidden + src_c]);
}

struct Df2GemvArgs {
  uint rows, K, N, out_stride;
  uint in_bf16, pack_rank, pack_tp, pack_slot_u2;
};

// Fast BF16 matmul for rows=1..8 (hproj_w: 256t, 16 outs/TG, half4 TG cache + half4 dot).
kernel void df2_bf16_gemv_n(const device float4* Xf4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                            constant Df2GemvArgs& a [[buffer(2)]], device float* Y [[buffer(3)]],
                            device uint2* send_base [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint tid [[thread_index_in_threadgroup]]) {
  (void)send_base;
  threadgroup half4 xs[8 * 64];
  const uint rows = min(a.rows, 8u), o0 = tg * 16u + sg * 2u;
  const bool a0 = o0 < a.N, a1 = (o0 + 1u) < a.N;
  const uint k8 = a.K >> 3, k4 = a.K >> 2, c256 = a.K >> 8;
  const device uint4* W0 = W4 + ulong(a0 ? o0 : 0u) * k8, *W1 = W4 + ulong(a1 ? (o0 + 1u) : 0u) * k8;
  float acc0[8], acc1[8];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 8u; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
  for (uint c = 0u; c < c256; ++c) {
    const uint x_base = c * 64u;
    #pragma clang loop unroll(full)
    for (uint it = 0u; it < 2u; ++it) {
      const uint idx = it * 256u + tid, r = idx >> 6, col = idx & 63u;
      xs[idx] = (r < rows) ? half4(clamp(Xf4[ulong(r) * k4 + x_base + col], -65504.0f, 65504.0f)) : half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a0) {
      const uint4 q0 = W0[c * 32u + lane], q1 = a1 ? W1[c * 32u + lane] : uint4(0u);
      const half4 w0a = df2_u2_h4(uint2(q0.x, q0.y)), w0b = df2_u2_h4(uint2(q0.z, q0.w));
      const half4 w1a = df2_u2_h4(uint2(q1.x, q1.y)), w1b = df2_u2_h4(uint2(q1.z, q1.w));
      #pragma clang loop unroll(full)
      for (uint r = 0u; r < 8u; ++r) {
        const half4 xa = xs[r * 64u + lane * 2u], xb = xs[r * 64u + lane * 2u + 1u];
        acc0[r] += float(dot(w0a, xa) + dot(w0b, xb));
        if (a1) acc1[r] += float(dot(w1a, xa) + dot(w1b, xb));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = 0u; r < rows; ++r) {
    const float v0 = df2_sum32(acc0[r]), v1 = a1 ? df2_sum32(acc1[r]) : 0.0f;
    if (lane == 0u) { if (a0) Y[ulong(r) * a.out_stride + o0] = v0; if (a1) Y[ulong(r) * a.out_stride + o0 + 1u] = v1; }
  }
}

// Fast tiled BF16-MMA GEMM for prompt absorb (rows > 8, 128t, 32x64 output tile, zero FP16 overflow).
kernel void df2_bf16_gemm(const device uchar* X_raw [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                          constant Df2GemvArgs& a [[buffer(2)]], device float* Y [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint XLD = 40u, WLD = 40u;
  threadgroup bfloat xs[32u * XLD], ws[4u * 16u * WLD];
  threadgroup bfloat* w16 = ws + sg * 16u * WLD;
  const uint r0 = tg.y * 32u, o0 = tg.x * 64u + sg * 16u;
  const uint cnt = min(32u, a.rows > r0 ? a.rows - r0 : 0u);
  if (cnt == 0u) return;
  const bool sg_act = o0 < a.N;
  const uint xr = tid >> 2, xc4 = (tid & 3u) << 1, lr = lane >> 1, lc = (lane & 1u) << 4;
  const uint k4 = a.K >> 2, k_steps = a.K >> 5;
  const device float4* Xf4 = reinterpret_cast<const device float4*>(X_raw);
  const device uint2* Xb2 = reinterpret_cast<const device uint2*>(X_raw);
  simdgroup_matrix<float, 8, 8> acc[2][4];
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 4u; ++m) acc[i][m] = simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint kb = 0u; kb < k_steps; ++kb) {
    const uint k_base = kb * 32u;
    float4 v0 = float4(0.0f), v1 = float4(0.0f);
    if (xr < cnt) {
      const ulong xb = ulong(r0 + xr) * k4 + (k_base >> 2) + xc4;
      v0 = a.in_bf16 ? df2_u2_f4(Xb2[xb]) : Xf4[xb];
      v1 = a.in_bf16 ? df2_u2_f4(Xb2[xb + 1u]) : Xf4[xb + 1u];
    }
    const uint k0 = xc4 * 4u;
    xs[(k0 + 0u) * XLD + xr] = bfloat(v0.x); xs[(k0 + 1u) * XLD + xr] = bfloat(v0.y);
    xs[(k0 + 2u) * XLD + xr] = bfloat(v0.z); xs[(k0 + 3u) * XLD + xr] = bfloat(v0.w);
    xs[(k0 + 4u) * XLD + xr] = bfloat(v1.x); xs[(k0 + 5u) * XLD + xr] = bfloat(v1.y);
    xs[(k0 + 6u) * XLD + xr] = bfloat(v1.z); xs[(k0 + 7u) * XLD + xr] = bfloat(v1.w);
    if (sg_act && (o0 + lr) < a.N) {
      const device uint4* w4 = reinterpret_cast<const device uint4*>(W + ulong(o0 + lr) * a.K + k_base + lc);
      threadgroup uint4* d4 = reinterpret_cast<threadgroup uint4*>(w16 + lr * WLD + lc);
      d4[0] = w4[0]; d4[1] = w4[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_act) {
      #pragma clang loop unroll(full)
      for (uint ks = 0u; ks < 4u; ++ks) {
        simdgroup_matrix<bfloat, 8, 8> w0, w1, b;
        simdgroup_load(w0, w16, WLD, ulong2(8u * ks, 0), false);
        simdgroup_load(w1, w16, WLD, ulong2(8u * ks, 8), false);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < 4u; ++m) {
          simdgroup_load(b, xs + (8u * ks) * XLD + m * 8u, XLD, ulong2(0, 0), false);
          simdgroup_multiply_accumulate(acc[0][m], w0, b, acc[0][m]);
          simdgroup_multiply_accumulate(acc[1][m], w1, b, acc[1][m]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!sg_act) return;
  threadgroup float* stg = reinterpret_cast<threadgroup float*>(w16);
  const uint hl = lane >> 4, ll = lane & 15u;
  for (uint hm = 0u; hm < 2u; ++hm) {
    if (hm * 16u >= cnt) break;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
      for (uint dm = 0u; dm < 2u; ++dm)
        simdgroup_store(acc[i][hm * 2u + dm], stg + (dm * 8u) * 16u + (i * 8u), 16u, ulong2(0, 0), true);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0u; row < 16u; row += 2u) {
      const uint r = row + hl;
      if (hm * 16u + r < cnt && (o0 + ll) < a.N)
        Y[ulong(r0 + hm * 16u + r) * a.out_stride + o0 + ll] = stg[r * 16u + ll];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}

inline float4 df2_bf4(float4 v) { return float4(float(bfloat(v.x)), float(bfloat(v.y)), float(bfloat(v.z)), float(bfloat(v.w))); }

struct Df2NormArgs { uint rows, width, add; float eps; };

// Qwen3/DFlash2 RMSNorm with optional bf16-rounded residual add: R += X (if add), out = w * bf16(R * rsqrt(mean+eps)).
kernel void df2_rms_norm(device float4* R4 [[buffer(0)]], const device float4* X4 [[buffer(1)]], const device uint2* W2 [[buffer(2)]], constant Df2NormArgs& a [[buffer(3)]], device float4* out4 [[buffer(4)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const uint n4 = a.width >> 2;
  const ulong off = ulong(row) * n4;
  float sum = 0.0f;
  for (uint i = t; i < n4; i += 256u) {
    float4 v = R4[off + i];
    if (a.add != 0u) { v = df2_bf4(v + X4[off + i]); R4[off + i] = v; }
    sum += dot(v, v);
  }
  sum = df2_sum32(sum);
  if (lane == 0u) part[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (uint s = 0u; s < 8u; ++s) tot += part[s];
  const float inv = precise::rsqrt(tot / float(a.width) + a.eps);
  for (uint i = t; i < n4; i += 256u) out4[off + i] = df2_bf4(df2_u2_f4(W2[i]) * df2_bf4(R4[off + i] * inv));
}

struct Df2ConvArgs { uint rows, width, block, groups, side, pad0, pad1, pad2; };

// Grouped two-tap block convolution (DFlashGroupedConv): writes pre-scaled half (y * 2^-5) for 0-barrier Q8 GEMV!
kernel void df2_conv(const device float* X [[buffer(0)]], const device float* D [[buffer(1)]], const device bfloat* base [[buffer(2)]], constant Df2ConvArgs& a [[buffer(3)]], device half* Y [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const uint ch = p.x, r = p.y, g = ch / (a.width / a.groups);
  const device float* d = D + ulong(r) * 4u * a.groups + a.side * 2u * a.groups;
  const float c0 = float(bfloat(float(base[ulong(a.side * 2u) * a.width + ch]) + float(bfloat(d[g]))));
  const float c1 = float(bfloat(float(base[ulong(a.side * 2u + 1u) * a.width + ch]) + float(bfloat(d[a.groups + g]))));
  float y = float(bfloat(c0 * float(bfloat(X[ulong(r) * a.width + ch]))));
  if (r % a.block >= 1u) y = float(bfloat(y + float(bfloat(c1 * float(bfloat(X[ulong(r - 1u) * a.width + ch]))))));
  Y[ulong(r) * a.width + ch] = half(clamp(y * 0.03125f, -65504.0f, 65504.0f));
}

struct Df2ConvNormArgs { uint rows, width, block, groups; float eps; uint pad0, pad1, pad2; };

// Fused DFlashGroupedConv (side=1) + bf16 residual add + RMSNorm: saves 12 dispatches/barriers & yc traffic!
kernel void df2_conv_add_rms(device float4* R4 [[buffer(0)]], const device float4* Y4 [[buffer(1)]], const device float* D [[buffer(2)]], const device uint2* base2 [[buffer(3)]], const device uint2* W2 [[buffer(4)]], constant Df2ConvNormArgs& a [[buffer(5)]], device float4* out4 [[buffer(6)]], uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const uint n4 = a.width >> 2, g_div = (a.width / a.groups) >> 2;
  const ulong off = ulong(r) * n4;
  const device float* d = D + ulong(r) * 4u * a.groups + 2u * a.groups;
  const bool has_prev = (r % a.block) >= 1u;
  float sum = 0.0f;
  for (uint i = t; i < n4; i += 256u) {
    const uint g = i / g_div;
    const float4 c0 = df2_bf4(df2_u2_f4(base2[2u * n4 + i]) + float(bfloat(d[g])));
    float4 yc = df2_bf4(c0 * df2_bf4(Y4[off + i]));
    if (has_prev) {
      const float4 c1 = df2_bf4(df2_u2_f4(base2[3u * n4 + i]) + float(bfloat(d[a.groups + g])));
      yc = df2_bf4(yc + df2_bf4(c1 * df2_bf4(Y4[ulong(r - 1u) * n4 + i])));
    }
    const float4 v = df2_bf4(R4[off + i] + yc);
    R4[off + i] = v;
    sum += dot(v, v);
  }
  sum = df2_sum32(sum);
  if (lane == 0u) part[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (uint s = 0u; s < 8u; ++s) tot += part[s];
  const float inv = precise::rsqrt(tot / float(a.width) + a.eps);
  for (uint i = t; i < n4; i += 256u) out4[off + i] = df2_bf4(df2_u2_f4(W2[i]) * df2_bf4(R4[off + i] * inv));
}

struct Df2PrepArgs {
  uint rows, heads, kv_heads, pos0;
  uint ring, dyn_meta;
  float eps, theta;
};

// Qwen3 per-head RMSNorm (128) + split-half RoPE (dyn_meta: 1=absorb<=meta[0], 2=draft at meta[2]).
kernel void df2_prep_qkv(const device float* Q_in [[buffer(0)]], const device float* K_in [[buffer(1)]],
                         const device float* V_in [[buffer(2)]], const device bfloat* qw [[buffer(3)]],
                         const device bfloat* kw [[buffer(4)]], constant Df2PrepArgs& a [[buffer(5)]],
                         device bfloat* Q_out [[buffer(6)]], device bfloat* K_ring [[buffer(7)]],
                         device bfloat* V_ring [[buffer(8)]], const device uint* meta [[buffer(9)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint unit = tg.x, row = tg.y;
  const uint lim = (a.dyn_meta == 1u) ? min(a.rows, meta[0]) : a.rows;
  if (unit >= a.heads + a.kv_heads || row >= lim) return;
  const bool is_q = unit < a.heads;
  const uint head = is_q ? unit : (unit - a.heads);
  const device float* src = is_q ? (Q_in + (ulong(row) * a.heads + head) * 128u)
                                 : (K_in + (ulong(row) * a.kv_heads + head) * 128u);
  const device bfloat* w = is_q ? qw : kw;
  const uint p = ((a.dyn_meta == 2u) ? meta[2] : a.pos0) + row;
  const ulong slot = ulong(p % a.ring);
  device bfloat* y = is_q ? (Q_out + (ulong(row) * a.heads + head) * 128u)
                          : (K_ring + (slot * a.kv_heads + head) * 128u);
  float v[4];
  float sum = 0.0f;
  for (uint j = 0u; j < 4u; ++j) {
    v[j] = float(bfloat(src[lane + 32u * j]));
    sum = fma(v[j], v[j], sum);
  }
  const float inv = precise::rsqrt(df2_sum32(sum) / 128.0f + a.eps);
  threadgroup float nrm[128];
  for (uint j = 0u; j < 4u; ++j) nrm[lane + 32u * j] = float(bfloat(float(w[lane + 32u * j]) * float(bfloat(v[j] * inv))));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0u; j < 4u; ++j) {
    const uint d = lane + 32u * j, i = d & 63u;
    const float ang = float(p) * (1.0f / precise::pow(a.theta, float(2u * i) / 128.0f));
    const float cs = float(bfloat(precise::cos(ang))), sn = float(bfloat(precise::sin(ang)));
    const float rot = (d < 64u) ? -nrm[d + 64u] : nrm[d - 64u];
    y[d] = bfloat(float(bfloat(nrm[d] * cs)) + float(bfloat(rot * sn)));
  }
  if (is_q) return;
  const device float* vs = V_in + (ulong(row) * a.kv_heads + head) * 128u;
  device bfloat* vd = V_ring + (slot * a.kv_heads + head) * 128u;
  for (uint j = 0u; j < 4u; ++j) vd[lane + 32u * j] = bfloat(vs[lane + 32u * j]);
}

struct Df2AttnArgs {
  uint rows, heads, kv_heads, gqa;
  uint lo, hi, ring, n_splits;
  float scale;
  uint dyn_meta, window, sl_from;
  uint split_sz, pad0, pad1, pad2;
};

// Non-causal sliding-window GQA attention split (32 or 128 keys/split, 8 SGs share 1 KV head, dynamic window).
kernel void df2_attend_split(const device bfloat* Q [[buffer(0)]], const device bfloat* K_ring [[buffer(1)]],
                             const device bfloat* V_ring [[buffer(2)]], constant Df2AttnArgs& a [[buffer(3)]],
                             device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                             device float4* po [[buffer(6)]], const device uint* meta [[buffer(7)]],
                             uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint kv_h = tg.x, row = tg.y, sp = tg.z;
  const uint qh = kv_h * a.gqa + sg;
  if (kv_h >= a.kv_heads || row >= a.rows || sp >= a.n_splits || sg >= a.gqa) return;
  uint lo = a.lo, hi = a.hi, act_sp = a.n_splits;
  if (a.dyn_meta != 0u) {
    const uint ap = meta[2];
    lo = max((ap + 1u > a.window) ? (ap + 1u - a.window) : 0u, a.sl_from);
    hi = ap + a.rows;
    act_sp = ((hi - lo) + a.split_sz - 1u) / a.split_sz;
    if (sp >= act_sp) return;
  }
  const uint k_start = lo + sp * a.split_sz, k_end = min(hi, k_start + a.split_sz);
  const device uint2* q2 = reinterpret_cast<const device uint2*>(Q + (ulong(row) * a.heads + qh) * 128u);
  const float4 qv = df2_u2_f4(q2[lane]);
  float4 ov = float4(0.0f);
  float m = -INFINITY, l = 0.0f;
  for (uint pos = k_start; pos < k_end; ++pos) {
    const ulong slot = ulong(pos % a.ring);
    const ulong kv_off = (slot * a.kv_heads + kv_h) * 128u;
    const float4 kv = df2_u2_f4(reinterpret_cast<const device uint2*>(K_ring + kv_off)[lane]);
    const float4 vv = df2_u2_f4(reinterpret_cast<const device uint2*>(V_ring + kv_off)[lane]);
    const float sc = df2_sum32(dot(qv, kv)) * a.scale;
    const float mn = max(m, sc);
    const float corr = (m == -INFINITY) ? 0.0f : exp(m - mn), ev = exp(sc - mn);
    l = fma(l, corr, ev);
    ov = fma(float4(ev), vv, ov * corr);
    m = mn;
  }
  const ulong p_idx = (ulong(row) * a.heads + qh) * a.n_splits + sp;
  if (lane == 0u) { pm[p_idx] = m; pl[p_idx] = l; }
  po[p_idx * 32u + lane] = ov;
}

kernel void df2_attend_merge(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                             const device float4* po [[buffer(2)]], constant Df2AttnArgs& a [[buffer(3)]],
                             device half4* out4 [[buffer(4)]], const device uint* meta [[buffer(5)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint qh = tg.x, row = tg.y;
  if (qh >= a.heads || row >= a.rows) return;
  uint act_sp = a.n_splits;
  if (a.dyn_meta != 0u) {
    const uint ap = meta[2];
    const uint lo = max((ap + 1u > a.window) ? (ap + 1u - a.window) : 0u, a.sl_from);
    act_sp = ((ap + a.rows - lo) + a.split_sz - 1u) / a.split_sz;
  }
  const ulong base = (ulong(row) * a.heads + qh) * a.n_splits;
  float M = -INFINITY;
  for (uint sp = 0u; sp < act_sp; ++sp) M = max(M, pm[base + sp]);
  float L = 0.0f;
  float4 acc = float4(0.0f);
  for (uint sp = 0u; sp < act_sp; ++sp) {
    const float ls = pl[base + sp];
    if (ls > 0.0f) {
      const float f = exp(pm[base + sp] - M);
      L += ls * f;
      acc = fma(po[(base + sp) * 32u + lane], float4(f), acc);
    }
  }
  const float4 o = (L > 0.0f) ? (acc / L) : float4(0.0f);
  const float4 ob = float4(float(bfloat(o.x)), float(bfloat(o.y)), float(bfloat(o.z)), float(bfloat(o.w)));
  out4[(ulong(row) * a.heads + qh) * 32u + lane] = half4(clamp(ob * 0.03125f, -65504.0f, 65504.0f));
}

struct Df2QuantArgs { uint N, K, pad0, pad1; };

// Load-time GPU quantization: BF16 [N, K] -> Q8_g64 int8 [N, K] + half scales [N, K/64] (1 SG per 64-elem group).
kernel void df2_quant_q8_g64(const device bfloat* W [[buffer(0)]], constant Df2QuantArgs& a [[buffer(1)]],
                             device char* Q [[buffer(2)]], device half* S [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg.x, row = tg.y;
  const uint g_cols = a.K >> 6;
  if (g >= g_cols || row >= a.N) return;
  const ulong base = ulong(row) * a.K + ulong(g) * 64u;
  const float v0 = float(W[base + lane]), v1 = float(W[base + lane + 32u]);
  const float amax = simd_max(max(fabs(v0), fabs(v1)));
  const half hs = half((amax > 0.0f) ? (amax / 127.0f) : 0.0f);
  const float inv_s = (float(hs) > 0.0f) ? (1.0f / float(hs)) : 0.0f;
  Q[base + lane] = char(clamp(rint(v0 * inv_s), -127.0f, 127.0f));
  Q[base + lane + 32u] = char(clamp(rint(v1 * inv_s), -127.0f, 127.0f));
  if (lane == 0u) S[ulong(row) * g_cols + g] = hs;
}

inline float df2_q8_dot16(uint4 q, half4 x0, half4 x1, half4 x2, half4 x3) {
  return float(dot(half4(as_type<char4>(q.x)), x0) + dot(half4(as_type<char4>(q.y)), x1) +
               dot(half4(as_type<char4>(q.z)), x2) + dot(half4(as_type<char4>(q.w)), x3));
}

// 513+ GB/s Q8_g64 GEMV for rows=1..8 (in_bf16: 0=f4, 1=bf16 fc_q8, 2=half4 o_w/down_w + uint2 pack).
kernel void df2_q8_gemv_n(const device uchar* X_raw [[buffer(0)]], const device uint4* W4 [[buffer(1)]], const device half* S [[buffer(2)]], constant Df2GemvArgs& a [[buffer(3)]], device float* Y [[buffer(4)]], device uint2* send_base [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half4 xs[8 * 128];
  const uint rows = min(a.rows, 8u), o0 = tg * 16u + sg * 2u;
  if (o0 >= a.N) return;
  const uint k16 = a.K >> 4, k4 = a.K >> 2, c512 = a.K >> 9, g_cols = a.K >> 6, sub_g = lane >> 2;
  const device uint4* W0 = W4 + ulong(o0) * k16, *W1 = W0 + k16;
  const device half* S0 = S + ulong(o0) * g_cols, *S1 = S0 + g_cols;
  const device float4* Xf4 = reinterpret_cast<const device float4*>(X_raw);
  const device uint2* Xb2 = reinterpret_cast<const device uint2*>(X_raw);
  const device half4* Xh = reinterpret_cast<const device half4*>(X_raw);
  const float x_sc = (a.in_bf16 == 1u) ? 0.00390625f : 0.03125f;
  float acc0[8], acc1[8];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 8u; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
  for (uint c = 0u; c < c512; ++c) {
    const uint x_base = c * 128u;
    #pragma clang loop unroll(full)
    for (uint it = 0u; it < 4u; ++it) {
      const uint idx = it * 256u + tid, r = idx >> 7, col = idx & 127u;
      if (r < rows) {
        const ulong xo = ulong(r) * k4 + x_base + col;
        xs[idx] = (a.in_bf16 == 2u) ? Xh[xo]
                : half4(clamp(((a.in_bf16 == 1u) ? df2_u2_f4(Xb2[xo]) : Xf4[xo]) * x_sc, -65504.0f, 65504.0f));
      } else xs[idx] = half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint gi = c * 8u + sub_g;
    const float sc0 = float(S0[gi]), sc1 = float(S1[gi]);
    const uint4 q0 = W0[c * 32u + lane], q1 = W1[c * 32u + lane];
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; ++r) {
      const uint b = r * 128u + lane * 4u;
      const half4 x0 = xs[b], x1 = xs[b + 1u], x2 = xs[b + 2u], x3 = xs[b + 3u];
      acc0[r] = fma(df2_q8_dot16(q0, x0, x1, x2, x3), sc0, acc0[r]);
      acc1[r] = fma(df2_q8_dot16(q1, x0, x1, x2, x3), sc1, acc1[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float* y_stg = reinterpret_cast<threadgroup float*>(xs);
  const float scale_out = (a.in_bf16 == 1u) ? 256.0f : 32.0f;
  for (uint r = 0u; r < rows; ++r) {
    const float v0 = df2_sum32(acc0[r]) * scale_out, v1 = df2_sum32(acc1[r]) * scale_out;
    if (lane == 0u) { Y[ulong(r) * a.out_stride + o0] = v0; Y[ulong(r) * a.out_stride + o0 + 1u] = v1; y_stg[r * 16u + sg * 2u] = v0; y_stg[r * 16u + sg * 2u + 1u] = v1; }
  }
  if (a.pack_tp > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < rows * 4u) {
      const uint r = tid >> 2, q = tid & 3u, col4 = tg * 4u + q, sb = r * 16u + q * 4u;
      const uint2 pk = uint2(df2_bf16x2(y_stg[sb], y_stg[sb + 1u]), df2_bf16x2(y_stg[sb + 2u], y_stg[sb + 3u]));
      const ulong idx2 = ulong(r) * (a.out_stride >> 2) + col4;
      for (uint p = 0u; p < a.pack_tp; ++p) if (p != a.pack_rank) send_base[ulong(p) * a.pack_slot_u2 + idx2] = pk;
    }
  }
}

// Zero-inner-barrier K-split Q=2 256t Q8 GEMV for narrow-N (in_bf16==2: half4, else float4): 4-5x faster!
kernel void df2_q8_gemv_q1(const device uchar* X_raw [[buffer(0)]], const device uint4* W4 [[buffer(1)]], const device half* S [[buffer(2)]], constant Df2GemvArgs& a [[buffer(3)]], device float* Y [[buffer(4)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2 * 8 * 8];
  const uint o0 = g * 2u, rows = min(a.rows, 8u);
  if (o0 >= a.N) return;
  const uint k16 = a.K >> 4, k4 = a.K >> 2, c512 = a.K >> 9, g_cols = a.K >> 6, sub_g = lane >> 2;
  const device uint4* W0 = W4 + ulong(o0) * k16, *W1 = W0 + k16;
  const device half* S0 = S + ulong(o0) * g_cols, *S1 = S0 + g_cols;
  const device float4* X4 = reinterpret_cast<const device float4*>(X_raw);
  const device half4* Xh = reinterpret_cast<const device half4*>(X_raw);
  float s0[8], s1[8];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 8u; ++r) { s0[r] = 0.0f; s1[r] = 0.0f; }
  for (uint c = sg; c < c512; c += 8u) {
    const uint gi = c * 8u + sub_g;
    const float sc0 = float(S0[gi]), sc1 = float(S1[gi]);
    const uint4 q0 = W0[c * 32u + lane], q1 = W1[c * 32u + lane];
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; ++r) if (r < rows) {
      const ulong xo = ulong(r) * k4 + c * 128u + lane * 4u;
      const half4 x0 = (a.in_bf16 == 2u) ? Xh[xo] : half4(clamp(X4[xo] * 0.03125f, -65504.0f, 65504.0f));
      const half4 x1 = (a.in_bf16 == 2u) ? Xh[xo + 1u] : half4(clamp(X4[xo + 1u] * 0.03125f, -65504.0f, 65504.0f));
      const half4 x2 = (a.in_bf16 == 2u) ? Xh[xo + 2u] : half4(clamp(X4[xo + 2u] * 0.03125f, -65504.0f, 65504.0f));
      const half4 x3 = (a.in_bf16 == 2u) ? Xh[xo + 3u] : half4(clamp(X4[xo + 3u] * 0.03125f, -65504.0f, 65504.0f));
      s0[r] = fma(df2_q8_dot16(q0, x0, x1, x2, x3), sc0, s0[r]);
      s1[r] = fma(df2_q8_dot16(q1, x0, x1, x2, x3), sc1, s1[r]);
    }
  }
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 8u; ++r) if (r < rows) {
    const float v0 = df2_sum32(s0[r]), v1 = df2_sum32(s1[r]);
    if (lane == 0u) { part[r * 8u + sg] = v0; part[(8u + r) * 8u + sg] = v1; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane < 16u && (lane & 7u) < rows) {
    float tot = part[lane * 8u];
    #pragma clang loop unroll(full)
    for (uint k = 1u; k < 8u; ++k) tot += part[lane * 8u + k];
    Y[ulong(lane & 7u) * a.out_stride + o0 + (lane >> 3)] = tot * 32.0f;
  }
}

// Zero-barrier Q=2 256t Q8_g64 GEMV reading pre-scaled half4 Xh from L2 (q_w: 66.8 -> 47.2 us!).
kernel void df2_q8_gemv_q2_xh(const device half4* Xh [[buffer(0)]], const device uint4* W4 [[buffer(1)]], const device half* S [[buffer(2)]], constant Df2GemvArgs& a [[buffer(3)]], device float* Y [[buffer(4)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = min(a.rows, 8u), o0 = tg * 16u + sg * 2u;
  if (o0 >= a.N) return;
  const uint k16 = a.K >> 4, k4 = a.K >> 2, c512 = a.K >> 9, g_cols = a.K >> 6, sub_g = lane >> 2;
  const device uint4* W0 = W4 + ulong(o0) * k16, *W1 = W0 + k16;
  const device half* S0 = S + ulong(o0) * g_cols, *S1 = S0 + g_cols;
  float acc0[8], acc1[8];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 8u; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
  for (uint c = 0u; c < c512; ++c) {
    const uint gi = c * 8u + sub_g, xb = c * 128u + lane * 4u;
    const float sc0 = float(S0[gi]), sc1 = float(S1[gi]);
    const uint4 q0 = W0[c * 32u + lane], q1 = W1[c * 32u + lane];
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; ++r) {
      const device half4* xp = Xh + ulong(r) * k4 + xb;
      const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
      acc0[r] = fma(df2_q8_dot16(q0, x0, x1, x2, x3), sc0, acc0[r]);
      acc1[r] = fma(df2_q8_dot16(q1, x0, x1, x2, x3), sc1, acc1[r]);
    }
  }
  for (uint r = 0u; r < rows; ++r) {
    const float v0 = df2_sum32(acc0[r]) * 32.0f, v1 = df2_sum32(acc1[r]) * 32.0f;
    if (lane == 0u) { Y[ulong(r) * a.out_stride + o0] = v0; Y[ulong(r) * a.out_stride + o0 + 1u] = v1; }
  }
}

// Fused Q8_g64 gate + up + SiLU*up for rows=1..8 (256t, 8 outs/TG, reads half4 Xh, writes pre-scaled half act!).
kernel void df2_q8_gu_silu_n(const device half4* Xh [[buffer(0)]], const device uint4* GW4 [[buffer(1)]], const device half* GS [[buffer(2)]], const device uint4* UW4 [[buffer(3)]], const device half* US [[buffer(4)]], constant Df2GemvArgs& a [[buffer(5)]], device half* act [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half4 xs[8 * 128];
  const uint rows = min(a.rows, 8u), o = tg * 8u + sg;
  if (o >= a.N) return;
  const uint k16 = a.K >> 4, k4 = a.K >> 2, c512 = a.K >> 9, g_cols = a.K >> 6, sub_g = lane >> 2;
  const device uint4* GW = GW4 + ulong(o) * k16, *UW = UW4 + ulong(o) * k16;
  const device half* GSp = GS + ulong(o) * g_cols, *USp = US + ulong(o) * g_cols;
  float ag[8], au[8];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 8u; ++r) { ag[r] = 0.0f; au[r] = 0.0f; }
  for (uint c = 0u; c < c512; ++c) {
    const uint x_base = c * 128u;
    #pragma clang loop unroll(full)
    for (uint it = 0u; it < 4u; ++it) {
      const uint idx = it * 256u + tid, r = idx >> 7, col = idx & 127u;
      xs[idx] = (r < rows) ? Xh[ulong(r) * k4 + x_base + col] : half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint gi = c * 8u + sub_g;
    const float scg = float(GSp[gi]), scu = float(USp[gi]);
    const uint4 qg = GW[c * 32u + lane], qu = UW[c * 32u + lane];
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; ++r) {
      const uint b = r * 128u + lane * 4u;
      const half4 x0 = xs[b], x1 = xs[b + 1u], x2 = xs[b + 2u], x3 = xs[b + 3u];
      ag[r] = fma(df2_q8_dot16(qg, x0, x1, x2, x3), scg, ag[r]);
      au[r] = fma(df2_q8_dot16(qu, x0, x1, x2, x3), scu, au[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = 0u; r < rows; ++r) {
    const float gv = df2_sum32(ag[r]) * 32.0f, uv = df2_sum32(au[r]) * 32.0f;
    if (lane == 0u) {
      const float e = precise::exp(-fabs(gv)), y = 1.0f / (1.0f + e), s = gv < 0.0f ? e * y : y;
      const float v = float(bfloat(float(bfloat(gv * s)) * float(bfloat(uv))));
      act[ulong(r) * a.out_stride + o] = half(clamp(v * 0.03125f, -65504.0f, 65504.0f));
    }
  }
}

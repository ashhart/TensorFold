// GLM-5.3 specialized FP8 dense GEMV/fused3/pack + single-read MLA merge/UV + MMA absorb.
#include <metal_stdlib>
#if __METAL_VERSION__ >= 310
#include <metal_simdgroup_matrix>
#endif
using namespace metal;

constant constexpr uint AF_LAT = 512, AF_NOPE = 192, AF_ROPE = 64, AF_QK = 256, AF_VDIM = 256;

struct MlaRow { uint slot, pos, list, nkeys; };
struct Fp8MatArgs { uint outs, ins, rows, x_stride, y_stride, y_col, sb_cols, pad0; ulong w_off, s_off; };
struct Fp8Mat3Args { uint o0, o1, o2, ins, rows, x_stride, y_stride, sb_cols, yc0, yc1, yc2, g0, g1, g2, pad0, pad1; ulong w0, s0, w1, s1, w2, s2; };
struct MlaArgs { uint rows, heads, head0, slot_keys, proj_stride, q_stride, pad2, kvb_in_blocks; float eps, scale, pad0, pad1; ulong kvb, kvb_scale; uint tree_base, tree_p0, tree_p1, tree_p2, tree_anc[8]; };

inline half4 af_fp8x4_h(uint q) {
  const half2 a = as_type<half2>(((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8));
  const half2 b = as_type<half2>((((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8));
  return half4(a.x, b.x, a.y, b.y);
}
inline uint af_bf16x2(float a, float b) {
  return uint(as_type<ushort>(bfloat(a))) | (uint(as_type<ushort>(bfloat(b))) << 16);
}
inline float af_sum32(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

// Constant-arg FP8-only R=1 GEMV helper (Q=2 128t, 8 outs/TG, direct X4, outs%16==0): 567-613 GB/s cold DRAM!
inline void af_fp8_gemv1_core(const device float4* X4, const device uchar* W, ulong wo, ulong so, uint outs, uint ins,
                              uint yc, uint sb_cols, device float* Y, uint lg, uint sg, uint lane) {
  const uint o0 = lg * 8u + sg * 2u;
  if (o0 >= outs) return;
  const uint c512 = ins >> 9, sub_sb = lane >> 3;
  const device float* S = reinterpret_cast<const device float*>(W + so);
  const ulong row0 = wo + ulong(o0) * ins, row1 = row0 + ins, sc0 = ulong(o0 >> 7) * sb_cols;
  float a0 = 0.0f, a1 = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    const uint xo = c * 128u + lane * 4u;
    const half4 x0 = half4(X4[xo]), x1 = half4(X4[xo + 1u]), x2 = half4(X4[xo + 2u]), x3 = half4(X4[xo + 3u]);
    const float s0 = S[sc0 + c * 4u + sub_sb] * 256.0f;
    const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + c * 512u + lane * 16u);
    const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row1 + c * 512u + lane * 16u);
    a0 = fma(float(dot(af_fp8x4_h(q0.x), x0)) + float(dot(af_fp8x4_h(q0.y), x1)) +
             float(dot(af_fp8x4_h(q0.z), x2)) + float(dot(af_fp8x4_h(q0.w), x3)), s0, a0);
    a1 = fma(float(dot(af_fp8x4_h(q1.x), x0)) + float(dot(af_fp8x4_h(q1.y), x1)) +
             float(dot(af_fp8x4_h(q1.z), x2)) + float(dot(af_fp8x4_h(q1.w), x3)), s0, a1);
  }
  const float v0 = af_sum32(a0), v1 = af_sum32(a1);
  if (lane == 0u) *reinterpret_cast<device float2*>(Y + ulong(yc + o0)) = float2(v0, v1);
}

kernel void glm_fp8_gemv_1(const device float4* X4 [[buffer(0)]], const device uchar* W [[buffer(1)]],
                           constant Fp8MatArgs& a [[buffer(2)]], device float* Y [[buffer(3)]],
                           uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  af_fp8_gemv1_core(X4, W, a.w_off, a.s_off, a.outs, a.ins, a.y_col, a.sb_cols, Y, g, sg, lane);
}

// Constant-arg FP8-only R=1 Q=4 128t (16 outs/TG, 0 TG barriers, float4+uint2 pack): 659 GB/s cold DRAM on o_proj!
kernel void glm_fp8_o_pack_1(const device float4* X4 [[buffer(0)]], const device uchar* W [[buffer(1)]],
                             constant Fp8MatArgs& a [[buffer(2)]], device float* Y [[buffer(3)]],
                             constant uint4& pk [[buffer(4)]], device uint2* send_base [[buffer(5)]],
                             uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint o0 = g * 16u + sg * 4u;
  if (o0 >= a.outs) return;
  const uint c512 = a.ins >> 9, sub_sb = lane >> 3;
  const device float* S = reinterpret_cast<const device float*>(W + a.s_off);
  const ulong row0 = a.w_off + ulong(o0) * a.ins;
  const ulong sc0 = ulong(o0 >> 7) * a.sb_cols;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    const uint xo = c * 128u + lane * 4u;
    const half4 x0 = half4(X4[xo]), x1 = half4(X4[xo + 1u]), x2 = half4(X4[xo + 2u]), x3 = half4(X4[xo + 3u]);
    const float s0 = S[sc0 + c * 4u + sub_sb] * 256.0f;
    const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + c * 512u + lane * 16u);
    const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row0 + a.ins + c * 512u + lane * 16u);
    const uint4 q2 = *reinterpret_cast<const device uint4*>(W + row0 + 2ul * a.ins + c * 512u + lane * 16u);
    const uint4 q3 = *reinterpret_cast<const device uint4*>(W + row0 + 3ul * a.ins + c * 512u + lane * 16u);
    a0 = fma(float(dot(af_fp8x4_h(q0.x), x0)) + float(dot(af_fp8x4_h(q0.y), x1)) +
             float(dot(af_fp8x4_h(q0.z), x2)) + float(dot(af_fp8x4_h(q0.w), x3)), s0, a0);
    a1 = fma(float(dot(af_fp8x4_h(q1.x), x0)) + float(dot(af_fp8x4_h(q1.y), x1)) +
             float(dot(af_fp8x4_h(q1.z), x2)) + float(dot(af_fp8x4_h(q1.w), x3)), s0, a1);
    a2 = fma(float(dot(af_fp8x4_h(q2.x), x0)) + float(dot(af_fp8x4_h(q2.y), x1)) +
             float(dot(af_fp8x4_h(q2.z), x2)) + float(dot(af_fp8x4_h(q2.w), x3)), s0, a2);
    a3 = fma(float(dot(af_fp8x4_h(q3.x), x0)) + float(dot(af_fp8x4_h(q3.y), x1)) +
             float(dot(af_fp8x4_h(q3.z), x2)) + float(dot(af_fp8x4_h(q3.w), x3)), s0, a3);
  }
  const float v0 = af_sum32(a0), v1 = af_sum32(a1), v2 = af_sum32(a2), v3 = af_sum32(a3);
  if (lane != 0u) return;
  const ulong at0 = ulong(a.y_col + o0);
  *reinterpret_cast<device float4*>(Y + at0) = float4(v0, v1, v2, v3);
  if (pk.y > 1u) {
    const uint2 p2 = uint2(af_bf16x2(v0, v1), af_bf16x2(v2, v3));
    for (uint p = 0u; p < pk.y; ++p) if (p != pk.x) send_base[ulong(p) * pk.z + (at0 >> 2)] = p2;
  }
}

// Constant-arg FP8-only R=6..8 Q=2 256t (16 outs/TG, outs%16==0) GEMV reading pre-cast half4 Xh w/ uint2 peer pack.
template <uint R, bool PACK>
inline void af_fp8_q2_256(const device half4* Xh, const device uchar* W, ulong wo, ulong so, uint outs, uint ins,
                          uint x_stride, uint y_stride, uint yc, uint sb_cols, device float* Y, uint4 pk,
                          device uint2* send_base, uint lg, uint sg, uint lane, uint tid, threadgroup half4* xh4) {
  const uint o0 = lg * 16u + sg * 2u;
  if (o0 >= outs) return;
  const uint c512 = ins >> 9, sub_sb = lane >> 3, xs4 = x_stride >> 2;
  const device float* S = reinterpret_cast<const device float*>(W + so);
  const ulong row0 = wo + ulong(o0) * ins, row1 = row0 + ins, sc0 = ulong(o0 >> 7) * sb_cols;
  float a0[R], a1[R];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) a0[r] = 0.0f, a1[r] = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    for (uint idx = tid; idx < R * 128u; idx += 256u) {
      const uint rr = idx >> 7, col4 = idx & 127u;
      xh4[idx] = Xh[ulong(rr) * xs4 + c * 128u + col4];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s0 = S[sc0 + c * 4u + sub_sb] * 256.0f;
    const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + c * 512u + lane * 16u);
    const half4 w00 = af_fp8x4_h(q0.x), w01 = af_fp8x4_h(q0.y), w02 = af_fp8x4_h(q0.z), w03 = af_fp8x4_h(q0.w);
    const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row1 + c * 512u + lane * 16u);
    const half4 w10 = af_fp8x4_h(q1.x), w11 = af_fp8x4_h(q1.y), w12 = af_fp8x4_h(q1.z), w13 = af_fp8x4_h(q1.w);
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < R; ++r) {
      const uint xb = r * 128u + lane * 4u;
      const half4 x0 = xh4[xb], x1 = xh4[xb + 1u], x2 = xh4[xb + 2u], x3 = xh4[xb + 3u];
      a0[r] = fma(float(dot(w00, x0)) + float(dot(w01, x1)) + float(dot(w02, x2)) + float(dot(w03, x3)), s0, a0[r]);
      a1[r] = fma(float(dot(w10, x0)) + float(dot(w11, x1)) + float(dot(w12, x2)) + float(dot(w13, x3)), s0, a1[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float* y_stg = reinterpret_cast<threadgroup float*>(xh4);
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) {
    const float v0 = af_sum32(a0[r]), v1 = af_sum32(a1[r]);
    if (lane == 0u) {
      *reinterpret_cast<device float2*>(Y + ulong(r) * y_stride + yc + o0) = float2(v0, v1);
      if (PACK && pk.y > 1u) { y_stg[r * 16u + sg * 2u] = v0; y_stg[r * 16u + sg * 2u + 1u] = v1; }
    }
  }
  if (PACK && pk.y > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < R * 4u) {
      const uint r = tid >> 2, q = tid & 3u, sb = r * 16u + q * 4u;
      const uint2 p2 = uint2(af_bf16x2(y_stg[sb], y_stg[sb + 1u]), af_bf16x2(y_stg[sb + 2u], y_stg[sb + 3u]));
      const ulong idx2 = (ulong(r) * y_stride + yc + lg * 16u + q * 4u) >> 2;
      for (uint p = 0u; p < pk.y; ++p) if (p != pk.x) send_base[ulong(p) * pk.z + idx2] = p2;
    }
  }
}

// Zero-barrier R=3..5 Q=2 FP8 GEMV reading pre-cast half4 Xh directly (SG_N=4 128t for x_g/q_b3/o5, SG_N=8 256t o3/o4).
template <uint R, uint SG_N, bool PACK, bool DIRECT_U1>
inline void af_fp8_dir_xh(const device half4* Xh, const device uchar* W, ulong wo, ulong so, uint outs, uint ins,
                          uint x_stride, uint y_stride, uint yc, uint sb_cols, device float* Y, uint4 pk,
                          device uint2* send_base, uint lg, uint sg, uint lane, uint tid, threadgroup float* y_stg) {
  const uint o0 = lg * (SG_N * 2u) + sg * 2u;
  if (o0 >= outs) return;
  const uint c512 = ins >> 9, sub_sb = lane >> 3, xs4 = x_stride >> 2;
  const device float* S = reinterpret_cast<const device float*>(W + so);
  const ulong row0 = wo + ulong(o0) * ins, row1 = row0 + ins, sc0 = ulong(o0 >> 7) * sb_cols;
  float a0[R], a1[R];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) a0[r] = 0.0f, a1[r] = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    const float s0 = S[sc0 + c * 4u + sub_sb] * 256.0f;
    const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + c * 512u + lane * 16u);
    const half4 w00 = af_fp8x4_h(q0.x), w01 = af_fp8x4_h(q0.y), w02 = af_fp8x4_h(q0.z), w03 = af_fp8x4_h(q0.w);
    const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row1 + c * 512u + lane * 16u);
    const half4 w10 = af_fp8x4_h(q1.x), w11 = af_fp8x4_h(q1.y), w12 = af_fp8x4_h(q1.z), w13 = af_fp8x4_h(q1.w);
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < R; ++r) {
      const ulong xb = ulong(r) * xs4 + c * 128u + lane * 4u;
      const half4 x0 = Xh[xb], x1 = Xh[xb + 1u], x2 = Xh[xb + 2u], x3 = Xh[xb + 3u];
      a0[r] = fma(float(dot(w00, x0)) + float(dot(w01, x1)) + float(dot(w02, x2)) + float(dot(w03, x3)), s0, a0[r]);
      a1[r] = fma(float(dot(w10, x0)) + float(dot(w11, x1)) + float(dot(w12, x2)) + float(dot(w13, x3)), s0, a1[r]);
    }
  }
  device uint* send_u1 = reinterpret_cast<device uint*>(send_base);
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) {
    const float v0 = af_sum32(a0[r]), v1 = af_sum32(a1[r]);
    if (lane == 0u) {
      const ulong at = ulong(r) * y_stride + yc + o0;
      *reinterpret_cast<device float2*>(Y + at) = float2(v0, v1);
      if (PACK && DIRECT_U1 && pk.y > 1u) {
        const uint p1 = af_bf16x2(v0, v1);
        const ulong st1 = ulong(pk.z) * 2ul;
        for (uint p = 0u; p < pk.y; ++p) if (p != pk.x) send_u1[ulong(p) * st1 + (at >> 1)] = p1;
      } else if (PACK && !DIRECT_U1 && pk.y > 1u) {
        y_stg[r * (SG_N * 2u) + sg * 2u] = v0; y_stg[r * (SG_N * 2u) + sg * 2u + 1u] = v1;
      }
    }
  }
  if (PACK && !DIRECT_U1 && pk.y > 1u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < R * (SG_N / 2u)) {
      const uint r = tid / (SG_N / 2u), q = tid % (SG_N / 2u), sb = r * (SG_N * 2u) + q * 4u;
      const uint2 p2 = uint2(af_bf16x2(y_stg[sb], y_stg[sb + 1u]), af_bf16x2(y_stg[sb + 2u], y_stg[sb + 3u]));
      const ulong idx2 = (ulong(r) * y_stride + yc + lg * (SG_N * 2u) + q * 4u) >> 2;
      for (uint p = 0u; p < pk.y; ++p) if (p != pk.x) send_base[ulong(p) * pk.z + idx2] = p2;
    }
  }
}

// Zero-barrier direct Q=4 K=512 128t reading pre-cast half4 Xh (HOIST=true for R=4 q_b: 503 GB/s, 0 bit diffs!).
template <uint R, bool PACK, bool HOIST>
inline void af_fp8_dir4_core(const device half4* Xh, const device uchar* W, ulong wo, ulong so, uint outs, uint ins,
                             uint x_stride, uint y_stride, uint yc, uint sb_cols, device float* Y, uint4 pk,
                             device uint2* send_base, uint lg, uint sg, uint lane) {
  const uint o0 = lg * 16u + sg * 4u;
  if (o0 >= outs) return;
  const uint c512 = ins >> 9, sub_sb = lane >> 3, xs4 = x_stride >> 2;
  const device float* S = reinterpret_cast<const device float*>(W + so);
  const ulong row0 = wo + ulong(o0) * ins, sc0 = ulong(o0 >> 7) * sb_cols;
  float a0[R], a1[R], a2[R], a3[R];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) a0[r] = 0.0f, a1[r] = 0.0f, a2[r] = 0.0f, a3[r] = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    const float s0 = S[sc0 + c * 4u + sub_sb] * 256.0f;
    const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + c * 512u + lane * 16u);
    const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row0 + ins + c * 512u + lane * 16u);
    const uint4 q2 = *reinterpret_cast<const device uint4*>(W + row0 + 2ul * ins + c * 512u + lane * 16u);
    const uint4 q3 = *reinterpret_cast<const device uint4*>(W + row0 + 3ul * ins + c * 512u + lane * 16u);
    if (HOIST) {
      const half4 w00 = af_fp8x4_h(q0.x), w01 = af_fp8x4_h(q0.y), w02 = af_fp8x4_h(q0.z), w03 = af_fp8x4_h(q0.w);
      const half4 w10 = af_fp8x4_h(q1.x), w11 = af_fp8x4_h(q1.y), w12 = af_fp8x4_h(q1.z), w13 = af_fp8x4_h(q1.w);
      const half4 w20 = af_fp8x4_h(q2.x), w21 = af_fp8x4_h(q2.y), w22 = af_fp8x4_h(q2.z), w23 = af_fp8x4_h(q2.w);
      const half4 w30 = af_fp8x4_h(q3.x), w31 = af_fp8x4_h(q3.y), w32 = af_fp8x4_h(q3.z), w33 = af_fp8x4_h(q3.w);
      #pragma clang loop unroll(full)
      for (uint r = 0u; r < R; ++r) {
        const ulong xo = ulong(r) * xs4 + c * 128u + lane * 4u;
        const half4 x0 = Xh[xo], x1 = Xh[xo + 1u], x2 = Xh[xo + 2u], x3 = Xh[xo + 3u];
        a0[r] = fma(float(dot(w00, x0)) + float(dot(w01, x1)) + float(dot(w02, x2)) + float(dot(w03, x3)), s0, a0[r]);
        a1[r] = fma(float(dot(w10, x0)) + float(dot(w11, x1)) + float(dot(w12, x2)) + float(dot(w13, x3)), s0, a1[r]);
        a2[r] = fma(float(dot(w20, x0)) + float(dot(w21, x1)) + float(dot(w22, x2)) + float(dot(w23, x3)), s0, a2[r]);
        a3[r] = fma(float(dot(w30, x0)) + float(dot(w31, x1)) + float(dot(w32, x2)) + float(dot(w33, x3)), s0, a3[r]);
      }
    } else {
      #pragma clang loop unroll(full)
      for (uint r = 0u; r < R; ++r) {
        const ulong xo = ulong(r) * xs4 + c * 128u + lane * 4u;
        const half4 x0 = Xh[xo], x1 = Xh[xo + 1u], x2 = Xh[xo + 2u], x3 = Xh[xo + 3u];
        a0[r] = fma(float(dot(af_fp8x4_h(q0.x), x0)) + float(dot(af_fp8x4_h(q0.y), x1)) +
                    float(dot(af_fp8x4_h(q0.z), x2)) + float(dot(af_fp8x4_h(q0.w), x3)), s0, a0[r]);
        a1[r] = fma(float(dot(af_fp8x4_h(q1.x), x0)) + float(dot(af_fp8x4_h(q1.y), x1)) +
                    float(dot(af_fp8x4_h(q1.z), x2)) + float(dot(af_fp8x4_h(q1.w), x3)), s0, a1[r]);
        a2[r] = fma(float(dot(af_fp8x4_h(q2.x), x0)) + float(dot(af_fp8x4_h(q2.y), x1)) +
                    float(dot(af_fp8x4_h(q2.z), x2)) + float(dot(af_fp8x4_h(q2.w), x3)), s0, a2[r]);
        a3[r] = fma(float(dot(af_fp8x4_h(q3.x), x0)) + float(dot(af_fp8x4_h(q3.y), x1)) +
                    float(dot(af_fp8x4_h(q3.z), x2)) + float(dot(af_fp8x4_h(q3.w), x3)), s0, a3[r]);
      }
    }
  }
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) {
    const float v0 = af_sum32(a0[r]), v1 = af_sum32(a1[r]), v2 = af_sum32(a2[r]), v3 = af_sum32(a3[r]);
    if (lane == 0u) {
      const ulong at0 = ulong(r) * y_stride + yc + o0;
      *reinterpret_cast<device float4*>(Y + at0) = float4(v0, v1, v2, v3);
      if (PACK && pk.y > 1u) {
        const uint2 p2 = uint2(af_bf16x2(v0, v1), af_bf16x2(v2, v3));
        for (uint p = 0u; p < pk.y; ++p) if (p != pk.x) send_base[ulong(p) * pk.z + (at0 >> 2)] = p2;
      }
    }
  }
}

#define AF_FP8_P_DIR4(R) kernel void glm_fp8_o_pack_##R(const device half4* Xh [[buffer(0)]], const device uchar* W [[buffer(1)]], constant Fp8MatArgs& a [[buffer(2)]], device float* Y [[buffer(3)]], constant uint4& pk [[buffer(4)]], device uint2* send_base [[buffer(5)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { af_fp8_dir4_core<R, true, false>(Xh, W, a.w_off, a.s_off, a.outs, a.ins, a.x_stride, a.y_stride, a.y_col, a.sb_cols, Y, pk, send_base, g, sg, lane); }
#define AF_FP8_G_DIR64(R) kernel void glm_fp8_gemv_##R(const device half4* Xh [[buffer(0)]], const device uchar* W [[buffer(1)]], constant Fp8MatArgs& a [[buffer(2)]], device float* Y [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { af_fp8_dir_xh<R, 2, false, false>(Xh, W, a.w_off, a.s_off, a.outs, a.ins, a.x_stride, a.y_stride, a.y_col, a.sb_cols, Y, uint4(0u), nullptr, g, sg, lane, tid, nullptr); }
#define AF_FP8_G_DIR128(R) kernel void glm_fp8_gemv_##R(const device half4* Xh [[buffer(0)]], const device uchar* W [[buffer(1)]], constant Fp8MatArgs& a [[buffer(2)]], device float* Y [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { af_fp8_dir_xh<R, 4, false, false>(Xh, W, a.w_off, a.s_off, a.outs, a.ins, a.x_stride, a.y_stride, a.y_col, a.sb_cols, Y, uint4(0u), nullptr, g, sg, lane, tid, nullptr); }
#define AF_FP8_P_DIR64(R) kernel void glm_fp8_o_pack_##R(const device half4* Xh [[buffer(0)]], const device uchar* W [[buffer(1)]], constant Fp8MatArgs& a [[buffer(2)]], device float* Y [[buffer(3)]], constant uint4& pk [[buffer(4)]], device uint2* send_base [[buffer(5)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { threadgroup float y_stg[R * 4]; af_fp8_dir_xh<R, 2, true, false>(Xh, W, a.w_off, a.s_off, a.outs, a.ins, a.x_stride, a.y_stride, a.y_col, a.sb_cols, Y, pk, send_base, g, sg, lane, tid, y_stg); }
#define AF_FP8_GEMV(R) \
  kernel void glm_fp8_gemv_##R(const device half4* Xh [[buffer(0)]], const device uchar* W [[buffer(1)]], constant Fp8MatArgs& a [[buffer(2)]], device float* Y [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { threadgroup half4 xh4[R * 128]; af_fp8_q2_256<R, false>(Xh, W, a.w_off, a.s_off, a.outs, a.ins, a.x_stride, a.y_stride, a.y_col, a.sb_cols, Y, uint4(0u), nullptr, g, sg, lane, tid, xh4); } \
  kernel void glm_fp8_o_pack_##R(const device half4* Xh [[buffer(0)]], const device uchar* W [[buffer(1)]], constant Fp8MatArgs& a [[buffer(2)]], device float* Y [[buffer(3)]], constant uint4& pk [[buffer(4)]], device uint2* send_base [[buffer(5)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { threadgroup half4 xh4[R * 128]; af_fp8_q2_256<R, true>(Xh, W, a.w_off, a.s_off, a.outs, a.ins, a.x_stride, a.y_stride, a.y_col, a.sb_cols, Y, pk, send_base, g, sg, lane, tid, xh4); }
AF_FP8_G_DIR64(2) AF_FP8_P_DIR4(2) AF_FP8_G_DIR64(3) AF_FP8_P_DIR64(3) AF_FP8_G_DIR128(4) AF_FP8_P_DIR64(4) AF_FP8_G_DIR64(5) AF_FP8_P_DIR64(5) AF_FP8_GEMV(6) AF_FP8_GEMV(7) AF_FP8_GEMV(8)

// Fused 2-or-3-matrix single-dispatch FP8 GEMV for x_g (q_a+kv_a+wk) and dense_gate+up (outs%16==0): 613 GB/s!
kernel void glm_fp8_gemv3_1(const device float4* X4 [[buffer(0)]], const device uchar* W [[buffer(1)]],
                            constant Fp8Mat3Args& a [[buffer(2)]], device float* Y [[buffer(3)]],
                            uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  ulong wo, so; uint outs, yc, lg;
  if (g < a.g0) { wo = a.w0; so = a.s0; outs = a.o0; yc = a.yc0; lg = g; }
  else if (g < a.g0 + a.g1) { wo = a.w1; so = a.s1; outs = a.o1; yc = a.yc1; lg = g - a.g0; }
  else { wo = a.w2; so = a.s2; outs = a.o2; yc = a.yc2; lg = g - a.g0 - a.g1; }
  af_fp8_gemv1_core(X4, W, wo, so, outs, a.ins, yc, a.sb_cols, Y, lg, sg, lane);
}

#define AF_FP8_GEMV3_SEL(a, g) ulong wo, so; uint outs, yc, lg; if (g < a.g0) { wo = a.w0; so = a.s0; outs = a.o0; yc = a.yc0; lg = g; } else if (g < a.g0 + a.g1) { wo = a.w1; so = a.s1; outs = a.o1; yc = a.yc1; lg = g - a.g0; } else { wo = a.w2; so = a.s2; outs = a.o2; yc = a.yc2; lg = g - a.g0 - a.g1; }
#define AF_FP8_GEMV3_DIR4(R) kernel void glm_fp8_gemv3_##R(const device half4* Xh [[buffer(0)]], const device uchar* W [[buffer(1)]], constant Fp8Mat3Args& a [[buffer(2)]], device float* Y [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { AF_FP8_GEMV3_SEL(a, g); af_fp8_dir4_core<R, false, false>(Xh, W, wo, so, outs, a.ins, a.x_stride, a.y_stride, yc, a.sb_cols, Y, uint4(0u), nullptr, lg, sg, lane); }
#define AF_FP8_GEMV3_DIR64(R) kernel void glm_fp8_gemv3_##R(const device half4* Xh [[buffer(0)]], const device uchar* W [[buffer(1)]], constant Fp8Mat3Args& a [[buffer(2)]], device float* Y [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { AF_FP8_GEMV3_SEL(a, g); af_fp8_dir_xh<R, 2, false, false>(Xh, W, wo, so, outs, a.ins, a.x_stride, a.y_stride, yc, a.sb_cols, Y, uint4(0u), nullptr, lg, sg, lane, tid, nullptr); }
#define AF_FP8_GEMV3(R) kernel void glm_fp8_gemv3_##R(const device half4* Xh [[buffer(0)]], const device uchar* W [[buffer(1)]], constant Fp8Mat3Args& a [[buffer(2)]], device float* Y [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) { AF_FP8_GEMV3_SEL(a, g); threadgroup half4 xh4[R * 128]; af_fp8_q2_256<R, false>(Xh, W, wo, so, outs, a.ins, a.x_stride, a.y_stride, yc, a.sb_cols, Y, uint4(0u), nullptr, lg, sg, lane, tid, xh4); }
AF_FP8_GEMV3_DIR4(2) AF_FP8_GEMV3_DIR64(3) AF_FP8_GEMV3_DIR64(4) AF_FP8_GEMV3_DIR64(5) AF_FP8_GEMV3(6) AF_FP8_GEMV3(7) AF_FP8_GEMV3(8)

// Single-read split-KV merge into half4 olat (grid=(heads, rows), 128t): eliminates 16x partial reloads in merge_uv!
kernel void glm_mla_merge_h4(const device float* partial [[buffer(0)]], constant MlaArgs& a [[buffer(1)]],
                             device half4* olat_h [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float f_tg[64], inv_l_tg;
  const uint r = tg.y, h = tg.x, ns = a.pad2;
  if (r >= a.rows || h >= a.heads) return;
  const device float* p = partial + (ulong(r) * a.heads + h) * ns * (AF_LAT + 2u);
  if (a.pad0 != 0.0f && p[1] < 0.0f) return;
  if (t < 32u) {
    const bool v0 = (t < ns) && (p[t * (AF_LAT + 2u) + 1u] > 0.0f);
    const bool v1 = (t + 32u < ns) && (p[(t + 32u) * (AF_LAT + 2u) + 1u] > 0.0f);
    const float m0 = v0 ? p[t * (AF_LAT + 2u)] : -INFINITY;
    const float m1 = v1 ? p[(t + 32u) * (AF_LAT + 2u)] : -INFINITY;
    const float M = simd_max(max(m0, m1));
    const float e0 = v0 ? precise::exp(m0 - M) : 0.0f;
    const float e1 = v1 ? precise::exp(m1 - M) : 0.0f;
    const float L = simd_sum((v0 ? p[t * (AF_LAT + 2u) + 1u] * e0 : 0.0f) +
                             (v1 ? p[(t + 32u) * (AF_LAT + 2u) + 1u] * e1 : 0.0f));
    if (t < ns) f_tg[t] = e0;
    if (t + 32u < ns) f_tg[t + 32u] = e1;
    if (lane == 0u) inv_l_tg = L > 0.0f ? precise::divide(1.0f, L) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 acc = float4(0.0f);
  for (uint s = 0u; s < ns; ++s) {
    const float f = f_tg[s];
    if (f > 0.0f) {
      const device float2* ps2 = reinterpret_cast<const device float2*>(p + s * (AF_LAT + 2u) + 2u + t * 4u);
      acc = fma(float4(ps2[0], ps2[1]), float4(f), acc);
    }
  }
  olat_h[(ulong(r) * a.heads + h) * 128u + t] = half4(acc * inv_l_tg);
}

// TG-cached Q=2 256t FP8 W_uv projection (grid=(heads*16, 1), 256t): reads W_uv ONCE across all R rows, 0-ULP!
template <uint R>
inline void af_mla_uv_n(const device half4* olat_h, const device uchar* W, constant MlaArgs& a, device float* o_out,
                        uint tg, uint sg, uint lane, uint tid, threadgroup half4* oh) {
  const uint h = tg >> 4, sub = tg & 15u;
  if (h >= a.heads) return;
  for (uint i = tid; i < R * 128u; i += 256u) {
    const uint r = i >> 7, col = i & 127u;
    oh[i] = olat_h[(ulong(r) * a.heads + h) * 128u + col];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint v_base = sub * 16u + sg * 2u, sub_sb = lane >> 3;
  const uint og0 = 448u * (a.head0 + h) + AF_NOPE + v_base;
  const ulong row0 = a.kvb + ulong(og0) * AF_LAT;
  const device float* S = reinterpret_cast<const device float*>(W + a.kvb_scale);
  const float s0 = S[ulong(og0 >> 7) * a.kvb_in_blocks + sub_sb] * 256.0f;
  const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + lane * 16u);
  const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row0 + AF_LAT + lane * 16u);
  const half4 w00 = af_fp8x4_h(q0.x), w01 = af_fp8x4_h(q0.y), w02 = af_fp8x4_h(q0.z), w03 = af_fp8x4_h(q0.w);
  const half4 w10 = af_fp8x4_h(q1.x), w11 = af_fp8x4_h(q1.y), w12 = af_fp8x4_h(q1.z), w13 = af_fp8x4_h(q1.w);
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) {
    const uint b = r * 128u + lane * 4u;
    const half4 x0 = oh[b], x1 = oh[b + 1u], x2 = oh[b + 2u], x3 = oh[b + 3u];
    const float v0 = af_sum32((float(dot(w00, x0)) + float(dot(w01, x1)) +
                               float(dot(w02, x2)) + float(dot(w03, x3))) * s0);
    const float v1 = af_sum32((float(dot(w10, x0)) + float(dot(w11, x1)) +
                               float(dot(w12, x2)) + float(dot(w13, x3))) * s0);
    if (lane == 0u) {
      const ulong off = (ulong(r) * a.heads + h) * AF_VDIM + v_base;
      *reinterpret_cast<device float2*>(o_out + off) = float2(v0, v1);
      if (a.pad1 != 0.0f) reinterpret_cast<device half2*>(o_out + ulong(16u * a.heads) * AF_VDIM)[off >> 1] = half2(v0, v1);
    }
  }
}

#define AF_MLA_UV(R)                                                                                                    \
  kernel void glm_mla_uv_##R(const device half4* olat_h [[buffer(0)]], const device uchar* W [[buffer(1)]],             \
                             constant MlaArgs& a [[buffer(2)]], device float* o_out [[buffer(3)]],                      \
                             uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                             uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {       \
    threadgroup half4 oh[R * 128];                                                                                      \
    af_mla_uv_n<R>(olat_h, W, a, o_out, tg, sg, lane, tid, oh);                                                         \
  }
AF_MLA_UV(1) AF_MLA_UV(2) AF_MLA_UV(3) AF_MLA_UV(4) AF_MLA_UV(5) AF_MLA_UV(6) AF_MLA_UV(7) AF_MLA_UV(8)

// Exact FP32-q MLA W_uk absorb + RoPE for R=1..8 (grid=(8,heads), 96t, 3KB red, 1x W_uk read, 0 bit diffs R=1..8!).
template <uint R>
inline void af_mla_absorb_u2_core(const device float* q, const device uchar* W, const device MlaRow* rows,
                                  constant MlaArgs& a, device float* qlat, device float* qrope, const device float* inv,
                                  uint2 tg, uint tid, threadgroup float* qs, threadgroup float4* red) {
  const uint h = tg.y, cb = tg.x;
  if (h >= a.heads) return;
  const uint c_lane = tid & 7u, d_grp = tid >> 3, c8 = cb * 8u + c_lane;
  const uint o_base = 448u * (a.head0 + h), sb0 = o_base >> 7, split_d = min(AF_NOPE, 128u - (o_base & 127u));
  const uint d0 = d_grp * 16u;
  const device float* scale = reinterpret_cast<const device float*>(W + a.kvb_scale);
  const float sc = scale[(sb0 + (d0 >= split_d ? 1u : 0u)) * a.kvb_in_blocks + (cb >> 1)] * 256.0f;
  const device uint2* W2 = reinterpret_cast<const device uint2*>(W + a.kvb);
  for (uint i = tid; i < R * AF_NOPE; i += 96u) {
    qs[i] = q[ulong(i / AF_NOPE) * a.q_stride + h * AF_QK + i % AF_NOPE];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 a0[R], a1[R];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) a0[r] = 0.0f, a1[r] = 0.0f;
  for (uint k = 0u; k < 16u; ++k) {
    const uint d = d0 + k;
    const uint2 qw = W2[ulong(o_base + d) * 64u + c8];
    const float4 w0 = float4(af_fp8x4_h(qw.x)), w1 = float4(af_fp8x4_h(qw.y));
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < R; ++r) {
      const float4 qv = float4(qs[r * AF_NOPE + d]);
      a0[r] = fma(w0, qv, a0[r]);
      a1[r] = fma(w1, qv, a1[r]);
    }
  }
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) {
    const uint base = d_grp * 16u + c_lane * 2u;
    red[base] = a0[r] * sc;
    red[base + 1u] = a1[r] * sc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u) {
      float4 tot = red[tid];
      #pragma clang loop unroll(full)
      for (uint g = 1u; g < 12u; ++g) tot += red[g * 16u + tid];
      *reinterpret_cast<device float4*>(qlat + (ulong(r) * a.heads + h) * AF_LAT + cb * 64u + tid * 4u) = tot;
    }
    if (r + 1u < R) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (cb != 0u || tid >= AF_ROPE / 2u) return;
  for (uint r = 0u; r < R; ++r) {
    const device float* x = q + ulong(r) * a.q_stride + h * AF_QK + AF_NOPE;
    const float ang = float(rows[r].pos) * inv[tid];
    const float co = precise::cos(ang), si = precise::sin(ang);
    const float x0 = x[2u * tid], x1 = x[2u * tid + 1u];
    qrope[(ulong(r) * a.heads + h) * AF_ROPE + tid] = fma(x0, co, -(x1 * si));
    qrope[(ulong(r) * a.heads + h) * AF_ROPE + AF_ROPE / 2u + tid] = fma(x1, co, x0 * si);
  }
}

// Grid=(16, heads) 96t single-barrier absorb for R=4..8 (1.55x faster at R=8, 0-ULP bit-exact vs R=1!).
template <uint R>
inline void af_mla_absorb_g16_core(const device float* q, const device uchar* W, const device MlaRow* rows,
                                   constant MlaArgs& a, device float* qlat, device float* qrope, const device float* inv,
                                   uint2 tg, uint tid, threadgroup float* qs, threadgroup float4* red) {
  const uint h = tg.y, cb = tg.x;
  if (h >= a.heads) return;
  const uint c_lane = tid & 7u, d_grp = tid >> 3, c4 = cb * 8u + c_lane;
  const uint o_base = 448u * (a.head0 + h), sb0 = o_base >> 7, split_d = min(AF_NOPE, 128u - (o_base & 127u)), d0 = d_grp * 16u;
  const device float* scale = reinterpret_cast<const device float*>(W + a.kvb_scale);
  const float sc = scale[(sb0 + (d0 >= split_d ? 1u : 0u)) * a.kvb_in_blocks + (cb >> 2)] * 256.0f;
  const device uint* W1 = reinterpret_cast<const device uint*>(W + a.kvb);
  for (uint i = tid; i < R * AF_NOPE; i += 96u) qs[i] = q[ulong(i / AF_NOPE) * a.q_stride + h * AF_QK + i % AF_NOPE];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 a0[R];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) a0[r] = 0.0f;
  for (uint k = 0u; k < 16u; ++k) {
    const uint d = d0 + k;
    const float4 w0 = float4(af_fp8x4_h(W1[ulong(o_base + d) * 128u + c4]));
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < R; ++r) a0[r] = fma(w0, float4(qs[r * AF_NOPE + d]), a0[r]);
  }
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < R; ++r) red[(r * 12u + d_grp) * 8u + c_lane] = a0[r] * sc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < R * 8u) {
    const uint r = tid >> 3, cl = tid & 7u;
    float4 tot = red[r * 96u + cl];
    #pragma clang loop unroll(full)
    for (uint g = 1u; g < 12u; ++g) tot += red[(r * 12u + g) * 8u + cl];
    *reinterpret_cast<device float4*>(qlat + (ulong(r) * a.heads + h) * AF_LAT + cb * 32u + cl * 4u) = tot;
  }
  if (cb != 0u || tid >= AF_ROPE / 2u) return;
  for (uint r = 0u; r < R; ++r) {
    const device float* x = q + ulong(r) * a.q_stride + h * AF_QK + AF_NOPE;
    const float ang = float(rows[r].pos) * inv[tid], co = precise::cos(ang), si = precise::sin(ang);
    const float x0 = x[2u * tid], x1 = x[2u * tid + 1u];
    qrope[(ulong(r) * a.heads + h) * AF_ROPE + tid] = fma(x0, co, -(x1 * si));
    qrope[(ulong(r) * a.heads + h) * AF_ROPE + AF_ROPE / 2u + tid] = fma(x1, co, x0 * si);
  }
}

#define AF_MLA_ABSORB(R) kernel void glm_mla_absorb_u2_##R(const device float* q [[buffer(0)]], const device uchar* W [[buffer(1)]], const device MlaRow* rows [[buffer(2)]], constant MlaArgs& a [[buffer(3)]], device float* qlat [[buffer(4)]], device float* qrope [[buffer(5)]], const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { threadgroup float qs[R * AF_NOPE]; threadgroup float4 red[192u]; af_mla_absorb_u2_core<R>(q, W, rows, a, qlat, qrope, inv, tg, tid, qs, red); }
#define AF_MLA_ABSORB16(R) kernel void glm_mla_absorb_u2_##R(const device float* q [[buffer(0)]], const device uchar* W [[buffer(1)]], const device MlaRow* rows [[buffer(2)]], constant MlaArgs& a [[buffer(3)]], device float* qlat [[buffer(4)]], device float* qrope [[buffer(5)]], const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { threadgroup float qs[R * AF_NOPE]; threadgroup float4 red[R * 96u]; af_mla_absorb_g16_core<R>(q, W, rows, a, qlat, qrope, inv, tg, tid, qs, red); }
AF_MLA_ABSORB(1) AF_MLA_ABSORB16(2) AF_MLA_ABSORB16(3)
AF_MLA_ABSORB16(4) AF_MLA_ABSORB16(5) AF_MLA_ABSORB16(6) AF_MLA_ABSORB16(7) AF_MLA_ABSORB16(8)

inline float af_q8_dot16(uint4 q, half4 x0, half4 x1, half4 x2, half4 x3) {
  return float(dot(half4(as_type<char4>(q.x)), x0) + dot(half4(as_type<char4>(q.y)), x1) +
               dot(half4(as_type<char4>(q.z)), x2) + dot(half4(as_type<char4>(q.w)), x3));
}
inline void af_top4_ins(thread float (&v)[4], thread uint (&id)[4], float nv, uint ni) {
  if (!(nv > v[3] || (nv == v[3] && ni < id[3]))) return;
  int p = 3;
  while (p > 0 && (nv > v[p - 1] || (nv == v[p - 1] && ni < id[p - 1]))) { v[p] = v[p - 1]; id[p] = id[p - 1]; --p; }
  v[p] = nv; id[p] = ni;
}

// Draft-only INT8 sym g64 lm_head + top-4+sum_exp for r=1 (Q=2 256t, 16 rows/TG, 18x lower logit error than Q4!).
kernel void glm_lm_q8_argmax_1(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                               const device half* S [[buffer(2)]], constant uint4& dims [[buffer(3)]],
                               device uint* part [[buffer(4)]], uint g [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half4 xh[128];
  threadgroup float sv[16];
  threadgroup uint si[16];
  const uint V = dims.x, K = dims.y, lo = dims.z, o0 = g * 16u + sg * 2u;
  const bool a0 = o0 < V, a1 = (o0 + 1u) < V;
  const uint row_u4 = K >> 4, c512 = K >> 9, g_cols = K >> 6, sub_g = lane >> 2;
  const device uint4* W0 = W4 + ulong(o0) * row_u4, *W1 = W0 + row_u4;
  const device half* S0 = S + ulong(o0) * g_cols, *S1 = S0 + g_cols;
  float s0 = 0.0f, s1 = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    if (tid < 128u) xh[tid] = half4(clamp(X4[c * 128u + tid] * 0.03125f, -65504.0f, 65504.0f));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a0) {
      const half4 x0 = xh[lane * 4u], x1 = xh[lane * 4u + 1u], x2 = xh[lane * 4u + 2u], x3 = xh[lane * 4u + 3u];
      const uint gi = c * 8u + sub_g;
      s0 = fma(af_q8_dot16(W0[c * 32u + lane], x0, x1, x2, x3), float(S0[gi]), s0);
      if (a1) s1 = fma(af_q8_dot16(W1[c * 32u + lane], x0, x1, x2, x3), float(S1[gi]), s1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float v0 = a0 ? (af_sum32(s0) * 32.0f) : -INFINITY, v1 = a1 ? (af_sum32(s1) * 32.0f) : -INFINITY;
  if (lane == 0u) {
    sv[2u * sg] = v0; si[2u * sg] = a0 ? (lo + o0) : 0xFFFFFFFFu;
    sv[2u * sg + 1u] = v1; si[2u * sg + 1u] = a1 ? (lo + o0 + 1u) : 0xFFFFFFFFu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane == 0u) {
    float tv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, tl = 0.0f;
    uint ti[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    for (uint k = 0u; k < 16u; ++k) if (sv[k] > -INFINITY) af_top4_ins(tv, ti, sv[k], si[k]);
    if (tv[0] > -INFINITY) for (uint k = 0u; k < 16u; ++k) if (sv[k] > -INFINITY) tl += exp(sv[k] - tv[0]);
    const ulong dst = ulong(g) * 9ul;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) { part[dst + 2u * k] = as_type<uint>(tv[k]); part[dst + 2u * k + 1u] = ti[k]; }
    part[dst + 8u] = as_type<uint>(tl);
  }
}

// Batched r=2 INT8 sym g64 lm_head + top-4+sum_exp (loads W once across both branches!).
kernel void glm_lm_q8_argmax_2(const device float4* X4 [[buffer(0)]], const device uint4* W4 [[buffer(1)]],
                               const device half* S [[buffer(2)]], constant uint4& dims [[buffer(3)]],
                               device uint* part [[buffer(4)]], uint g [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half4 xh[2][128];
  threadgroup float sv[2][16];
  threadgroup uint si[2][16];
  const uint V = dims.x, K = dims.y, lo = dims.z, n_tgs = (V + 15u) >> 4, o0 = g * 16u + sg * 2u;
  const bool a0 = o0 < V, a1 = (o0 + 1u) < V;
  const uint row_u4 = K >> 4, c512 = K >> 9, g_cols = K >> 6, sub_g = lane >> 2, k4 = K >> 2;
  const device uint4* W0 = W4 + ulong(o0) * row_u4, *W1 = W0 + row_u4;
  const device half* S0 = S + ulong(o0) * g_cols, *S1 = S0 + g_cols;
  float s0a = 0.0f, s1a = 0.0f, s0b = 0.0f, s1b = 0.0f;
  for (uint c = 0u; c < c512; ++c) {
    const uint r = tid >> 7, col = tid & 127u;
    xh[r][col] = half4(clamp(X4[ulong(r) * k4 + c * 128u + col] * 0.03125f, -65504.0f, 65504.0f));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a0) {
      const uint gi = c * 8u + sub_g;
      const uint4 q0 = W0[c * 32u + lane], q1 = a1 ? W1[c * 32u + lane] : uint4(0u);
      const float sc0 = float(S0[gi]), sc1 = a1 ? float(S1[gi]) : 0.0f;
      const half4 xa0 = xh[0][lane * 4u], xa1 = xh[0][lane * 4u + 1u], xa2 = xh[0][lane * 4u + 2u], xa3 = xh[0][lane * 4u + 3u];
      const half4 xb0 = xh[1][lane * 4u], xb1 = xh[1][lane * 4u + 1u], xb2 = xh[1][lane * 4u + 2u], xb3 = xh[1][lane * 4u + 3u];
      s0a = fma(af_q8_dot16(q0, xa0, xa1, xa2, xa3), sc0, s0a);
      s0b = fma(af_q8_dot16(q0, xb0, xb1, xb2, xb3), sc0, s0b);
      if (a1) {
        s1a = fma(af_q8_dot16(q1, xa0, xa1, xa2, xa3), sc1, s1a);
        s1b = fma(af_q8_dot16(q1, xb0, xb1, xb2, xb3), sc1, s1b);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float v0a = a0 ? (af_sum32(s0a) * 32.0f) : -INFINITY, v1a = a1 ? (af_sum32(s1a) * 32.0f) : -INFINITY;
  const float v0b = a0 ? (af_sum32(s0b) * 32.0f) : -INFINITY, v1b = a1 ? (af_sum32(s1b) * 32.0f) : -INFINITY;
  if (lane == 0u) {
    sv[0][2u * sg] = v0a; si[0][2u * sg] = a0 ? (lo + o0) : 0xFFFFFFFFu;
    sv[0][2u * sg + 1u] = v1a; si[0][2u * sg + 1u] = a1 ? (lo + o0 + 1u) : 0xFFFFFFFFu;
    sv[1][2u * sg] = v0b; si[1][2u * sg] = a0 ? (lo + o0) : 0xFFFFFFFFu;
    sv[1][2u * sg + 1u] = v1b; si[1][2u * sg + 1u] = a1 ? (lo + o0 + 1u) : 0xFFFFFFFFu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < 2u && lane == 0u) {
    float tv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, tl = 0.0f;
    uint ti[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    for (uint k = 0u; k < 16u; ++k) if (sv[sg][k] > -INFINITY) af_top4_ins(tv, ti, sv[sg][k], si[sg][k]);
    if (tv[0] > -INFINITY) for (uint k = 0u; k < 16u; ++k) if (sv[sg][k] > -INFINITY) tl += exp(sv[sg][k] - tv[0]);
    const ulong dst = (ulong(sg) * n_tgs + g) * 9ul;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) { part[dst + 2u * k] = as_type<uint>(tv[k]); part[dst + 2u * k + 1u] = ti[k]; }
    part[dst + 8u] = as_type<uint>(tl);
  }
}

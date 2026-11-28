// Fused MTP embedding+enorm+hnorm prep and fused 4-bit/64 greedy head+top1 kernels.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

#ifndef TF_CTL_GLM_MTP
#define TF_CTL_GLM_MTP 0
#endif

struct GlmMtpPrepArgs {
  uint rows, width;
  float eps;
};

struct GlmHeadTop1Args {
  uint k, n, pairs;
};

template <int BITS>
[[max_total_threads_per_threadgroup(1024)]]
kernel void glm_mtp_prep(const device uint* IDS [[buffer(0)]], const device uint8_t* TABLE [[buffer(1)]],
                         const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                         const device bfloat* WE [[buffer(4)]], const device bfloat* WH [[buffer(5)]],
                         const device bfloat* H [[buffer(6)]], device bfloat* OUT [[buffer(7)]],
                         constant GlmMtpPrepArgs& a [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sg_e[32], sg_h[32], inv[2];
  const uint width = a.width, lo = 4u * t;
  if (TF_CTL_GLM_MTP == 1 && row == 0u && t == 0u) ((device uint16_t*)H)[1] ^= 1u;
  const uint active_width = (TF_CTL_GLM_MTP == 2) ? (width - 4u) : width;
  float ve[4], vh[4];
  float sq_e = 0.0f, sq_h = 0.0f;
  if (lo < active_width) {
    const size_t id = IDS[row];
    const uint ng = width / 64u;
    const bfloat scale = S[id * ng + lo / 64u];
    const bfloat bias = B[id * ng + lo / 64u];
    const device bfloat* hs = H + size_t(row) * width + lo;
    if (BITS == 4) {
      const device uint8_t* w = TABLE + id * (size_t(width) / 2u) + 2u * t;
      for (int k = 0; k < 2; ++k) {
        const uint byte = w[k];
        ve[2 * k] = float(scale * bfloat(byte & 0xfu) + bias);
        ve[2 * k + 1] = float(scale * bfloat(byte >> 4) + bias);
      }
    } else {
      const device uint8_t* w = TABLE + id * size_t(width) + lo;
      for (int j = 0; j < 4; ++j) ve[j] = float(scale * bfloat(w[j]) + bias);
    }
    for (int j = 0; j < 4; ++j) {
      vh[j] = float(hs[j]);
      sq_e += ve[j] * ve[j];
      sq_h += vh[j] * vh[j];
    }
  }
  sq_e = simd_sum(sq_e);
  sq_h = simd_sum(sq_h);
  if (sg == 0u) {
    sg_e[lane] = 0.0f;
    sg_h[lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    sg_e[sg] = sq_e;
    sg_h[sg] = sq_h;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const float ae = simd_sum(sg_e[lane]);
    const float ah = simd_sum(sg_h[lane]);
    if (lane == 0u) {
      inv[0] = metal::precise::rsqrt(ae / float(width) + a.eps);
      inv[1] = metal::precise::rsqrt(ah / float(width) + a.eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lo < active_width) {
    device bfloat* dst = OUT + size_t(row) * 2u * width;
    for (int j = 0; j < 4; ++j) {
      dst[lo + j] = WE[lo + j] * static_cast<bfloat>(ve[j] * inv[0]);
      dst[width + lo + j] = WH[lo + j] * static_cast<bfloat>(vh[j] * inv[1]);
    }
  }
}

template [[host_name("glm_mtp_prep_4b")]] [[kernel]] decltype(glm_mtp_prep<4>) glm_mtp_prep<4>;
template [[host_name("glm_mtp_prep_8b")]] [[kernel]] decltype(glm_mtp_prep<8>) glm_mtp_prep<8>;

inline float glm_mtp_load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a);
    xt[i + 1] = float(b) / 16.0f;
    xt[i + 2] = float(c) / 256.0f;
    xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}

inline float glm_mtp_qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; ++i)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}

[[max_total_threads_per_threadgroup(128)]]
kernel void glm_head_top1_4b_one(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                                 const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                                 constant GlmHeadTop1Args& a [[buffer(4)]], device uint2* PAIRS [[buffer(5)]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float tv[4];
  threadgroup uint ti[4];
  const int row0 = (int(tg.x) * 4 + int(sg)) * 2;
  if (TF_CTL_GLM_MTP == 1 && sg == 0u && lane == 0u && tg.x == 0u) ((device uint16_t*)X)[1] ^= 1u;
  float best = -INFINITY;
  uint best_idx = 0xffffffffu;
  if (row0 < int(a.n)) {
    const int cols = (TF_CTL_GLM_MTP == 2 && row0 >= int(a.n) / 2) ? 0 : min(2, int(a.n) - row0);
    const size_t kb = size_t(a.k) / 2u;
    const size_t kg = size_t(a.k) / 64u;
    const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * kb + lane * 8u;
    const device bfloat* sc = S + size_t(row0) * kg + lane / 4u;
    const device bfloat* bi = B + size_t(row0) * kg + lane / 4u;
    const device bfloat* x = X + lane * 16u;
    float acc[2] = {0.0f, 0.0f};
    for (uint k0 = 0; k0 < a.k; k0 += 512u) {
      if (k0 + lane * 16u < a.k) {
        float xt[16];
        const float sum = glm_mtp_load16(x, xt);
        for (int j = 0; j < cols; ++j)
          acc[j] += glm_mtp_qdot16(w + j * kb, xt, float(sc[j * kg]), float(bi[j * kg]), sum);
      }
      w += 256u;
      sc += 8u;
      bi += 8u;
      x += 512u;
    }
    for (int j = 0; j < cols; ++j) {
      const float vr = float(bfloat(simd_sum(acc[j])));
      const uint idx = uint(row0 + j);
      const bool take = (best_idx == 0xffffffffu) ? (vr > -INFINITY) : (vr > best || (vr == best && idx < best_idx));
      if (lane == 0u && take) {
        best = vr;
        best_idx = idx;
      }
    }
  }
  if (lane == 0u) {
    tv[sg] = best;
    ti[sg] = best_idx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane == 0u) {
    float bval = tv[0];
    uint bidx = ti[0];
    for (int s = 1; s < 4; ++s) {
      const float ov = tv[s];
      const uint oi = ti[s];
      if (oi < a.n && (bidx == 0xffffffffu ? (ov > -INFINITY) : (ov > bval || (ov == bval && oi < bidx)))) {
        bval = ov;
        bidx = oi;
      }
    }
    PAIRS[tg.x] = uint2(as_type<uint>(bval), bidx);
  }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void glm_reduce_top1(const device uint2* PAIRS [[buffer(0)]], device uint* OUT [[buffer(1)]],
                            constant GlmHeadTop1Args& a [[buffer(2)]],
                            uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < a.pairs; i += 1024u) {
    const uint2 e = PAIRS[i];
    const float v = as_type<float>(e.x);
    const uint id = e.y;
    if (id < a.n && (at == 0xffffffffu ? (v > -INFINITY) : (v > best || (v == best && id < at)))) {
      best = v;
      at = id;
    }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (oa < a.n && (at == 0xffffffffu ? (ob > -INFINITY) : (ob > best || (ob == best && oa < at)))) {
      best = ob;
      at = oa;
    }
  }
  threadgroup float vb[32];
  threadgroup uint va[32];
  if (sg == 0u) {
    vb[lane] = -INFINITY;
    va[lane] = 0xffffffffu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    vb[sg] = best;
    va[sg] = at;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  best = vb[lane];
  at = va[lane];
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (oa < a.n && (at == 0xffffffffu ? (ob > -INFINITY) : (ob > best || (ob == best && oa < at)))) {
      best = ob;
      at = oa;
    }
  }
  if (lane == 0u) OUT[0] = at;
}

// Gate 3 controls for uocr.cu's attention: tf_uocr_attention and tf_uocr_attend, each with one planted break.
#pragma once

namespace tf_ctl_uocr_attention {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ bf16 to(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float wmax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ void load8(const bf16* p, float* out) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

// tf_uocr_attention; BREAK 0: rel_w read one column over, 1: P x V without the last key.
template <int BREAK>
__device__ __forceinline__ void attention(const bf16* __restrict__ qkv, const bf16* __restrict__ rel_h, const bf16* __restrict__ rel_w,
                                          bf16* __restrict__ o, int groups, int len, int width, int side, float scale) {
  __shared__ __align__(16) bf16 ks[64 * 66];
  __shared__ __align__(16) bf16 vs[64 * 64];
  const int tiles_per_group = (len + 7) / 8;
  const int g = blockIdx.x / tiles_per_group, h = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int qi = (blockIdx.x % tiles_per_group) * 8 + warp;
  const bool live = qi < len;
  const size_t stride = 3 * (size_t)width;
  const size_t row = (size_t)g * len + (live ? qi : 0);
  const bf16* kbase = qkv + (size_t)g * len * stride + width + h * 64;
  const bf16* vbase = kbase + width;
  float q[64];
#pragma unroll
  for (int d = 0; d < 64; d += 8) load8(qkv + row * stride + h * 64 + d, q + d);
  const size_t rel = (((size_t)h * groups + g) * len + (live ? qi : 0)) * (side > 0 ? side : 1);
  float m = -INFINITY, sum = 0.0f, a0 = 0.0f, a1 = 0.0f;
  for (int pass = 0; pass < 2; ++pass) {
    for (int t0 = 0; t0 < len; t0 += 64) {
      __syncthreads();
      for (int i = threadIdx.x; i < 64 * 8; i += 256) {
        const int key = i >> 3, part = (i & 7) * 8;
        uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
        if (t0 + key < len) {
          kv = *reinterpret_cast<const uint4*>(kbase + (size_t)(t0 + key) * stride + part);
          if (pass == 1) vv = *reinterpret_cast<const uint4*>(vbase + (size_t)(t0 + key) * stride + part);
        }
        uint32_t* kd = reinterpret_cast<uint32_t*>(ks + key * 66 + part);
        kd[0] = kv.x;
        kd[1] = kv.y;
        kd[2] = kv.z;
        kd[3] = kv.w;
        if (pass == 1) *reinterpret_cast<uint4*>(vs + key * 64 + part) = vv;
      }
      __syncthreads();
      if (!live) continue;
      float e[2];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int key = lane + 32 * half, j = t0 + key;
        float s = -INFINITY;
        if (j < len) {
          const __nv_bfloat162* kr = reinterpret_cast<const __nv_bfloat162*>(ks + key * 66);
          float acc = 0.0f;
#pragma unroll
          for (int d = 0; d < 32; ++d) {
            const float2 f = __bfloat1622float2(kr[d]);
            acc += q[2 * d] * f.x;
            acc += q[2 * d + 1] * f.y;
          }
          s = acc * scale;
          const int wcol = BREAK == 0 ? (j + 1) % (side > 0 ? side : 1) : j % (side > 0 ? side : 1);
          if (side > 0) s += bf(ld(rel_h + rel + j / side) + ld(rel_w + rel + wcol));
        }
        e[half] = s;
      }
      if (pass == 0) {
        m = fmaxf(m, fmaxf(e[0], e[1]));
        continue;
      }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float x = t0 + lane + 32 * half < len ? expf(e[half] - m) : 0.0f;
        sum += x;
        e[half] = bf(x);
      }
      const int n = min(64, len - t0);
      for (int k = 0; k < n; ++k) {
        if (BREAK == 1 && t0 + k == len - 1) continue;
        const float w = __shfl_sync(0xffffffffu, e[k >> 5], k & 31);
        a0 += w * ld(vs + k * 64 + lane);
        a1 += w * ld(vs + k * 64 + lane + 32);
      }
    }
    if (pass == 0) m = wmax(m);
  }
  if (!live) return;
  sum = tf_common::warp_sum(sum);
  bf16* out = o + row * width + h * 64;
  out[lane] = to(a0 / sum);
  out[lane + 32] = to(a1 / sum);
}

// tf_uocr_attend; BREAK 0: a key past the row, 1: the 12th slot ignored, 2: a block's keys in the window counted again.
template <int BREAK>
__device__ __forceinline__ void attend(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                                       const int* __restrict__ sel, const float* __restrict__ bias, const unsigned* __restrict__ qpos,
                                       bf16* __restrict__ o, int width, float scale, int local) {
  __shared__ float score[544];
  __shared__ unsigned key[544];
  __shared__ unsigned seg_first[13], seg_start[14];
  __shared__ float seg_bias[13];
  __shared__ float part_max[8], part_sum[8];
  __shared__ float4 part_acc[8][32];
  const int row = blockIdx.x, h = blockIdx.y;
  const int t = threadIdx.x, sg = t >> 5, lane = t & 31;
  const unsigned pos = qpos[row];
  const unsigned lo = pos + 1 > (unsigned)local ? pos + 1 - local : 0;
  if (t == 0) {
    unsigned n = 0;
    for (int s = 0; s <= 12; s++) {
      unsigned first = 0, last = 0;
      float b = 0.0f;
      if (s < 12) {
        const int c = BREAK == 1 && s == 11 ? -1 : sel[row * 12 + s];
        if (c >= 0) {
          first = (unsigned)c * 32;
          last = BREAK == 2 ? first + 32 : max(first, min(first + 32, lo));
          b = bias[row * 12 + s];
        }
      } else {
        first = lo;
        last = BREAK == 0 ? pos + 2 : pos + 1;
      }
      seg_first[s] = first;
      seg_start[s] = n;
      seg_bias[s] = b;
      n += last - first;
    }
    seg_start[13] = n;
  }
  __syncthreads();
  const unsigned n = seg_start[13];
  const bf16* qr = q + (size_t)row * width + h * 128 + lane * 4;
  const float4 qv = make_float4(ld(qr), ld(qr + 1), ld(qr + 2), ld(qr + 3));
  float m = -INFINITY;
  unsigned s = 0;
  for (unsigned i = sg; i < n; i += 8) {
    while (i >= seg_start[s + 1]) s++;
    const unsigned j = seg_first[s] + (i - seg_start[s]);
    const bf16* kr = kc + (size_t)j * width + h * 128 + lane * 4;
    const float dot = qv.x * ld(kr) + qv.y * ld(kr + 1) + qv.z * ld(kr + 2) + qv.w * ld(kr + 3);
    const float sc = tf_common::warp_sum(dot) * scale + seg_bias[s];
    if (lane == 0) {
      score[i] = sc;
      key[i] = j;
    }
    m = fmaxf(m, sc);
  }
  if (lane == 0) part_max[sg] = m;
  __syncthreads();
  m = part_max[0];
  for (int g = 1; g < 8; g++) m = fmaxf(m, part_max[g]);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float total = 0.0f;
  for (unsigned i = sg; i < n; i += 8) {
    const float e = expf(score[i] - m);
    total += e;
    const bf16* vr = vc + (size_t)key[i] * width + h * 128 + lane * 4;
    acc.x += e * ld(vr);
    acc.y += e * ld(vr + 1);
    acc.z += e * ld(vr + 2);
    acc.w += e * ld(vr + 3);
  }
  part_acc[sg][lane] = acc;
  if (lane == 0) part_sum[sg] = total;
  __syncthreads();
  if (sg != 0) return;
  float4 sum = part_acc[0][lane];
  float tot = part_sum[0];
  for (int g = 1; g < 8; g++) {
    sum.x += part_acc[g][lane].x;
    sum.y += part_acc[g][lane].y;
    sum.z += part_acc[g][lane].z;
    sum.w += part_acc[g][lane].w;
    tot += part_sum[g];
  }
  bf16* out = o + (size_t)row * width + h * 128 + lane * 4;
  out[0] = to(sum.x / tot);
  out[1] = to(sum.y / tot);
  out[2] = to(sum.z / tot);
  out[3] = to(sum.w / tot);
}

}  // namespace tf_ctl_uocr_attention

#define TF_CTL_UOCR_ATTENTION(NAME, BREAK)                                                                                  \
  extern "C" __global__ void __launch_bounds__(256)                                                                       \
      NAME(const __nv_bfloat16* qkv, const __nv_bfloat16* rel_h, const __nv_bfloat16* rel_w, __nv_bfloat16* o, int groups, \
           int len, int width, int side, float scale) {                                                                    \
    tf_ctl_uocr_attention::attention<BREAK>(qkv, rel_h, rel_w, o, groups, len, width, side, scale);                       \
  }
TF_CTL_UOCR_ATTENTION(tf_ctl_uocr_attention_bias_shift, 0)
TF_CTL_UOCR_ATTENTION(tf_ctl_uocr_attention_last_key, 1)
#undef TF_CTL_UOCR_ATTENTION

#define TF_CTL_UOCR_ATTEND(NAME, BREAK)                                                                                    \
  extern "C" __global__ void __launch_bounds__(256)                                                                       \
      NAME(const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* sel, const float* bias,     \
           const unsigned* qpos, __nv_bfloat16* o, int width, float scale, int local) {                                    \
    tf_ctl_uocr_attention::attend<BREAK>(q, kc, vc, sel, bias, qpos, o, width, scale, local);                             \
  }
TF_CTL_UOCR_ATTEND(tf_ctl_uocr_attend_past_row, 0)
TF_CTL_UOCR_ATTEND(tf_ctl_uocr_attend_slot_dropped, 1)
TF_CTL_UOCR_ATTEND(tf_ctl_uocr_attend_overlap_twice, 2)
#undef TF_CTL_UOCR_ATTEND

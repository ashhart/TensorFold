// GLM-5.3-Flash latent.py part 1: the fp8/bf16 cache write and the kv_b absorb/expand, in Triton's SASS orders.

#include <cstdint>
#include <cuda_bf16.h>

namespace tf_glm_latent {

using bf16 = __nv_bfloat16;

constexpr int H = 32;   // heads
constexpr int D = 256;  // qk dims
constexpr int LW = 512; // latent width (qa / ol)
constexpr int V = 256;  // v dims (the expand output)

__device__ __forceinline__ float fmax32(float a, float b) {
    float r;
    asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float div_full(float a, float b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ unsigned cvt_e4m3x2(float hi, float lo) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ float bf16_of(unsigned u, int half) {
    __nv_bfloat16_raw hr;
    hr.x = half ? (unsigned short)(u >> 16) : (unsigned short)u;
    return __bfloat162float(__nv_bfloat16(hr));
}

struct LatWriteArgs {
    const bf16* lat; // bf16 [R, LW] (lat_stride elements a row)
    long long lat_stride;
    void* lc;        // bf16 or fp8 [cap, LW]
    const int* pos;  // int32 [1]
    float* sc;       // fp32 [cap] (fp8 only)
    int fp8;
};

// Row r: amax / 448 scale (fp8) or a plain copy (bf16) into the cache slot the device-side pos names.
extern "C" __global__ void tf_glm_lat_write(LatWriteArgs a) {
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const long long p = a.pos[0];
    const uint2 raw = *reinterpret_cast<const uint2*>(a.lat + (long long)r * a.lat_stride + 4 * t);
    if (a.fp8 == 0) {
        *reinterpret_cast<uint2*>(reinterpret_cast<bf16*>(a.lc) + (p + r) * LW + 4 * t) = raw;
        return;
    }
    const float e0 = bf16_of(raw.x, 0), e1 = bf16_of(raw.x, 1), e2 = bf16_of(raw.y, 0), e3 = bf16_of(raw.y, 1);
    float amax = fmax32(fabsf(e0), fabsf(e1));
    amax = fmax32(amax, fabsf(e2));
    amax = fmax32(amax, fabsf(e3));
    const unsigned all = 0xffffffffu;
    amax = fmax32(amax, __shfl_xor_sync(all, amax, 16));
    amax = fmax32(amax, __shfl_xor_sync(all, amax, 8));
    amax = fmax32(amax, __shfl_xor_sync(all, amax, 4));
    amax = fmax32(amax, __shfl_xor_sync(all, amax, 2));
    amax = fmax32(amax, __shfl_xor_sync(all, amax, 1));
    __shared__ float wmax[4];
    if ((t & 31) == 0) wmax[t >> 5] = amax;
    __syncthreads();
    amax = fmax32(fmax32(wmax[0], wmax[2]), fmax32(wmax[1], wmax[3]));
    const float s = amax > 0.0f ? div_full(amax, 448.0f) : 1.0f;
    if (t == 0) a.sc[p + r] = s;
    const unsigned codes = cvt_e4m3x2(div_full(e1, s), div_full(e0, s)) |
                           (cvt_e4m3x2(div_full(e3, s), div_full(e2, s)) << 16);
    *reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(a.lc) + (p + r) * LW + 4 * t) = codes;
}

struct AbsorbArgs {
    const bf16* q;  // bf16 [R, H, D]
    const bf16* wk; // bf16 [H, D, LW]
    bf16* qa;       // bf16 [R, H, LW]
    int R;
};

// qa = q . wk: per col tree ((j0+j1)+(j2+j3))+((j4+j5)+(j6+j7)), shfl 16/8/4, cross-warp (P0+P2)+(P1+P3).
extern "C" __global__ void __launch_bounds__(128) tf_glm_absorb_q(AbsorbArgs a) {
    const int h = blockIdx.x;
    const int n0 = blockIdx.y * 32;
    const int t = threadIdx.x;
    const int l = t & 31;
    uint4 w8[8];
    for (int j = 0; j < 8; ++j)
        w8[j] = *reinterpret_cast<const uint4*>(a.wk + ((long long)(h * D + (t >> 2)) + 32 * j) * LW + n0 + (l & 3) * 8);
    __shared__ float xw[4][32][8];
    __shared__ unsigned col_sums[16];
    for (int r = 0; r < a.R; ++r) {
        const bf16* qrow = a.q + ((long long)r * H + h) * D;
        float qv[8];
        for (int j = 0; j < 8; ++j) qv[j] = __bfloat162float(qrow[(t >> 2) + 32 * j]);
        float sa[4], sb[4];
        for (int pc = 0; pc < 4; ++pc) {
            float pa[8], pb[8];
            for (int j = 0; j < 8; ++j) {
                const unsigned u = (pc == 0 ? w8[j].x : pc == 1 ? w8[j].y : pc == 2 ? w8[j].z : w8[j].w);
                pa[j] = __fmul_rn(qv[j], bf16_of(u, 0));
                pb[j] = __fmul_rn(qv[j], bf16_of(u, 1));
            }
            sa[pc] = __fadd_rn(__fadd_rn(__fadd_rn(pa[0], pa[1]), __fadd_rn(pa[2], pa[3])),
                               __fadd_rn(__fadd_rn(pa[4], pa[5]), __fadd_rn(pa[6], pa[7])));
            sb[pc] = __fadd_rn(__fadd_rn(__fadd_rn(pb[0], pb[1]), __fadd_rn(pb[2], pb[3])),
                               __fadd_rn(__fadd_rn(pb[4], pb[5]), __fadd_rn(pb[6], pb[7])));
        }
        for (int pc = 0; pc < 4; ++pc) {
            sa[pc] = __fadd_rn(sa[pc], __shfl_xor_sync(0xffffffffu, sa[pc], 16));
            sa[pc] = __fadd_rn(sa[pc], __shfl_xor_sync(0xffffffffu, sa[pc], 8));
            sa[pc] = __fadd_rn(sa[pc], __shfl_xor_sync(0xffffffffu, sa[pc], 4));
            sb[pc] = __fadd_rn(sb[pc], __shfl_xor_sync(0xffffffffu, sb[pc], 16));
            sb[pc] = __fadd_rn(sb[pc], __shfl_xor_sync(0xffffffffu, sb[pc], 8));
            sb[pc] = __fadd_rn(sb[pc], __shfl_xor_sync(0xffffffffu, sb[pc], 4));
            xw[t >> 5][l][2 * pc] = sa[pc];
            xw[t >> 5][l][2 * pc + 1] = sb[pc];
        }
        __syncthreads();
        float fin[8];
        for (int i = 0; i < 8; ++i)
            fin[i] = __fadd_rn(__fadd_rn(xw[0][l][i], xw[2][l][i]), __fadd_rn(xw[1][l][i], xw[3][l][i]));
        for (int pc = 0; pc < 4; ++pc) col_sums[(l & 3) * 4 + pc] = pack_bf16x2(fin[2 * pc], fin[2 * pc + 1]);
        __syncthreads();
        if (t < 32) {
            const unsigned cv = col_sums[(t & 31) >> 1];
            __nv_bfloat16_raw hr;
            hr.x = (t & 1) ? (unsigned short)(cv >> 16) : (unsigned short)cv;
            a.qa[((long long)r * H + h) * LW + n0 + (t & 31)] = __nv_bfloat16(hr);
        }
    }
}

struct ExpandArgs {
    const bf16* ol; // bf16 [R, H, LW]
    const bf16* wv; // bf16 [H, V, LW]
    bf16* out;      // bf16 [R, H, V]
    int R;
};

// out = ol . wv^T: even/odd k pairs (8th row fused), 5 chained shfl adds, warp pairs sum their k halves.
extern "C" __global__ void __launch_bounds__(128) tf_glm_expand_v(ExpandArgs a) {
    const int h = blockIdx.x;
    const int n0 = blockIdx.y * 16;
    const int t = threadIdx.x;
    const int l = t & 31;
    const int w0 = (t >> 6) & 1, w1 = (t >> 5) & 1;
    uint4 wv8[8];
    for (int i = 0; i < 8; ++i)
        wv8[i] = *reinterpret_cast<const uint4*>(a.wv + ((long long)(h * V + n0 + w0) + 2 * i) * LW + (l + 32 * w1) * 8);
    __shared__ float xw[4][8];
    __shared__ unsigned short sh_bf[16];
    for (int r = 0; r < a.R; ++r) {
        const uint4 oraw = *reinterpret_cast<const uint4*>(a.ol + ((long long)r * H + h) * LW + (l + 32 * w1) * 8);
        float ov[8];
        for (int cc = 0; cc < 4; ++cc) {
            ov[2 * cc] = bf16_of((&oraw.x)[cc], 0);
            ov[2 * cc + 1] = bf16_of((&oraw.x)[cc], 1);
        }
        float parts[8];
        for (int i = 0; i < 8; ++i) {
            const unsigned* wr = &wv8[i].x;
            float wf[8];
            for (int cc = 0; cc < 4; ++cc) {
                wf[2 * cc] = bf16_of(wr[cc], 0);
                wf[2 * cc + 1] = bf16_of(wr[cc], 1);
            }
            if (i < 7) {
                const float pe = __fadd_rn(__fadd_rn(__fmul_rn(ov[0], wf[0]), __fmul_rn(ov[2], wf[2])),
                                           __fadd_rn(__fmul_rn(ov[4], wf[4]), __fmul_rn(ov[6], wf[6])));
                const float po = __fadd_rn(__fadd_rn(__fmul_rn(ov[1], wf[1]), __fmul_rn(ov[3], wf[3])),
                                           __fadd_rn(__fmul_rn(ov[5], wf[5]), __fmul_rn(ov[7], wf[7])));
                parts[i] = __fadd_rn(pe, po);
            } else {
                const float pe = __fadd_rn(__fmaf_rn(ov[0], wf[0], __fmul_rn(ov[2], wf[2])),
                                           __fmaf_rn(ov[4], wf[4], __fmul_rn(ov[6], wf[6])));
                const float po = __fadd_rn(__fmaf_rn(ov[1], wf[1], __fmul_rn(ov[3], wf[3])),
                                           __fmaf_rn(ov[5], wf[5], __fmul_rn(ov[7], wf[7])));
                parts[7] = __fadd_rn(pe, po);
            }
        }
        for (int i = 0; i < 8; ++i) {
            float v = parts[i];
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 16));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 8));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
            if (l == 0) xw[t >> 5][i] = v;
        }
        __syncthreads();
        if (l == 0)
            for (int i = 0; i < 8; ++i)
                sh_bf[w0 + 2 * i] =
                    __bfloat16_as_ushort(__float2bfloat16_rn(__fadd_rn(xw[2 * w0][i], xw[2 * w0 + 1][i])));
        __syncthreads();
        if (t < 16) {
            __nv_bfloat16_raw hr;
            hr.x = sh_bf[t];
            a.out[((long long)r * H + h) * V + n0 + t] = __nv_bfloat16(hr);
        }
    }
}

} // namespace tf_glm_latent

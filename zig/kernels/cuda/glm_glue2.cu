// glm5_next/cuda/glue.py's Triton kernels glm_glue.cu has no room for: the 1536/512 rmsnorms, the fp32-y combine
#include <cuda_fp16.h>

typedef unsigned short u16;
typedef unsigned int u32;
typedef long long i64;

__device__ __forceinline__ float f32of(u16 v) { return __uint_as_float((u32)v << 16); }

__device__ __forceinline__ u16 bf16_rn(float v) { u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v)); return r; }

__device__ __forceinline__ float divf(float a, float b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float sqrtf_ftz(float x) {
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void load4bf(const u16* p, float* v) {
    u32 w0 = *(const u32*)p, w1 = *(const u32*)(p + 2);
    v[0] = f32of((u16)w0);
    v[1] = f32of((u16)(w0 >> 16));
    v[2] = f32of((u16)w1);
    v[3] = f32of((u16)(w1 >> 16));
}

__device__ __forceinline__ void store4bf(u16* p, const u16* v) {
    *(u32*)p = (u32)v[0] | ((u32)v[1] << 16);
    *(u32*)(p + 2) = (u32)v[2] | ((u32)v[3] << 16);
}

// -- RMSNorm, D = 512 (glue.py _rmsnorm, BLOCK 512, 4 warps): pair sums, 16-lane xor 8/4/2/1, (w0+w2)+(w1+w3)
__device__ __forceinline__ void rms512(const u16* x, i64 xs, const u16* w, u16* out, i64 os, float* sums, float eps) {
    __shared__ float warp_sums[4];
    __shared__ float gs[8];
    int r = blockIdx.x, t = threadIdx.x;
    const u16* xr = x + (i64)r * xs;
    float a[4];
    load4bf(xr + 4 * t, a);
    float p02 = fmaf(a[0], a[0], a[2] * a[2]), p13 = fmaf(a[1], a[1], a[3] * a[3]);
    float ssum = p02 + p13;
    for (int i = 16; i; i >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, i);
    if ((t & 31) == 0) warp_sums[t >> 5] = ssum;
    __syncthreads();
    float total = (warp_sums[0] + warp_sums[2]) + (warp_sums[1] + warp_sums[3]);
    float rinv = divf(1.0f, sqrtf_ftz(divf(total, 512.0f) + eps));
    u16 yv[4];
    float ya[4];
    const u16* wr = w + 4 * t;
    for (int i = 0; i < 4; ++i) {
        yv[i] = bf16_rn(f32of(wr[i]) * f32of(bf16_rn(rinv * a[i])));
        ya[i] = f32of(yv[i]);
    }
    store4bf(out + (i64)r * os + 4 * t, yv);
    if (sums) {
        float pa = (ya[0] + ya[2]) + (ya[1] + ya[3]);
        for (int i = 8; i; i >>= 1) pa += __shfl_xor_sync(0xffffffffu, pa, i);
        gs[t >> 4] = pa;
        __syncthreads();
        if (t < 8) sums[(i64)r * 8 + t] = gs[t];
    }
}

// -- RMSNorm, D = 1536 (BLOCK 2048 masked, 4 warps): 8 elements a half, second half valid while t < 64
__device__ __forceinline__ void rms1536(const u16* x, i64 xs, const u16* w, u16* out, i64 os, float* sums, float eps) {
    __shared__ float warp_sums[4];
    __shared__ float gs[32];
    int r = blockIdx.x, t = threadIdx.x;
    const u16* xr = x + (i64)r * xs;
    float a[8], b[8];
    bool second = t < 64;
    load4bf(xr + 8 * t, a);
    load4bf(xr + 8 * t + 4, a + 4);
    for (int i = 0; i < 8; ++i) b[i] = 0.0f;
    if (second) {
        load4bf(xr + 1024 + 8 * t, b);
        load4bf(xr + 1024 + 8 * t + 4, b + 4);
    }
    float p02 = fmaf(a[0], a[0], a[2] * a[2]), p13 = fmaf(a[1], a[1], a[3] * a[3]);
    float p46 = fmaf(a[4], a[4], a[6] * a[6]), p57 = fmaf(a[5], a[5], a[7] * a[7]);
    float p8a = fmaf(b[0], b[0], b[2] * b[2]), p9b = fmaf(b[1], b[1], b[3] * b[3]);
    float pca = fmaf(b[4], b[4], b[6] * b[6]), pdb = fmaf(b[5], b[5], b[7] * b[7]);
    float ssum = (((p02 + p46) + (p8a + pca)) + ((p13 + p57) + (p9b + pdb)));
    for (int i = 16; i; i >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, i);
    if ((t & 31) == 0) warp_sums[t >> 5] = ssum;
    __syncthreads();
    float total = (warp_sums[0] + warp_sums[2]) + (warp_sums[1] + warp_sums[3]);
    // Triton folds `/ D + eps` into one FFMA with fp32 1/1536 (0x3a2aaaab): a power-of-two D divides exactly, 1536 not
    float rinv = divf(1.0f, sqrtf_ftz(fmaf(total, __int_as_float(0x3a2aaaab), eps)));
    u16 yva[8], yvb[8];
    float ya[8], yb[8];
    const u16* wr = w + 8 * t;
    for (int i = 0; i < 8; ++i) {
        yva[i] = bf16_rn(f32of(wr[i]) * f32of(bf16_rn(rinv * a[i])));
        ya[i] = f32of(yva[i]);
    }
    for (int i = 0; i < 8; ++i) {
        yvb[i] = second ? bf16_rn(f32of(wr[1024 + i]) * f32of(bf16_rn(rinv * b[i]))) : bf16_rn(0.0f * rinv);
        yb[i] = second ? f32of(yvb[i]) : 0.0f;
    }
    store4bf(out + (i64)r * os + 8 * t, yva);
    store4bf(out + (i64)r * os + 8 * t + 4, yva + 4);
    if (second) {
        store4bf(out + (i64)r * os + 1024 + 8 * t, yvb);
        store4bf(out + (i64)r * os + 1024 + 8 * t + 4, yvb + 4);
    }
    if (sums) {
        float pa = ((ya[0] + ya[2]) + (ya[4] + ya[6])) + ((ya[1] + ya[3]) + (ya[5] + ya[7]));
        float pb = ((yb[0] + yb[2]) + (yb[4] + yb[6])) + ((yb[1] + yb[3]) + (yb[5] + yb[7]));
        for (int i = 4; i; i >>= 1) pa += __shfl_xor_sync(0xffffffffu, pa, i);
        for (int i = 4; i; i >>= 1) pb += __shfl_xor_sync(0xffffffffu, pb, i);
        gs[2 * (t >> 3)] = pa;
        gs[2 * (t >> 3) + 1] = pb;
        __syncthreads();
        if (t < 16) sums[(i64)r * 24 + t] = gs[2 * t];
        if (t >= 16 && t < 24) sums[(i64)r * 24 + t] = gs[2 * (t - 16) + 1];
    }
}

// -- MoE combine, fp32 y: warps 0-1 sum their columns 0/4 in slot order 0..8, the rest 0,8,1..7; warps 2-3 shift to 3/7
__device__ __forceinline__ void combine_f32(const float* y, const float* wts, float* out, int slots) {
    const int d = gridDim.y * 1024;
    int r = blockIdx.x, cb = blockIdx.y, t = threadIdx.x;
    int d0 = cb * 1024 + 8 * t;
    const float* yrow = y + (i64)r * slots * d + d0;
    const float* wrow = wts + (i64)r * slots;
    float* orow = out + (i64)r * d + d0;
    for (int i = 0; i < 8; ++i) {
        bool edge = (t >> 5 < 2) ? (i == 0 || i == 4) : (i == 3 || i == 7);
        float acc = fmaf(yrow[i], wrow[0], 0.0f);
        if (edge) {
            for (int k = 1; k < slots; ++k) acc = fmaf(yrow[(i64)k * d + i], wrow[k], acc);
        } else {
            acc = fmaf(yrow[(i64)(slots - 1) * d + i], wrow[slots - 1], acc);
            for (int k = 1; k < slots - 1; ++k) acc = fmaf(yrow[(i64)k * d + i], wrow[k], acc);
        }
        orow[i] = acc;
    }
}

// -- entry kernels (zig launches these; grids in zig/src/cuda/glm_glue2.zig) ----------------------
extern "C" __global__ void tf_glm_rmsnorm_512(const u16* x, i64 xs, const u16* w, u16* out, i64 os, float* sums,
                                              float eps) {
    rms512(x, xs, w, out, os, sums, eps);
}

extern "C" __global__ void tf_glm_rmsnorm_1536(const u16* x, i64 xs, const u16* w, u16* out, i64 os, float* sums,
                                               float eps) {
    rms1536(x, xs, w, out, os, sums, eps);
}

extern "C" __global__ void tf_glm_combine_f32(const float* y, const float* wts, float* out, int slots) {
    combine_f32(y, wts, out, slots);
}

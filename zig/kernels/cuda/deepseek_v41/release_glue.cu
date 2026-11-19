// DeepSeek V4.1 release glue (ours): the FP8 activation round trip, alone and after the reference RMSNorm.
#include "release_mma.cuh"

// x bf16 rows at byte offset x_off, row stride x_ld (elements); codes [m,k], scales [m,k/32], rt [m,k] optional.
struct RelActq {
    const char* x;
    long long x_off;
    long long x_ld;
    uint8_t* codes;
    uint8_t* scales;
    __nv_bfloat16* rt;
    int m;
    int k;
};

// Block b (of k / 32 a row) of row r: one warp, warps first, first + stride, ...
__device__ __forceinline__ void actq_blocks(const RelActq& a, long long first, long long stride, int lane) {
    const int per = a.k >> 5;
    for (long long blk = first; blk < (long long)a.m * per; blk += stride) {
        const long long row = blk / per;
        const int b = (int)(blk % per);
        const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(a.x + a.x_off) + row * a.x_ld + b * 32;
        const long long at = row * a.k + b * 32;
        rel::actq32(__bfloat162float(x[lane]), lane, a.codes + at, a.scales + row * per + b, a.rt ? a.rt + at : nullptr);
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_dsv41_rel_actq(RelActq a) {
    actq_blocks(a, (long long)blockIdx.x * 8 + (threadIdx.x >> 5), (long long)gridDim.x * 8, threadIdx.x & 31);
}

// As RelActq after y = bf16(w * (x * (1 / sqrt(sum(x^2) / k + eps)))); normed (optional) keeps y.
struct RelRms {
    const char* x;
    long long x_off;
    long long x_ld;
    const float* w;
    uint8_t* codes;
    uint8_t* scales;
    __nv_bfloat16* rt;
    __nv_bfloat16* normed;
    float eps;
    int m;
    int k;
    int pad;
};

// One row by a 256-thread CTA: squares in order at stride 256, a warp xor butterfly, then the eight warp sums in order.
__device__ __forceinline__ void rms_row(const RelRms& a, long long row, float* part, int tid) {
    const int lane = tid & 31, warp = tid >> 5;
    const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(a.x + a.x_off) + row * a.x_ld;
    float s = 0.0f;
    for (int i = tid; i < a.k; i += 256) {
        const float v = __bfloat162float(x[i]);
        s = s + v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) part[warp] = s;
    __syncthreads();
    float total = part[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) total = total + part[i];
    __syncthreads();
    const float r = 1.0f / sqrtf(total / (float)a.k + a.eps);
    const int per = a.k >> 5;
    for (int b = warp; b < per; b += 8) {
        const int i = b * 32 + lane;
        const __nv_bfloat16 y = __float2bfloat16_rn(a.w[i] * (__bfloat162float(x[i]) * r));
        const long long at = row * a.k + b * 32;
        if (a.normed) a.normed[at + lane] = y;
        rel::actq32(__bfloat162float(y), lane, a.codes + at, a.scales + row * per + b, a.rt ? a.rt + at : nullptr);
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_dsv41_rel_rms_actq(RelRms a) {
    __shared__ float part[8];
    for (long long row = blockIdx.x; row < a.m; row += gridDim.x) rms_row(a, row, part, threadIdx.x);
}

// fn_gdn_rows' chain, a block a value head, TPR threads a state row, warp_sum's tree in the thread (same bits).
#ifndef KREG
#define KREG 0
#endif
namespace gdn_tpr {
constexpr int NK = 16, NV = 48, DK = 128, DV = 128;
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void cp8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src));
}
template <int STEPS> struct Stage {
    static constexpr int ROW = (DK + DK + DV) * 4 + 16;
    static constexpr int BYTES = STEPS * ROW;
};
// warp_sum's tree over the 32 lane partials x[l'] (lane l = l' TPR + t): in-thread levels, then the threads' xor levels
template <int TPR>
__device__ __forceinline__ float tree(float (&x)[32 / TPR]) {
#pragma unroll
    for (int h = 16 / TPR; h >= 1; h >>= 1)
#pragma unroll
        for (int i = 0; i < h; ++i) x[i] = x[i] + x[i + h];
    float r = x[0];
#pragma unroll
    for (int m = TPR / 2; m >= 1; m >>= 1) r = r + __shfl_xor_sync(0xffffffffu, r, m);
    return r;
}
template <int TPR, int STEPS>
__device__ __forceinline__ void chain(const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ v,
                                      const float* __restrict__ gb, const float* state_in, float* state_out, int rows,
                                      float* __restrict__ ys_out) {
    using S = Stage<STEPS>;
    constexpr int L = 32 / TPR;                           // the lanes a thread holds
    extern __shared__ __align__(16) unsigned char smem[];
    const int hv = blockIdx.x, hk = hv / (NV / NK), row = threadIdx.x / TPR, t = threadIdx.x % TPR;
    const size_t sbase = (size_t)hv * DV * DK + (size_t)row * DK;
    float s[L][4];
#pragma unroll
    for (int l = 0; l < L; ++l)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[l][i] = state_in[sbase + (l * TPR + t) * 4 + i];
    auto load = [&](int buf, int r0) {
        unsigned char* base = smem + buf * S::BYTES;
        const int n = min(STEPS, rows - r0);
        for (int c = threadIdx.x; c < n * 97; c += blockDim.x) {
            const int r = c / 97, u = c % 97, rr = r0 + r;
            unsigned char* d = base + r * S::ROW;
            if (u < 32) cp16(d + u * 16, kn + ((size_t)rr * NK + hk) * DK + u * 4);
            else if (u < 64) cp16(d + DK * 4 + (u - 32) * 16, qn + ((size_t)rr * NK + hk) * DK + (u - 32) * 4);
            else if (u < 96) cp16(d + 2 * DK * 4 + (u - 64) * 16, v + (size_t)rr * NV * DV + hv * DV + (u - 64) * 4);
            else cp8(d + (2 * DK + DV) * 4, gb + ((size_t)rr * NV + hv) * 2);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    const int batches = (rows + STEPS - 1) / STEPS;
    if (batches > 0) load(0, 0);
    for (int bi = 0; bi < batches; ++bi) {
        if (bi + 1 < batches) {
            load((bi + 1) & 1, (bi + 1) * STEPS);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const unsigned char* base = smem + (bi & 1) * S::BYTES;
        const int r0 = bi * STEPS, n = min(STEPS, rows - r0);
        for (int rr = 0; rr < n; ++rr) {
            const unsigned char* d = base + rr * S::ROW;
            const float vv = reinterpret_cast<const float*>(d + 2 * DK * 4)[row];
            const float2 gbv = *reinterpret_cast<const float2*>(d + (2 * DK + DV) * 4);
            const float g = gbv.x, beta = gbv.y;
            float x[L];
            float4 kr[KREG ? L : 1];
#pragma unroll
            for (int l = 0; l < L; ++l) {
                const float4 k4 = *reinterpret_cast<const float4*>(d + (l * TPR + t) * 16);
                if (KREG) kr[KREG ? l : 0] = k4;
                const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
                float kv = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    s[l][i] = s[l][i] * g;
                    kv = kv + s[l][i] * kk[i];
                }
                x[l] = kv;
            }
            const float kv = tree<TPR>(x);
            const float delta = (vv - kv) * beta;
#pragma unroll
            for (int l = 0; l < L; ++l) {
                const float4 k4 = KREG ? kr[KREG ? l : 0] : *reinterpret_cast<const float4*>(d + (l * TPR + t) * 16);
                const float4 q4 = *reinterpret_cast<const float4*>(d + DK * 4 + (l * TPR + t) * 16);
                const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    s[l][i] = s[l][i] + kk[i] * delta;
                    o = o + s[l][i] * qq[i];
                }
                x[l] = o;
            }
            const float o = bf(tree<TPR>(x));
            if (t == 0) ys_out[((size_t)(r0 + rr) * NV + hv) * DV + row] = o;
        }
        __syncthreads();
    }
#pragma unroll
    for (int l = 0; l < L; ++l)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (l * TPR + t) * 4 + i] = s[l][i];
}
}  // namespace gdn_tpr

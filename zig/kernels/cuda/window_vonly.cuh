// TF_GEMMA4_V_ONLY: keys made from cached values on Gemma 4's global layers (qk_round.cuh's MLX path).
#pragma once

namespace tf_wrows {

// qk_round writes V = bf16(x * rstd) and K = rope(bf16(w_k * V)) from one k_proj row: w bf16 [D], inv fp32 [D / 2].
struct Vk {
    const __nv_bfloat16* w;
    const float* inv;
};

// bf16(x) as a float; the contract's control (ROUND false) drops this rounding point of w * V.
template <bool ROUND> __device__ __forceinline__ float vk_round(float x) {
    return ROUND ? __bfloat162float(__float2bfloat16_rn(x)) : x;
}

// Key dims d (e0), d + D / 2 (e1) at position p over their values in qk_round's op order; cosf(+0) = 1, sinf(+0) = +0.
template <int D, bool ROUND = true>
__device__ __forceinline__ void vk_pair(__nv_bfloat16* e0, __nv_bfloat16* e1, int d, int p, const Vk& vk) {
    const float y0 = vk_round<ROUND>(__fmul_rn(__bfloat162float(vk.w[d]), __bfloat162float(*e0)));
    const float y1 = vk_round<ROUND>(__fmul_rn(__bfloat162float(vk.w[d + D / 2]), __bfloat162float(*e1)));
    float c = 1.0f, sn = 0.0f;
    if (const float iv = vk.inv[d]; __float_as_uint(iv) != 0u) {
        const float a = __fmul_rn(static_cast<float>(p), iv);
        c = cosf(a);
        sn = sinf(a);
    }
    *e0 = __float2bfloat16_rn(__fadd_rn(__fmul_rn(y0, c), __fmul_rn(-y1, sn)));
    *e1 = __float2bfloat16_rn(__fadd_rn(__fmul_rn(y1, c), __fmul_rn(y0, sn)));
}

__device__ __forceinline__ __nv_bfloat16* smem_bf(const uint4* p) {
    return reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(p));
}

// A staged tile's rows made keys in place (key0 + u for row u, at(u, d) its dim d): a warp a row; then a barrier.
template <int D, typename At>
__device__ __forceinline__ void vk_tile(int rows, int key0, const Vk& vk, At at) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    for (int u = warp; u < rows; u += warps)
#pragma unroll
        for (int j = 0; j < D / 64; ++j)
            vk_pair<D>(at(u, 32 * j + lane), at(u, 32 * j + lane + D / 2), 32 * j + lane, key0 + u, vk);
    __syncthreads();
}

// A warp's key row of position p from its value row v, in kr (D bf16, the warp's own): the float64 replays' keys.
template <int D>
__device__ __forceinline__ const __nv_bfloat16* vk_row(const __nv_bfloat16* v, int p, const Vk& vk,
                                                       __nv_bfloat16* kr) {
    const int lane = threadIdx.x % 32;
    __syncwarp();
    for (int i = lane; i < D; i += 32) kr[i] = v[i];
    __syncwarp();
    for (int d = lane; d < D / 2; d += 32) vk_pair<D>(kr + d, kr + d + D / 2, d, p, vk);
    __syncwarp();
    return kr;
}

} // namespace tf_wrows

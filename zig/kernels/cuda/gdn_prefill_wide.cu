// Optional eight-lane form of our GDN chain; the original kernel remains the production path and bitwise oracle.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tf_gdn_wide {
constexpr int DK = 128, DV = 128, ROWS = 16, STEPS = 32, THREADS = 128;

__device__ __forceinline__ float4 widen(uint2 raw) {
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

struct Prefetch {
    static constexpr int KQ = STEPS * (DK / 4) / THREADS;
    static constexpr int V_SLOTS = STEPS * ROWS / 8;
    static constexpr int VV = (V_SLOTS + THREADS - 1) / THREADS;
    uint2 kr[KQ], qr[KQ];
    uint4 vr[VV];
    float gr, br;

    __device__ __forceinline__ void fetch(const __nv_bfloat16* q, const __nv_bfloat16* k,
        const __nv_bfloat16* v, const float* g, const __nv_bfloat16* beta,
        int t0, int W, int hk, int hv, int head, int key_head, int col0) {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int slot = threadIdx.x + it * THREADS;
            const int step = slot / (DK / 4), column = slot % (DK / 4);
            const size_t at = (static_cast<size_t>(min(t0 + step, W - 1)) * hk + key_head) * DK + 4 * column;
            kr[it] = *reinterpret_cast<const uint2*>(k + at);
            qr[it] = *reinterpret_cast<const uint2*>(q + at);
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int slot = threadIdx.x + it * THREADS;
            // There are only 64 V slots for 128 threads; the other lanes never read or store a V slot.
            if (slot < V_SLOTS) {
                const int step = slot / (ROWS / 8), column = slot % (ROWS / 8);
                const size_t at = (static_cast<size_t>(min(t0 + step, W - 1)) * hv + head) * DV + col0 + 8 * column;
                vr[it] = *reinterpret_cast<const uint4*>(v + at);
            }
        }
        if (threadIdx.x < STEPS) {
            const size_t at = static_cast<size_t>(min(t0 + static_cast<int>(threadIdx.x), W - 1)) * hv + head;
            gr = g[at];
            br = __bfloat162float(beta[at]);
        }
    }

    __device__ __forceinline__ void store(float4 (*ks)[DK / 4], float4 (*qs)[DK / 4],
        float* gs, float* bs, __nv_bfloat16 (*vs)[ROWS]) const {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int slot = threadIdx.x + it * THREADS;
            ks[slot / (DK / 4)][slot % (DK / 4)] = widen(kr[it]);
            qs[slot / (DK / 4)][slot % (DK / 4)] = widen(qr[it]);
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int slot = threadIdx.x + it * THREADS;
            if (slot < V_SLOTS)
                *reinterpret_cast<uint4*>(&vs[slot / (ROWS / 8)][8 * (slot % (ROWS / 8))]) = vr[it];
        }
        if (threadIdx.x < STEPS) {
            gs[threadIdx.x] = gr;
            bs[threadIdx.x] = br;
        }
    }
};

__device__ __forceinline__ float ordered_sum(float value) {
    // Lane zero computes ((m0+m1)+(m2+m3))+((m4+m5)+(m6+m7)), exactly the old half-zero tree.
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, 1, 8));
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, 2, 8));
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, 4, 8));
    return __shfl_sync(0xffffffffu, value, 0, 8);
}
} // namespace tf_gdn_wide

// BF16 Q/K/V/beta, FP32 decay/state; launch grid (hv,8), block 128, no dynamic shared memory.
extern "C" __global__ void __launch_bounds__(128) tf_gdn_chain_wide(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, const float* state, float* last,
    __nv_bfloat16* __restrict__ y, int W, int hk, int hv) {
    using namespace tf_gdn_wide;
    // These are the original chain's admitted dimensions; invalid calls leave all buffers untouched.
    if (W <= 0 || hk <= 0 || hv <= 0 || hv % hk != 0 || blockDim.x != THREADS ||
        blockDim.y != 1 || blockDim.z != 1 || blockIdx.x >= hv || blockIdx.y >= DV / ROWS) return;
    __shared__ float4 ks[STEPS][DK / 4], qs[STEPS][DK / 4];
    __shared__ float gs[STEPS], bs[STEPS];
    __shared__ __align__(16) __nv_bfloat16 vs[STEPS][ROWS];
    const int head = blockIdx.x, lane = threadIdx.x & 7;
    const int row = blockIdx.y * ROWS + (threadIdx.x >> 3);
    const int key_head = head / (hv / hk);
    float s[16];
    const float* s0 = state + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j) s[j] = s0[8 * j + lane];
    Prefetch pf;
    pf.fetch(q, k, v, g, beta, 0, W, hk, hv, head, key_head, blockIdx.y * ROWS);
    for (int t0 = 0; t0 < W; t0 += STEPS) {
        const int count = min(STEPS, W - t0);
        __syncthreads();
        pf.store(ks, qs, gs, bs, vs);
        __syncthreads();
        if (t0 + STEPS < W)
            pf.fetch(q, k, v, g, beta, t0 + STEPS, W, hk, hv, head, key_head, blockIdx.y * ROWS);
        for (int step = 0; step < count; ++step) {
            const float gt = gs[step], bt = bs[step];
            const float vt = __bfloat162float(vs[step][threadIdx.x >> 3]);
            const float* key = reinterpret_cast<const float*>(ks[step]);
            const float* query = reinterpret_cast<const float*>(qs[step]);
            float memory = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                // One lane keeps the old r-th accumulator's exact serial j order and multiply rounding point.
                s[j] = __fmul_rn(s[j], gt);
                memory = __fmaf_rn(s[j], key[8 * j + lane], memory);
            }
            const float delta = __fmul_rn(__fsub_rn(vt, ordered_sum(memory)), bt);
            float output = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                s[j] = __fmaf_rn(key[8 * j + lane], delta, s[j]);
                output = __fmaf_rn(s[j], query[8 * j + lane], output);
            }
            output = ordered_sum(output);
            if (lane == 0) {
                const size_t at = (static_cast<size_t>(t0 + step) * hv + head) * DV + row;
                y[at] = __float2bfloat16_rn(output);
            }
        }
    }
    float* s1 = last + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j) s1[8 * j + lane] = s[j];
}

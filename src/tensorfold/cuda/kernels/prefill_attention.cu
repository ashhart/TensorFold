// Prompt attention with _attend's bits (prefill_attention.py): one CTA takes HPC query heads of one KV head for
// 16-row blocks, so each 32-key slot of keys or values staged in shared memory serves all of them.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int BN = 64, HALF = 32;             // keys a tile (the fold's unit), keys a staging slot
constexpr float LOG2E = 1.4426950408889634f;
constexpr int SMEM_MAX = 96 * 1024;

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(qmm_frag::smem(p)));
}

// Triton's exp on sm_121: a round-to-nearest log2(e) multiply, then ex2.approx
__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

// Triton's fp32 division (div.full, not IEEE round-to-nearest)
__device__ __forceinline__ float tdiv(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// 16-byte chunk c of row r, XOR-swizzled so ldmatrix's eight rows hit eight bank groups
template <int D>
__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }

// _tile's row sum under Triton's warps [1, 8]: pairs in-thread, lanes xor 2 then 1, then warps j and j + 4, ...
__device__ __forceinline__ float rowsum(const float (&c)[8]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float x = __fadd_rn(c[j], __shfl_xor_sync(0xffffffffu, c[j], 2));
        w[j] = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                     __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
pattn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
             const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
             int G, float scale) {
    constexpr int C = D / 8, K16 = D / 16, NB = D / 8, RB = WARPS / HPC, THREADS = 32 * WARPS;
    constexpr bool QREG = D <= 128;               // the queries' fragments stay in registers, else in shared memory
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * HALF * C;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int groups = G / HPC, hk = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int row0 = blockIdx.x * 16 * RB, last = min(row0 + 16 * RB, W) - 1;
    const int head = hk * G + hg * HPC + warp % HPC, wrow = row0 + (warp / HPC) * 16;
    const int tiles = (p0 + last) / BN + 1, items = 4 * tiles, limit = p0 + W;
    // item i: tile i / 4's keys [0, 32), keys [32, 64), values [0, 32), values [32, 64)
    auto stage = [&](int item) {
        const __nv_bfloat16* src = item % 4 < 2 ? kc : vc;
        const int key0 = (item / 4) * BN + (item % 2) * HALF;
        uint4* dst = slots + (item % NS) * HALF * C;
        for (int i = threadIdx.x; i < HALF * C; i += THREADS) {
            const int key = key0 + i / C;
            const bool in = key < limit;             // past the cache's filled keys: zeros, never read
            cp16z(dst + sw<D>(i / C, i % C), src + ((int64_t)(in ? key : 0) * HK + hk) * D + (i % C) * 8, in);
        }
    };
    uint32_t qa[QREG ? K16 : 1][4];
    if constexpr (QREG) {
        const int r0 = wrow + g, r1 = r0 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r0, W - 1) * H + head) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r1, W - 1) * H + head) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < W ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < W ? q1[8 * k + qd + 4] : 0u;
        }
    } else {
        for (int i = threadIdx.x; i < WARPS * 16 * C; i += THREADS) {
            const int w = i / (16 * C), r = (i / C) % 16, row = row0 + (w / HPC) * 16 + r;
            const int hd = hk * G + hg * HPC + w % HPC;
            cp16z(qs + sw<D>(i / C, i % C), q + ((int64_t)min(row, W - 1) * H + hd) * D + (i % C) * 8, row < W);
        }
        commit();
    }
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float sc[8][4];
    uint32_t pa[4][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int pos0 = p0 + wrow + g, pos1 = pos0 + 8;
    auto take = [&](int it) -> const uint4* {           // item it's slot, once every thread's copy has landed
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * HALF * C;
    };
    auto done = [&](int it) {                            // the slot is free: stage the item NS ahead into it
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    // scores of a slot's 32 keys into n8 blocks s[0..3]: a chain over D's k16 steps from zero, as tl.dot
    auto scores = [&](float (&s)[4][4], const uint4* sl) {
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            uint32_t a[4];
            if constexpr (QREG) {
                a[0] = qa[k][0]; a[1] = qa[k][1]; a[2] = qa[k][2]; a[3] = qa[k][3];
            } else {
                ldmatrix4(a, qs + sw<D>(warp * 16 + lane % 16, 2 * k + lane / 16));
            }
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw<D>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(s[n], a, b[0], b[1]);
                    mma0(s[n + 1], a, b[2], b[3]);
                } else {
                    mma(s[n], a, b[0], b[1]);
                    mma(s[n + 1], a, b[2], b[3]);
                }
            }
        }
    };
    // a slot's 32 values: the tile's k16 steps p0 and p1 of the output chain, in key order
    auto values = [&](const uint32_t (&pa0)[4], const uint32_t (&pa1)[4], const uint4* sl) {
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<D>(8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa0, b[0], b[1]);
            mma(o[i + 1], pa0, b[2], b[3]);
        }
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<D>(16 + 8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa1, b[0], b[1]);
            mma(o[i + 1], pa1, b[2], b[3]);
        }
    };
    for (int t = 0; t < tiles; ++t) {
        const int it = 4 * t;
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[0]), take(it));
        done(it);
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[4]), take(it + 1));
        done(it + 1);
        {                                                // _tile's fold of the tile's 64 scores
            const int kb = t * BN + 2 * qd;
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = c <= pos0 ? __fmul_rn(sc[n][0], scale) : -INFINITY;
                sc[n][1] = c + 1 <= pos0 ? __fmul_rn(sc[n][1], scale) : -INFINITY;
                sc[n][2] = c <= pos1 ? __fmul_rn(sc[n][2], scale) : -INFINITY;
                sc[n][3] = c + 1 <= pos1 ? __fmul_rn(sc[n][3], scale) : -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
                tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
            const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
            const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
            const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
            float c0[8], c1[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = act0 && c <= pos0 ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
                sc[n][1] = act0 && c + 1 <= pos0 ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
                sc[n][2] = act1 && c <= pos1 ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
                sc[n][3] = act1 && c + 1 <= pos1 ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
                c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
                c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
            }
            l0 = __fmaf_rn(l0, a0, rowsum(c0));
            l1 = __fmaf_rn(l1, a1, rowsum(c1));
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
                pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
                pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
        values(pa[0], pa[1], take(it + 2));
        done(it + 2);
        values(pa[2], pa[3], take(it + 3));
        done(it + 3);
    }
    wait_group<0>();
    const int r0 = wrow + g, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * H + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * H + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}

template <int D, int WARPS, int HPC>
void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, int p0, float scale) {
    constexpr int QBYTES = D <= 128 ? 0 : WARPS * 16 * D * 2, SLOT = HALF * D * 2;
    constexpr int NS = (SMEM_MAX - QBYTES) / SLOT < 8 ? (SMEM_MAX - QBYTES) / SLOT : 8;
    static_assert(NS >= 2, "shared memory for two slots");
    constexpr int smem = NS * SLOT + QBYTES;
    static const cudaError_t configured =
        cudaFuncSetAttribute(pattn_kernel<D, WARPS, HPC, NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(configured == cudaSuccess, "prompt attention needs ", smem, " bytes of shared memory a block");
    const int W = static_cast<int>(q.size(0)), H = static_cast<int>(q.size(1)), HK = static_cast<int>(k.size(1));
    const int G = H / HK, rows = 16 * (WARPS / HPC);
    const dim3 grid((W + rows - 1) / rows, HK * (G / HPC));
    pattn_kernel<D, WARPS, HPC, NS><<<grid, 32 * WARPS, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), p0,
        W, H, HK, G, scale);
}

template <int D>
void dispatch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, int p0, float scale,
              int hpc) {
    switch (hpc) {                                     // eight warps: HPC heads x (8 / HPC) 16-row blocks
        case 1: launch<D, 8, 1>(q, k, v, out, p0, scale); break;
        case 2: launch<D, 8, 2>(q, k, v, out, p0, scale); break;
        case 4: launch<D, 8, 4>(q, k, v, out, p0, scale); break;
        default: launch<D, 8, 8>(q, k, v, out, p0, scale); break;
    }
}

}  // namespace

void prefill_attention_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, int p0,
                            float scale, int hpc) {
    if (q.size(0) == 0) return;
    if (q.size(2) == 256) dispatch<256>(q, k, v, out, p0, scale, hpc);
    else dispatch<128>(q, k, v, out, p0, scale, hpc);
}

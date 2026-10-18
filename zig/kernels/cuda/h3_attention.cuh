// MiniMax H3's tile-routed attention in bf16 on mma.sync m16n8k16: fp32 scores, an fp32 online softmax, fp32 sums.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace tf_h3 {

__device__ __forceinline__ uint32_t smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem(p)));
}
__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem(p)));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

constexpr int head_dim = 128, tile_rows = 64;
// Bytes of one head's tile in shared memory, and the attention's dynamic shared bytes: q, k and v (two blocks an SM).
constexpr int tile_bytes = tile_rows * head_dim * 2;
constexpr int attention_shared = 3 * tile_bytes;

// softmax(q k^T scale) v over a query tile's key tiles in bf16, fp32 scores and sums; grid [query tiles, H] of 128.
__device__ __forceinline__ void attention_body(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
        const int* __restrict__ LIST, const int* __restrict__ SIZES, const int* __restrict__ ROWOF,
        __nv_bfloat16* __restrict__ Y, int slots, int heads, int queries, int keys, int first_query, int per_query,
        float scale) {
    constexpr int D = head_dim, T = tile_rows, THREADS = 128;
    extern __shared__ __align__(128) unsigned char buf[];
    unsigned char* const qsm = buf;
    unsigned char* const ksm = buf + tile_bytes;
    unsigned char* const vsm = buf + 2 * tile_bytes;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.y, qt = first_query + blockIdx.x;
    const size_t hs = static_cast<size_t>(head) * slots;
    const int* list = LIST + (per_query ? (static_cast<size_t>(head) * queries + blockIdx.x) * keys : 0);

    // A row is 256 bytes in 16 chunks; eight rows read at one chunk fall in different banks.
    auto at = [](int r, int c) { return r * 256 + ((c ^ (r & 7)) << 4); };
    auto load = [&](unsigned char* to, const __nv_bfloat16* from, int tile) {
        const __nv_bfloat16* src = from + (hs + static_cast<size_t>(tile) * T) * D;
        for (int c = tid; c < T * 16; c += THREADS) cp16(to + at(c >> 4, c & 15), src + c * 8);
    };

    // a key tile's k is fetched while the tile before it is weighed against its values, and its v while its scores form
    load(qsm, Q, qt);
    load(ksm, K, list[0]);
    commit();

    const int row_a = (lane & 7) + ((lane >> 3) & 1) * 8, chunk_a = lane >> 4;
    const int row_b = (lane & 7) + ((lane >> 4) & 1) * 8, chunk_b = (lane >> 3) & 1;
    const int r0 = warp * 16 + (lane >> 2), c0 = (lane & 3) * 2;
    const float sl = scale * 1.4426950408889634f;
    float out[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j][0] = out[j][1] = out[j][2] = out[j][3] = 0.0f;
    float top0 = -1e30f, top1 = -1e30f, mass0 = 0.0f, mass1 = 0.0f;
    uint32_t qa[8][4];

    for (int sel = 0; sel < keys; ++sel) {
        const int tile = list[sel];
        wait<0>();
        __syncthreads();
        load(vsm, V, tile);
        commit();
        if (sel == 0) {
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) ldmatrix4(qa[ks], qsm + at(warp * 16 + row_a, ks * 2 + chunk_a));
        }
        const unsigned char* pk = ksm;
        const unsigned char* pv = vsm;
        const int size = SIZES[tile];

        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b[4];
                ldmatrix4(b, pk + at(j * 16 + row_b, ks * 2 + chunk_b));
                mma_bf16(s[2 * j], qa[ks], b[0], b[1]);
                mma_bf16(s[2 * j + 1], qa[ks], b[2], b[3]);
            }
        }
        __syncthreads();
        if (sel + 1 < keys) load(ksm, K, list[sel + 1]);
        commit();
        // each row's 64 scores sit in the four lanes of one group: its maximum needs two shuffles
        float m0 = -1e30f, m1 = -1e30f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + c0;
            s[j][0] = col < size ? s[j][0] * sl : -1e30f;
            s[j][1] = col + 1 < size ? s[j][1] * sl : -1e30f;
            s[j][2] = col < size ? s[j][2] * sl : -1e30f;
            s[j][3] = col + 1 < size ? s[j][3] * sl : -1e30f;
            m0 = fmaxf(m0, fmaxf(s[j][0], s[j][1]));
            m1 = fmaxf(m1, fmaxf(s[j][2], s[j][3]));
        }
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        const float n0 = fmaxf(top0, m0), n1 = fmaxf(top1, m1);
        const float f0 = exp2f(top0 - n0), f1 = exp2f(top1 - n1);
        // the weights in fp32 for the softmax's mass, rounded to bf16 as the value product's A operand
        float l0 = 0.0f, l1 = 0.0f;
        uint32_t pa[4][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float p0 = exp2f(s[j][0] - n0), p1 = exp2f(s[j][1] - n0);
            const float p2 = exp2f(s[j][2] - n1), p3 = exp2f(s[j][3] - n1);
            l0 += p0 + p1;
            l1 += p2 + p3;
            pa[j >> 1][(j & 1) * 2] = pack2(p0, p1);
            pa[j >> 1][(j & 1) * 2 + 1] = pack2(p2, p3);
        }
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        mass0 = mass0 * f0 + l0;
        mass1 = mass1 * f1 + l1;
        top0 = n0;
        top1 = n1;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            out[j][0] *= f0;
            out[j][1] *= f0;
            out[j][2] *= f1;
            out[j][3] *= f1;
        }
        wait<1>();
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int jd = 0; jd < 8; ++jd) {
                uint32_t b[4];
                ldmatrix4t(b, pv + at(kk * 16 + row_a, jd * 2 + chunk_a));
                mma_bf16(out[2 * jd], pa[kk], b[0], b[1]);
                mma_bf16(out[2 * jd + 1], pa[kk], b[2], b[3]);
            }
        }
    }
    const int to0 = ROWOF[qt * T + r0], to1 = ROWOF[qt * T + r0 + 8];
    const float i0 = 1.0f / mass0, i1 = 1.0f / mass1;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int col = head * D + j * 8 + c0;
        if (to0 >= 0) *reinterpret_cast<uint32_t*>(Y + static_cast<size_t>(to0) * heads * D + col) = pack2(out[j][0] * i0, out[j][1] * i0);
        if (to1 >= 0) *reinterpret_cast<uint32_t*>(Y + static_cast<size_t>(to1) * heads * D + col) = pack2(out[j][2] * i1, out[j][3] * i1);
    }
}

} // namespace tf_h3

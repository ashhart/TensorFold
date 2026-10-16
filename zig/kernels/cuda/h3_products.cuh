// MiniMax H3's int8 products and tile attention on mma.sync m16n8k32: templates included by h3.cu.
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
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_u8s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Block order: bands of `group` row tiles; inside a band one column tile's row tiles run next to each other.
__device__ __forceinline__ int2 tile_of(int b, int rows_t, int cols_t, int group) {
    const int band = group * cols_t;
    const int first = b / band * group, in_band = b % band, height = min(group, rows_t - first);
    return make_int2(first + in_band % height, in_band / height);
}

enum { plain = 0, swiglu = 1, wide = 2 };
constexpr int wide_group = 1024;

// D[m, n] = sum over k of X[m, k] W[n, k] in int8, K in two-stage slices of KC bytes; plain, swiglu and wide epilogues.
template <int MODE, int BM, int BN, int WM, int WN, int KC = 128>
__device__ __forceinline__ void gemm_i8(
        const int8_t* __restrict__ X, const float* __restrict__ XS, const int8_t* __restrict__ W,
        const float* __restrict__ WS, __nv_bfloat16* __restrict__ Y, int M, int N, int K, int group) {
    constexpr int CH = KC / 16, THREADS = WM * WN * 32;
    constexpr int MT = BM / WM / 16, NT = BN / WN / 8;
    constexpr int XB = BM * KC, STAGE = (BM + BN) * KC;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
    const int rows_t = (M + BM - 1) / BM, cols_t = N / BN;
    const int2 at = tile_of(blockIdx.x, rows_t, cols_t, group);
    const int m0 = at.x * BM, n0 = at.y * BN;

    // where a row's 16-byte chunk sits in a stage: eight rows read at one chunk fall in different banks
    auto spot = [](int r, int ch) {
        return KC == 64 ? (r >> 1) * 128 + (r & 1) * 64 + ((ch ^ ((r >> 1) & 3)) << 4) : r * KC + ((ch ^ (r & 7)) << 4);
    };
    auto load = [&](int s, int k0) {
        unsigned char* px = buf + s * STAGE;
        unsigned char* pw = px + XB;
        for (int c = tid; c < BM * CH; c += THREADS) {
            const int r = c / CH, ch = c % CH;
            cp16(px + spot(r, ch), X + static_cast<size_t>(m0 + r) * K + k0 + ch * 16);
        }
        for (int c = tid; c < BN * CH; c += THREADS) {
            const int r = c / CH, ch = c % CH;
            cp16(pw + spot(r, ch), W + static_cast<size_t>(n0 + r) * K + k0 + ch * 16);
        }
    };

    int acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;
    float facc[MODE == wide ? MT : 1][MODE == wide ? NT : 1][4];
    if (MODE == wide) {
#pragma unroll
        for (int i = 0; i < (MODE == wide ? MT : 1); ++i)
#pragma unroll
            for (int j = 0; j < (MODE == wide ? NT : 1); ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) facc[i][j][e] = 0.0f;
    }
    const int row_a = (lane & 7) + ((lane >> 3) & 1) * 8, chunk_a = lane >> 4;
    const int row_b = (lane & 7) + ((lane >> 4) & 1) * 8, chunk_b = (lane >> 3) & 1;
    const int rbase = m0 + wm * (BM / WM) + (lane >> 2), cbase = n0 + wn * (BN / WN) + (lane & 3) * 2;

    load(0, 0);
    commit();
    const int slices = K / KC;
    for (int s = 0; s < slices; ++s) {
        if (s + 1 < slices) load((s + 1) & 1, (s + 1) * KC);
        commit();
        wait<1>();
        __syncthreads();
        const unsigned char* px = buf + (s & 1) * STAGE + wm * (BM / WM) * KC;
        const unsigned char* pw = buf + (s & 1) * STAGE + XB + wn * (BN / WN) * KC;
#pragma unroll
        for (int ks = 0; ks < KC / 32; ++ks) {
            uint32_t a[MT][4], b[NT / 2][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = i * 16 + row_a;
                ldmatrix4(a[i], px + spot(r, ks * 2 + chunk_a));
            }
#pragma unroll
            for (int j = 0; j < NT / 2; ++j) {
                const int r = j * 16 + row_b;
                ldmatrix4(b[j], pw + spot(r, ks * 2 + chunk_b));
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) mma_s8(acc[i][j], a[i], b[j / 2][(j & 1) * 2], b[j / 2][(j & 1) * 2 + 1]);
        }
        if (MODE == wide && ((s + 1) * KC) % wide_group == 0) {
            const int g = ((s + 1) * KC) / wide_group - 1, groups = K / wide_group;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int row = rbase + i * 16;
                const float s0 = XS[static_cast<size_t>(row) * groups + g], s1 = XS[static_cast<size_t>(row + 8) * groups + g];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    facc[MODE == wide ? i : 0][MODE == wide ? j : 0][0] += float(acc[i][j][0]) * s0;
                    facc[MODE == wide ? i : 0][MODE == wide ? j : 0][1] += float(acc[i][j][1]) * s0;
                    facc[MODE == wide ? i : 0][MODE == wide ? j : 0][2] += float(acc[i][j][2]) * s1;
                    facc[MODE == wide ? i : 0][MODE == wide ? j : 0][3] += float(acc[i][j][3]) * s1;
                    acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = rbase + i * 16 + h * 8;
            if (row >= M) continue;
            const float xs = MODE == wide ? 1.0f : XS[row];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int col = cbase + j * 8;
                float v0, v1;
                if (MODE == wide) {
                    v0 = facc[MODE == wide ? i : 0][MODE == wide ? j : 0][2 * h] * WS[col];
                    v1 = facc[MODE == wide ? i : 0][MODE == wide ? j : 0][2 * h + 1] * WS[col + 1];
                } else {
                    v0 = float(acc[i][j][2 * h]) * xs * WS[col];
                    v1 = float(acc[i][j][2 * h + 1]) * xs * WS[col + 1];
                }
                if (MODE == swiglu) {
                    Y[static_cast<size_t>(row) * (N / 2) + col / 2] = __float2bfloat16(v1 / (1.0f + __expf(-v1)) * v0);
                } else {
                    __nv_bfloat162 o;
                    o.x = __float2bfloat16(v0);
                    o.y = __float2bfloat16(v1);
                    *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(row) * N + col) = o;
                }
            }
        }
    }
}

constexpr int head_dim = 128, tile_rows = 64;

// softmax(q k^T) v over a query tile's key tiles, int8 scores and 8-bit weights; grid [query tiles, H], 128 threads.
__device__ __forceinline__ void attention_body(
        const int8_t* __restrict__ Q, const float* __restrict__ QS, const int8_t* __restrict__ K,
        const float* __restrict__ KS, const int8_t* __restrict__ VT, const float* __restrict__ VS,
        const int* __restrict__ LIST, const int* __restrict__ SIZES, const int* __restrict__ ROWOF,
        __nv_bfloat16* __restrict__ Y, int slots, int tiles, int heads, int queries, int keys, int first_query,
        int per_query, float scale, float* __restrict__ SOUT) {
    constexpr int D = head_dim, T = tile_rows, THREADS = 128;
    constexpr int QB = T * D, KB = T * D, VB = D * T, STAGE = KB + VB, PB = T * T;
    extern __shared__ __align__(128) unsigned char buf[];
    unsigned char* const qsm = buf;
    unsigned char* const psm = buf + QB;
    unsigned char* const stages = buf + QB + PB;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.y, qt = first_query + blockIdx.x;
    const size_t hs = static_cast<size_t>(head) * slots, ht = static_cast<size_t>(head) * tiles;
    const int* list = LIST + (per_query ? (static_cast<size_t>(head) * queries + blockIdx.x) * keys : 0);

    // Chunk positions that keep eight rows read at one chunk in different banks.
    auto at128 = [](int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); };
    auto at64 = [](int r, int c) { return (r >> 1) * 128 + (r & 1) * 64 + ((c ^ ((r >> 1) & 3)) << 4); };
    auto load = [&](int s, int tile) {
        unsigned char* pk = stages + s * STAGE;
        unsigned char* pv = pk + KB;
        const int8_t* k = K + (hs + static_cast<size_t>(tile) * T) * D;
        const int8_t* v = VT + (ht + tile) * VB;
        for (int c = tid; c < T * 8; c += THREADS) cp16(pk + at128(c >> 3, c & 7), k + c * 16);
        for (int c = tid; c < D * 4; c += THREADS) cp16(pv + at64(c >> 2, c & 3), v + c * 16);
    };

    const int8_t* q = Q + (hs + static_cast<size_t>(qt) * T) * D;
    for (int c = tid; c < T * 8; c += THREADS) cp16(qsm + at128(c >> 3, c & 7), q + c * 16);
    load(0, list[0]);
    commit();

    const int row_a = (lane & 7) + ((lane >> 3) & 1) * 8, chunk_a = lane >> 4;
    const int row_b = (lane & 7) + ((lane >> 4) & 1) * 8, chunk_b = (lane >> 3) & 1;
    const int r0 = warp * 16 + (lane >> 2), c0 = (lane & 3) * 2;
    const float qs0 = QS[hs + qt * T + r0] * scale * 1.4426950408889634f;
    const float qs1 = QS[hs + qt * T + r0 + 8] * scale * 1.4426950408889634f;
    float out[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j][0] = out[j][1] = out[j][2] = out[j][3] = 0.0f;
    float top0 = -1e30f, top1 = -1e30f, mass0 = 0.0f, mass1 = 0.0f;
    uint32_t qa[4][4];

    for (int sel = 0; sel < keys; ++sel) {
        const int tile = list[sel];
        if (sel + 1 < keys) load((sel + 1) & 1, list[sel + 1]);
        commit();
        wait<1>();
        __syncthreads();
        if (sel == 0) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int r = warp * 16 + row_a;
                ldmatrix4(qa[ks], qsm + at128(r, ks * 2 + chunk_a));
            }
        }
        const unsigned char* pk = stages + (sel & 1) * STAGE;
        const unsigned char* pv = pk + KB;
        const int size = SIZES[tile];
        const float ks_t = KS[ht + tile], vs_t = VS[ht + tile];

        int sc[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b[4];
                ldmatrix4(b, pk + at128(j * 16 + row_b, ks * 2 + chunk_b));
                mma_s8(sc[2 * j], qa[ks], b[0], b[1]);
                mma_s8(sc[2 * j + 1], qa[ks], b[2], b[3]);
            }
        }
        // each row's 64 scores sit in the four lanes of one group: its maximum needs two shuffles
        float s[8][4], m0 = -1e30f, m1 = -1e30f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + c0;
            const float o0 = col < size ? 0.0f : -1e30f, o1 = col + 1 < size ? 0.0f : -1e30f;
            s[j][0] = float(sc[j][0]) * (qs0 * ks_t) + o0;
            s[j][1] = float(sc[j][1]) * (qs0 * ks_t) + o1;
            s[j][2] = float(sc[j][2]) * (qs1 * ks_t) + o0;
            s[j][3] = float(sc[j][3]) * (qs1 * ks_t) + o1;
            m0 = fmaxf(m0, fmaxf(s[j][0], s[j][1]));
            m1 = fmaxf(m1, fmaxf(s[j][2], s[j][3]));
            if (SOUT) {
                float* o = SOUT + (hs + qt * T + r0) * slots + tile * T + col;
                o[0] = s[j][0];
                o[1] = s[j][1];
                o[8 * static_cast<size_t>(slots)] = s[j][2];
                o[8 * static_cast<size_t>(slots) + 1] = s[j][3];
            }
        }
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        int l0 = 0, l1 = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + c0;
            const int u0 = min(255, int(exp2f(s[j][0] - m0) * 255.0f + 0.5f)), u1 = min(255, int(exp2f(s[j][1] - m0) * 255.0f + 0.5f));
            const int u2 = min(255, int(exp2f(s[j][2] - m1) * 255.0f + 0.5f)), u3 = min(255, int(exp2f(s[j][3] - m1) * 255.0f + 0.5f));
            l0 += u0 + u1;
            l1 += u2 + u3;
            *reinterpret_cast<uint16_t*>(psm + at64(r0, col >> 4) + (col & 15)) = uint16_t(u0 | (u1 << 8));
            *reinterpret_cast<uint16_t*>(psm + at64(r0 + 8, col >> 4) + (col & 15)) = uint16_t(u2 | (u3 << 8));
        }
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        __syncwarp();

        int pr[16][4];
#pragma unroll
        for (int j = 0; j < 16; ++j) pr[j][0] = pr[j][1] = pr[j][2] = pr[j][3] = 0;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t a[4];
            ldmatrix4(a, psm + at64(warp * 16 + row_a, ks * 2 + chunk_a));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t b[4];
                ldmatrix4(b, pv + at64(j * 16 + row_b, ks * 2 + chunk_b));
                mma_u8s8(pr[2 * j], a, b[0], b[1]);
                mma_u8s8(pr[2 * j + 1], a, b[2], b[3]);
            }
        }
        const float n0 = fmaxf(top0, m0), n1 = fmaxf(top1, m1);
        const float f0 = exp2f(top0 - n0), f1 = exp2f(top1 - n1), e0 = exp2f(m0 - n0), e1 = exp2f(m1 - n1);
        const float g0 = e0 * vs_t, g1 = e1 * vs_t;
        mass0 = mass0 * f0 + e0 * float(l0);
        mass1 = mass1 * f1 + e1 * float(l1);
        top0 = n0;
        top1 = n1;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            out[j][0] = out[j][0] * f0 + float(pr[j][0]) * g0;
            out[j][1] = out[j][1] * f0 + float(pr[j][1]) * g0;
            out[j][2] = out[j][2] * f1 + float(pr[j][2]) * g1;
            out[j][3] = out[j][3] * f1 + float(pr[j][3]) * g1;
        }
        __syncthreads();
    }
    const int to0 = ROWOF[qt * T + r0], to1 = ROWOF[qt * T + r0 + 8];
    const float i0 = 1.0f / fmaxf(mass0, 1e-30f), i1 = 1.0f / fmaxf(mass1, 1e-30f);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int col = head * D + j * 8 + c0;
        __nv_bfloat162 o;
        if (to0 >= 0) {
            o.x = __float2bfloat16(out[j][0] * i0);
            o.y = __float2bfloat16(out[j][1] * i0);
            *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(to0) * heads * D + col) = o;
        }
        if (to1 >= 0) {
            o.x = __float2bfloat16(out[j][2] * i1);
            o.y = __float2bfloat16(out[j][3] * i1);
            *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(to1) * heads * D + col) = o;
        }
    }
}

} // namespace tf_h3

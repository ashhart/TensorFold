// hpattn8(_whole)_kernel<256, 8, 2, 2>'s rows, bit for bit, for a chunk of at most 64 rows: the keys over many blocks.
#pragma once
#include "prefill_attention.cuh"

namespace tf_prefill_split {

using namespace tf_prefill_attention;

constexpr int D = 256, C = D / 8, K16 = D / 16, CS = C;  // the head's 16-byte chunks, 16-dim steps, staged chunks
constexpr int G6 = 6;                                    // query heads a key head (Qwen3.8-27B: 24 / 4)
constexpr int AW = G6;                                   // scores: a warp a query head of the key head, 16 rows
constexpr int ATHREADS = 32 * AW, ANS = 2;               // the body's two staging slots
constexpr int A_SMEM = (ANS * HALF * CS + AW * 16 * CS) * 16;  // slots and staged queries

struct Geo {
    const __nv_bfloat16* q;  // [W, H, D], the chunk's first row at 0
    const uint8_t *kc, *vc;  // e4m3 caches (qwen35_kv8.cu's layout)
    const float *ks, *vs;    // their scales
    int p0, W, H, HK, G;
    float scale;
    int CAP;
    const int* PT;  // pages (null: one whole window)
    long long PS;
    int PTOK;
};

// The cache row (and page byte offset) of key position `row` for key head hk: the body's row_of and page_of.
template <bool HPAGED>
__device__ __forceinline__ void locate(const Geo& g, int hk, int row, int64_t& r, int64_t& pg) {
    if constexpr (HPAGED) {
        r = g.PT ? (int64_t)hk * g.PTOK + row % g.PTOK : (int64_t)hk * g.CAP + row;
        pg = g.PT ? (int64_t)g.PT[row / g.PTOK] * g.PS : 0;
    } else {
        r = (int64_t)hk * g.CAP + row;
        pg = 0;
    }
}

// PASS 0 writes each tile's maxima to TM; PASS 1 writes its weights to PA and their sums to RS with N from the scan.
template <int PASS, bool HPAGED>
__global__ void __launch_bounds__(ATHREADS, 1)
scores_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
              const float* __restrict__ ks, const float* __restrict__ vs, int p0, int W, int H, int HK, int G,
              float scale, int CAP, const int* __restrict__ PT, long long PS, int PTOK, int R0, int LAST, int TPS,
              int TSTRIDE, float* TM, const float* N, uint32_t* __restrict__ PA, float* __restrict__ RS) {
    const Geo geo{q, kc, vc, ks, vs, p0, W, H, HK, G, scale, CAP, PT, PS, PTOK};
    constexpr int WARPS = AW, NS = ANS, THREADS = ATHREADS;
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * HALF * CS;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int hk = blockIdx.y, head = hk * G + warp;
    const int all_tiles = (p0 + LAST) / BN + 1, tb = (int)blockIdx.z * TPS;
    const int tiles = min(all_tiles - tb, TPS), items = 2 * tiles, limit = p0 + W;
    if (tiles <= 0) return;
    // the body's fp8 fetch and put over key halves only: item i is tile tb + i / 2's keys from (i % 2) * 32 on
    constexpr int PER = (HALF * C + THREADS - 1) / THREADS;
    uint2 pre[PER];
    float pre_s[PER];
    auto fetch = [&](int item) {
        const int key0 = (item / 2 + tb) * BN + (item % 2) * HALF;
        const int lo = HPAGED && geo.PT ? min(key0, limit - 1) / geo.PTOK * geo.PTOK : 0;
        const int64_t first = HPAGED && geo.PT ? (int64_t)geo.PT[lo / geo.PTOK] * geo.PS : 0;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int i = threadIdx.x + j * THREADS, key = key0 + i / C;
            pre[j] = make_uint2(0u, 0u);
            pre_s[j] = 0.0f;
            if (i < HALF * C && key < limit) {
                int64_t r, pg;
                if (HPAGED && geo.PT && key >= lo && key - lo < geo.PTOK) {
                    r = (int64_t)hk * geo.PTOK + (key - lo);
                    pg = first;
                } else locate<HPAGED>(geo, hk, key, r, pg);
                pre[j] = __ldg(reinterpret_cast<const uint2*>(kc + pg + r * D + (i % C) * 8));
                pre_s[j] = __ldg(reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(ks) + pg) + r);
            }
        }
    };
    auto put = [&](int item) {
        uint4* dst = slots + (item % NS) * HALF * CS;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int i = threadIdx.x + j * THREADS;
            if (i < HALF * C) dst[sw<8 * CS>(i / C, i % C)] = widen8(pre[j], pre_s[j]);
        }
    };
    // the body's staged queries (D = 256): warp w's 16 rows of head hk * G + w
    for (int i = threadIdx.x; i < WARPS * 16 * C; i += THREADS) {
        const int w = i / (16 * C), row = R0 + (i / C) % 16, hd = hk * G + w;
        cp16z(qs + sw<8 * CS>(i / C, i % C), q + ((int64_t)min(row, W - 1) * H + hd) * D + (i % C) * 8, row < W);
    }
    commit();
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) {
            fetch(i);
            put(i);
        }
        commit();
    }
    if (NS < items) fetch(NS);
    auto take = [&](int it) -> const uint4* {
        cp_wait<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * HALF * CS;
    };
    auto done = [&](int it) {
        __syncthreads();
        if (it + NS < items) {
            put(it + NS);
            if (it + NS + 1 < items) fetch(it + NS + 1);
        }
        commit();
    };
    auto scores = [&](float (&s)[4][4], const uint4* sl) {
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            uint32_t a[4];
            ldmatrix4(a, qs + sw<8 * CS>(warp * 16 + lane % 16, 2 * k + lane / 16));
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw<8 * CS>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
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
    float sc[8][4];
    const int pos0 = p0 + R0 + g, pos1 = pos0 + 8;  // the body's causal positions (wrow = R0)
    const int r0 = g, r1 = g + 8;
    for (int t = 0; t < tiles; ++t) {
        const int it = 2 * t, T = tb + t;
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[0]), take(it));
        done(it);
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[4]), take(it + 1));
        done(it + 1);
        const int kb = T * BN + 2 * qd;
        float tm0, tm1;
        tile_mask(sc, kb, pos0, pos1, scale, tm0, tm1);
        const size_t at = ((size_t)head * TSTRIDE + T) * 16;
        if constexpr (PASS == 0) {
            if (qd == 0) {
                TM[at + r0] = tm0;
                TM[at + r1] = tm1;
            }
        } else {
            float c0[8], c1[8];
            tile_weights(sc, kb, pos0, pos1, tm0 != -INFINITY, tm1 != -INFINITY, N[at + r0], N[at + r1], c0, c1);
            const float s0 = rowsum(c0), s1 = rowsum(c1);
            uint32_t pa[4][4];
            tile_pack(pa, sc);
            uint4* dst = reinterpret_cast<uint4*>(PA) + (((size_t)head * TSTRIDE + T) * 32 + lane) * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) dst[j] = make_uint4(pa[j][0], pa[j][1], pa[j][2], pa[j][3]);
            if (qd == 0) {
                RS[at + r0] = s0;
                RS[at + r1] = s1;
            }
        }
    }
    cp_wait<0>();
}

// The running maximum a (row, head) in tile order: n and a as the body forms them. N may share TM's storage.
__global__ void scan_kernel(const float* TM, float* N, float* __restrict__ A, int H, int tiles, int TSTRIDE) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= H * 16) return;
    const int head = id / 16, r = id % 16;
    float m = -INFINITY;
    for (int t0 = 0; t0 < tiles; t0 += 16) {
        float tmv[16];
#pragma unroll
        for (int u = 0; u < 16; ++u)
            tmv[u] = t0 + u < tiles ? TM[((size_t)head * TSTRIDE + t0 + u) * 16 + r] : -INFINITY;
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            if (t0 + u >= tiles) break;
            const size_t at = ((size_t)head * TSTRIDE + t0 + u) * 16 + r;
            const bool act = tmv[u] != -INFINITY;
            const float n = run_max(m, tmv[u], act);
            N[at] = n;
            A[at] = run_factor(m, n, act);
            m = n;
        }
    }
}

constexpr int CW = 2;                      // 8-dim output fragments a chain warp
constexpr int CNS = 4;                     // value tiles in flight
constexpr int CTHREADS = 32 * G6;
constexpr int C_SMEM = CNS * BN * CW * 16;  // widened value tiles: the block's CW chunks of each key, unswizzled

// The chain a (head, CW fragments): l = fma(l, a, sum), o = o * a, then the body's four mma(o, pa, V) a tile; tdiv.
template <bool HPAGED>
__global__ void __launch_bounds__(CTHREADS, 1)
chain_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
             const float* __restrict__ ks, const float* __restrict__ vs, int p0, int W, int H, int HK, int G,
             float scale, int CAP, const int* __restrict__ PT, long long PS, int PTOK, int R0, int LAST, int TSTRIDE,
             const uint32_t* __restrict__ PA, const float* __restrict__ RS, const float* __restrict__ A,
             __nv_bfloat16* __restrict__ out) {
    const Geo geo{q, kc, vc, ks, vs, p0, W, H, HK, G, scale, CAP, PT, PS, PTOK};
    extern __shared__ uint4 smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, mt = lane / 8;
    const int hk = blockIdx.x, head = hk * G + warp, f0 = blockIdx.y * CW;  // the block's first fragment
    const int tiles = (p0 + LAST) / BN + 1, limit = p0 + W;
    // a tile's 64 keys by CW chunks of e4m3 and their scales, widened as the body's put does
    constexpr int PERC = (BN * CW + CTHREADS - 1) / CTHREADS;
    uint2 pre[PERC];
    float pre_s[PERC];
    auto fetch = [&](int t) {
#pragma unroll
        for (int j = 0; j < PERC; ++j) {
            const int i = threadIdx.x + j * CTHREADS, key = t * BN + i / CW, ch = f0 + i % CW;
            pre[j] = make_uint2(0u, 0u);
            pre_s[j] = 0.0f;
            if (i < BN * CW && key < limit) {
                int64_t r, pg;
                locate<HPAGED>(geo, hk, key, r, pg);
                pre[j] = __ldg(reinterpret_cast<const uint2*>(vc + pg + r * D + ch * 8));
                pre_s[j] = __ldg(reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(vs) + pg) + r);
            }
        }
    };
    auto put = [&](int t) {
        uint4* dst = smem + (t % CNS) * BN * CW;
#pragma unroll
        for (int j = 0; j < PERC; ++j) {
            const int i = threadIdx.x + j * CTHREADS;
            if (i < BN * CW) dst[i] = widen8(pre[j], pre_s[j]);  // key i / CW, chunk f0 + i % CW
        }
    };
    float o[CW][4];
#pragma unroll
    for (int i = 0; i < CW; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float l0 = 0.0f, l1 = 0.0f;
    const int r0 = g, r1 = g + 8;
    for (int t = 0; t < CNS - 1 && t < tiles; ++t) {
        fetch(t);
        put(t);
    }
    if (CNS - 1 < tiles) fetch(CNS - 1);
    __syncthreads();
    const uint4* pa_src = reinterpret_cast<const uint4*>(PA) + ((size_t)head * TSTRIDE * 32 + lane) * 4;
    uint4 nw[4];
    float na0, na1, ns0, ns1;
    auto load = [&](int t) {  // tile t's weights, scale factors and sums, a tile ahead
        const size_t at = ((size_t)head * TSTRIDE + t) * 16;
        na0 = A[at + r0];
        na1 = A[at + r1];
        ns0 = RS[at + r0];
        ns1 = RS[at + r1];
        const uint4* src = pa_src + (size_t)t * 32 * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) nw[j] = src[j];
    };
    load(0);
    for (int t = 0; t < tiles; ++t) {
        const float a0 = na0, a1 = na1, s0 = ns0, s1 = ns1;
        uint32_t pa[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            pa[j][0] = nw[j].x;
            pa[j][1] = nw[j].y;
            pa[j][2] = nw[j].z;
            pa[j][3] = nw[j].w;
        }
        if (t + 1 < tiles) load(t + 1);
        l0 = __fmaf_rn(l0, a0, s0);
        l1 = __fmaf_rn(l1, a1, s1);
#pragma unroll
        for (int i = 0; i < CW; ++i) {
            o[i][0] = __fmul_rn(o[i][0], a0);
            o[i][1] = __fmul_rn(o[i][1], a0);
            o[i][2] = __fmul_rn(o[i][2], a1);
            o[i][3] = __fmul_rn(o[i][3], a1);
        }
        const uint4* tile = smem + (t % CNS) * BN * CW;
#pragma unroll
        for (int j = 0; j < 4; ++j) {  // keys 16 j to 16 j + 15, the body's values() order
#pragma unroll
            for (int i = 0; i < CW; i += 2) {
                uint32_t b[4];  // the body's ldmatrix4t matrices: keys 16 j + 8 (mt % 2) + lane % 8, chunk i + mt / 2
                ldmatrix4t(b, tile + (16 * j + 8 * (mt % 2) + lane % 8) * CW + ((i + mt / 2) ^ PATTN_VLANE));
                mma(o[i], pa[j], b[0], b[1]);
                mma(o[i + 1], pa[j], b[2], b[3]);
            }
        }
        // tile t + CNS - 1 goes to tile t - 1's slot, which every warp left before the last barrier
        if (t + CNS - 1 < tiles) {
            put(t + CNS - 1);
            if (t + CNS < tiles) fetch(t + CNS);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < CW; ++i) {
        const int d = 8 * (f0 + i) + 2 * (lane % 4), w0 = R0 + r0, w1 = R0 + r1;
        if (w0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)w0 * H + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (w1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)w1 * H + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}

// The bf16 words of two outputs that differ, counted into BAD[0].
__global__ void diff_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, int n, unsigned* BAD) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) atomicAdd(BAD, 1u);
}

}  // namespace tf_prefill_split

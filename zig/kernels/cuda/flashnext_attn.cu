// Flash Next's attention glue and its sparse indexer's pooling and scores: each row's own fixed-order sums.

// glue.py and attention.py's arithmetic and bf16 rounding points; the attention dots on mma tiles, K ascending.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"
#include "qmm_frag.cuh"

using namespace tf_glue;

namespace fn_attn {

constexpr int TK = 32;   // keys an attention tile holds
constexpr int HD = 256;  // head width the attention tiles are built for
constexpr int QR = 16;   // query heads a KV head's tile holds (G <= 16, the rest zero)

// A block's fixed-order sum: each warp's butterfly, then the warps' sums in warp order.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int warps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total = __fadd_rn(total, scratch[w]);
    __syncthreads();
    return total;
}

__device__ __forceinline__ float inv_rms(float ss, int width, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(width)), eps)));
}

// Rotate-half RoPE over the first 2 * half features at `pos`: cos and sin of the fp32 angle (full range reduction).
__device__ __forceinline__ float rope(float xn, float xpn, int d, int half, int pos, const float* __restrict__ inv) {
    if (d >= 2 * half) return xn;
    const int i = d < half ? d : d - half;
    float s, c;
    sincosf(__fmul_rn(static_cast<float>(pos), inv[i]), &s, &c);
    return d < half ? __fsub_rn(__fmul_rn(xn, c), __fmul_rn(xpn, s)) : __fadd_rn(__fmul_rn(xpn, s), __fmul_rn(xn, c));
}

// ldmatrix .x4 .trans: four 8x8 bf16 tiles of shared memory, each thread's pair down a column (the B layout of rows).
__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(qmm_frag::smem(p)));
}

__device__ __forceinline__ int partner(int d, int half) {
    return d < half ? d + half : (d < 2 * half ? d - half : d);
}

}  // namespace fn_attn

using namespace fn_attn;

__device__ __forceinline__ int media_pos(int pos, int d, int half, const int* rope3, int n, int delta) {
    if (!rope3) return pos;
    const int axis = (d < half ? d : d - half) % 3;
    return pos < n ? rope3[axis * n + pos] : pos + delta;
}

// glue._attn_prep (text positions, bf16 cache), program (row, head): heads normed, rounded, roped, rounded, stored.
__device__ __forceinline__ void fn_attn_prep_body(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                                                               const float* __restrict__ QW, const float* __restrict__ KW,
                                                               const float* __restrict__ IW, const float* __restrict__ INV,
                                                               __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC,
                                                               __nv_bfloat16* __restrict__ VC, __nv_bfloat16* __restrict__ IQ,
                                                               __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ, int NKV,
                                                               int NI, int IHD, int HALF, const int* rope3, int n, int delta) {
    __shared__ float scratch[32];
    const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int pos = POS0[0] + r;
    const long long row = static_cast<long long>(r) * PW;
    if (head >= NQ + NKV + NI) {
        if (d < IHD) IKC[static_cast<long long>(pos) * IHD + d] = P[row + NQ * 2 * HD + 2 * NKV * HD + NI * IHD + d];
        return;
    }
    const bool index = head >= NQ + NKV;
    const int width = index ? IHD : HD;
    const bool live = d < width;
    const long long src = row + (head < NQ ? head * 2 * HD : (head < NQ + NKV ? NQ * 2 * HD + (head - NQ) * HD : NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD));
    const float* w = head < NQ ? QW : (head < NQ + NKV ? KW : IW);
    const float x = live ? bf(P[src + d]) : 0.0f;
    const float rinv = inv_rms(block_sum(__fmul_rn(x, x), scratch), width, eps);
    if (!live) return;
    const int p = partner(d, HALF);
    const float xn = rbf(__fmul_rn(__fmul_rn(x, rinv), w[d]));
    const float xpn = rbf(__fmul_rn(__fmul_rn(bf(P[src + p]), rinv), w[p]));
    const __nv_bfloat16 out = __float2bfloat16_rn(rope(xn, xpn, d, HALF, media_pos(pos, d, HALF, rope3, n, delta), INV));
    if (head < NQ) {
        Q[(static_cast<long long>(r) * NQ + head) * HD + d] = out;
    } else if (!index) {
        const long long slot = static_cast<long long>(pos) * NKV + head - NQ;
        KC[slot * HD + d] = out;
        VC[slot * HD + d] = P[row + NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d];
    } else {
        IQ[(static_cast<long long>(r) * NI + head - NQ - NKV) * IHD + d] = out;
    }
}

extern "C" __global__ void __launch_bounds__(256) fn_attn_prep(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                                                               const float* __restrict__ QW, const float* __restrict__ KW,
                                                               const float* __restrict__ IW, const float* __restrict__ INV,
                                                               __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC,
                                                               __nv_bfloat16* __restrict__ VC, __nv_bfloat16* __restrict__ IQ,
                                                               __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ, int NKV,
                                                               int NI, int IHD, int HALF) {
    fn_attn_prep_body(P, POS0, QW, KW, IW, INV, Q, KC, VC, IQ, IKC, eps, PW, NQ, NKV, NI, IHD, HALF, nullptr, 0, 0);
}

extern "C" __global__ void __launch_bounds__(256) fn_attn_prep_media(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                                                               const float* __restrict__ QW, const float* __restrict__ KW,
                                                               const float* __restrict__ IW, const float* __restrict__ INV,
                                                               __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC,
                                                               __nv_bfloat16* __restrict__ VC, __nv_bfloat16* __restrict__ IQ,
                                                               __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ, int NKV,
                                                               int NI, int IHD, int HALF, const int* rope3, int n, int delta) {
    fn_attn_prep_body(P, POS0, QW, KW, IW, INV, Q, KC, VC, IQ, IKC, eps, PW, NQ, NKV, NI, IHD, HALF, rope3, n, delta);
}

// glue._attn_gate, program (row, head) of HD threads: bf16(o / (1 + e^-gate)) and its 32-group sums.
extern "C" __global__ void __launch_bounds__(256) fn_attn_gate(const __nv_bfloat16* __restrict__ O, const __nv_bfloat16* __restrict__ P,
                                                               __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, int PW, int NQ) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float o = bf(O[(static_cast<long long>(r) * NQ + h) * HD + d]);
    const float g = bf(P[static_cast<long long>(r) * PW + h * 2 * HD + HD + d]);
    const float out = rbf(__fdiv_rn(o, __fadd_rn(1.0f, tf_exp(-g))));
    OUT[static_cast<long long>(r) * NQ * HD + h * HD + d] = __float2bfloat16_rn(out);
    const float sum = warp_sum(out);
    if ((d & 31) == 0) XS[static_cast<long long>(r) * (NQ * HD / 32) + h * (HD / 32) + d / 32] = sum;
}

// attention._pool, a block a completed block of RATIO indexer keys: their mean, rounded, normed and roped at its start.
__device__ __forceinline__ void fn_qsa_pool_body(const __nv_bfloat16* __restrict__ IKC, __nv_bfloat16* __restrict__ POOLED,
                                                              const int* __restrict__ POS0, const float* __restrict__ W,
                                                              const float* __restrict__ INV, float eps, int R, int DI, int HALF,
                                                              int RATIO, const int* rope3, int n, int delta) {
    __shared__ float scratch[32];
    __shared__ float xs[256];
    const int p0 = POS0[0];
    const int b = p0 / RATIO + blockIdx.x, d = threadIdx.x;
    if (RATIO * b + RATIO > p0 + R) return;
    float acc = bf(IKC[static_cast<long long>(RATIO) * b * DI + d]);
    for (int k = 1; k < RATIO; ++k) acc = __fadd_rn(acc, bf(IKC[(static_cast<long long>(RATIO) * b + k) * DI + d]));
    const float x = rbf(__fdiv_rn(acc, static_cast<float>(RATIO)));
    xs[d] = x;
    const float rinv = inv_rms(block_sum(__fmul_rn(x, x), scratch), DI, eps);
    const int p = partner(d, HALF);
    const float xn = rbf(__fmul_rn(__fmul_rn(x, rinv), W[d]));
    const float xpn = rbf(__fmul_rn(__fmul_rn(xs[p], rinv), W[p]));
    POOLED[static_cast<long long>(b) * DI + d] = __float2bfloat16_rn(rope(xn, xpn, d, HALF, media_pos(RATIO * b, d, HALF, rope3, n, delta), INV));
}

extern "C" __global__ void __launch_bounds__(128) fn_qsa_pool(const __nv_bfloat16* __restrict__ IKC, __nv_bfloat16* __restrict__ POOLED,
                                                              const int* __restrict__ POS0, const float* __restrict__ W,
                                                              const float* __restrict__ INV, float eps, int R, int DI, int HALF,
                                                              int RATIO) {
    fn_qsa_pool_body(IKC, POOLED, POS0, W, INV, eps, R, DI, HALF, RATIO, nullptr, 0, 0);
}

extern "C" __global__ void __launch_bounds__(128) fn_qsa_pool_media(const __nv_bfloat16* __restrict__ IKC, __nv_bfloat16* __restrict__ POOLED,
                                                              const int* __restrict__ POS0, const float* __restrict__ W,
                                                              const float* __restrict__ INV, float eps, int R, int DI, int HALF,
                                                              int RATIO, const int* rope3, int n, int delta) {
    fn_qsa_pool_body(IKC, POOLED, POS0, W, INV, eps, R, DI, HALF, RATIO, rope3, n, delta);
}

// attention._scores, program (row, 64 blocks): the heads' sum of relu(q . k) / sqrt(DI), rows past TOP (DI 128).
extern "C" __global__ void __launch_bounds__(256) fn_qsa_scores(const __nv_bfloat16* __restrict__ IQ, const __nv_bfloat16* __restrict__ POOLED,
                                                                const int* __restrict__ POS0, float* __restrict__ SC, int NB, int HI,
                                                                int RATIO, int TOP) {
    constexpr int DI = 128;
    __shared__ float qs[8][DI];
    const int r = blockIdx.x, j = blockIdx.y;
    const int complete = (POS0[0] + r + 1) / RATIO;
    if (complete <= TOP || j * 64 >= complete) return;
    for (int e = threadIdx.x; e < HI * DI; e += blockDim.x) qs[e / DI][e % DI] = bf(IQ[static_cast<long long>(r) * HI * DI + e]);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float scale = __fsqrt_rn(static_cast<float>(DI));
    uint2 raw[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int b = j * 64 + warp * 8 + i;
        raw[i] = b < complete ? *reinterpret_cast<const uint2*>(POOLED + static_cast<long long>(b) * DI + 4 * lane) : make_uint2(0, 0);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int b = j * 64 + warp * 8 + i;
        if (b >= complete) break;
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&raw[i]);
        float total = 0.0f;
        for (int h = 0; h < HI; ++h) {
            float part = 0.0f;
#pragma unroll
            for (int e = 0; e < 4; ++e) part = __fmaf_rn(bf(kb[e]), qs[h][4 * lane + e], part);
            total = __fadd_rn(total, fmaxf(warp_sum(part), 0.0f));
        }
        if (lane == 0) SC[static_cast<long long>(r) * NB + b] = __fdiv_rn(total, scale);
    }
}

// attention._chunks, program (row, KV head, chunk): the chunk's keys TK at a time, online softmax, partial o, m, l.
template <int NB, bool ONE = false, bool FUSE = false>
__device__ __forceinline__ void attn_chunks(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                            const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                            float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                            const int* __restrict__ IDS, const int* __restrict__ NKR, const int* __restrict__ SPR,
                                            int H, int HK, int G, int CH, int NCH, float SCALE, int IDW, int QSA) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    typedef __nv_bfloat16 Row[HD + 8];
    Row* qs = reinterpret_cast<Row*>(smem_raw);                  // [QR] the row's query heads
    Row* tiles_kv = qs + QR;                                     // [NB][2][TK] each buffer's keys, then values
    // ONE: one [TK] buffer, the tile's keys and then, once the scores are taken, its values (fewer bytes, more blocks)
    __nv_bfloat16 (*ps)[TK + 8] = reinterpret_cast<__nv_bfloat16 (*)[TK + 8]>(tiles_kv + (ONE ? TK : NB * 2 * TK));
    __shared__ float red_m[4][QR], red_l[4][QR];
    __shared__ int key_ids[NB][TK];
    const int r = blockIdx.x, hk = blockIdx.y, c = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int n = POS0[0] + r + 1;
    bool sparse = false;
    if (QSA) {
        sparse = SPR[r] != 0;
        if (sparse) n = NKR[r];
    }
    // FUSE: one program a row takes every chunk in order and merges them as fn_attn_merge does (no partials)
    const int c_end = FUSE ? (n + CH - 1) / CH : c + 1;
    if (c * CH >= n) return;
    int start = c * CH;
    for (int e = threadIdx.x; e < QR * HD / 8; e += blockDim.x) {
        const int i = e / (HD / 8), v = e % (HD / 8);
        qmm_frag::cp16z(&qs[i][8 * v], Q + (static_cast<long long>(r) * H + hk * G + min(i, G - 1)) * HD + 8 * v, i < G);
    }
    // a tile's key ids, then its keys and values staged as two cp.async groups
    auto load = [&](int t0, int buf) {
        if (threadIdx.x < TK) {
            const int ki = start + t0 + threadIdx.x;
            key_ids[buf][threadIdx.x] = ki < n ? (sparse ? IDS[static_cast<long long>(r) * IDW + ki] : ki) : -1;
        }
        __syncthreads();
        for (int pass = 0; pass < 2; ++pass) {
            const __nv_bfloat16* src = pass == 0 ? KC : VC;
            Row* dst = tiles_kv + (buf * 2 + pass) * TK;
            for (int e = threadIdx.x; e < TK * HD / 8; e += blockDim.x) {
                const int k = e / (HD / 8), v = e % (HD / 8), key = key_ids[buf][k];
                qmm_frag::cp16z(&dst[k][8 * v], src + (static_cast<long long>(max(key, 0)) * HK + hk) * HD + 8 * v, key >= 0);
            }
            qmm_frag::commit();
        }
    };
    // ONE: the keys (pass 0) or the values (pass 1) of the tile whose ids are in key_ids[0], into the one buffer
    auto load_one = [&](int pass) {
        const __nv_bfloat16* src = pass == 0 ? KC : VC;
        for (int e = threadIdx.x; e < TK * HD / 8; e += blockDim.x) {
            const int k = e / (HD / 8), v = e % (HD / 8), key = key_ids[0][k];
            qmm_frag::cp16z(&tiles_kv[k][8 * v], src + (static_cast<long long>(max(key, 0)) * HK + hk) * HD + 8 * v, key >= 0);
        }
        qmm_frag::commit();
    };
    float mm[2] = {-INFINITY, -INFINITY}, ll[2] = {0.0f, 0.0f}, oo[FUSE ? 8 : 1][4] = {};
    for (int cc = c; cc < c_end; ++cc) {
    start = cc * CH;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    float o[8][4] = {};
    const int tiles = min(n - start, CH);
    if (NB == 2) load(0, 0);
    for (int t0 = 0, it = 0; t0 < tiles; t0 += TK, ++it) {
        const int buf = NB == 2 ? it & 1 : 0;
        const bool ahead = NB == 2 && t0 + TK < tiles;  // the next tile's loads go out before this tile's math
        if constexpr (ONE) {
            if (threadIdx.x < TK) {
                const int ki = start + t0 + threadIdx.x;
                key_ids[0][threadIdx.x] = ki < n ? (sparse ? IDS[static_cast<long long>(r) * IDW + ki] : ki) : -1;
            }
            __syncthreads();
            load_one(0);
        } else {
            if (NB == 1) load(t0, 0);
            if (ahead) load(t0 + TK, buf ^ 1);
        }
        Row* kt = tiles_kv + (ONE ? 0 : buf * 2 * TK);
        Row* vt = ONE ? kt : kt + TK;
        const int* keys = key_ids[buf];
        if constexpr (ONE) qmm_frag::wait<0>();
        else if (ahead) qmm_frag::wait<3>(); else qmm_frag::wait<1>();
        __syncthreads();
        constexpr int JT = TK / 32;  // a warp's 8-key tiles
        float s[JT][4] = {};
#pragma unroll 4
        for (int ks = 0; ks < HD / 16; ks += 2) {
            uint32_t a0[4], a1[4];
            qmm_frag::ldmatrix4(a0, &qs[lane % 16][ks * 16 + (lane / 16) * 8]);
            qmm_frag::ldmatrix4(a1, &qs[lane % 16][(ks + 1) * 16 + (lane / 16) * 8]);
#pragma unroll
            for (int jt = 0; jt < JT; ++jt) {
                uint32_t b[4];  // two k16 steps of the tile's 8 keys
                qmm_frag::ldmatrix4(b, &kt[warp * 8 * JT + jt * 8 + lane % 8][ks * 16 + (lane / 8) * 8]);
                qmm_frag::mma(s[jt], a0, b[0], b[1]);
                qmm_frag::mma(s[jt], a1, b[2], b[3]);
            }
        }
        if constexpr (ONE) {                              // every warp has read the keys: the values take their place
            __syncthreads();
            load_one(1);
        }
        float tile_m[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int jt = 0; jt < JT; ++jt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool valid = keys[warp * 8 * JT + jt * 8 + 2 * t + (e & 1)] >= 0;
                s[jt][e] = valid ? __fmul_rn(s[jt][e], SCALE) : -INFINITY;
                tile_m[e >> 1] = fmaxf(tile_m[e >> 1], s[jt][e]);
            }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            tile_m[h] = fmaxf(tile_m[h], __shfl_xor_sync(0xffffffffu, tile_m[h], 1));
            tile_m[h] = fmaxf(tile_m[h], __shfl_xor_sync(0xffffffffu, tile_m[h], 2));
            if (t == 0) red_m[warp][g + 8 * h] = tile_m[h];
        }
        __syncthreads();
        float next_m[2], alpha[2], rows[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = g + 8 * h;
            const float tm = fmaxf(fmaxf(red_m[0][row], red_m[1][row]), fmaxf(red_m[2][row], red_m[3][row]));
            const bool active = tm != -INFINITY;
            next_m[h] = active ? fmaxf(m[h], tm) : m[h];
            alpha[h] = active ? (m[h] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m[h], next_m[h]))) : 1.0f;
            rows[h] = 0.0f;
        }
#pragma unroll
        for (int jt = 0; jt < JT; ++jt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int h = e >> 1;
                const float p = s[jt][e] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(s[jt][e], next_m[h]));
                rows[h] = __fadd_rn(rows[h], p);
                ps[g + 8 * h][warp * 8 * JT + jt * 8 + 2 * t + (e & 1)] = __float2bfloat16_rn(p);
            }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            rows[h] = __fadd_rn(rows[h], __shfl_xor_sync(0xffffffffu, rows[h], 1));
            rows[h] = __fadd_rn(rows[h], __shfl_xor_sync(0xffffffffu, rows[h], 2));
            if (t == 0) red_l[warp][g + 8 * h] = rows[h];
        }
        if (ONE) qmm_frag::wait<0>(); else if (ahead) qmm_frag::wait<2>(); else qmm_frag::wait<0>();
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = g + 8 * h;
            const float sum = __fadd_rn(__fadd_rn(__fadd_rn(red_l[0][row], red_l[1][row]), red_l[2][row]), red_l[3][row]);
            l[h] = __fmaf_rn(l[h], alpha[h], sum);
            m[h] = next_m[h];
        }
#pragma unroll
        for (int q = 0; q < 8; ++q)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[q][e] = __fmul_rn(o[q][e], alpha[e >> 1]);
        for (int ks = 0; ks < TK / 16; ++ks) {
            uint32_t a[4];
            qmm_frag::ldmatrix4(a, &ps[lane % 16][ks * 16 + (lane / 16) * 8]);
#pragma unroll
            for (int q = 0; q < 8; q += 2) {
                uint32_t b[4];  // two 8-column tiles of the values, transposed into the B layout
                ldmatrix4t(b, &vt[ks * 16 + lane % 16][warp * 64 + q * 8 + (lane / 16) * 8]);
                qmm_frag::mma(o[q], a, b[0], b[1]);
                qmm_frag::mma(o[q + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    if constexpr (FUSE) {                                  // fn_attn_merge's step for this chunk, element by element
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const bool active = l[h] > 0.0f;
            const float next = active ? fmaxf(mm[h], m[h]) : mm[h];
            const float a = active ? (mm[h] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(mm[h], next))) : 1.0f;
            const float bb = active ? tf_exp(__fsub_rn(m[h], next)) : 0.0f;
#pragma unroll
            for (int q = 0; q < 8; ++q)
#pragma unroll
                for (int e = 0; e < 2; ++e) oo[FUSE ? q : 0][2 * h + e] = __fmaf_rn(o[q][2 * h + e], bb, __fmul_rn(oo[FUSE ? q : 0][2 * h + e], a));
            ll[h] = __fmaf_rn(l[h], bb, __fmul_rn(ll[h], a));
            mm[h] = next;
        }
        continue;
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int i = g + 8 * h;
        if (i >= G) continue;
        const long long base = (static_cast<long long>(r) * NCH + cc) * H + hk * G + i;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            float2 v = make_float2(o[q][2 * h], o[q][2 * h + 1]);
            *reinterpret_cast<float2*>(PO + base * HD + warp * 64 + q * 8 + 2 * t) = v;
        }
        if (warp == 0 && t == 0) {
            PM[base] = m[h];
            PL[base] = l[h];
        }
    }
    }
    if constexpr (FUSE) {                                  // o / l, rounded once, as fn_attn_merge writes OUT
        __nv_bfloat16* OUT = reinterpret_cast<__nv_bfloat16*>(PO);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int i = g + 8 * h;
            if (i >= G) continue;
            const long long row = (static_cast<long long>(r) * H + hk * G + i) * HD;
#pragma unroll
            for (int q = 0; q < 8; ++q)
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    OUT[row + warp * 64 + q * 8 + 2 * t + e] = __float2bfloat16_rn(__fdiv_rn(oo[q][2 * h + e], ll[h]));
        }
    }
}

// attn_chunks one tile at a time (prompt chunks: two blocks a multiprocessor).
extern "C" __global__ void __launch_bounds__(128) fn_attn_chunks(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                                                 const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                                                 float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                 const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                 const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                 float SCALE, int IDW, int QSA) {
    attn_chunks<1>(Q, KC, VC, POS0, PO, PM, PL, IDS, NKR, SPR, H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

// attn_chunks on one buffer, a row's chunks merged in place (prompt rows): OUT bf16 in PO's place, no partials.
extern "C" __global__ void __launch_bounds__(128) fn_attn_fused(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                                                const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                                                float* __restrict__ OUT, const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                const int* __restrict__ SPR, int H, int HK, int G, int CH,
                                                                float SCALE, int IDW, int QSA) {
    attn_chunks<1, true, true>(Q, KC, VC, POS0, OUT, nullptr, nullptr, IDS, NKR, SPR, H, HK, G, CH, 1, SCALE, IDW, QSA);
}

// attn_chunks with one buffer, keys then values (prompt chunks: three blocks a multiprocessor), the same sums.
extern "C" __global__ void __launch_bounds__(128) fn_attn_chunks_one(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                                                     const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                                                     float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                     const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                     const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                     float SCALE, int IDW, int QSA) {
    attn_chunks<1, true>(Q, KC, VC, POS0, PO, PM, PL, IDS, NKR, SPR, H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

// attn_chunks with the next tile's loads in flight (a decode round's few rows): the same sums.
extern "C" __global__ void __launch_bounds__(128) fn_attn_chunks_ahead(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                                                       const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                                                       float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                       const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                       const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                       float SCALE, int IDW, int QSA) {
    attn_chunks<2>(Q, KC, VC, POS0, PO, PM, PL, IDS, NKR, SPR, H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

// attention._merge, program (row, head) of HD threads: the head's chunk partials merged in chunk order, o / l.
extern "C" __global__ void __launch_bounds__(256) fn_attn_merge(const float* __restrict__ PO, const float* __restrict__ PM,
                                                                const float* __restrict__ PL, const int* __restrict__ POS0,
                                                                __nv_bfloat16* __restrict__ OUT, const int* __restrict__ NKR,
                                                                const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                int QSA) {
    const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    int n = POS0[0] + r + 1;
    if (QSA && SPR[r] != 0) n = NKR[r];
    const int chunks = (n + CH - 1) / CH;
    {
        float m = -INFINITY, l = 0.0f, o = 0.0f;
        for (int c = 0; c < chunks; ++c) {
            const long long base = (static_cast<long long>(r) * NCH + c) * H + head;
            const float cm = PM[base], cl = PL[base], co = PO[base * HD + d];
            const bool active = cl > 0.0f;
            const float next = active ? fmaxf(m, cm) : m;
            const float a = active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f;
            const float b = active ? tf_exp(__fsub_rn(cm, next)) : 0.0f;
            o = __fmaf_rn(co, b, __fmul_rn(o, a));
            l = __fmaf_rn(cl, b, __fmul_rn(l, a));
            m = next;
        }
        OUT[(static_cast<long long>(r) * H + head) * HD + d] = __float2bfloat16_rn(__fdiv_rn(o, l));
    }
}

#include "flashnext_qsa_tile.cuh"

// The indexer scores 8 rows a block, every head, the sums in fn_qsa_scores' order (its bits).
template __global__ void qsa_tile::scores<8, 4>(const __nv_bfloat16*, const __nv_bfloat16*, const int*, float*, int, int, int, int);

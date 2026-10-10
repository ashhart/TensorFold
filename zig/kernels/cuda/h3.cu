// MiniMax H3 / FastH3's transformer blocks in int8 on CUDA tensor cores: the row kernels and the kernels' entry points.
#include "h3_products.cuh"

namespace tf_h3 {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ int8_t q8(float x) { return int8_t(max(-127, min(127, __float2int_rn(x)))); }

// A value every thread of the block agrees on: the sum or the largest of what each holds.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < (blockDim.x >> 5); ++w) t += scratch[w];
    __syncthreads();
    return t;
}
__device__ __forceinline__ float block_max(float v, float* scratch) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < (blockDim.x >> 5); ++w) t = fmaxf(t, scratch[w]);
    __syncthreads();
    return t;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

} // namespace tf_h3

using namespace tf_h3;

extern "C" {

__global__ void __launch_bounds__(128) h3_gemm(const int8_t* X, const float* XS, const int8_t* W, const float* WS,
                                               __nv_bfloat16* Y, int M, int N, int K, int group) {
    gemm_i8<plain, 128, 128, 2, 2, 64>(X, XS, W, WS, Y, M, N, K, group);
}
__global__ void __launch_bounds__(256) h3_gemm_n256(const int8_t* X, const float* XS, const int8_t* W, const float* WS,
                                                    __nv_bfloat16* Y, int M, int N, int K, int group) {
    gemm_i8<plain, 128, 256, 2, 4>(X, XS, W, WS, Y, M, N, K, group);
}
__global__ void __launch_bounds__(128) h3_gemm_swiglu(const int8_t* X, const float* XS, const int8_t* W, const float* WS,
                                                      __nv_bfloat16* Y, int M, int N, int K, int group) {
    gemm_i8<swiglu, 128, 128, 2, 2, 64>(X, XS, W, WS, Y, M, N, K, group);
}
__global__ void __launch_bounds__(256) h3_gemm_wide(const int8_t* X, const float* XS, const int8_t* W, const float* WS,
                                                    __nv_bfloat16* Y, int M, int N, int K, int group) {
    gemm_i8<wide, 256, 128, 4, 2>(X, XS, W, WS, Y, M, N, K, group);
}

// W (N, K) bf16 to int8 rows with a scale per output; `half` interleaves a SwiGLU's [gate; value]. Grid [N].
__global__ void h3_quant_weight(const __nv_bfloat16* W, int8_t* W8, float* WS, int K, int row0, int half) {
    __shared__ float scratch[8];
    const int n = blockIdx.x;
    const __nv_bfloat16* w = W + static_cast<size_t>(n) * K;
    float top = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) top = fmaxf(top, fabsf(f(w[k])));
    top = fmaxf(block_max(top, scratch), 1e-12f);
    const int to = half > 0 ? (n >= half ? 2 * (n - half) : 2 * n + 1) : row0 + n;
    if (threadIdx.x == 0) WS[to] = top / 127.0f;
    const float inverse = 127.0f / top;
    int8_t* o = W8 + static_cast<size_t>(to) * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) o[k] = q8(f(w[k]) * inverse);
}

// A block's modulated RMSNorm to int8 with a scale per row, after an optional gated add of Y into X. Grid [MP].
__global__ void h3_norm_q8(__nv_bfloat16* X, const __nv_bfloat16* Y, const __nv_bfloat16* NW, const float* GT,
                           const float* TAB, const int* LINE, int8_t* Q, float* XS, int M, int C, int gate,
                           int scale, int shift, float eps) {
    __shared__ float scratch[8];
    const int row = blockIdx.x;
    int8_t* q = Q + static_cast<size_t>(row) * C;
    if (row >= M) {
        for (int c = threadIdx.x; c < C; c += blockDim.x) q[c] = 0;
        if (threadIdx.x == 0) XS[row] = 0.0f;
        return;
    }
    const int line = LINE[row];
    __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    float sq = 0.0f;
    if (gate >= 0) {
        const float* g = GT + (static_cast<size_t>(line) * 6 + gate) * C;
        const __nv_bfloat16* y = Y + static_cast<size_t>(row) * C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float v = bf(f(x[c]) + bf(bf(g[c]) * f(y[c])));
            x[c] = __float2bfloat16(v);
            sq += v * v;
        }
    } else {
        for (int c = threadIdx.x; c < C; c += blockDim.x) sq += f(x[c]) * f(x[c]);
    }
    const float inv = rsqrtf(block_sum(sq, scratch) / float(C) + eps);
    const float* s = TAB + (static_cast<size_t>(line) * 6 + scale) * C;
    const float* h = TAB + (static_cast<size_t>(line) * 6 + shift) * C;
    float top = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        top = fmaxf(top, fabsf(bf(bf(bf(f(x[c]) * inv * f(NW[c])) * bf(1.0f + s[c])) + bf(h[c]))));
    top = fmaxf(block_max(top, scratch), 1e-12f);
    if (threadIdx.x == 0) XS[row] = top / 127.0f;
    const float inverse = 127.0f / top;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        q[c] = q8(bf(bf(bf(f(x[c]) * inv * f(NW[c])) * bf(1.0f + s[c])) + bf(h[c])) * inverse);
}

// X += GT[line, part] * Y: the last block's MLP branch. Grid [M], 128 threads.
__global__ void h3_gate_add(__nv_bfloat16* X, const __nv_bfloat16* Y, const float* GT, const int* LINE, int C, int part) {
    const int row = blockIdx.x;
    const float* g = GT + (static_cast<size_t>(LINE[row]) * 6 + part) * C;
    __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    const __nv_bfloat16* y = Y + static_cast<size_t>(row) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) x[c] = __float2bfloat16(f(x[c]) + bf(bf(g[c]) * f(y[c])));
}

// q, k, v to int8 in tile order: head RMSNorm and rotary on q and k; scales per row (q) and per tile (k, v).
__global__ void __launch_bounds__(256) h3_heads(
        const __nv_bfloat16* QKV, const __nv_bfloat16* NQ, const __nv_bfloat16* NK, const float* COS, const float* SIN,
        const int* ROWOF, int8_t* TQ, float* TQS, int8_t* TK, int8_t* TVT, float* KS, float* VS, int H, int stride,
        int ROT, int slots, int tiles, float eps) {
    __shared__ float tops[16];
    const int tile = blockIdx.x, s = threadIdx.x >> 2, c0 = (threadIdx.x & 3) * 32, slot = tile * 64 + s;
    const int row = ROWOF[slot];
    const bool live = row >= 0;
    const float* cs = COS + static_cast<size_t>(live ? row : 0) * ROT;
    const float* sn = SIN + static_cast<size_t>(live ? row : 0) * ROT;
    for (int head = 0; head < H; ++head) {
        const __nv_bfloat16* qr = QKV + static_cast<size_t>(live ? row : 0) * stride + head * 128;
        const __nv_bfloat16* kr = qr + H * 128;
        const __nv_bfloat16* vr = kr + H * 128;
        float q[32], k[32], v[32], sq = 0.0f, sk = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            q[j] = live ? f(qr[c0 + j]) : 0.0f;
            k[j] = live ? f(kr[c0 + j]) : 0.0f;
            v[j] = live ? f(vr[c0 + j]) : 0.0f;
            sq += q[j] * q[j];
            sk += k[j] * k[j];
        }
        // a row's four quarters sit in neighbouring lanes
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        sk += __shfl_xor_sync(0xffffffffu, sk, 1);
        sk += __shfl_xor_sync(0xffffffffu, sk, 2);
        const float qi = rsqrtf(sq / 128.0f + eps), ki = rsqrtf(sk / 128.0f + eps);
        float qt = 0.0f, kt = 0.0f, vt = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int c = c0 + j;
            float a = bf(q[j] * qi * f(NQ[c])), b = bf(k[j] * ki * f(NK[c]));
            if (c < 2 * ROT && live) {
                const bool low = c < ROT;
                const int p = low ? c + ROT : c - ROT, t = low ? c : c - ROT;
                const float qp = bf(f(qr[p]) * qi * f(NQ[p])), kp = bf(f(kr[p]) * ki * f(NK[p]));
                a = low ? a * cs[t] - qp * sn[t] : qp * sn[t] + a * cs[t];
                b = low ? b * cs[t] - kp * sn[t] : kp * sn[t] + b * cs[t];
            }
            q[j] = a;
            k[j] = b;
            qt = fmaxf(qt, fabsf(a));
            kt = fmaxf(kt, fabsf(b));
            vt = fmaxf(vt, fabsf(v[j]));
        }
        qt = fmaxf(qt, __shfl_xor_sync(0xffffffffu, qt, 1));
        qt = fmaxf(fmaxf(qt, __shfl_xor_sync(0xffffffffu, qt, 2)), 1e-12f);
        // the tile's largest k and v: each warp's, then the eight warps'
        kt = warp_max(kt);
        vt = warp_max(vt);
        if ((threadIdx.x & 31) == 0) {
            tops[threadIdx.x >> 5] = kt;
            tops[8 + (threadIdx.x >> 5)] = vt;
        }
        __syncthreads();
        kt = vt = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            kt = fmaxf(kt, tops[w]);
            vt = fmaxf(vt, tops[8 + w]);
        }
        __syncthreads();
        kt = fmaxf(kt, 1e-12f);
        vt = fmaxf(vt, 1e-12f);
        const size_t ht = static_cast<size_t>(head) * tiles + tile, at = static_cast<size_t>(head) * slots + slot;
        if (threadIdx.x == 0) {
            KS[ht] = kt / 127.0f;
            VS[ht] = vt / 127.0f;
        }
        if (!live) continue;
        if ((threadIdx.x & 3) == 0) TQS[at] = qt / 127.0f;
        const float qv = 127.0f / qt, kv = 127.0f / kt, vv = 127.0f / vt;
        int8_t* vo = TVT + (ht * 128 + c0) * 64 + s;
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            uint32_t pq = 0, pk = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                pq |= uint32_t(uint8_t(q8(q[j + e] * qv))) << (8 * e);
                pk |= uint32_t(uint8_t(q8(k[j + e] * kv))) << (8 * e);
                vo[(j + e) * 64] = q8(v[j + e] * vv);
            }
            *reinterpret_cast<uint32_t*>(TQ + at * 128 + c0 + j) = pq;
            *reinterpret_cast<uint32_t*>(TK + at * 128 + c0 + j) = pk;
        }
    }
}

// Each tile's mean q, k and v over its real rows. QP, KP, VP: (H, tiles, 128) float. Grid [tiles, H], 32 threads.
__global__ void h3_pool(const int8_t* TQ, const float* TQS, const int8_t* TK, const float* KS, const int8_t* TVT,
                        const float* VS, const int* SIZES, float* QP, float* KP, float* VP, int slots, int tiles) {
    const int tile = blockIdx.x, head = blockIdx.y, size = SIZES[tile], c0 = 4 * threadIdx.x;
    const size_t first = static_cast<size_t>(head) * slots + tile * 64, ht = static_cast<size_t>(head) * tiles + tile;
    float q[4] = {0, 0, 0, 0};
    int k[4] = {0, 0, 0, 0}, v[4] = {0, 0, 0, 0};
    for (int s = 0; s < size; ++s) {
        const int8_t* q8p = TQ + (first + s) * 128 + c0;
        const int8_t* k8p = TK + (first + s) * 128 + c0;
        const float qs = TQS[first + s];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] += float(q8p[j]) * qs;
            k[j] += k8p[j];
            v[j] += TVT[(ht * 128 + c0 + j) * 64 + s];
        }
    }
    const float inv = 1.0f / float(size), ks = KS[ht] * inv, vs = VS[ht] * inv;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        QP[ht * 128 + c0 + j] = q[j] * inv;
        KP[ht * 128 + c0 + j] = float(k[j]) * ks;
        VP[ht * 128 + c0 + j] = float(v[j]) * vs;
    }
}

// The tiles' means as int8 rows for h3_attention, 64 tiles a group. Grid [groups, H], 64 threads.
__global__ void h3_pool_quant(const float* QP, const float* KP, const float* VP, int8_t* PQ, float* PQS, int8_t* PK,
                              float* PKS, int8_t* PVT, float* PVS, int tiles, int groups) {
    __shared__ float scratch[8];
    const int group = blockIdx.x, head = blockIdx.y, tile = group * 64 + threadIdx.x;
    const bool real = tile < tiles;
    const size_t from = (static_cast<size_t>(head) * tiles + tile) * 128;
    float qt = 0.0f, kt = 0.0f, vt = 0.0f;
    if (real) {
        for (int c = 0; c < 128; ++c) {
            qt = fmaxf(qt, fabsf(QP[from + c]));
            kt = fmaxf(kt, fabsf(KP[from + c]));
            vt = fmaxf(vt, fabsf(VP[from + c]));
        }
    }
    qt = fmaxf(qt, 1e-12f);
    kt = fmaxf(block_max(kt, scratch), 1e-12f);
    vt = fmaxf(block_max(vt, scratch), 1e-12f);
    const size_t hg = static_cast<size_t>(head) * groups + group, to = (static_cast<size_t>(head) * groups * 64 + tile) * 128;
    if (threadIdx.x == 0) {
        PKS[hg] = kt / 127.0f;
        PVS[hg] = vt / 127.0f;
    }
    PQS[static_cast<size_t>(head) * groups * 64 + tile] = real ? qt / 127.0f : 0.0f;
    const float qv = 127.0f / qt, kv = 127.0f / kt, vv = 127.0f / vt;
    for (int c = 0; c < 128; ++c) {
        PQ[to + c] = real ? q8(QP[from + c] * qv) : 0;
        PK[to + c] = real ? q8(KP[from + c] * kv) : 0;
        PVT[(hg * 128 + c) * 64 + threadIdx.x] = real ? q8(VP[from + c] * vv) : 0;
    }
}

// A video tile's key tiles: the prefix, then its KEEP best video tiles by score in tile order. Grid [video tiles, H].
__global__ void h3_topk(const float* S, int* IDX, int tiles, int span, int P0, int KEEP) {
    const int NV = tiles - P0, tq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const float* s = S + (static_cast<size_t>(head) * span + P0 + tq) * span + P0;
    int* out = IDX + (static_cast<size_t>(head) * NV + tq) * (P0 + KEEP);
    for (int i = lane; i < P0; i += 32) out[i] = i;
    auto key = [&](int j) {
        const uint32_t b = __float_as_uint(s[j]);
        return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    };
    uint32_t thr = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t cand = thr | (1u << bit);
        uint32_t mine = 0;
        for (int j = lane; j < NV; j += 32) mine += key(j) >= cand;
        if (__reduce_add_sync(0xffffffffu, mine) >= uint32_t(KEEP)) thr = cand;
    }
    // a kept tile's place is the number kept before it: lanes count in order, each after the lanes before it
    uint32_t above = 0;
    for (int j = lane; j < NV; j += 32) above += key(j) > thr;
    const int ties = KEEP - int(__reduce_add_sync(0xffffffffu, above));
    int kept = 0, tied = 0;
    for (int j0 = 0; j0 < NV; j0 += 32) {
        const int j = j0 + lane;
        const uint32_t k = j < NV ? key(j) : 0u;
        const uint32_t eq = __ballot_sync(0xffffffffu, j < NV && k == thr);
        const uint32_t gt = __ballot_sync(0xffffffffu, j < NV && k > thr);
        // of this round's equal scores, the first `ties - tied` are kept
        const int my_tie = tied + __popc(eq & ((1u << lane) - 1));
        const uint32_t take = __ballot_sync(0xffffffffu, (gt >> lane & 1) || ((eq >> lane & 1) && my_tie < ties));
        if (take >> lane & 1) out[P0 + kept + __popc(take & ((1u << lane) - 1))] = P0 + j;
        kept += __popc(take);
        tied += __popc(eq);
    }
}

// The pooled branch gated into the attention output, then the row to int8 with a scale per row. Grid [MP].
__global__ void h3_mix_quant(const __nv_bfloat16* Y, const __nv_bfloat16* G, const __nv_bfloat16* C, const int* SLOT,
                             int8_t* Q, float* XS, int R, int W, int stride) {
    __shared__ float scratch[8];
    const int row = blockIdx.x;
    int8_t* q = Q + static_cast<size_t>(row) * W;
    if (row >= R) {
        for (int c = threadIdx.x; c < W; c += blockDim.x) q[c] = 0;
        if (threadIdx.x == 0) XS[row] = 0.0f;
        return;
    }
    const __nv_bfloat16* y = Y + static_cast<size_t>(row) * W;
    const __nv_bfloat16* g = G + static_cast<size_t>(row) * stride;
    const __nv_bfloat16* coarse = C ? C + static_cast<size_t>(SLOT[row] >> 6) * W : nullptr;
    auto mixed = [&](int c) {
        if (!coarse) return f(y[c]);
        return bf(f(y[c]) + bf(f(coarse[c]) * f(g[c])));
    };
    float top = 0.0f;
    for (int c = threadIdx.x; c < W; c += blockDim.x) top = fmaxf(top, fabsf(mixed(c)));
    top = fmaxf(block_max(top, scratch), 1e-12f);
    if (threadIdx.x == 0) XS[row] = top / 127.0f;
    const float inverse = 127.0f / top;
    for (int c = threadIdx.x; c < W; c += blockDim.x) q[c] = q8(mixed(c) * inverse);
}

// The MLP's wide rows to int8 with a scale per (row, 1024 columns). Grid [MP], 128 threads.
__global__ void h3_quant_wide(const __nv_bfloat16* X, int8_t* Q, float* XS, int R, int W) {
    const int row = blockIdx.x, lane = threadIdx.x & 31, groups = W / wide_group;
    for (int g = threadIdx.x >> 5; g < groups; g += blockDim.x >> 5) {
        int8_t* q = Q + static_cast<size_t>(row) * W + g * wide_group;
        if (row >= R) {
            for (int c = lane * 4; c < wide_group; c += 128) *reinterpret_cast<uint32_t*>(q + c) = 0;
            if (lane == 0) XS[static_cast<size_t>(row) * groups + g] = 0.0f;
            continue;
        }
        const __nv_bfloat16* x = X + static_cast<size_t>(row) * W + g * wide_group;
        float top = 0.0f;
        for (int c = lane; c < wide_group; c += 32) top = fmaxf(top, fabsf(f(x[c])));
        top = fmaxf(warp_max(top), 1e-12f);
        if (lane == 0) XS[static_cast<size_t>(row) * groups + g] = top / 127.0f;
        const float inverse = 127.0f / top;
        for (int c = lane * 4; c < wide_group; c += 128) {
            uint32_t p = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) p |= uint32_t(uint8_t(q8(f(x[c + j]) * inverse))) << (8 * j);
            *reinterpret_cast<uint32_t*>(q + c) = p;
        }
    }
}

__global__ void __launch_bounds__(128) h3_attention(
        const int8_t* Q, const float* QS, const int8_t* K, const float* KS, const int8_t* VT, const float* VS,
        const int* LIST, const int* SIZES, const int* ROWOF, __nv_bfloat16* Y, int slots, int tiles, int heads,
        int queries, int keys, int first_query, int per_query, float scale, float* SOUT) {
    attention_body(Q, QS, K, KS, VT, VS, LIST, SIZES, ROWOF, Y, slots, tiles, heads, queries, keys, first_query, per_query, scale, SOUT);
}

} // extern "C"

// MiniMax H3 / FastH3's transformer blocks on CUDA in the checkpoint's bf16: the row, routing and attention kernels.
#include "h3_attention.cuh"

namespace tf_h3 {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float f(__nv_bfloat16 x) { return __bfloat162float(x); }

// A value every thread of the block agrees on: the sum of what each holds.
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

// The modulated norm of one value, rounded to bf16 where torch's bf16 ops round: norm, 1 + scale, product, shift.
__device__ __forceinline__ float modulated(float x, float inv, float w, float scale, float shift) {
    return bf(bf(bf(x * inv * w) * bf(1.0f + bf(scale))) + bf(shift));
}

} // namespace tf_h3

using namespace tf_h3;

extern "C" {

// H spans of `words` 4-byte words, SRC + h * sp to DST + h * dp: ranks' exchange. Grid [words / 16384, H], 256 threads.
__global__ void h3_spans(uint32_t* DST, const uint32_t* SRC, long long dp, long long sp, int words) {
    uint32_t* d = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(DST) + blockIdx.y * dp);
    const uint32_t* s = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(SRC) + blockIdx.y * sp);
    const int first = blockIdx.x * 16384, last = min(first + 16384, words);
    for (int i = first + threadIdx.x; i < last; i += blockDim.x) d[i] = s[i];
}

// A block's modulated RMSNorm to bf16 rows O, after an optional gated add of Y into X. Grid [M], 256 threads.
__global__ void __launch_bounds__(256) h3_norm(__nv_bfloat16* X, const __nv_bfloat16* Y, const __nv_bfloat16* NW,
                                               const float* GT, const float* TAB, const int* LINE, __nv_bfloat16* O,
                                               int C, int gate, int scale, int shift, float eps) {
    __shared__ float scratch[8];
    const int row = blockIdx.x, line = LINE[row];
    __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    float sq = 0.0f;
    if (gate >= 0) {
        const float* g = GT + (static_cast<size_t>(line) * 6 + gate) * C;
        const __nv_bfloat16* y = Y + static_cast<size_t>(row) * C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float v = bf(f(x[c]) + f(y[c]) * bf(g[c]));
            x[c] = __float2bfloat16(v);
            sq += v * v;
        }
    } else {
        for (int c = threadIdx.x; c < C; c += blockDim.x) sq += f(x[c]) * f(x[c]);
    }
    const float inv = rsqrtf(block_sum(sq, scratch) / float(C) + eps);
    const float* s = TAB + (static_cast<size_t>(line) * 6 + scale) * C;
    const float* h = TAB + (static_cast<size_t>(line) * 6 + shift) * C;
    __nv_bfloat16* o = O + static_cast<size_t>(row) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) o[c] = __float2bfloat16(modulated(f(x[c]), inv, f(NW[c]), s[c], h[c]));
}

// X += GT[line, part] * Y in one rounding, as torch's addcmul: the last block's MLP branch. Grid [M], 256 threads.
__global__ void __launch_bounds__(256) h3_gate_add(__nv_bfloat16* X, const __nv_bfloat16* Y, const float* GT,
                                                   const int* LINE, int C, int part) {
    const int row = blockIdx.x;
    const float* g = GT + (static_cast<size_t>(LINE[row]) * 6 + part) * C;
    __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    const __nv_bfloat16* y = Y + static_cast<size_t>(row) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) x[c] = __float2bfloat16(f(x[c]) + f(y[c]) * bf(g[c]));
}

// q, k, v to bf16 in tile order: head RMSNorm, then split-half rotary rounded as the reference's bf16 tables round it.
__global__ void __launch_bounds__(256) h3_heads(
        const __nv_bfloat16* QKV, const __nv_bfloat16* NQ, const __nv_bfloat16* NK, const float* COS, const float* SIN,
        const int* ROWOF, __nv_bfloat16* TQ, __nv_bfloat16* TK, __nv_bfloat16* TV, int H, int stride, int ROT, int slots,
        float eps) {
    const int tile = blockIdx.x, s = threadIdx.x >> 2, c0 = (threadIdx.x & 3) * 32, slot = tile * 64 + s;
    const int row = ROWOF[slot];
    const bool live = row >= 0;
    const float* cs = COS + static_cast<size_t>(live ? row : 0) * ROT;
    const float* sn = SIN + static_cast<size_t>(live ? row : 0) * ROT;
    for (int head = 0; head < H; ++head) {
        const __nv_bfloat16* qr = QKV + static_cast<size_t>(live ? row : 0) * stride + head * 128;
        const __nv_bfloat16* kr = qr + H * 128;
        const __nv_bfloat16* vr = kr + H * 128;
        float q[32], k[32], sq = 0.0f, sk = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            q[j] = live ? f(qr[c0 + j]) : 0.0f;
            k[j] = live ? f(kr[c0 + j]) : 0.0f;
            sq += q[j] * q[j];
            sk += k[j] * k[j];
        }
        // a row's four quarters sit in neighbouring lanes
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        sk += __shfl_xor_sync(0xffffffffu, sk, 1);
        sk += __shfl_xor_sync(0xffffffffu, sk, 2);
        if (!live) continue;
        const float qi = rsqrtf(sq / 128.0f + eps), ki = rsqrtf(sk / 128.0f + eps);
        const size_t at = (static_cast<size_t>(head) * slots + slot) * 128 + c0;
#pragma unroll
        for (int j = 0; j < 32; j += 2) {
            float a[2], b[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int c = c0 + j + e;
                a[e] = bf(q[j + e] * qi * f(NQ[c]));
                b[e] = bf(k[j + e] * ki * f(NK[c]));
                if (c < 2 * ROT) {
                    const bool low = c < ROT;
                    const int p = low ? c + ROT : c - ROT, t = low ? c : c - ROT;
                    const float sign = low ? -1.0f : 1.0f, co = bf(cs[t]), si = bf(sn[t]);
                    const float qp = bf(f(qr[p]) * qi * f(NQ[p])), kp = bf(f(kr[p]) * ki * f(NK[p]));
                    a[e] = bf(a[e] * co) + bf(sign * qp * si);
                    b[e] = bf(b[e] * co) + bf(sign * kp * si);
                }
            }
            *reinterpret_cast<uint32_t*>(TQ + at + j) = pack2(a[0], a[1]);
            *reinterpret_cast<uint32_t*>(TK + at + j) = pack2(b[0], b[1]);
            *reinterpret_cast<uint32_t*>(TV + at + j) = *reinterpret_cast<const uint32_t*>(vr + c0 + j);
        }
    }
}

// Each tile's mean q, k and v over its real rows, in fp32. QP, KP, VP: (H, tiles, 128). Grid [tiles, H], 32 threads.
__global__ void h3_pool(const __nv_bfloat16* TQ, const __nv_bfloat16* TK, const __nv_bfloat16* TV, const int* SIZES,
                        float* QP, float* KP, float* VP, int slots, int tiles) {
    const int tile = blockIdx.x, head = blockIdx.y, size = SIZES[tile], c0 = 4 * threadIdx.x;
    const size_t first = static_cast<size_t>(head) * slots + tile * 64, ht = static_cast<size_t>(head) * tiles + tile;
    float q[4] = {0, 0, 0, 0}, k[4] = {0, 0, 0, 0}, v[4] = {0, 0, 0, 0};
    for (int s = 0; s < size; ++s) {
        const size_t at = (first + s) * 128 + c0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] += f(TQ[at + j]);
            k[j] += f(TK[at + j]);
            v[j] += f(TV[at + j]);
        }
    }
    const float n = float(size);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        QP[ht * 128 + c0 + j] = q[j] / n;
        KP[ht * 128 + c0 + j] = k[j] / n;
        VP[ht * 128 + c0 + j] = v[j] / n;
    }
}

// Tile against tile: S[h, tq, tk] = QP[h, tq] . KP[h, tk] / sqrt(128), a warp a pair. Grid [tiles / 8, query tiles, H].
__global__ void __launch_bounds__(256) h3_tile_scores(const float* QP, const float* KP, float* S, int tiles) {
    const int tk = blockIdx.x * 8 + (threadIdx.x >> 5), tq = blockIdx.y, head = blockIdx.z, c0 = 4 * (threadIdx.x & 31);
    if (tk >= tiles) return;
    const float* q = QP + (static_cast<size_t>(head) * tiles + tq) * 128 + c0;
    const float* k = KP + (static_cast<size_t>(head) * tiles + tk) * 128 + c0;
    const float sum = warp_sum(q[0] * k[0] + q[1] * k[1] + q[2] * k[2] + q[3] * k[3]);
    if ((threadIdx.x & 31) == 0) S[(static_cast<size_t>(head) * tiles + tq) * tiles + tk] = sum * 0.08838834764831845f;
}

// A video tile's keys: the prefix, then its KEEP best tiles (NaN as -inf, ties to the lower tile) in tile order.
__global__ void h3_topk(const float* S, int* IDX, int tiles, int P0, int KEEP) {
    const int NV = tiles - P0, tq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const float* s = S + (static_cast<size_t>(head) * tiles + P0 + tq) * tiles + P0;
    int* out = IDX + (static_cast<size_t>(head) * NV + tq) * (P0 + KEEP);
    for (int i = lane; i < P0; i += 32) out[i] = i;
    auto key = [&](int j) {
        const float v = s[j];
        const uint32_t b = __float_as_uint(isnan(v) ? -INFINITY : v);
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

// The pooled branch: C[h, tq] = softmax over every tile of S[h, tq] times VP[h], in fp32. Grid [query tiles, H], 32.
__global__ void h3_coarse(const float* S, const float* VP, float* C, int tiles) {
    const int tq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x, c0 = 4 * lane;
    const float* s = S + (static_cast<size_t>(head) * tiles + tq) * tiles;
    float top = -3.0e38f;
    for (int i = lane; i < tiles; i += 32) top = fmaxf(top, s[i]);
    top = warp_max(top);
    float mass = 0.0f;
    for (int i = lane; i < tiles; i += 32) mass += expf(s[i] - top);
    const float inv = 1.0f / warp_sum(mass);
    float acc[4] = {0, 0, 0, 0};
    for (int i = 0; i < tiles; ++i) {
        const float w = expf(s[i] - top);
        const float* v = VP + (static_cast<size_t>(head) * tiles + i) * 128 + c0;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] += w * v[j];
    }
    float* o = C + (static_cast<size_t>(head) * tiles + tq) * 128 + c0;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = acc[j] * inv;
}

// The pooled branch gated into the attention rows in place: Y[r, c] += bf16(C[c / 128, tile of r] G[r, c]). Grid [M].
__global__ void __launch_bounds__(256) h3_gate_mix(__nv_bfloat16* Y, const __nv_bfloat16* G, const float* C,
                                                   const int* SLOT, int W, int stride, int tiles) {
    const int row = blockIdx.x, tile = SLOT[row] >> 6;
    __nv_bfloat16* y = Y + static_cast<size_t>(row) * W;
    const __nv_bfloat16* g = G + static_cast<size_t>(row) * stride;
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        const float cc = C[(static_cast<size_t>(c >> 7) * tiles + tile) * 128 + (c & 127)];
        y[c] = __float2bfloat16(f(y[c]) + bf(cc * f(g[c])));
    }
}

// silu(gate) * value as torch rounds it in bf16, from the first projection's rows [gate | value]. Grid-stride.
__global__ void __launch_bounds__(256) h3_swiglu(const __nv_bfloat16* F, __nv_bfloat16* A, int M, int N) {
    const size_t total = static_cast<size_t>(M) * N;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t r = i / N, n = i % N;
        const float g = f(F[r * 2 * N + n]), v = f(F[r * 2 * N + N + n]);
        A[i] = __float2bfloat16(bf(g / (1.0f + expf(-g))) * v);
    }
}

__global__ void __launch_bounds__(128) h3_attention(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, const int* LIST, const int* SIZES,
        const int* ROWOF, __nv_bfloat16* Y, int slots, int heads, int queries, int keys, int first_query, int per_query,
        float scale) {
    attention_body(Q, K, V, LIST, SIZES, ROWOF, Y, slots, heads, queries, keys, first_query, per_query, scale);
}

} // extern "C"

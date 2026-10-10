// Prefill attention for long prompts: 64 query rows share a 64-key tile; dot2 products and online softmax in fp32.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "decode/pages.hpp"

namespace {

constexpr int kRows = 64;     // query rows a block
constexpr int kKeys = 64;     // keys a tile
constexpr int kChunk = 64;    // head dimensions a staged key or value chunk
constexpr int kThreads = 256;
static_assert(kKeys == kPageTokens, "a key tile is one page");

// A cache element type's pair for dot2, the rounding into it and the product.
template <typename Cache>
struct Dot;

template <>
struct Dot<__half> {
    using elem = _Float16;
    using pair = _Float16 __attribute__((ext_vector_type(2)));
    static __device__ __forceinline__ float dot(pair a, pair b, float acc) {
        return __builtin_amdgcn_fdot2(a, b, acc, false);
    }
};

template <>
struct Dot<hip_bfloat16> {
    using elem = __bf16;
    using pair = __bf16 __attribute__((ext_vector_type(2)));
    static __device__ __forceinline__ float dot(pair a, pair b, float acc) {
#if TF_DOT2_BF16
        return __builtin_amdgcn_fdot2_f32_bf16(a, b, acc, false);
#else
        // RDNA2 has no bf16 dot2: its caches are fp16, this only keeps the kernel building
        return fmaf(static_cast<float>(a.y), static_cast<float>(b.y), fmaf(static_cast<float>(a.x), static_cast<float>(b.x), acc));
#endif
    }
};

// A 16-byte LDS read as four pairs.
template <typename Pair>
__device__ __forceinline__ void pairs4(const void* at, Pair out[4]) {
    const uint4 w = *reinterpret_cast<const uint4*>(at);
    out[0] = __builtin_bit_cast(Pair, w.x);
    out[1] = __builtin_bit_cast(Pair, w.y);
    out[2] = __builtin_bit_cast(Pair, w.z);
    out[3] = __builtin_bit_cast(Pair, w.w);
}

// q rounds to the cache type once and stays in LDS; with PAGED, k and v are a layer's pools and a 64-key tile a page.
template <typename Cache, bool PAGED>
__device__ void fa_dot2(const float* __restrict__ q, const Cache* __restrict__ k, const Cache* __restrict__ v,
                        float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d, float scale,
                        int q_pos0, long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh,
                        long long v_ss, const unsigned* table, int count) {
    constexpr int kQChunks = 256 / 8;      // 16-byte chunks a q row
    constexpr int kKChunks = kChunk / 8;   // a staged key row's
    constexpr int kPLd = kRows + 4;        // a key pair's probabilities, row-indexed
    __shared__ uint4 sQ[kRows * kQChunks];
    __shared__ uint4 sKV[kKeys * kKChunks];   // a key chunk (key, 8 dims) or a value chunk (key pair, 4 dims)
    using T = Dot<Cache>;
    using elem = typename T::elem;
    using pair_t = typename T::pair;
    __shared__ alignas(16) pair_t sH[kKeys / 2 * kPLd];
    __shared__ float sA[kRows], sL[kRows];

    const int q0 = blockIdx.x * kRows;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (q0 >= qlen) return;
    const int tid = threadIdx.x;
    int q_hi = qlen - q0;
    if (q_hi > kRows) q_hi = kRows;
    int max_vis = q_pos0 + q0 + q_hi;
    if (max_vis > span) max_vis = span;
    const int kv = head / (heads / kv_heads);
    const float* qrows = q + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    float* orows = out + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    const Cache* kbase = PAGED ? k : k + static_cast<long long>(batch) * k_sb + static_cast<long long>(kv) * k_sh;
    const Cache* vbase = PAGED ? v : v + static_cast<long long>(batch) * v_sb + static_cast<long long>(kv) * v_sh;
    const long long kstride = PAGED ? d : k_ss;
    const long long vstride = PAGED ? d : v_ss;
    const int chunks = d / 8;

    for (int e = tid; e < kRows * chunks; e += kThreads) {
        const int row = e / chunks, c = e % chunks;
        elem h[8];
#pragma unroll
        for (int m = 0; m < 8; ++m)
            h[m] = static_cast<elem>(row < q_hi ? qrows[static_cast<long long>(row) * d + 8 * c + m] : 0.f);
        sQ[row * kQChunks + (c ^ (row & 7))] = __builtin_bit_cast(uint4, h);
    }

    // scores: rows sr + 16i against keys 4sk .. 4sk + 3; PV: rows pr .. pr + 3 at dims 4pc .. 4pc + 3 of each chunk
    const int sr = tid / 16, sk = tid % 16;
    const int pr = (tid / 16) * 4, pc = tid % 16;
    float m_run[4], l_run[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) m_run[i] = -INFINITY, l_run[i] = 0.f;
    float o[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int m = 0; m < 16; ++m) o[i][m] = 0.f;

    for (int k0 = 0; k0 < max_vis; k0 += kKeys) {
        // the tile's first key row in the keys' and the values' storage
        const Cache* ktile = PAGED ? kbase + page_row(table[k0 / kPageTokens], count, kv, 0, d) : kbase + static_cast<long long>(k0) * k_ss;
        const Cache* vtile = PAGED ? vbase + page_row(table[k0 / kPageTokens], count, kv, 0, d) : vbase + static_cast<long long>(k0) * v_ss;
        float acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
        for (int c0 = 0; c0 < chunks; c0 += kKChunks) {
            uint4 staged[2];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int e = tid + t * kThreads, key = e / kKChunks, c = e % kKChunks;
                staged[t] = k0 + key < span
                                ? *reinterpret_cast<const uint4*>(ktile + static_cast<long long>(key) * kstride + 8 * (c0 + c))
                                : uint4{0, 0, 0, 0};
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int e = tid + t * kThreads, key = e / kKChunks, c = e % kKChunks;
                sKV[key * kKChunks + (c ^ ((key >> 2) & 7))] = staged[t];
            }
            __syncthreads();
#pragma unroll 2
            for (int c = 0; c < kKChunks; ++c) {
                pair_t qv[4][4], kf[4][4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = sr + 16 * i;
                    pairs4(&sQ[row * kQChunks + ((c0 + c) ^ (row & 7))], qv[i]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int key = 4 * sk + j;
                    pairs4(&sKV[key * kKChunks + (c ^ ((key >> 2) & 7))], kf[j]);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int m = 0; m < 4; ++m) acc[i][j] = T::dot(qv[i][m], kf[j][m], acc[i][j]);
            }
        }
        // the online softmax in registers: a row's 64 keys sit on its 16 lanes
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = sr + 16 * i;
            int vis = q_pos0 + q0 + row + 1;
            if (vis > span) vis = span;
            float tile_max = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = (row < q_hi && k0 + 4 * sk + j < vis) ? acc[i][j] * scale : -INFINITY;
                tile_max = fmaxf(tile_max, acc[i][j]);
            }
#pragma unroll
            for (int lane = 8; lane > 0; lane >>= 1) tile_max = fmaxf(tile_max, __shfl_xor(tile_max, lane, 16));
            const float next_m = fmaxf(m_run[i], tile_max);
            const float alpha = next_m == -INFINITY ? 1.f : __expf(m_run[i] - next_m);
            float p[4], sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[j] = next_m == -INFINITY ? 0.f : __expf(acc[i][j] - next_m);
                sum += p[j];
            }
#pragma unroll
            for (int lane = 8; lane > 0; lane >>= 1) sum += __shfl_xor(sum, lane, 16);
            l_run[i] = alpha * l_run[i] + sum;
            m_run[i] = next_m;
            sH[(2 * sk) * kPLd + row] = pair_t{static_cast<elem>(p[0]), static_cast<elem>(p[1])};
            sH[(2 * sk + 1) * kPLd + row] = pair_t{static_cast<elem>(p[2]), static_cast<elem>(p[3])};
            if (sk == 0) sA[row] = alpha;
        }
#pragma unroll
        for (int at = 0; at < 256 / kChunk; ++at) {
            if (at * kKChunks >= chunks) break;
            const int c0 = at * kKChunks;
            // a value chunk: key pair `pair`, dims 8g .. 8g + 7 interleaved as (key, key + 1) fp16 pairs
            const int pair = tid / kKChunks, g = tid % kKChunks;
            const int key = k0 + 2 * pair;
            const uint4 a = key < span ? *reinterpret_cast<const uint4*>(vtile + static_cast<long long>(2 * pair) * vstride + 8 * (c0 + g))
                                       : uint4{0, 0, 0, 0};
            const uint4 b = key + 1 < span
                                ? *reinterpret_cast<const uint4*>(vtile + static_cast<long long>(2 * pair + 1) * vstride + 8 * (c0 + g))
                                : uint4{0, 0, 0, 0};
            __syncthreads();
            const elem* ha = reinterpret_cast<const elem*>(&a);
            const elem* hb = reinterpret_cast<const elem*>(&b);
            pair_t lo[4], hi[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) lo[m] = pair_t{ha[m], hb[m]}, hi[m] = pair_t{ha[4 + m], hb[4 + m]};
            sKV[pair * 16 + 2 * g] = __builtin_bit_cast(uint4, lo);
            sKV[pair * 16 + 2 * g + 1] = __builtin_bit_cast(uint4, hi);
            __syncthreads();
            float pv[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) pv[i][m] = 0.f;
#pragma unroll 4
            for (int pp = 0; pp < kKeys / 2; ++pp) {
                pair_t pr4[4], vv[4];
                pairs4(sH + pp * kPLd + pr, pr4);
                pairs4(&sKV[pp * 16 + pc], vv);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int m = 0; m < 4; ++m) pv[i][m] = T::dot(pr4[i], vv[m], pv[i][m]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    float& acc_o = o[i][4 * at + m];
                    acc_o = fmaf(sA[pr + i], acc_o, pv[i][m]);
                }
        }
    }
    if (sk == 0)
#pragma unroll
        for (int i = 0; i < 4; ++i) sL[sr + 16 * i] = l_run[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = pr + i;
        if (row >= q_hi) continue;
        const float inv = sL[row] == 0.f ? 0.f : 1.f / sL[row];
#pragma unroll
        for (int at = 0; at < 256 / kChunk; ++at) {
            if (at * kKChunks >= chunks) break;
#pragma unroll
            for (int m = 0; m < 4; ++m)
                orows[static_cast<long long>(row) * d + 64 * at + 4 * pc + m] = o[i][4 * at + m] * inv;
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
               int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss,
               long long v_sb, long long v_sh, long long v_ss) {
    fa_dot2<__half, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    fa_dot2<hip_bfloat16, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb,
                                 v_sh, v_ss, nullptr, 0);
}

// The same tile over a stream's pages: `k` and `v` are a layer's pools, `table` the stream's page ids.
extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_paged_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
                int kv_heads, int d, float scale, int q_pos0, const unsigned* table, int count) {
    fa_dot2<__half, true>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, 0, 0, 0, 0, 0, 0, table, count);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_paged_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                 int heads, int kv_heads, int d, float scale, int q_pos0, const unsigned* table, int count) {
    fa_dot2<hip_bfloat16, true>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, 0, 0, 0, 0, 0, 0, table, count);
}

// The 64-row prefill tile's body, shared by its entry points (prefill.hip) and the contract's controls.
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "decode/pages.hpp"

#ifndef TF_DOT2_SIGNED_ZERO
#error "the caps macros come from the build"
#endif

namespace {

constexpr int kRows = 64;     // query rows a block
constexpr int kKeys = 64;     // keys a tile
constexpr int kChunk = 64;    // head dimensions a staged key or value chunk
constexpr int kThreads = 256;
using word4 = unsigned __attribute__((ext_vector_type(4)));   // a 16-byte load held as one value
static_assert(kKeys == kPageTokens, "a key tile is one page");

// A cache element type's pair for dot2, the rounding into it and the product.
template <typename Cache>
struct Dot;

template <>
struct Dot<__half> {
    using elem = _Float16;
    static constexpr unsigned top = 0x7c00u;   // the exponent bits, all set in an infinity or NaN
    using pair = _Float16 __attribute__((ext_vector_type(2)));
    static __device__ __forceinline__ float dot(pair a, pair b, float acc) {
        return __builtin_amdgcn_fdot2(a, b, acc, false);
    }
};

template <>
struct Dot<hip_bfloat16> {
    using elem = __bf16;
    static constexpr unsigned top = 0x7f80u;
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

// A 16-byte load of row `row` at element `col`; unless FULL, a row from `rows` on reads row `rows` - 1 instead.
template <bool FULL, typename Cache>
__device__ __forceinline__ word4 load16(const Cache* base, long long stride, int row, int rows, int col) {
    return *reinterpret_cast<const word4*>(base + (FULL || row < rows ? row : rows - 1) * stride + col);
}

// A staged piece, zero for a row past the span.
__device__ __forceinline__ uint4 kept(word4 w, bool in) {
    return __builtin_bit_cast(uint4, in ? w : word4{0, 0, 0, 0});
}

// x from lane ^ M of its 16-lane row (M < 16): a DPP row_xmask move, all lanes active.
template <int M>
__device__ __forceinline__ float row_xor(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x160 + M, 0xf, 0xf, false));
}

// A 16-lane row's max and sum by xor butterflies over lane distances 8, 4, 2, 1, each lane taking own then partner.
__device__ __forceinline__ float row_max16(float x) {
    x = fmaxf(x, row_xor<8>(x));
    x = fmaxf(x, row_xor<4>(x));
    x = fmaxf(x, row_xor<2>(x));
    return fmaxf(x, row_xor<1>(x));
}

__device__ __forceinline__ float row_sum16(float x) {
    x += row_xor<8>(x);
    x += row_xor<4>(x);
    x += row_xor<2>(x);
    return x + row_xor<1>(x);
}

// One lane's x on every lane of its 16-lane row, `sel` naming that lane in each nibble: v_permlane16.
template <typename P>
__device__ __forceinline__ P row_lane(P x, unsigned sel) {
    const int w = __builtin_bit_cast(int, x);
    return __builtin_bit_cast(P, __builtin_amdgcn_permlane16(w, w, sel, sel, false, false));
}

// Either 16-bit half of `w` has every exponent bit of `top` set: an infinity or NaN.
__device__ __forceinline__ bool nonfinite2(unsigned w, unsigned top) {
    return (w & top) == top || (w & (top << 16)) == (top << 16);
}

// q rounds to the cache type once into LDS, p stays in registers; PAGED: pooled k, v; !CAUSAL: rows see every key.
template <typename Cache, bool PAGED, bool CAUSAL, bool MASKED, bool EDGE>
__device__ void fa_dot2(const float* __restrict__ q, const Cache* __restrict__ k, const Cache* __restrict__ v,
                        float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d, float scale,
                        int q_pos0, long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh,
                        long long v_ss, const unsigned* table, int count) {
    constexpr int kQChunks = 256 / 8;      // 16-byte chunks a q row
    constexpr int kKChunks = kChunk / 8;   // a staged key row's
    __shared__ uint4 sQ[kRows * kQChunks];
    __shared__ uint4 sKV[kKeys * kKChunks];   // a key chunk (key, 8 dims) or a value chunk (key pair, 4 dims)
    using T = Dot<Cache>;
    using elem = typename T::elem;
    using pair_t = typename T::pair;

    const int q0 = blockIdx.x * kRows;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (q0 >= qlen) return;
    const int tid = threadIdx.x;
    int q_hi = qlen - q0;
    if (q_hi > kRows) q_hi = kRows;
    int max_vis = CAUSAL ? q_pos0 + q0 + q_hi : span;
    if (max_vis > span) max_vis = span;
    const int kv = head / (heads / kv_heads);
    const float* qrows = q + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    float* orows = out + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    const Cache* kbase = PAGED ? k : k + static_cast<long long>(batch) * k_sb + static_cast<long long>(kv) * k_sh;
    const Cache* vbase = PAGED ? v : v + static_cast<long long>(batch) * v_sb + static_cast<long long>(kv) * v_sh;
    const long long kstride = PAGED ? d : k_ss;
    const long long vstride = PAGED ? d : v_ss;
    const int chunks = d / 8;

    if constexpr (EDGE) {
        // the keys a live row is masked from: past row 0's edge, up to the end of the last tile read
        const int lo = q_pos0 + q0 + 1;
        int hi = (max_vis + kKeys - 1) / kKeys * kKeys;
        if (hi > span) hi = span;
        // at most 126 keys span three pages: their ids load once, uniform, before the per-thread reads
        const int p0 = lo / kPageTokens;
        unsigned pg[3] = {0, 0, 0};
        if constexpr (PAGED) {
#pragma unroll
            for (int i = 0; i < 3; ++i)
                if (lo < hi && p0 + i <= (hi - 1) / kPageTokens) pg[i] = table[p0 + i];
        }
        int bad = 0;
        for (int e = tid; e < (hi - lo) * chunks; e += kThreads) {
            const int key = lo + e / chunks, c = e % chunks;
            const int at = key / kPageTokens - p0;
            const unsigned id = at == 0 ? pg[0] : at == 1 ? pg[1] : pg[2];
            const Cache* row = PAGED ? vbase + page_row(id, count, kv, key % kPageTokens, d) : vbase + key * vstride;
            const word4 w = *reinterpret_cast<const word4*>(row + 8 * c);
#pragma unroll
            for (int j = 0; j < 4; ++j) bad |= nonfinite2(w[j], T::top);
        }
        if (!__syncthreads_or(bad)) return;
    }

    for (int e = tid; e < kRows * chunks; e += kThreads) {
        const int row = e / chunks, c = e % chunks;
        elem h[8];
#pragma unroll
        for (int m = 0; m < 8; ++m)
            h[m] = static_cast<elem>(row < q_hi ? qrows[static_cast<long long>(row) * d + 8 * c + m] : 0.f);
        sQ[row * kQChunks + (c ^ (row & 7))] = __builtin_bit_cast(uint4, h);
    }

    // rows sr + 16i: scores against keys 4sk .. 4sk + 3, PV at dims 4sk .. 4sk + 3 of each chunk
    const int sr = tid / 16, sk = tid % 16;
    float m_run[4], l_run[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) m_run[i] = -INFINITY, l_run[i] = 0.f;
    float o[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int m = 0; m < 16; ++m) o[i][m] = 0.f;

    // a tile ending by full_end is in range and visible to every stored row, so FULL drops its guards and masks
    int full_end = CAUSAL ? q_pos0 + q0 + 1 : span;
    if (full_end > span) full_end = span;
    // the first key row of the tile at k0 (on page `pg` when paged) in the keys' and the values' storage
    const auto ktile_at = [&](int k0, unsigned pg) __attribute__((always_inline)) {
        return PAGED ? kbase + page_row(pg, count, kv, 0, d) : kbase + static_cast<long long>(k0) * k_ss;
    };
    const auto vtile_at = [&](int k0, unsigned pg) __attribute__((always_inline)) {
        return PAGED ? vbase + page_row(pg, count, kv, 0, d) : vbase + static_cast<long long>(k0) * v_ss;
    };
    // each staged chunk's loads issue before the previous chunk's products, which hide their latency
    word4 kst[2], va, vb;
    const auto load_k = [&]<bool FULL>(const Cache* ktile, int rows, int c0) __attribute__((always_inline)) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int e = tid + t * kThreads, key = e / kKChunks, c = e % kKChunks;
            kst[t] = load16<FULL>(ktile, kstride, key, rows, 8 * (c0 + c));
        }
    };
    // a value chunk: key pair `pair`, dims 8g .. 8g + 7 of both keys
    const auto load_v = [&]<bool FULL>(const Cache* vtile, int rows, int c0) __attribute__((always_inline)) {
        const int pair = tid / kKChunks, g = tid % kKChunks;
        va = load16<FULL>(vtile, vstride, 2 * pair, rows, 8 * (c0 + g));
        vb = load16<FULL>(vtile, vstride, 2 * pair + 1, rows, 8 * (c0 + g));
    };
    unsigned page = PAGED && max_vis > 0 ? table[0] : 0;
    if (max_vis > 0) load_k.template operator()<false>(ktile_at(0, page), span, 0);
    const auto tile = [&]<bool FULL>(int k0) __attribute__((always_inline)) {
        const Cache* ktile = ktile_at(k0, page);
        const Cache* vtile = vtile_at(k0, page);
        const int rows = span - k0;
        const bool more = k0 + kKeys < max_vis;
        unsigned next = page;
        float acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
        for (int c0 = 0; c0 < chunks; c0 += kKChunks) {
            __syncthreads();
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int e = tid + t * kThreads, key = e / kKChunks, c = e % kKChunks;
                sKV[key * kKChunks + (c ^ ((key >> 2) & 7))] = kept(kst[t], FULL || key < rows);
            }
            __syncthreads();
            if (PAGED && c0 == 0 && more) {
                // a vector index keeps the page id out of scalar registers, so its load is awaited only where used
                int at_page = (k0 + kKeys) / kPageTokens;
                asm volatile("" : "+v"(at_page));
                next = table[at_page];
            }
            if (c0 + kKChunks < chunks) load_k.template operator()<FULL>(ktile, rows, c0 + kKChunks);
            else load_v.template operator()<FULL>(vtile, rows, 0);
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
        // the online softmax in registers: a row's 64 keys sit on its 16 lanes, key pair pp's p on lane pp / 2
        pair_t ph[4][2];
        float alpha_r[4];
        int seen[4];   // the keys of this tile each row sees: none to 0, all from 64
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = sr + 16 * i;
            int vis = CAUSAL ? q_pos0 + q0 + row + 1 : span;
            if (vis > span) vis = span;
            if constexpr (MASKED) seen[i] = vis - k0;
            float tile_max = -INFINITY;
            int nan_row = 0; // !CAUSAL: a NaN score makes its row NaN, as in Torch
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = (FULL || (row < q_hi && k0 + 4 * sk + j < vis)) ? acc[i][j] * scale : -INFINITY;
                if constexpr (!CAUSAL) nan_row |= isnan(acc[i][j]);
                tile_max = fmaxf(tile_max, acc[i][j]);
            }
            tile_max = row_max16(tile_max);
            if constexpr (!CAUSAL) nan_row = row_max16(nan_row ? 1.f : 0.f) > 0.f;
            const float next_m = nan_row ? NAN : fmaxf(m_run[i], tile_max);
            const float alpha = next_m == -INFINITY ? 1.f : expf(m_run[i] - next_m);
            float p[4], sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[j] = next_m == -INFINITY ? 0.f : expf(acc[i][j] - next_m);
                sum += p[j];
            }
            sum = row_sum16(sum);
            l_run[i] = alpha * l_run[i] + sum;
            m_run[i] = next_m;
            ph[i][0] = pair_t{static_cast<elem>(p[0]), static_cast<elem>(p[1])};
            ph[i][1] = pair_t{static_cast<elem>(p[2]), static_cast<elem>(p[3])};
            alpha_r[i] = alpha;
        }
#pragma unroll
        for (int at = 0; at < 256 / kChunk; ++at) {
            if (at * kKChunks >= chunks) break;
            // the chunk's key pairs as (key, key + 1) pairs at each of its dims
            const int pair = tid / kKChunks, g = tid % kKChunks;
            using elem8 = elem __attribute__((ext_vector_type(8)));
            // MASKED: a value past the span reads the span's last (load16); the products zero each key past a row's edge
            const uint4 ua = MASKED ? __builtin_bit_cast(uint4, va) : kept(va, FULL || 2 * pair < rows);
            const uint4 ub = MASKED ? __builtin_bit_cast(uint4, vb) : kept(vb, FULL || 2 * pair + 1 < rows);
            const elem8 ha = __builtin_bit_cast(elem8, ua);
            const elem8 hb = __builtin_bit_cast(elem8, ub);
            pair_t lo[4], hi[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) lo[m] = pair_t{ha[m], hb[m]}, hi[m] = pair_t{ha[4 + m], hb[4 + m]};
            __syncthreads();
            sKV[pair * 16 + 2 * g] = __builtin_bit_cast(uint4, lo);
            sKV[pair * 16 + 2 * g + 1] = __builtin_bit_cast(uint4, hi);
            __syncthreads();
            if ((at + 1) * kKChunks < chunks) load_v.template operator()<FULL>(vtile, rows, (at + 1) * kKChunks);
            else if (more) load_k.template operator()<false>(ktile_at(k0 + kKeys, next), rows - kKeys, 0);
            float pv[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) pv[i][m] = 0.f;
#pragma unroll 4
            for (int pp = 0; pp < kKeys / 2; ++pp) {
                pair_t vv[4];
                pairs4(&sKV[pp * 16 + sk], vv);
                const unsigned lane = (pp / 2) * 0x11111111u;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const pair_t pp4 = row_lane(ph[i][pp % 2], lane);
                    if constexpr (MASKED) {
                        // a key past the row's edge takes value 0: RDNA 3.5's dot2 rounds a -0 lane off +0's
                        const int left = FULL ? 2 : seen[i] - 2 * pp;
                        const unsigned keep = left >= 2 ? ~0u : left == 1 ? 0xffffu : 0u;   // key 2pp is the low half
#pragma unroll
                        for (int m = 0; m < 4; ++m)
                            pv[i][m] = T::dot(pp4, __builtin_bit_cast(pair_t, __builtin_bit_cast(unsigned, vv[m]) & keep),
                                              pv[i][m]);
                    } else {
#pragma unroll
                        for (int m = 0; m < 4; ++m) pv[i][m] = T::dot(pp4, vv[m], pv[i][m]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    float& acc_o = o[i][4 * at + m];
                    acc_o = fmaf(alpha_r[i], acc_o, pv[i][m]);
                }
        }
        page = next;
    };
    int k0 = 0;
    for (; k0 + kKeys <= full_end; k0 += kKeys) tile.template operator()<true>(k0);
    for (; k0 < max_vis; k0 += kKeys) tile.template operator()<false>(k0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = sr + 16 * i;
        if (row >= q_hi) continue;
        const float inv = l_run[i] == 0.f ? 0.f : 1.f / l_run[i];
#pragma unroll
        for (int at = 0; at < 256 / kChunk; ++at) {
            if (at * kKChunks >= chunks) break;
#pragma unroll
            for (int m = 0; m < 4; ++m)
                orows[static_cast<long long>(row) * d + 64 * at + 4 * sk + m] = o[i][4 * at + m] * inv;
        }
    }
}

}  // namespace

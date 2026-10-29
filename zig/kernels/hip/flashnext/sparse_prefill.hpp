#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../common/arch.hpp"
#include "../ops/common.hpp"

// Sparse prefill attention on gfx11 WMMA: a tile of query rows times a kv head's heads shares every gathered key.

namespace {

constexpr int kFxRowsPer = 4;                     // query rows a block, times the kv head's query heads
constexpr int kFxMostGroup = 12;                  // the whole model's query heads a kv head, as sparse.hpp's cap
constexpr int kFxRows = kFxRowsPer * kFxMostGroup;   // 48, three 16-row WMMA tiles
constexpr int kFxRowTiles = kFxRows / 16;
constexpr int kFxKeys = 16;                       // keys a tile: one WMMA column tile, ratio 4 selection blocks
constexpr int kFxRatio = 4;                       // the indexer's compress_ratio
constexpr int kFxKeyBlocks = kFxKeys / kFxRatio;
constexpr int kFxMostDims = 256;                  // the sparse attention's head dim, as sparse.hpp's
constexpr int kFxKSteps = kFxMostDims / 16;
constexpr int kFxDimTiles = kFxMostDims / 16;
constexpr int kFxThreads = 256;
constexpr int kFxWaves = kFxThreads / 32;
constexpr int kFxKStepsPerWave = kFxKSteps / 2; // two waves split one S tile's head dimension
constexpr int kFxSoftLanes = 4;                  // threads a probability row, one selection block apiece
constexpr int kFxKStride = kFxMostDims + 8;      // staged key rows, so 16-byte fragment reads miss no bank group
constexpr int kFxVtStride = kFxKeys + 8;          // V transposed: a dimension's row of keys, padded for the reads
constexpr int kFxPStride = kFxKeys + 8;
constexpr int kFxScoreStride = kFxKeys + 4;      // a float4 of scores per softmax lane
constexpr int kFxMostWords = 160;                // a part's block range: 5,120 blocks, 20,480 tokens at ratio 4

typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));  // vec.hpp's lives in tf::rocm, not this namespace
typedef float f32x8 __attribute__((ext_vector_type(8)));

// The sixteen contiguous k of one WMMA fragment: two 16-byte LDS reads joined by shufflevector.
__device__ __forceinline__ f16x16 fxfrag16(const uint16_t* p) {
    const u32x4 lo = *reinterpret_cast<const u32x4*>(p);
    const u32x4 hi = *reinterpret_cast<const u32x4*>(p + 8);
    return __builtin_bit_cast(f16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

__device__ __forceinline__ bf16x16 fxfrag16bf(const uint16_t* p) {
    const u32x4 lo = *reinterpret_cast<const u32x4*>(p);
    const u32x4 hi = *reinterpret_cast<const u32x4*>(p + 8);
    return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

// The cache kind on the matrix cores: its fragment type, its narrowing into the cache type and its fp32 mma.
template <int KIND>
struct FxWmma;

template <>
struct FxWmma<1> {
    using frag = f16x16;
    static __device__ __forceinline__ uint16_t near(float v) { return __half_as_ushort(__float2half_rn(v)); }
    static __device__ __forceinline__ bool finite(uint16_t b) { return (b & 0x7c00u) != 0x7c00u; }
    static __device__ __forceinline__ f32x8 mma(frag a, frag b, f32x8 c) {
#if TF_DEVICE_WMMA_GFX11
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
        return c;  // the launcher takes this kernel only where the caps table says matrix = wmma11
#endif
    }
    static __device__ __forceinline__ frag load(const uint16_t* p) { return fxfrag16(p); }
};

template <>
struct FxWmma<2> {
    using frag = bf16x16;
    static __device__ __forceinline__ uint16_t near(float v) { return float_to_bf16(v); }
    static __device__ __forceinline__ bool finite(uint16_t b) { return (b & 0x7f80u) != 0x7f80u; }
    static __device__ __forceinline__ f32x8 mma(frag a, frag b, f32x8 c) {
#if TF_DEVICE_WMMA_GFX11
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
        return c;  // the launcher takes this kernel only where the caps table says matrix = wmma11
#endif
    }
    static __device__ __forceinline__ frag load(const uint16_t* p) { return fxfrag16bf(p); }
};

// WMMA column c is query row c % kFxRowsPer and head c / kFxRowsPer; on a tail tile live columns are scattered.
__device__ __forceinline__ int fxCol(int i, int rows_in) { return (i / rows_in) * kFxRowsPer + (i % rows_in); }

// Whether column c maps to a row and a head inside the tile; equals c < group * rows_in only on a full tile.
__device__ __forceinline__ bool fxLive(int c, int rows_in, int group) {
    return (c % kFxRowsPer) < rows_in && (c / kFxRowsPer) < group;
}

// Block (tile, g, part): query rows tile * kFxRowsPer .. of kv head g, this part's range of the cache's blocks.
template <int KIND>
__device__ void sparsePrefillWmma(const float* __restrict__ q, const uint16_t* __restrict__ k,
                                   const uint16_t* __restrict__ v, const int* __restrict__ ids, long long id_stride,
                                   const int* __restrict__ count, const int* __restrict__ sparse,
                                   const int* __restrict__ ends, float* __restrict__ parts_o,
                                   float* __restrict__ parts_ms, int rows, int heads, int kv_heads, int dims,
                                   long long cap, float scale, int parts) {
    using T = FxWmma<KIND>;
    const int tile = blockIdx.x, g = blockIdx.y, part = blockIdx.z;
    const int group = heads / kv_heads;
    const int tiles = (group * kFxRowsPer + 15) / 16;  // 16-row WMMA tiles with a live column, of kFxRowTiles
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5, sub = lane & 15, half = lane >> 4;
    const int q0 = tile * kFxRowsPer;
    const int rows_in = min(kFxRowsPer, rows - q0);
    const int cols = group * rows_in;  // how many of the 48 WMMA columns are live; see fxCol for WHICH ones
    const long long head_base = static_cast<long long>(g) * cap * dims;

    // The part's block range, word-aligned outward: every block of the cache falls in exactly one part's range.
    const int nblocks = static_cast<int>((cap + kFxRatio - 1) / kFxRatio);
    const int blo = static_cast<int>((static_cast<long long>(part) * nblocks) / parts);
    const int bhi = min(nblocks, static_cast<int>((static_cast<long long>(part + 1) * nblocks) / parts));
    const int wbase = blo >> 5;
    const int words = bhi > blo ? (((bhi - 1) >> 5) - wbase + 1) : 0;

    __shared__ unsigned maps[kFxRowsPer * kFxMostWords];  // the tile's rows' blocks, always kFxMostWords apart
    __shared__ unsigned uni[kFxMostWords];                 // their union: the key walk
    __shared__ alignas(16) uint16_t stage[kFxMostDims * kFxVtStride];  // K row-major, then V transposed: one buffer
    __shared__ alignas(16) uint16_t p_lds[kFxRows * kFxPStride];
    __shared__ alignas(16) float s_lds[2][kFxRowTiles][16][kFxScoreStride];
    __shared__ float row_max[kFxRows], row_sum[kFxRows], row_scale[kFxRows];
    __shared__ unsigned sel_keys[kFxRows];                 // per column: the tile keys it selected, one bit each
    __shared__ unsigned v_bad[kFxMostDims];                // per dim: the tile keys whose V there is not finite
    __shared__ float new_max[kFxRows], new_sum[kFxRows];  // staged across the softmax's barrier, then folded in

    // K occupies stage[0 .. kFxKeys * kFxKStride); V^T is staged into the same buffer after the scores are in LDS.
    uint16_t* k_lds = stage;
    uint16_t* v_lds = stage;

    for (int i = t; i < kFxRowsPer * kFxMostWords; i += kFxThreads) maps[i] = 0u;
    // p_lds' padded columns feed P V too: they hold probability zero, so a masked-out key contributes nothing
    for (int i = t; i < tiles * 16 * kFxPStride; i += kFxThreads) p_lds[i] = 0u;
    for (int i = t; i < kFxRows; i += kFxThreads) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.f;
        row_scale[i] = 1.f;
        sel_keys[i] = 0u;
    }
    __syncthreads();

    // The rows' bitmaps: the block of every listed token, and for a dense row every block with a token below ends.
    for (int rl = 0; rl < kFxRowsPer; ++rl) {
        const int r = q0 + rl;
        if (r < rows) {
            unsigned* map = maps + rl * kFxMostWords;
            const bool listed = sparse != nullptr && ids != nullptr && sparse[r] != 0;
            const int end = ends[r];
            if (listed) {
                const int n = count[r];
                for (int j = t; j < n; j += kFxThreads) {
                    const int b = ids[r * id_stride + j] >> 2;
                    const int iw = (b >> 5) - wbase;
                    if (iw >= 0 && iw < words) atomicOr(&map[iw], 1u << (b & 31));
                }
            } else {
                const int last = min(bhi, (end + kFxRatio - 1) / kFxRatio);
                for (int b = blo + t; b < last; b += kFxThreads) atomicOr(&map[(b >> 5) - wbase], 1u << (b & 31));
            }
        }
        __syncthreads();  // every row's atomicOr of this pass is visible before the union folds them
    }
    for (int iw = t; iw < words; iw += kFxThreads) {
        unsigned u = 0u;
        for (int rl = 0; rl < kFxRowsPer; ++rl) u |= maps[rl * kFxMostWords + iw];
        const int base = (wbase + iw) << 5;
        if (base + 32 > bhi) u &= (1u << (bhi - base)) - 1u;  // past the part is another part's
        uni[iw] = u;
    }
    __syncthreads();

    // Q into fragments: wave w scores row tile w / 2 over head dimension half w % 2; its A row is sub, its k the half.
    const int rb = wave >> 1, dh = wave & 1;
    typename T::frag q_frag[kFxKStepsPerWave];
    {
        const int c = rb * 16 + sub;
        const bool live = (c % kFxRowsPer) < rows_in && c / kFxRowsPer < group;  // not c < cols: see fxCol
        const float* qrow = q + (static_cast<long long>(live ? q0 + (c % kFxRowsPer) : 0) * heads +
                                 g * group + (live ? c / kFxRowsPer : 0)) *
                                    dims;
#pragma unroll
        for (int ks = 0; ks < kFxKStepsPerWave; ++ks) {
            const int d0 = (dh * kFxKStepsPerWave + ks) * 16;
            alignas(16) uint16_t tmp[16];  // T::load reads it as two uint4: 16 bytes, so it must be aligned
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int d = d0 + i;
                tmp[i] = (live && d < dims) ? T::near(qrow[d] * scale) : static_cast<uint16_t>(0);
            }
            q_frag[ks] = T::load(&tmp[0]);
        }
    }

    f32x8 o_acc[kFxRowTiles][2];
#pragma unroll
    for (int rt = 0; rt < kFxRowTiles; ++rt)
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) o_acc[rt][dt] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};

    // The union walk, uniform across the block: set bits of uni in increasing block order, [blo, bhi) only.
    int cw = 0;
    unsigned bits = words > 0 ? (uni[0] & (~0u << (blo & 31))) : 0u;

    while (true) {
        int blk[kFxKeyBlocks];
        int n = 0;
        while (n < kFxKeyBlocks) {
            while (bits == 0u) {
                ++cw;
                if (cw >= words) break;
                bits = uni[cw];
            }
            if (bits == 0u) break;
            blk[n++] = ((wbase + cw) << 5) + __builtin_ctz(bits);
            bits &= bits - 1u;
        }
        if (n == 0) break;                        // the part's union is spent: every other part holds these blocks
        while (n < kFxKeyBlocks) blk[n++] = -1;  // pad: its keys read key 0, score -inf, hold probability zero

        // The tile's keys: key i is block blk[i / ratio] at offset i % ratio, from uniform blk[].
        int key[kFxKeys];
#pragma unroll
        for (int i = 0; i < kFxKeys; ++i) {
            const int b = blk[i / kFxRatio];
            key[i] = b < 0 ? 0 : ((b << 2) + (i % kFxRatio));
        }
        for (int d = t; d < kFxMostDims; d += kFxThreads) v_bad[d] = 0u;
        __syncthreads();

        // stage K: one 16-byte load per lane, a wave over one key row, rows conflict-free at stride kFxKStride
        constexpr int kKRegs = (kFxKeys * (kFxMostDims / 8)) / kFxThreads;
#pragma unroll
        for (int r = 0; r < kKRegs; ++r) {
            const int idx = t + r * kFxThreads;
            const int krow = idx / (kFxMostDims / 8), d8 = (idx % (kFxMostDims / 8)) * 8;
            if (d8 + 8 <= dims) {
                const uint4 packed =
                    *reinterpret_cast<const uint4*>(k + head_base + static_cast<long long>(key[krow]) * dims + d8);
                *reinterpret_cast<uint4*>(&k_lds[krow * kFxKStride + d8]) = packed;
            }
        }
        __syncthreads();

        // S = Q K^T: wave w scores row tile rb against the 16 keys over head half dh.
        if (rb < tiles) {
            f32x8 s_acc = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < kFxKStepsPerWave; ++ks) {
                const int d0 = (dh * kFxKStepsPerWave + ks) * 16;
                if (d0 >= dims) break;
                const auto kf = T::load(&k_lds[sub * kFxKStride + d0]);
                s_acc = T::mma(q_frag[ks], kf, s_acc);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) s_lds[dh][rb][2 * i + half][sub] = s_acc[i];
        }
        __syncthreads();

        // The fp32 online softmax of every live column: kFxSoftLanes threads a column, one selection block each.
        if (t < cols * kFxSoftLanes) {
            const int c = fxCol(t / kFxSoftLanes, rows_in), seg = t % kFxSoftLanes;
            const int rt = c >> 4, row = c & 15;
            const int rl = c % kFxRowsPer, r = q0 + rl;
            const int b = blk[seg];
            const int iw = (b >> 5) - wbase;
            const bool marked =
                b >= 0 && iw >= 0 && iw < words && ((maps[rl * kFxMostWords + iw] >> (b & 31)) & 1u) != 0u;
            float vals[kFxSoftLanes];
            unsigned chosen = 0u;
#pragma unroll
            for (int m = 0; m < kFxSoftLanes; ++m) {
                const int kpos = (b << 2) + m;
                chosen |= (marked && kpos < ends[r]) ? 1u << (seg * kFxSoftLanes + m) : 0u;
                vals[m] = (marked && kpos < ends[r])
                            ? s_lds[0][rt][row][seg * kFxSoftLanes + m] + s_lds[1][rt][row][seg * kFxSoftLanes + m]
                            : -INFINITY;
            }
            float pmax = -INFINITY;
#pragma unroll
            for (int m = 0; m < kFxSoftLanes; ++m) pmax = fmaxf(pmax, vals[m]);
#pragma unroll
            for (int off = 1; off < kFxSoftLanes; off <<= 1) pmax = fmaxf(pmax, __shfl_xor(pmax, off, kFxSoftLanes));
            const float prev = row_max[c];
            const float top = fmaxf(prev, pmax);
            const float f = prev == -INFINITY ? 0.f : expf(prev - top);
            float psum = 0.f;
#pragma unroll
            for (int m = 0; m < kFxSoftLanes; ++m) {
                const float e = vals[m] == -INFINITY ? 0.f : expf(vals[m] - top);
                psum += e;
                p_lds[c * kFxPStride + seg * kFxSoftLanes + m] = T::near(e);
            }
#pragma unroll
            for (int off = 1; off < kFxSoftLanes; off <<= 1) psum += __shfl_xor(psum, off, kFxSoftLanes);
            // the running state is staged in new_* and folded after the barrier, since every lane reads row_max above
            for (int off = 1; off < kFxSoftLanes; off <<= 1) chosen |= __shfl_xor(chosen, off, kFxSoftLanes);
            if ((lane & (kFxSoftLanes - 1)) == 0) {
                sel_keys[c] = chosen;
                new_max[c] = top;
                new_sum[c] = row_sum[c] * f + psum;
                row_scale[c] = f;
            }
        }
        __syncthreads();
        // the same packed→column walk the softmax used: on a tail tile the live columns are not contiguous
        for (int i = t; i < cols; i += kFxThreads) {
            const int c = fxCol(i, rows_in);
            row_max[c] = new_max[c];
            row_sum[c] = new_sum[c];
        }

        // Stage V transposed into the buffer K vacated: a thread takes one key and 16 dims, storing v_lds[dim][key].
        constexpr int kVRegs = (kFxKeys * kFxMostDims) / (kFxThreads * 8);
        const int v_key = t % kFxKeys;
        const int v_off = t / kFxKeys;  // 0 .. 15: which 16-dim span of the head this thread covers
#pragma unroll
        for (int r = 0; r < kVRegs; ++r) {
            const int d0 = v_off * (kVRegs * 8) + r * 8;
            if (d0 + 8 > dims) break;
            const uint4 packed =
                *reinterpret_cast<const uint4*>(v + head_base + static_cast<long long>(key[v_key]) * dims + d0);
            const uint16_t* e = reinterpret_cast<const uint16_t*>(&packed);
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const bool bad = !T::finite(e[m]);  // staged as zero: P's zero weight times a NaN would reach every row
                if (bad) atomicOr(&v_bad[d0 + m], 1u << v_key);
                v_lds[(d0 + m) * kFxVtStride + v_key] = bad ? static_cast<uint16_t>(0) : e[m];
            }
        }
        __syncthreads();

        // O = P V on the accumulator rescaled by the tile factors; wave w takes dim tiles w and w + 8.
#pragma unroll
        for (int rt = 0; rt < kFxRowTiles; ++rt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float f = row_scale[rt * 16 + 2 * i + half];
                o_acc[rt][0][i] *= f;
                o_acc[rt][1][i] *= f;
            }
        }
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
            const int dim_tile = wave + dt * kFxWaves;
            if (dim_tile * 16 >= dims) continue;
            const auto vf = T::load(&v_lds[(dim_tile * 16 + sub) * kFxVtStride]);
#pragma unroll
            for (int rt = 0; rt < kFxRowTiles; ++rt) {
                if (rt >= tiles) continue;
                const auto pf = T::load(&p_lds[(rt * 16 + sub) * kFxPStride]);
                o_acc[rt][dt] = T::mma(pf, vf, o_acc[rt][dt]);
            }
            const unsigned bad = v_bad[dim_tile * 16 + sub];
            if (bad != 0u) {  // a row that selected a non-finite V gets NaN in that dim, as the scalar kernel's 0 * NaN
#pragma unroll
                for (int rt = 0; rt < kFxRowTiles; ++rt) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if ((sel_keys[rt * 16 + 2 * i + half] & bad) != 0u) o_acc[rt][dt][i] = NAN;
                    }
                }
            }
        }
        __syncthreads();
    }

    // The part's unnormalized output and its (max, sum) for tf_fn_sparse_merge; an empty part stores zeros, -inf, 0.
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
        const int dim_tile = wave + dt * kFxWaves;
        if (dim_tile * 16 >= dims) continue;
#pragma unroll
        for (int rt = 0; rt < kFxRowTiles; ++rt) {
            if (rt >= tiles) continue;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int c = rt * 16 + 2 * i + half;
                if (!fxLive(c, rows_in, group)) continue;  // padded columns belong to no row: store nothing
                const int d = dim_tile * 16 + sub;
                if (d >= dims) continue;
                const long long out =
                    (static_cast<long long>(q0 + (c % kFxRowsPer)) * heads + g * group + (c / kFxRowsPer)) * parts +
                    part;
                parts_o[out * dims + d] = reinterpret_cast<const float*>(&o_acc[rt][dt])[i];
            }
        }
    }
    if (t < cols) {
        const int c = fxCol(t, rows_in);
        const long long out =
            (static_cast<long long>(q0 + (c % kFxRowsPer)) * heads + g * group + (c / kFxRowsPer)) * parts + part;
        parts_ms[out * 2] = row_max[c];
        parts_ms[out * 2 + 1] = row_sum[c];
    }
}

// The two 16-bit cache kinds; cache_kind stays in the signature so the argument list matches tf_fn_sparse_parts.
extern "C" __global__ void __launch_bounds__(kFxThreads)
tf_fn_sparse_prefill_wmma_bf16(const float* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v,
                                int cache_kind, const int* __restrict__ ids, long long id_stride,
                                const int* __restrict__ count, const int* __restrict__ sparse,
                                const int* __restrict__ ends, float* __restrict__ parts_o, float* __restrict__ parts_ms,
                                int rows, int heads, int kv_heads, int dims, long long cap, float scale, int parts) {
    (void)cache_kind;
    sparsePrefillWmma<2>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride, count,
                        sparse, ends, parts_o, parts_ms, rows, heads, kv_heads, dims, cap, scale, parts);
}

extern "C" __global__ void __launch_bounds__(kFxThreads)
tf_fn_sparse_prefill_wmma_f16(const float* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v,
                               int cache_kind, const int* __restrict__ ids, long long id_stride,
                               const int* __restrict__ count, const int* __restrict__ sparse,
                               const int* __restrict__ ends, float* __restrict__ parts_o, float* __restrict__ parts_ms,
                               int rows, int heads, int kv_heads, int dims, long long cap, float scale, int parts) {
    (void)cache_kind;
    sparsePrefillWmma<1>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride, count,
                        sparse, ends, parts_o, parts_ms, rows, heads, kv_heads, dims, cap, scale, parts);
}

}  // namespace

#pragma once

#include "../ops/common.hpp"

// sparse_parts with the same fp32 operations in the same order: 16-bit staging, a lane a key and several heads.

namespace {

constexpr int kLanesTiles = 1;
constexpr int kLanesRow = kSparseMostDims + 8;

template <int kKind>
__device__ __forceinline__ float lanes_widen(uint32_t h) {
    if constexpr (kKind == 1) return __half2float(__ushort_as_half(static_cast<uint16_t>(h)));
    return bf16_to_float(static_cast<uint16_t>(h));
}

// Block (g, r, p) as sparse_parts over a kKind cache (1 fp16, 2 bf16): a lane a key and kHeads heads, 2 dims a thread.
template <int kSparseGroup, int kKind>
__device__ void sparse_lanes(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                             const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads,
                             int kv_heads, int dims, long long cap, float scale, int parts) {
    constexpr int kWavesTile = kSparseWaves / kLanesTiles;
    constexpr int kHeads = (kSparseGroup + kWavesTile - 1) / kWavesTile;
    constexpr int kHalf = (kSparseGroup + 1) / 2;
    __shared__ __attribute__((aligned(16))) float qs[kSparseGroup][kSparseMostDims];
    __shared__ __attribute__((aligned(16))) uint16_t kv[kLanesTiles][kSparseTile][kLanesRow];
    __shared__ __attribute__((aligned(16))) float ps[kLanesTiles][kSparseGroup][kSparseTile];
    __shared__ float tree[kLanesTiles][kSparseGroup], tops[kLanesTiles][kSparseGroup], sums[kLanesTiles][kSparseGroup];
    __shared__ float fs[kLanesTiles][kSparseGroup], m[kSparseGroup], l[kSparseGroup];
    __shared__ long long keys[kLanesTiles][kSparseTile];
    const int g = blockIdx.x, r = blockIdx.y, part = blockIdx.z;
    const int group = heads / kv_heads;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int n = count[r];
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int lo = static_cast<int>(static_cast<long long>(part) * n / parts);
    const int hi = static_cast<int>(static_cast<long long>(part + 1) * n / parts);
    const long long head_base = static_cast<long long>(g) * cap * dims;
    const uint16_t* k16 = static_cast<const uint16_t*>(k);
    const uint16_t* v16 = static_cast<const uint16_t*>(v);
    for (int i = t; i < group * dims; i += kSparseWaves * 32) {
        const int h = i / dims, d = i % dims;
        qs[h][d] = q[(static_cast<long long>(r) * heads + g * group + h) * dims + d] * scale;
    }
    if (t < kSparseGroup) {
        m[t] = -INFINITY;
        l[t] = 0.f;
    }
    // scores: this wave's tile and first head; values: this thread's dim pair and first head
    const int slot = wave / kWavesTile, h0 = (wave % kWavesTile) * kHeads;
    const int dp = (t & 127) * 2, hv0 = (t >> 7) * kHalf;
    float o[kHalf][2];
#pragma unroll
    for (int h = 0; h < kHalf; ++h) o[h][0] = o[h][1] = 0.f;
    const int chunks = dims / 8;
    for (int t0 = lo; t0 < hi; t0 += kLanesTiles * kSparseTile) {
        int tn[kLanesTiles];
#pragma unroll
        for (int s = 0; s < kLanesTiles; ++s) tn[s] = max(0, min(kSparseTile, hi - t0 - s * kSparseTile));
        __syncthreads();
        if (t < kLanesTiles * kSparseTile) {
            const int s = t / kSparseTile, j = t % kSparseTile, at = t0 + s * kSparseTile + j;
            if (j < tn[s]) keys[s][j] = listed ? ids[r * id_stride + at] : at;
        }
        __syncthreads();
        for (int i = t; i < kLanesTiles * kSparseTile * chunks; i += kSparseWaves * 32) {
            const int s = i / (kSparseTile * chunks), j = (i / chunks) % kSparseTile, c = i % chunks;
            if (j < tn[s])
                *reinterpret_cast<uint4*>(&kv[s][j][c * 8]) =
                    *reinterpret_cast<const uint4*>(k16 + head_base + keys[s][j] * dims + c * 8);
        }
        __syncthreads();
        float sc[kHeads];
#pragma unroll
        for (int a = 0; a < kHeads; ++a) sc[a] = 0.f;
        const bool live = lane < tn[slot];
        if (tn[slot] > 0) {
#pragma unroll 4
            for (int d = 0; d < dims; d += 4) {
                const uint2 w = *reinterpret_cast<const uint2*>(&kv[slot][lane][d]);
                const float k0 = lanes_widen<kKind>(w.x & 0xffff), k1 = lanes_widen<kKind>(w.x >> 16);
                const float k2 = lanes_widen<kKind>(w.y & 0xffff), k3 = lanes_widen<kKind>(w.y >> 16);
#pragma unroll
                for (int a = 0; a < kHeads; ++a) {
                    const float4 x = *reinterpret_cast<const float4*>(&qs[min(h0 + a, kSparseGroup - 1)][d]);
                    sc[a] += x.x * k0;
                    sc[a] += x.y * k1;
                    sc[a] += x.z * k2;
                    sc[a] += x.w * k3;
                }
            }
#pragma unroll
            for (int a = 0; a < kHeads; ++a) {
                float top = live ? sc[a] : -INFINITY;
#pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) top = fmaxf(top, __shfl_xor(top, mask, 32));
                if (lane == 0 && h0 + a < group) tree[slot][h0 + a] = top;
            }
        }
        __syncthreads();
        // values replace the keys; each tile's top is folded into the running max in tile order
        for (int i = t; i < kLanesTiles * kSparseTile * chunks; i += kSparseWaves * 32) {
            const int s = i / (kSparseTile * chunks), j = (i / chunks) % kSparseTile, c = i % chunks;
            if (j < tn[s])
                *reinterpret_cast<uint4*>(&kv[s][j][c * 8]) =
                    *reinterpret_cast<const uint4*>(v16 + head_base + keys[s][j] * dims + c * 8);
        }
        if (tn[slot] > 0) {
#pragma unroll
            for (int a = 0; a < kHeads; ++a) {
                const int h = h0 + a;
                if (h >= group) continue;
                float top = fmaxf(tree[0][h], m[h]);
                for (int s = 1; s <= slot; ++s) top = fmaxf(tree[s][h], top);
                const float e = live ? expf(sc[a] - top) : 0.f;
                float sum = e;
#pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
                ps[slot][h][lane] = e;
                if (lane == 0) {
                    tops[slot][h] = top;
                    sums[slot][h] = sum;
                }
            }
        }
        __syncthreads();
        if (t < group) {
#pragma unroll
            for (int s = 0; s < kLanesTiles; ++s) {
                if (tn[s] == 0) break;
                const float top = tops[s][t];
                const float f = expf(m[t] - top);
                fs[s][t] = f;
                l[t] = l[t] * f + sums[s][t];
                m[t] = top;
            }
        }
        __syncthreads();
        if (dp < dims) {
#pragma unroll
            for (int s = 0; s < kLanesTiles; ++s) {
                if (tn[s] == 0) break;
#pragma unroll
                for (int h = 0; h < kHalf; ++h) {
                    const float f = fs[s][hv0 + h];
                    o[h][0] *= f;
                    o[h][1] *= f;
                }
                for (int j = 0; j < tn[s]; ++j) {
                    const uint32_t w = *reinterpret_cast<const uint32_t*>(&kv[s][j][dp]);
                    const float v0 = lanes_widen<kKind>(w & 0xffff), v1 = lanes_widen<kKind>(w >> 16);
#pragma unroll
                    for (int h = 0; h < kHalf; ++h) {
                        const float p = ps[s][hv0 + h][j];
                        o[h][0] += p * v0;
                        o[h][1] += p * v1;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int h = 0; h < kHalf; ++h) {
        if (hv0 + h >= group || dp >= dims) continue;
        const long long out = (static_cast<long long>(r) * heads + g * group + hv0 + h) * parts + part;
        parts_o[out * dims + dp] = o[h][0];
        parts_o[out * dims + dp + 1] = o[h][1];
    }
    if (t < group) {
        const long long out = (static_cast<long long>(r) * heads + g * group + t) * parts + part;
        parts_ms[out * 2] = m[t];
        parts_ms[out * 2 + 1] = l[t];
    }
}

extern "C" __global__ void __launch_bounds__(kSparseWaves * 32, 2)
tf_fn_sparse_lanes6_f16(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                          const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads,
                          int kv_heads, int dims, long long cap, float scale, int parts) {
    sparse_lanes<6, 1>(q, k, v, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap, scale,
                           parts);
}

extern "C" __global__ void __launch_bounds__(kSparseWaves * 32, 2)
tf_fn_sparse_lanes6_bf16(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                          const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads,
                          int kv_heads, int dims, long long cap, float scale, int parts) {
    sparse_lanes<6, 2>(q, k, v, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap, scale,
                           parts);
}

extern "C" __global__ void __launch_bounds__(kSparseWaves * 32, 2)
tf_fn_sparse_lanes12_f16(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                          const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads,
                          int kv_heads, int dims, long long cap, float scale, int parts) {
    sparse_lanes<12, 1>(q, k, v, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap, scale,
                           parts);
}

extern "C" __global__ void __launch_bounds__(kSparseWaves * 32, 2)
tf_fn_sparse_lanes12_bf16(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                          const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads,
                          int kv_heads, int dims, long long cap, float scale, int parts) {
    sparse_lanes<12, 2>(q, k, v, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap, scale,
                           parts);
}

}  // namespace

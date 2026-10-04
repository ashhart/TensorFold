#include <cuda_runtime.h>
#include <stdint.h>

#include "argmax_order.h"

// Raw encodings preserve subnormals and select the first NaN or first maximum.
__device__ __forceinline__ TFMaxCandidate tf_max_warp(TFMaxCandidate candidate) {
    for (int offset = 16; offset != 0; offset >>= 1) {
        TFMaxCandidate other;
        other.key = __shfl_down_sync(0xffffffffu, candidate.key, offset);
        other.nan = __shfl_down_sync(0xffffffffu, candidate.nan, offset);
        const uint32_t lo = __shfl_down_sync(0xffffffffu, uint32_t(candidate.column), offset);
        const uint32_t hi = __shfl_down_sync(0xffffffffu, uint32_t(candidate.column >> 32), offset);
        other.column = (uint64_t(hi) << 32) | lo;
        candidate = tf_max_pick(candidate, other);
    }
    return candidate;
}

// Eight BF16 columns from one 16-byte word, the low half of each 32-bit lane first.
__device__ __forceinline__ TFMaxCandidate tf_max_bf16x8(TFMaxCandidate candidate, uint4 q, uint64_t column) {
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        candidate = tf_max_pick(candidate, tf_max_bits(words[j] << 16, column + 2 * j));
        candidate = tf_max_pick(candidate, tf_max_bits(words[j] & 0xffff0000u, column + 2 * j + 1));
    }
    return candidate;
}

// Any block of whole warps; an aligned BF16 row is read 16 bytes a load, four loads in flight a thread.
__device__ uint64_t tf_row_max_column(const void* input, int64_t row, int64_t vocab,
                                      int64_t stride, int32_t dtype) {
    TFMaxCandidate candidate = tf_max_empty();
    const int64_t step = int64_t(blockDim.x);
    const uint16_t* half = static_cast<const uint16_t*>(input) + row * stride;
    if (dtype == 0 && vocab % 8 == 0 && (reinterpret_cast<uintptr_t>(half) & 15u) == 0) {
        const uint4* vectors = reinterpret_cast<const uint4*>(half);
        const int64_t count = vocab / 8;
        int64_t v = int64_t(threadIdx.x);
        for (; v + 3 * step < count; v += 4 * step) {
            uint4 q[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) q[u] = vectors[v + u * step];
#pragma unroll
            for (int u = 0; u < 4; ++u) candidate = tf_max_bf16x8(candidate, q[u], uint64_t(v + u * step) * 8u);
        }
        for (; v < count; v += step) candidate = tf_max_bf16x8(candidate, vectors[v], uint64_t(v) * 8u);
    } else {
        for (int64_t column = int64_t(threadIdx.x); column < vocab;) {
            const int64_t at = row * stride + column;
            const uint32_t bits = dtype == 0 ? uint32_t(static_cast<const uint16_t*>(input)[at]) << 16
                                             : static_cast<const uint32_t*>(input)[at];
            candidate = tf_max_pick(candidate, tf_max_bits(bits, uint64_t(column)));
            if (vocab - column <= step) break;
            column += step;
        }
    }
    candidate = tf_max_warp(candidate);
    __shared__ TFMaxCandidate partials[32];
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    if (lane == 0) partials[warp] = candidate;
    __syncthreads();
    if (warp == 0) {
        candidate = lane < int(blockDim.x >> 5) ? partials[lane] : tf_max_empty();
        candidate = tf_max_warp(candidate);
    }
    return candidate.column;
}

extern "C" __global__ __launch_bounds__(1024) void tf_argmax_rows_kernel(
        const void* input, int64_t* ids, int64_t rows, int64_t vocab, int64_t stride, int32_t dtype) {
    const int64_t row = int64_t(blockIdx.x);
    if (row >= rows) return;
    if (vocab < 1 || stride < vocab || (dtype != 0 && dtype != 1)) {
        if (threadIdx.x == 0) ids[row] = -1;
        return;
    }
    const uint64_t column = tf_row_max_column(input, row, vocab, stride, dtype);
    if (threadIdx.x == 0) ids[row] = int64_t(column);
}

extern "C" __global__ __launch_bounds__(1024) void tf_argmax_rows_i32_kernel(
        const void* input, int32_t* ids, int64_t rows, int64_t vocab, int64_t stride, int32_t dtype) {
    const int64_t row = int64_t(blockIdx.x);
    if (row >= rows) return;
    if (vocab < 1 || stride < vocab || (dtype != 0 && dtype != 1)) {
        if (threadIdx.x == 0) ids[row] = -1;
        return;
    }
    const uint64_t column = tf_row_max_column(input, row, vocab, stride, dtype);
    if (threadIdx.x == 0) ids[row] = int32_t(column);
}

extern "C" int tf_argmax_rows(const void* input, int64_t* ids, int64_t rows, int64_t vocab,
                               int64_t stride, int32_t dtype, cudaStream_t stream) {
    if (!tf_argmax_dimensions(rows, vocab, stride, dtype)) return int(cudaErrorInvalidValue);
    if (rows == 0) return int(cudaSuccess);
    const uint64_t bytes = dtype == 0 ? 2u : 4u;
    const uint64_t elements = uint64_t(rows - 1) * uint64_t(stride) + uint64_t(vocab);
    if (elements > SIZE_MAX / bytes || !input || !ids || uintptr_t(input) % bytes || uintptr_t(ids) % 8u)
        return int(cudaErrorInvalidValue);
    if (!tf_argmax_disjoint(uintptr_t(input), uintptr_t(ids), elements * bytes, uint64_t(rows) * 8u))
        return int(cudaErrorInvalidValue);
    tf_argmax_rows_kernel<<<unsigned(rows), 256, 0, stream>>>(input, ids, rows, vocab, stride, dtype);
    return int(cudaGetLastError());
}


extern "C" int tf_argmax_rows_i32(const void* input, int32_t* ids, int64_t rows, int64_t vocab,
                                  int64_t stride, int32_t dtype, cudaStream_t stream) {
    if (!tf_argmax_dimensions(rows, vocab, stride, dtype) || vocab > int64_t(INT32_MAX) + 1)
        return int(cudaErrorInvalidValue);
    if (rows == 0) return int(cudaSuccess);
    const uint64_t bytes = dtype == 0 ? 2u : 4u;
    const uint64_t elements = uint64_t(rows - 1) * uint64_t(stride) + uint64_t(vocab);
    if (elements > SIZE_MAX / bytes || !input || !ids || uintptr_t(input) % bytes || uintptr_t(ids) % 4u)
        return int(cudaErrorInvalidValue);
    if (!tf_argmax_disjoint(uintptr_t(input), uintptr_t(ids), elements * bytes, uint64_t(rows) * 4u))
        return int(cudaErrorInvalidValue);
    tf_argmax_rows_i32_kernel<<<unsigned(rows), 256, 0, stream>>>(input, ids, rows, vocab, stride, dtype);
    return int(cudaGetLastError());
}

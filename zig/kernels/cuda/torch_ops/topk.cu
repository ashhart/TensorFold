#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

namespace {

constexpr uint32_t topk_threads = 256;
constexpr uint32_t topk_tile = 4096;

__device__ __forceinline__ uint32_t selection_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return UINT32_MAX;
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}

__device__ __forceinline__ float input_value(const float* input, uint64_t row, uint64_t column,
                                          uint64_t row_bytes, uint64_t column_bytes) {
    const char* address = reinterpret_cast<const char*>(input) + row * row_bytes + column * column_bytes;
    return *reinterpret_cast<const float*>(address);
}

struct Workspace {
    uint32_t* histogram;
    uint32_t* threshold;
    uint32_t* remaining;
    uint32_t* greater_counts;
    uint32_t* equal_counts;
    uint32_t* greater_prefix;
    uint32_t* equal_prefix;
    uint32_t* greater_total;
};

uint64_t workspace_words(uint64_t rows, uint64_t tiles) {
    const uint64_t tile_words = tiles * 260u;
    if (tile_words > UINT64_MAX - 3u || rows > UINT64_MAX / (tile_words + 3u)) return 0;
    return rows * (tile_words + 3u);
}

Workspace workspace(void* scratch, uint64_t rows, uint64_t tiles) {
    uint32_t* cursor = static_cast<uint32_t*>(scratch);
    Workspace result{};
    result.histogram = cursor;
    cursor += rows * tiles * 256u;
    result.threshold = cursor;
    cursor += rows;
    result.remaining = cursor;
    cursor += rows;
    result.greater_counts = cursor;
    cursor += rows * tiles;
    result.equal_counts = cursor;
    cursor += rows * tiles;
    result.greater_prefix = cursor;
    cursor += rows * tiles;
    result.equal_prefix = cursor;
    cursor += rows * tiles;
    result.greater_total = cursor;
    return result;
}

}

extern "C" __global__ void tf_topk_f32_histogram_kernel(
    const float* input, uint32_t* histogram, const uint32_t* threshold,
    uint64_t columns, uint64_t tiles, uint64_t row_bytes, uint64_t column_bytes, uint32_t shift) {
    __shared__ uint32_t counts[256];
    const uint64_t row = blockIdx.y;
    const uint64_t tile = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    counts[lane] = 0;
    __syncthreads();
    const uint32_t mask = shift == 24 ? 0u : UINT32_MAX << (shift + 8u);
    const uint32_t wanted = shift == 24 ? 0u : threshold[row];
    const uint64_t begin = tile * topk_tile;
    const uint64_t limit = begin + topk_tile < columns ? begin + topk_tile : columns;
    for (uint64_t column = begin + lane; column < limit; column += topk_threads) {
        const uint32_t key = selection_key(input_value(input, row, column, row_bytes, column_bytes));
        if ((key & mask) == wanted) atomicAdd(counts + ((key >> shift) & 255u), 1u);
    }
    __syncthreads();
    histogram[(row * tiles + tile) * 256u + lane] = counts[lane];
}

extern "C" __global__ void tf_topk_f32_choose_digit_kernel(
    const uint32_t* histogram, uint32_t* threshold, uint32_t* remaining,
    uint64_t tiles, uint32_t k, uint32_t shift) {
    __shared__ uint32_t totals[256];
    const uint64_t row = blockIdx.x;
    const uint32_t digit = threadIdx.x;
    uint32_t count = 0;
    for (uint64_t tile = 0; tile < tiles; ++tile) count += histogram[(row * tiles + tile) * 256u + digit];
    totals[digit] = count;
    __syncthreads();
    if (digit == 0) {
        uint32_t rank = shift == 24 ? k : remaining[row];
        uint32_t prefix = shift == 24 ? 0u : threshold[row];
        for (int d = 255; d >= 0; --d) {
            if (rank <= totals[d]) {
                threshold[row] = prefix | (uint32_t(d) << shift);
                remaining[row] = rank;
                break;
            }
            rank -= totals[d];
        }
    }
}

extern "C" __global__ void tf_topk_f32_count_kernel(
    const float* input, const uint32_t* threshold, uint32_t* greater_counts, uint32_t* equal_counts,
    uint64_t columns, uint64_t tiles, uint64_t row_bytes, uint64_t column_bytes) {
    __shared__ uint32_t greater[256];
    __shared__ uint32_t equal[256];
    const uint64_t row = blockIdx.y;
    const uint64_t tile = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t selected = threshold[row];
    const uint64_t begin = tile * topk_tile;
    const uint64_t limit = begin + topk_tile < columns ? begin + topk_tile : columns;
    uint32_t above = 0, tied = 0;
    for (uint64_t column = begin + lane; column < limit; column += topk_threads) {
        const uint32_t key = selection_key(input_value(input, row, column, row_bytes, column_bytes));
        above += key > selected;
        tied += key == selected;
    }
    greater[lane] = above;
    equal[lane] = tied;
    __syncthreads();
    for (uint32_t distance = 128; distance != 0; distance >>= 1) {
        if (lane < distance) {
            greater[lane] += greater[lane + distance];
            equal[lane] += equal[lane + distance];
        }
        __syncthreads();
    }
    if (lane == 0) {
        greater_counts[row * tiles + tile] = greater[0];
        equal_counts[row * tiles + tile] = equal[0];
    }
}

extern "C" __global__ void tf_topk_f32_prefix_kernel(
    const uint32_t* greater_counts, const uint32_t* equal_counts, uint32_t* greater_prefix,
    uint32_t* equal_prefix, uint32_t* greater_total, uint64_t tiles) {
    const uint64_t row = blockIdx.x;
    if (threadIdx.x != 0) return;
    uint32_t above = 0, tied = 0;
    for (uint64_t tile = 0; tile < tiles; ++tile) {
        const uint64_t at = row * tiles + tile;
        greater_prefix[at] = above;
        equal_prefix[at] = tied;
        above += greater_counts[at];
        tied += equal_counts[at];
    }
    greater_total[row] = above;
}

extern "C" __global__ void tf_topk_f32_compact_kernel(
    const float* input, float* values, int64_t* indices, const uint32_t* threshold,
    const uint32_t* greater_prefix, const uint32_t* equal_prefix, const uint32_t* greater_total,
    uint64_t columns, uint64_t tiles, uint64_t row_bytes, uint64_t column_bytes, uint32_t k) {
    __shared__ uint32_t warp_greater[8];
    __shared__ uint32_t warp_equal[8];
    const uint64_t row = blockIdx.y;
    const uint64_t tile = blockIdx.x;
    const uint32_t thread = threadIdx.x;
    const uint32_t lane = thread & 31u;
    const uint32_t warp = thread >> 5u;
    const uint32_t preceding = lane == 0 ? 0u : (1u << lane) - 1u;
    const uint32_t selected = threshold[row];
    uint32_t above_start = greater_prefix[row * tiles + tile];
    uint32_t tied_start = greater_total[row] + equal_prefix[row * tiles + tile];
    const uint64_t begin = tile * topk_tile;
    for (uint32_t stripe = 0; stripe < topk_tile; stripe += topk_threads) {
        const uint64_t column = begin + stripe + thread;
        const bool valid = column < columns;
        const float value = valid ? input_value(input, row, column, row_bytes, column_bytes) : 0.0f;
        const uint32_t key = valid ? selection_key(value) : 0u;
        const bool above = valid && key > selected;
        const bool tied = valid && key == selected;
        const uint32_t above_mask = __ballot_sync(UINT32_MAX, above);
        const uint32_t tied_mask = __ballot_sync(UINT32_MAX, tied);
        if (lane == 0) {
            warp_greater[warp] = __popc(above_mask);
            warp_equal[warp] = __popc(tied_mask);
        }
        __syncthreads();
        uint32_t above_at = above_start + __popc(above_mask & preceding);
        uint32_t tied_at = tied_start + __popc(tied_mask & preceding);
        uint32_t above_count = 0, tied_count = 0;
        for (uint32_t w = 0; w < 8; ++w) {
            if (w < warp) {
                above_at += warp_greater[w];
                tied_at += warp_equal[w];
            }
            above_count += warp_greater[w];
            tied_count += warp_equal[w];
        }
        if (above) {
            values[row * k + above_at] = value;
            indices[row * k + above_at] = int64_t(column);
        }
        if (tied && tied_at < k) {
            values[row * k + tied_at] = value;
            indices[row * k + tied_at] = int64_t(column);
        }
        above_start += above_count;
        tied_start += tied_count;
        __syncthreads();
    }
}

extern "C" uint64_t tf_topk_f32_unsorted_scratch_bytes(uint64_t rows, uint64_t columns) {
    if (rows == 0 || columns == 0 || rows > 65535u || columns > UINT32_MAX) return 0;
    const uint64_t tiles = columns / topk_tile + (columns % topk_tile != 0);
    const uint64_t words = workspace_words(rows, tiles);
    return words == 0 || words > UINT64_MAX / sizeof(uint32_t) ? 0 : words * sizeof(uint32_t);
}

extern "C" cudaError_t tf_topk_f32_unsorted(
    const float* input, float* values, int64_t* indices, uint64_t rows, uint64_t columns, uint64_t k,
    uint64_t row_bytes, uint64_t column_bytes, void* scratch, uint64_t scratch_bytes, cudaStream_t stream) {
    if (k > columns || rows > 65535u || columns > UINT32_MAX) return cudaErrorInvalidValue;
    if (rows == 0 || k == 0) return cudaSuccess;
    const uint64_t needed = tf_topk_f32_unsorted_scratch_bytes(rows, columns);
    if (!input || !values || !indices || !scratch || needed == 0 || scratch_bytes < needed ||
        column_bytes < sizeof(float) || row_bytes % sizeof(float) || column_bytes % sizeof(float) ||
        uintptr_t(input) % alignof(float) || uintptr_t(values) % alignof(float) ||
        uintptr_t(indices) % alignof(int64_t) || uintptr_t(scratch) % alignof(uint32_t) ||
        columns - 1 > (UINT64_MAX - sizeof(float)) / column_bytes ||
        (rows - 1 && row_bytes > (UINT64_MAX - sizeof(float) - (columns - 1) * column_bytes) / (rows - 1)))
        return cudaErrorInvalidValue;
    const uint64_t tiles = columns / topk_tile + (columns % topk_tile != 0);
    Workspace w = workspace(scratch, rows, tiles);
    const dim3 tile_grid{uint32_t(tiles), uint32_t(rows)};
    cudaError_t status;
    for (int shift = 24; shift >= 0; shift -= 8) {
        tf_topk_f32_histogram_kernel<<<tile_grid, topk_threads, 0, stream>>>(
            input, w.histogram, w.threshold, columns, tiles, row_bytes, column_bytes, uint32_t(shift));
        if ((status = cudaGetLastError()) != cudaSuccess) return status;
        tf_topk_f32_choose_digit_kernel<<<uint32_t(rows), topk_threads, 0, stream>>>(
            w.histogram, w.threshold, w.remaining, tiles, uint32_t(k), uint32_t(shift));
        if ((status = cudaGetLastError()) != cudaSuccess) return status;
    }
    tf_topk_f32_count_kernel<<<tile_grid, topk_threads, 0, stream>>>(
        input, w.threshold, w.greater_counts, w.equal_counts, columns, tiles, row_bytes, column_bytes);
    if ((status = cudaGetLastError()) != cudaSuccess) return status;
    tf_topk_f32_prefix_kernel<<<uint32_t(rows), 1, 0, stream>>>(
        w.greater_counts, w.equal_counts, w.greater_prefix, w.equal_prefix, w.greater_total, tiles);
    if ((status = cudaGetLastError()) != cudaSuccess) return status;
    tf_topk_f32_compact_kernel<<<tile_grid, topk_threads, 0, stream>>>(
        input, values, indices, w.threshold, w.greater_prefix, w.equal_prefix, w.greater_total,
        columns, tiles, row_bytes, column_bytes, uint32_t(k));
    return cudaGetLastError();
}

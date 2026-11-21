// Hy4-flash's 192-wide query and 256-wide value projections on the shared MLA tiles.
#include <cuda_bf16.h>
#include <stdint.h>
#include "../common/mla.cuh"
#include "../common/reduce.cuh"

struct Hy4RopeParameters { float frequencies[32]; uint32_t axes[32]; };
struct Hy4RopeView { uint32_t rows, heads, input_stride, output_stride; };
struct Hy4CacheWrite { uint32_t rows, capacity; float epsilon; };

// A request's write slots address distinct live rows in the retained latent arena.
extern "C" __global__ void __launch_bounds__(128) tf_hy4_cache_keys(
    const __nv_bfloat16 *input, const __nv_bfloat16 *gain, const __nv_bfloat16 *table, const int32_t *slots,
    __nv_bfloat16 *cache, Hy4CacheWrite a) {
    const uint32_t row = blockIdx.x, t = threadIdx.x;
    const int slot = slots[row];
    if (slot < 0 || uint32_t(slot) >= a.capacity) return;
    const __nv_bfloat16 *x = input + uint64_t(row) * 576;
    __nv_bfloat16 *y = cache + uint64_t(slot) * 576;
    float v[4], sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __bfloat162float(x[t + i * 128]);
        sq = __fmaf_rn(v[i], v[i], sq);
    }
    __shared__ float partial[4];
    const float inverse = rsqrtf(tf_common::block_sum_tree(sq, partial, 4) / 512.0f + a.epsilon);
    for (int i = 0; i < 4; ++i) {
        const uint32_t col = t + i * 128;
        const float normalized = __bfloat162float(__float2bfloat16_rn(v[i] * inverse));
        y[col] = __float2bfloat16_rn(normalized * __bfloat162float(gain[col]));
    }
    if (t < 32) {
        const uint32_t col = 2 * t;
        const float u = __bfloat162float(x[512 + col]), v1 = __bfloat162float(x[513 + col]);
        const float c = __bfloat162float(table[uint64_t(row) * 64 + col]);
        const float s = __bfloat162float(table[uint64_t(row) * 64 + col + 1]);
        y[512 + col] = __float2bfloat16_rn(u * c - v1 * s);
        y[513 + col] = __float2bfloat16_rn(v1 * c + u * s);
    }
}

// Per-value gate on the expanded heads. Sigmoid rounds to BF16 before the product.
extern "C" __global__ void __launch_bounds__(128) tf_hy4_gate_values(
    __nv_bfloat16 *values, const __nv_bfloat16 *logits, uint32_t columns) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= columns) return;
    const uint64_t i = uint64_t(blockIdx.y) * columns + col;
    const float gate = 1.0f / (1.0f + expf(-__bfloat162float(logits[i])));
    values[i] = __float2bfloat16_rn(__bfloat162float(values[i]) * __bfloat162float(__float2bfloat16_rn(gate)));
}

extern "C" __global__ void __launch_bounds__(32) tf_hy4_rotary_table(
    const int32_t *positions, __nv_bfloat16 *table, Hy4RopeParameters p) {
    const uint32_t pair = threadIdx.x, row = blockIdx.x;
    const float phase = float(positions[uint64_t(row) * 3 + p.axes[pair]]) * p.frequencies[pair];
    float s, c;
    sincosf(phase, &s, &c);
    table[uint64_t(row) * 64 + 2 * pair] = __float2bfloat16_rn(c);
    table[uint64_t(row) * 64 + 2 * pair + 1] = __float2bfloat16_rn(s);
}

extern "C" __global__ void __launch_bounds__(32) tf_hy4_rotate(
    const __nv_bfloat16 *input, const __nv_bfloat16 *table, __nv_bfloat16 *output, Hy4RopeView v) {
    const uint32_t row = blockIdx.x, head = blockIdx.y, col = 2 * threadIdx.x;
    const uint64_t row_head = uint64_t(row) * v.heads + head;
    const uint64_t i = row_head * v.input_stride + col, o = row_head * v.output_stride + col;
    const uint64_t base = uint64_t(row) * 64 + col;
    const float a = __bfloat162float(input[i]), b = __bfloat162float(input[i + 1]);
    const float c = __bfloat162float(table[base]), s = __bfloat162float(table[base + 1]);
    output[o] = __float2bfloat16_rn(a * c - b * s);
    output[o + 1] = __float2bfloat16_rn(b * c + a * s);
}

extern "C" __global__ void __launch_bounds__(128)
tf_hy4_absorb_q(const __nv_bfloat16 *q, const __nv_bfloat16 *wk, __nv_bfloat16 *out,
                int64_t rows, int64_t heads, int64_t rows_per_block) {
    tf_mla::absorb_q<192>(q, wk, out, rows, heads, rows_per_block, 256, 576);
}

extern "C" __global__ void __launch_bounds__(128)
tf_hy4_expand_v(const __nv_bfloat16 *o, const __nv_bfloat16 *wv, __nv_bfloat16 *out,
                int64_t rows, int64_t heads, int64_t rows_per_block) {
    tf_mla::expand_v<256>(o, wv, out, rows, heads, rows_per_block);
}

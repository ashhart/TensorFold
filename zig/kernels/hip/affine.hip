// Four-bit BF16 row projection with FP32 sums and explicit BF16 RNE output.
// Compile with -ffp-contract=off. Each row uses the same reduction regardless of M.
#include <hip/hip_runtime.h>
#include <stdint.h>

// Bounded independent-stream test load; writes only its own scratch and progress.
extern "C" __global__ void affine_test_load(float* scratch, unsigned* progress) {
    const unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0001f * (at + 1);
    for (unsigned i = 0; i < 200000; ++i)
        value = value * 0.999999f + 0.000001f;
    scratch[at] = value;
    atomicAdd(progress, 1U);
}

__device__ static float from_bf16(uint16_t x) {
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
}

__device__ static uint16_t to_bf16(float x) {
    const uint32_t u = __float_as_uint(x);
    // Preserve NaNs rather than rounding a small NaN payload to infinity.
    if ((u & 0x7fffffffU) > 0x7f800000U)
        return static_cast<uint16_t>((u >> 16) | 0x40U);
    return static_cast<uint16_t>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}

__device__ static uint32_t unpack(const uint32_t* words, size_t index, unsigned bits) {
    const size_t bit = index * bits;
    const unsigned shift = bit % 32;
    uint32_t value = words[bit / 32] >> shift;
    if (shift + bits > 32) value |= words[bit / 32 + 1] << (32 - shift);
    return value & ((1U << bits) - 1U);
}

// core/row_projection Sum.f32 at upstream 7ae6df7: 16 inputs per lane,
// four-product chains, scale/bias applied per lane, then a 32-lane reduction.
// Reduction-order equivalence to Metal must be measured, not assumed.
extern "C" __global__ void affine_row4_f32(
    const uint16_t* x, const uint32_t* words, const uint16_t* scales,
    const uint16_t* biases, uint16_t* out,
    size_t m, size_t n, size_t k, unsigned bits, unsigned group) {
    const unsigned lane = threadIdx.x & 31;
    const size_t col = static_cast<size_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const size_t row = blockIdx.y;
    if (row >= m || col >= n || bits != 4 || (group != 32 && group != 64)) return;
    float acc = 0.0f;
    const uint32_t* w = words + col * (k / 8);
    for (size_t at = lane * 16; at < k; at += 512) {
        float dot = 0.0f, sum = 0.0f;
        for (unsigned j = 0; j < 16; j += 4) {
            float product[4];
            for (unsigned h = 0; h < 4; ++h) {
                const float v = from_bf16(x[row * k + at + j + h]);
                sum = sum + v;
                product[h] = v * static_cast<float>(unpack(w, at + j + h, 4));
            }
            dot = dot + (((product[0] + product[1]) + product[2]) + product[3]);
        }
        const size_t g = col * (k / group) + at / group;
        acc = acc + (from_bf16(scales[g]) * dot + sum * from_bf16(biases[g]));
    }
    for (unsigned offset = 16; offset != 0; offset /= 2)
        acc = acc + __shfl_down(acc, offset, 32);
    if (lane == 0) out[row * n + col] = to_bf16(acc);
}

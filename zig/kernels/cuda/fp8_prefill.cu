// Independently implemented FP8 prompt projections over TensorFold's existing weight layouts.
#include "fp8_prefill_common.cuh"

namespace tf_fp8_fast {

// SATFINITE E4M3 conversion: finite overflow and infinity clamp, while NaN remains NaN.
__device__ __forceinline__ void quantize(const __nv_bfloat16* x, uint8_t* out,
                                        int m, int k, int ldx, float inverse) {
    const size_t group = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= static_cast<size_t>(m) * (k / 16)) return;
    const int row = group / (k / 16), column = (group % (k / 16)) * 16;
    const __nv_bfloat16* source = x + static_cast<size_t>(row) * ldx + column;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const float a = __bfloat162float(source[pair * 2]);
        const float b = __bfloat162float(source[pair * 2 + 1]);
        out[static_cast<size_t>(row) * k + column + input_byte(pair * 2)] =
            __nv_cvt_float_to_fp8(__fmul_rn(a, inverse), __NV_SATFINITE, __NV_E4M3);
        out[static_cast<size_t>(row) * k + column + input_byte(pair * 2 + 1)] =
            __nv_cvt_float_to_fp8(__fmul_rn(b, inverse), __NV_SATFINITE, __NV_E4M3);
    }
}

// Each warp quantizes a 64-value group and writes its scale and the sum of its represented E4M3 codes.
__device__ __forceinline__ void quantize_affine(const __nv_bfloat16* x, uint8_t* out,
    float* scales, float* sums, int m, int k, int ldx) {
    const int lane = threadIdx.x & 31;
    const size_t group = static_cast<size_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (group >= static_cast<size_t>(m) * (k / 64)) return;
    const int row = group / (k / 64), column = (group % (k / 64)) * 64;
    const __nv_bfloat16* source = x + static_cast<size_t>(row) * ldx + column + lane * 2;
    const float a = __bfloat162float(source[0]), b = __bfloat162float(source[1]);
    const bool bad = __ballot_sync(0xffffffffu, !isfinite(a) || !isfinite(b)) != 0;
    float maximum = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
    for (int step = 16; step > 0; step >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, step));
    const float scale = bad ? nanf("") : __fdiv_rn(maximum, 448.0f);
    const uint8_t qa = bad ? 0x7f : __nv_cvt_float_to_fp8(scale == 0 ? a : __fdiv_rn(a, scale), __NV_SATFINITE, __NV_E4M3);
    const uint8_t qb = bad ? 0x7f : __nv_cvt_float_to_fp8(scale == 0 ? b : __fdiv_rn(b, scale), __NV_SATFINITE, __NV_E4M3);
    const int at = column + input_byte(lane * 2);
    *reinterpret_cast<uint16_t*>(out + static_cast<size_t>(row) * k + at) = uint16_t(qa) | (uint16_t(qb) << 8);
    float sum = __fadd_rn(decode(qa), decode(qb));
#pragma unroll
    for (int step = 16; step > 0; step >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, step));
    if (lane == 0) { scales[group] = scale; sums[group] = sum; }
}

// One block a row; any nonfinite input gives that whole row a NaN scale and NaN codes.
__device__ __forceinline__ void quantize_rows(const __nv_bfloat16* x, uint8_t* out,
    float* row_scales, int m, int k, int ldx) {
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (row >= m) return;
    __shared__ float maxima[8];
    __shared__ uint32_t invalid[8];
    const __nv_bfloat16* source = x + static_cast<size_t>(row) * ldx;
    float maximum = 0;
    uint32_t bad = 0;
    for (int column = tid; column < k; column += blockDim.x) {
        const float value = __bfloat162float(source[column]);
        bad |= !isfinite(value);
        maximum = fmaxf(maximum, fabsf(value));
    }
#pragma unroll
    for (int step = 16; step > 0; step >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, step));
        bad |= __shfl_xor_sync(0xffffffffu, bad, step);
    }
    if (lane == 0) { maxima[warp] = maximum; invalid[warp] = bad; }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < 8 ? maxima[lane] : 0;
        bad = lane < 8 ? invalid[lane] : 0;
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, step));
            bad |= __shfl_xor_sync(0xffffffffu, bad, step);
        }
        if (lane == 0) {
            maxima[0] = bad ? nanf("") : __fdiv_rn(maximum, 448.0f);
            invalid[0] = bad;
            row_scales[row] = maxima[0];
        }
    }
    __syncthreads();
    const float scale = maxima[0];
    const bool invalid_row = invalid[0] != 0;
    for (int column = tid; column < k; column += blockDim.x) {
        const float value = __bfloat162float(source[column]);
        out[static_cast<size_t>(row) * k + input_byte(column)] = invalid_row ? 0x7f :
            __nv_cvt_float_to_fp8(scale == 0 ? value : __fdiv_rn(value, scale), __NV_SATFINITE, __NV_E4M3);
    }
}

template<bool F32, bool RowScaled = false>
__device__ __forceinline__ void direct(const uint8_t* x, const uint8_t* weight, void* output,
    int m, int n, int k, int npad, float alpha, const float* row_scales = nullptr,
    const float* weight_scales = nullptr) {
    constexpr int BM = 64, BN = 128, BK = 128, WM = 2, WN = 4, MT = 2, NT = 4, THREADS = 256;
    struct Shared {
        alignas(16) uint8_t a[2][BM * BK];
        alignas(16) uint8_t b[2][BN * BK];
    };
    __shared__ Shared shared;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row_tiles = (m + BM - 1) / BM;
    const int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int m0 = (tile_id % row_tiles) * BM, n0 = (tile_id / row_tiles) * BN;
    const int kgroups = k / 64, tiles = (k + BK - 1) / BK;
    auto load = [&](int stage, int tile) {
        for (int task = tid; task < BM * (BK / 16); task += THREADS) {
            const int row = task / (BK / 16), column = task % (BK / 16);
            const int source_k = tile * BK + column * 16;
            const bool valid = m0 + row < m && source_k < k;
            qmm_frag::cp16z(shared.a[stage] + row * BK + chunk(row, column) * 16,
                x + static_cast<size_t>(valid ? m0 + row : 0) * k + (valid ? source_k : 0), valid);
        }
        for (int task = tid; task < BN * BK / 16; task += THREADS) {
            const int square = task / 256, within = task % 256;
            const int half = square / 2, ntile = square % 2;
            const int column = n0 + ntile * 64, kg = tile * 2 + half;
            const bool valid = column < npad && kg < kgroups;
            const size_t index = (static_cast<size_t>(valid ? column / 64 : 0) * kgroups + (valid ? kg : 0)) * 4096 + within * 16;
            qmm_frag::cp16z(shared.b[stage] + task * 16, weight + index, valid);
        }
    };
    float sums[MT][NT][4] = {};
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
        if (stage < tiles) load(stage, stage);
        qmm_frag::commit();
    }
    for (int tile = 0; tile < tiles; ++tile) {
        qmm_frag::wait<1>(); __syncthreads();
        const int stage = tile & 1;
#pragma unroll
        for (int step = 0; step < BK / 32; ++step) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int row = (warp / WN) * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                qmm_frag::ldmatrix4(a[i], shared.a[stage] + row * BK + chunk(row, step * 2 + (lane >> 4)) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int ntile8 = (warp % WN) * NT + j;
                const int square = (step / 2) * 2 + ntile8 / 8;
                const uint2 b = *reinterpret_cast<const uint2*>(shared.b[stage] + square * 4096 + (ntile8 % 8) * 512 + lane * 16 + (step & 1) * 8);
#pragma unroll
                for (int i = 0; i < MT; ++i) mma(sums[i][j], a[i], b.x, b.y);
            }
        }
        __syncthreads();
        if (tile + 2 < tiles) load(stage, tile + 2);
        qmm_frag::commit();
    }
    qmm_frag::wait<0>();
    store_tile<BM, WN, MT, NT, F32, RowScaled>(sums, output, m0, n0, m, n, alpha,
                                             row_scales, weight_scales, k / 64);
}

template<bool F32>
__device__ __forceinline__ void affine(const uint8_t* x, const float* scales, const float* input_sum,
    const uint32_t* weight, const __nv_bfloat16* weight_scale, const __nv_bfloat16* bias,
    void* output, int m, int n, int k, int npad) {
    constexpr int BM = 64, BN = 128, BK = 64, WM = 2, WN = 4, MT = 2, NT = 4, THREADS = 256;
    struct Shared {
        alignas(16) uint8_t a[2][BM * BK];
        alignas(16) uint8_t b[2][BN * BK / 2];
        alignas(16) float as[2][BM], ax[2][BM];
        alignas(16) __nv_bfloat16 ws[2][BN], wb[2][BN];
    };
    __shared__ Shared shared;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row_tiles = (m + BM - 1) / BM;
    const int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int m0 = (tile_id % row_tiles) * BM, n0 = (tile_id / row_tiles) * BN, groups = k / BK;
    auto load = [&](int stage, int group) {
        for (int task = tid; task < BM * 4; task += THREADS) {
            const int row = task / 4, column = task % 4;
            const bool valid = m0 + row < m;
            qmm_frag::cp16z(shared.a[stage] + row * BK + chunk(row, column) * 16,
                x + static_cast<size_t>(valid ? m0 + row : 0) * k + group * BK + column * 16, valid);
        }
        for (int task = tid; task < BN * BK / 32; task += THREADS) {
            const int ntile = task / 128, within = task % 128, column = n0 + ntile * 64;
            const bool valid = column < npad;
            const size_t index = (static_cast<size_t>(valid ? column / 64 : 0) * groups + group) * 2048 + within * 16;
            qmm_frag::cp16z(shared.b[stage] + task * 16, reinterpret_cast<const uint8_t*>(weight) + index, valid);
        }
        for (int row = tid; row < BM; row += THREADS) {
            const bool valid = m0 + row < m;
            shared.as[stage][row] = valid ? scales[static_cast<size_t>(m0 + row) * groups + group] : 0;
            shared.ax[stage][row] = valid ? input_sum[static_cast<size_t>(m0 + row) * groups + group] : 0;
        }
        for (int column = tid; column < BN; column += THREADS) {
            const bool valid = n0 + column < npad;
            shared.ws[stage][column] = valid ? weight_scale[static_cast<size_t>(group) * npad + n0 + column] : __float2bfloat16_rn(0);
            shared.wb[stage][column] = valid ? bias[static_cast<size_t>(group) * npad + n0 + column] : __float2bfloat16_rn(0);
        }
    };
    float sums[MT][NT][4] = {};
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
        if (stage < groups) load(stage, stage);
        qmm_frag::commit();
    }
    for (int group = 0; group < groups; ++group) {
        qmm_frag::wait<1>(); __syncthreads();
        const int stage = group & 1;
        float partial[MT][NT][4] = {};
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int row = (warp / WN) * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                qmm_frag::ldmatrix4(a[i], shared.a[stage] + row * BK + chunk(row, step * 2 + (lane >> 4)) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int ntile8 = (warp % WN) * NT + j;
                const uint2 packed = reinterpret_cast<const uint2*>(shared.b[stage])[ntile8 * 32 + lane];
                const uint32_t word = step ? packed.y : packed.x;
                const uint32_t b0 = affine_fragment(word, 0), b1 = affine_fragment(word, 1);
#pragma unroll
                for (int i = 0; i < MT; ++i) mma(partial[i][j], a[i], b0, b1);
            }
        }
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = (warp / WN) * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                    const int column = (warp % WN) * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
                    for (int pair = 0; pair < 2; ++pair) {
                        const float ws = __bfloat162float(shared.ws[stage][column + pair]);
                        const float wb = __bfloat162float(shared.wb[stage][column + pair]);
                        const float value = __fadd_rn(__fmul_rn(ws, partial[i][j][h * 2 + pair]),
                                                     __fmul_rn(wb, shared.ax[stage][row]));
                        sums[i][j][h * 2 + pair] = __fadd_rn(sums[i][j][h * 2 + pair], __fmul_rn(shared.as[stage][row], value));
                    }
                }
        __syncthreads();
        if (group + 2 < groups) load(stage, group + 2);
        qmm_frag::commit();
    }
    qmm_frag::wait<0>();
    store_tile<BM, WN, MT, NT, F32>(sums, output, m0, n0, m, n, 1.0f);
}
}

extern "C" __global__ __launch_bounds__(256) void tf_fp8_prompt_quant(
    const __nv_bfloat16* x, uint8_t* out, int m, int k, int ldx, float inverse) {
    tf_fp8_fast::quantize(x, out, m, k, ldx, inverse);
}

extern "C" __global__ __launch_bounds__(256) void tf_affine4_fp8_quant(
    const __nv_bfloat16* x, uint8_t* out, float* scales, float* sums, int m, int k, int ldx) {
    tf_fp8_fast::quantize_affine(x, out, scales, sums, m, k, ldx);
}

extern "C" __global__ __launch_bounds__(256) void tf_fp8_row_quant(
    const __nv_bfloat16* x, uint8_t* out, float* row_scales, int m, int k, int ldx) {
    tf_fp8_fast::quantize_rows(x, out, row_scales, m, k, ldx);
}

#define TF_FP8_PROMPT(NAME, F32) \
extern "C" __global__ __launch_bounds__(256) void NAME( \
    const uint8_t* x, const uint8_t* w, void* y, int m, int n, int k, int npad, float alpha) { \
    tf_fp8_fast::direct<F32>(x, w, y, m, n, k, npad, alpha); \
}
TF_FP8_PROMPT(tf_fp8_prompt, false)
TF_FP8_PROMPT(tf_fp8_prompt_f32, true)
#undef TF_FP8_PROMPT

#define TF_FP8_ROW_PROMPT(NAME, F32) \
extern "C" __global__ __launch_bounds__(256) void NAME( \
    const uint8_t* x, const float* row_scales, const uint8_t* w, const float* weight_scales, \
    void* y, int m, int n, int k, int npad) { \
    tf_fp8_fast::direct<F32, true>(x, w, y, m, n, k, npad, 1.0f, row_scales, weight_scales); \
}
TF_FP8_ROW_PROMPT(tf_fp8_row_prompt, false)
TF_FP8_ROW_PROMPT(tf_fp8_row_prompt_f32, true)
#undef TF_FP8_ROW_PROMPT

#define TF_AFFINE_FP8_PROMPT(NAME, F32) \
extern "C" __global__ __launch_bounds__(256) void NAME( \
    const uint8_t* x, const float* as, const float* ax, const uint32_t* w, \
    const __nv_bfloat16* ws, const __nv_bfloat16* wb, void* y, int m, int n, int k, int npad) { \
    tf_fp8_fast::affine<F32>(x, as, ax, w, ws, wb, y, m, n, k, npad); \
}
TF_AFFINE_FP8_PROMPT(tf_affine4_fp8_prompt, false)
TF_AFFINE_FP8_PROMPT(tf_affine4_fp8_prompt_f32, true)
#undef TF_AFFINE_FP8_PROMPT

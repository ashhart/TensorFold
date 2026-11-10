// Bounded experiment: one exact integer-code expansion per projection, with unchanged affine FP8 arithmetic.
#include "fp8_prefill_common.cuh"

namespace tf_affine_prepared {
using namespace tf_fp8_fast;

// Each thread expands one existing 16-value uint2 slot to its native FP8 uint4 slot.
__device__ __forceinline__ void prepare(const uint32_t* source, uint8_t* destination, int k) {
    const size_t slot = (size_t(blockIdx.y) * (k / 64) + blockIdx.x) * 256 + threadIdx.x;
    const uint2 packed = reinterpret_cast<const uint2*>(source)[slot];
    const uint4 codes = make_uint4(affine_fragment(packed.x, 0), affine_fragment(packed.x, 1),
                                  affine_fragment(packed.y, 0), affine_fragment(packed.y, 1));
    reinterpret_cast<uint4*>(destination)[slot] = codes;
}

template<bool F32>
__device__ __forceinline__ void product(const uint8_t* x, const float* scales, const float* input_sum,
    const uint8_t* weight, const __nv_bfloat16* weight_scale, const __nv_bfloat16* bias,
    void* output, int m, int n, int k, int npad) {
    constexpr int BM = 64, BN = 128, BK = 64, WM = 2, WN = 4, MT = 2, NT = 4, THREADS = 256;
    struct Shared {
        alignas(16) uint8_t a[2][BM * BK];
        alignas(16) uint8_t b[2][BN * BK];
        alignas(16) float as[2][BM], ax[2][BM];
        alignas(16) __nv_bfloat16 ws[2][BN], wb[2][BN];
    };
    static_assert(sizeof(Shared) == 26624);
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
        for (int task = tid; task < BN * BK / 16; task += THREADS) {
            const int ntile = task / 256, within = task % 256, column = n0 + ntile * 64;
            const bool valid = column < npad;
            const size_t index = (static_cast<size_t>(valid ? column / 64 : 0) * groups + group) * 4096 + within * 16;
            qmm_frag::cp16z(shared.b[stage] + task * 16, weight + index, valid);
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
        // The original two K32 MMAs, group scaling, additions and BF16 rounding are unchanged.
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
                const uint2 codes = *reinterpret_cast<const uint2*>(shared.b[stage] +
                    ntile8 * 512 + lane * 16 + step * 8);
                const uint32_t b0 = codes.x, b1 = codes.y;
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
} // namespace tf_affine_prepared

// Launch grid (K / 64, Npad / 64), block 256; destination requires exactly Npad*K bytes.
extern "C" __global__ __launch_bounds__(256) void tf_affine_fp8_prepare(
    const uint32_t* source, uint8_t* destination, int k) {
    tf_affine_prepared::prepare(source, destination, k);
}

#define TF_AFFINE_PREPARED(NAME, F32) \
extern "C" __global__ __launch_bounds__(256) void NAME( \
    const uint8_t* x, const float* as, const float* ax, const uint8_t* weight, \
    const __nv_bfloat16* ws, const __nv_bfloat16* wb, void* output, int m, int n, int k, int npad) { \
    tf_affine_prepared::product<F32>(x, as, ax, weight, ws, wb, output, m, n, k, npad); \
}
TF_AFFINE_PREPARED(tf_affine_fp8_prepared, false)
TF_AFFINE_PREPARED(tf_affine_fp8_prepared_f32, true)
#undef TF_AFFINE_PREPARED

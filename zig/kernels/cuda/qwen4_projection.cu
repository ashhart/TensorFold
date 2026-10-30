// BF16-activation projections over raw BF16, MLX4/8, FP8 and NVFP4 checkpoint layouts.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#define TF_INLINE __device__ __forceinline__
#define TF_DATA
#define TF_INDEX unsigned long long
#define TF_FROM_BITS __uint_as_float
#define TF_TO_BITS __float_as_uint
#define TF_LDEXP ldexpf
#define TF_FMA __fmaf_rn
#define TF_HALF(bits) __half2float(__ushort_as_half(bits))
#include "../metal/core/weight_only.h"

template<int CONTROL>
__device__ __forceinline__ void tf_wo_run(const unsigned short* x, const unsigned char* w,
    const unsigned char* s, const unsigned char* b, unsigned short* out, TfWeightOnly p, int repetition) {
    const int row = blockIdx.y, lane = threadIdx.x & 31;
    const int col = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (col >= p.n || row >= p.rows) return;
    float total = 0.0f;
    for (int first = 0; first < p.k; first += p.group) {
        float partial = tf_wo_partial(x, w, s, b, p, row, col, first, lane, CONTROL);
        for (int offset = 16; offset; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffffu, partial, offset);
            if (CONTROL == 4) partial = tf_wo_bf(tf_wo_round(partial));
        }
        total += partial;
        if (CONTROL == 4) total = tf_wo_bf(tf_wo_round(total));
    }
    if (CONTROL == 1 && isnan(total)) total = 0.0f;
    if (CONTROL == 2) total += float(repetition);
    if (lane == 0) out[(TF_INDEX)row * p.n + col] = tf_wo_round(total);
}
#define TF_WO_KERNEL(NAME, CONTROL) \
extern "C" __global__ void __launch_bounds__(128) NAME(const unsigned short* x, const unsigned char* w, \
const unsigned char* s, const unsigned char* b, unsigned short* out, TfWeightOnly p, int repetition) { \
    tf_wo_run<CONTROL>(x, w, s, b, out, p, repetition); \
}
TF_WO_KERNEL(tf_qwen4_projection, 0)
TF_WO_KERNEL(tf_qwen4_projection_bad_nan, 1)
TF_WO_KERNEL(tf_qwen4_projection_bad_repeat, 2)
TF_WO_KERNEL(tf_qwen4_projection_bad_tail, 3)
TF_WO_KERNEL(tf_qwen4_projection_bad_sums, 4)

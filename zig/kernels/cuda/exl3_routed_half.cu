// Routed FP16 checkpoint rows: round each scaled input to FP16 before the shared FP32 Hadamard transform.
#include <cuda_fp16.h>
#include "exl3_decode.cuh"

extern "C" __global__ __launch_bounds__(32) void tf_exl3_routed_half_rotate(
    const half* x, const int* picks, const half* suh, half* out, int width, int input_slots, int experts) {
    const int pair = blockIdx.x, block = blockIdx.y, lane = threadIdx.x;
    const int expert = picks[pair];
    if (expert < 0 || expert >= experts) return;
    const int row = input_slots ? pair / input_slots : pair;
    const int c = block * 128 + lane * 4;
    float v[4];
    for (int j = 0; j < 4; ++j)
        v[j] = __half2float(__hmul(x[(size_t)row * width + c + j], suh[(size_t)expert * width + c + j]));
    tf_exl3::fwht128(v, lane);
    for (int j = 0; j < 4; ++j) out[(size_t)pair * width + c + j] =
        __float2half_rn(v[j] * tf_exl3::HAD_SCALE);
}

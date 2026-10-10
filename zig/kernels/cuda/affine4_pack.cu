// The MLX affine-4 repack into the lane matmul's tiles (cuda/qlinear.zig), moved verbatim from nemotron_ops.cu.

#include <stdint.h>

// MLX (n, k/8) words -> the tiled lane-matmul layout [npad/64][kg][8][32][2] (qmm.py's pack, groups of 64).
extern "C" __global__ void tf_pack_dense(const uint32_t* __restrict__ w, int n, int k8, int kg, uint32_t* __restrict__ out,
                                         long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int v = idx & 1, lane = (idx >> 1) & 31, j = (idx >> 6) & 7;
    const long long tg = idx >> 9;
    const int g = static_cast<int>(tg % kg);
    const long long t = tg / kg;
    const long long row = t * 64 + j * 8 + (lane >> 2);
    const int c = lane & 3;
    const int offs[8] = {0, 8, 16, 24, 1, 9, 17, 25};
    uint32_t word = 0;
    if (row < n) {
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int input = g * 64 + 32 * v + 2 * c + offs[p];
            const uint32_t src = w[row * k8 + (input >> 3)];
            word |= ((src >> (4 * (input & 7))) & 0xFu) << (4 * p);
        }
    }
    out[idx] = word;
}

// (n, kg) 16-bit -> (kg, npad) with zero columns past n: the tiled scales' and biases' layout.
extern "C" __global__ void tf_transpose_pad16(const uint16_t* __restrict__ x, int n, int kg, int npad,
                                              uint16_t* __restrict__ out) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(kg) * npad) return;
    const int g = static_cast<int>(idx / npad), col = static_cast<int>(idx % npad);
    out[idx] = col < n ? x[static_cast<long long>(col) * kg + g] : 0;
}

// Our own kernels for Nemotron's weight layouts and the serial feed (torch-op replacements live in torch_ops/).

#include <cuda_bf16.h>
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

// A serial round's end on the device: its token becomes the next window's input; pos += 1, parity flips, keep 1.
extern "C" __global__ void tf_serial_feed(const int* __restrict__ sampled, int* __restrict__ ids, int* __restrict__ meta,
                                          int* __restrict__ history) {
    const int tok = sampled[0];
    const int pos = meta[0];
    ids[0] = tok;
    history[pos + 1] = tok;
    meta[0] = pos + 1;
    meta[1] = meta[1] ^ 1;
    meta[2] = 1;
}

namespace {

// Exclusive prefix sum over a 128-thread block (four warps); `total` gets the sum.
__device__ __forceinline__ int scan128(int v, int* total) {
    __shared__ int ws[4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int s = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, s, o);
        if (lane >= o) s += u;
    }
    if (lane == 31) ws[warp] = s;
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += ws[w];
    *total = ws[0] + ws[1] + ws[2] + ws[3];
    __syncthreads();
    return before + s - v;
}

}  // namespace

// Routed slots retain plan_kernel item and member order for experts below E.
extern "C" __global__ void __launch_bounds__(128) tf_plan_routed(const int* __restrict__ picks, int rows, int slots, int routed,
                                                                int E, int T, int* __restrict__ members,
                                                                int* __restrict__ items, int* __restrict__ counts) {
    __shared__ int pk[16 * 8];
    const int tid = threadIdx.x, P = rows * slots;
    for (int p = tid; p < P; p += blockDim.x) pk[p] = p % slots < routed ? picks[p] : -1;
    __syncthreads();
    int c = 0;
    if (tid < E)
        for (int p = 0; p < P; ++p) c += pk[p] == tid;
    const int tiles = (c + T - 1) / T;
    int nmembers, ntiles, nused;
    const int off = scan128(c, &nmembers);
    const int ioff = scan128(tiles, &ntiles);
    scan128(c > 0, &nused);
    if (tid == 0) {
        counts[0] = ntiles;
        counts[1] = nused;
    }
    if (tid >= E) return;
    for (int j = 0; j < tiles; ++j) {
        int* it = items + 3 * (ioff + j);
        it[0] = tid;
        it[1] = off + T * j;
        it[2] = min(T, c - T * j);
    }
    int k = off;
    for (int p = 0; p < P && k < off + c; ++p)
        if (pk[p] == tid) members[k++] = p;
}

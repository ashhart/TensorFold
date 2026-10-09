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

// y[row] += R x[row] at rows r * row_mul + row_add (r < rows): a lesson's change past the 4-bit codes.
extern "C" __global__ void __launch_bounds__(128) tf_rest_rows(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                               const __nv_bfloat16* __restrict__ r, void* __restrict__ y,
                                                               int y_stride, int y_f32, int rows, int row_mul, int row_add,
                                                               int n, int k) {
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (j >= n) return;
    const __nv_bfloat16* rj = r + static_cast<size_t>(j) * k;
    for (int r0 = 0; r0 < rows; r0 += 16) {
        const int tile = min(16, rows - r0);
        float acc[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) acc[t] = 0.f;
        for (int i = lane * 8; i < k; i += 256) {
            const uint4 wv = *reinterpret_cast<const uint4*>(rj + i);
            const __nv_bfloat16* w8 = reinterpret_cast<const __nv_bfloat16*>(&wv);
            float wf[8];
#pragma unroll
            for (int q = 0; q < 8; ++q) wf[q] = __bfloat162float(w8[q]);
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                if (t < tile) {
                    const size_t row = static_cast<size_t>(r0 + t) * row_mul + row_add;
                    const uint4 xv = *reinterpret_cast<const uint4*>(x + row * x_stride + i);
                    const __nv_bfloat16* x8 = reinterpret_cast<const __nv_bfloat16*>(&xv);
                    float s = 0.f;
#pragma unroll
                    for (int q = 0; q < 8; ++q) s += wf[q] * __bfloat162float(x8[q]);
                    acc[t] += s;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (t < tile) {
                float v = acc[t];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) {
                    const size_t row = static_cast<size_t>(r0 + t) * row_mul + row_add;
                    if (y_f32) {
                        float* yp = static_cast<float*>(y) + row * y_stride + j;
                        *yp += v;
                    } else {
                        __nv_bfloat16* yp = static_cast<__nv_bfloat16*>(y) + row * y_stride + j;
                        *yp = __float2bfloat16(__bfloat162float(*yp) + v);
                    }
                }
            }
        }
    }
}

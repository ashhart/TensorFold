// Our own kernels for Nemotron's serial feed, routed plan and rests (torch-op replacements live in torch_ops/).

#include <cuda_bf16.h>
#include <stdint.h>

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

// The pocket in one CTA a row (n <= 64, fp32 up and down [n, d]): what tf_pocket_in then tf_pocket_out must equal.
extern "C" __global__ void __launch_bounds__(256) tf_pocket_rows(const __nv_bfloat16* __restrict__ y, int d,
                                                                 const float* __restrict__ up, const float* __restrict__ down,
                                                                 int n, void* __restrict__ out, int out_f32, int row_mul,
                                                                 int row_add) {
    __shared__ float act[64];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const __nv_bfloat16* yr = y + static_cast<size_t>(blockIdx.x) * d;
    for (int j = warp; j < n; j += 8) {
        const float* uj = up + static_cast<size_t>(j) * d;
        float s = 0.f;
        for (int i = lane; i < d; i += 32) s += uj[i] * __bfloat162float(yr[i]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) {
            const float z = fmaxf(s, 0.f);
            act[j] = z * z;
        }
    }
    __syncthreads();
    const size_t row = static_cast<size_t>(blockIdx.x) * row_mul + row_add;
    for (int k = threadIdx.x; k < d; k += 256) {
        float s = 0.f;
        for (int j = 0; j < n; ++j) s += act[j] * down[static_cast<size_t>(j) * d + k];
        if (out_f32) {
            static_cast<float*>(out)[row * d + k] += s;
        } else {
            __nv_bfloat16* o = static_cast<__nv_bfloat16*>(out) + row * d + k;
            *o = __float2bfloat16(__bfloat162float(*o) + s);
        }
    }
}

namespace {

constexpr int pocket_width = 64;  // act's row stride: the most neurons a pocket holds

// act [rows, 64] = relu^2(up y) over the first n neurons: a warp takes NPW neurons by R rows, its lanes' sums in order.
template <int NPW, int R>
__device__ __forceinline__ void pocket_in(const __nv_bfloat16* __restrict__ y, int d, const float* __restrict__ up,
                                          int n, float* __restrict__ act, int rows) {
    const int lane = threadIdx.x & 31;
    const int j0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * NPW;
    if (j0 >= n) return;
    const int r0 = blockIdx.y * R;
    const int tile = min(R, rows - r0);
    float s[NPW][R];
#pragma unroll
    for (int q = 0; q < NPW; ++q)
#pragma unroll
        for (int t = 0; t < R; ++t) s[q][t] = 0.f;
    for (int i = lane; i < d; i += 32) {
        float u[NPW], x[R];
#pragma unroll
        for (int q = 0; q < NPW; ++q) u[q] = j0 + q < n ? up[static_cast<size_t>(j0 + q) * d + i] : 0.f;
#pragma unroll
        for (int t = 0; t < R; ++t) x[t] = t < tile ? __bfloat162float(y[static_cast<size_t>(r0 + t) * d + i]) : 0.f;
#pragma unroll
        for (int q = 0; q < NPW; ++q)
#pragma unroll
            for (int t = 0; t < R; ++t) s[q][t] += u[q] * x[t];
    }
#pragma unroll
    for (int q = 0; q < NPW; ++q) {
#pragma unroll
        for (int t = 0; t < R; ++t) {
            float v = s[q][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && j0 + q < n && t < tile) {
                const float z = fmaxf(v, 0.f);
                act[static_cast<size_t>(r0 + t) * pocket_width + j0 + q] = z * z;
            }
        }
    }
}

// out (bf16, fp32 if out_f32) += act down at rows r * row_mul + row_add: a thread an output, R rows, neurons in order.
template <int R, int B>
__device__ __forceinline__ void pocket_out(const float* __restrict__ act, const float* __restrict__ down, int n, int d,
                                           void* __restrict__ out, int out_f32, int rows, int row_mul, int row_add) {
    if (n == 0) return;  // no neuron in use: the output keeps its bits, as when the launch is skipped
    __shared__ float a[R * pocket_width];
    const int r0 = blockIdx.y * R;
    const int tile = min(R, rows - r0);
    for (int i = threadIdx.x; i < tile * pocket_width; i += blockDim.x)
        if (i % pocket_width < n) a[i] = act[static_cast<size_t>(r0) * pocket_width + i];
    __syncthreads();
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= d) return;
    float s[R];
#pragma unroll
    for (int t = 0; t < R; ++t) s[t] = 0.f;
    for (int j0 = 0; j0 < n; j0 += B) {
        float w[B];
#pragma unroll
        for (int b = 0; b < B; ++b) w[b] = j0 + b < n ? down[static_cast<size_t>(j0 + b) * d + k] : 0.f;
#pragma unroll
        for (int t = 0; t < R; ++t) {
            if (t < tile) {
#pragma unroll
                for (int b = 0; b < B; ++b)
                    if (j0 + b < n) s[t] += a[t * pocket_width + j0 + b] * w[b];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < R; ++t) {
        if (t < tile) {
            const size_t row = static_cast<size_t>(r0 + t) * row_mul + row_add;
            if (out_f32) {
                static_cast<float*>(out)[row * d + k] += s[t];
            } else {
                __nv_bfloat16* o = static_cast<__nv_bfloat16*>(out) + row * d + k;
                *o = __float2bfloat16(__bfloat162float(*o) + s[t]);
            }
        }
    }
}

}  // namespace

// tf_pocket_rows' first half over the first *used neurons: a warp a neuron, a block row a row.
extern "C" __global__ void __launch_bounds__(128) tf_pocket_in(const __nv_bfloat16* __restrict__ y, int d,
                                                               const float* __restrict__ up,
                                                               const int* __restrict__ used,
                                                               float* __restrict__ act, int rows) {
    pocket_in<1, 1>(y, d, up, *used, act, rows);
}

// The same for many rows: a warp eight neurons by eight rows.
extern "C" __global__ void __launch_bounds__(256) tf_pocket_in_rows(const __nv_bfloat16* __restrict__ y, int d,
                                                                    const float* __restrict__ up,
                                                                    const int* __restrict__ used,
                                                                    float* __restrict__ act, int rows) {
    pocket_in<8, 8>(y, d, up, *used, act, rows);
}

// tf_pocket_rows' second half over the first *used neurons: a thread an output, a block row a row.
extern "C" __global__ void __launch_bounds__(128) tf_pocket_out(const float* __restrict__ act,
                                                                const float* __restrict__ down,
                                                                const int* __restrict__ used, int d,
                                                                void* __restrict__ out, int out_f32, int rows,
                                                                int row_mul, int row_add) {
    pocket_out<1, 16>(act, down, *used, d, out, out_f32, rows, row_mul, row_add);
}

// The same for many rows: a thread an output over 16 rows.
extern "C" __global__ void __launch_bounds__(128) tf_pocket_out_rows(const float* __restrict__ act,
                                                                     const float* __restrict__ down,
                                                                     const int* __restrict__ used, int d,
                                                                     void* __restrict__ out, int out_f32, int rows,
                                                                     int row_mul, int row_add) {
    pocket_out<16, 8>(act, down, *used, d, out, out_f32, rows, row_mul, row_add);
}

// A target step's nudge: y (bf16, or fp32 when y_f32) += x (fp32), n values, into a row of a layer's output.
extern "C" __global__ void tf_add_row(void* __restrict__ y, int y_f32, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (y_f32) {
        static_cast<float*>(y)[i] += x[i];
    } else {
        __nv_bfloat16* p = static_cast<__nv_bfloat16*>(y) + i;
        *p = __float2bfloat16(__bfloat162float(*p) + x[i]);
    }
}

// LatentMoE (Bolt): a row's routed outputs in the latent width, weighted and summed in fp32, then bf16 like the
// reference's index_add_ sum. Y [R * K, L] fp32 (pair rows in slot order), WT [R * K], Z [R, L]; one block a row.
extern "C" __global__ void tf_latent_sum(const float* __restrict__ Y, const float* __restrict__ WT, __nv_bfloat16* __restrict__ Z,
                                         int K, int L) {
    const int64_t r = blockIdx.x;
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc = __fmaf_rn(Y[(r * K + k) * L + i], WT[r * K + k], acc);
        Z[r * L + i] = __float2bfloat16(acc);
    }
}

// out = a + b in bf16 (b bf16, or fp32 when b_f32): the latent experts' projection plus the shared expert, n values.
extern "C" __global__ void tf_add_bf16(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ a, const void* __restrict__ b,
                                       int b_f32, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float y = b_f32 ? static_cast<const float*>(b)[i] : __bfloat162float(static_cast<const __nv_bfloat16*>(b)[i]);
    out[i] = __float2bfloat16(__bfloat162float(a[i]) + y);
}

// An image prompt's chunk: rows whose SRC names a feature row take it in place of their token's embedding.
// H [rows, D] bf16, F [image rows, D] bf16, SRC [rows] (-1: a text row); one block a row.
extern "C" __global__ void tf_splice_rows(__nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ F, const int* __restrict__ SRC, int D) {
    const int64_t r = blockIdx.x;
    const int s = SRC[r];
    if (s < 0) return;
    for (int i = threadIdx.x; i < D; i += blockDim.x) H[r * D + i] = F[static_cast<int64_t>(s) * D + i];
}

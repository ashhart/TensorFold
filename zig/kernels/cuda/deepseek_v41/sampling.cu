// DeepSeek-V4.1 keyed sampling, device part; --fmad=false: torch rounds v / t, the subtraction and exp separately.

// stats: per-row (m, s) in float64, sum order not torch's (the keep-count guard absorbs it); pack: [vals, cols + id0].

#include <cstdint>

namespace dsv41_sampling {

constexpr int NT = 1024;

__device__ __forceinline__ double warp_max(double v) {
    for (int o = 16; o > 0; o >>= 1) {
        const double u = __shfl_xor_sync(0xFFFFFFFFu, v, o);
        v = u > v ? u : v;
    }
    return v;
}

__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

// One CTA a row: rows [rows, C] fp32 (row stride ld), out float64 [rows, 2].
__global__ void __launch_bounds__(NT) stats_kernel(const float* __restrict__ lg, long long ld, int C, double t,
                                                   double* __restrict__ out) {
    __shared__ double red[NT / 32];
    __shared__ double s_m;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* row = lg + (long long)blockIdx.x * ld;
    double m = -__longlong_as_double(0x7FF0000000000000LL);   // -inf
    for (int c = tid; c < C; c += NT) {
        const double v = (double)row[c];
        m = v > m ? v : m;
    }
    m = warp_max(m);
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (warp == 0) {
        double v = red[lane];
        v = warp_max(v);
        if (lane == 0) s_m = v;
    }
    __syncthreads();
    const double mx = s_m;
    const double mt = mx / t;                                  // (m / t)[:, None]
    double s = 0.0;
    for (int c = tid; c < C; c += NT) s += exp((double)row[c] / t - mt);
    s = warp_sum(s);
    __syncthreads();                                           // red is reused
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (warp == 0) {
        double v = red[lane];
        v = warp_sum(v);
        if (lane == 0) {
            out[2 * (long long)blockIdx.x] = mx;
            out[2 * (long long)blockIdx.x + 1] = v;
        }
    }
}

// One CTA a row: vals fp32 [rows, k], cols int64 [rows, k] -> out fp32 [rows, 2k].
__global__ void __launch_bounds__(256) pack_kernel(const float* __restrict__ vals, const long long* __restrict__ cols,
                                                   int k, int id0, float* __restrict__ out) {
    const long long r = blockIdx.x;
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        out[r * 2 * k + j] = vals[r * k + j];
        out[r * 2 * k + k + j] = __int_as_float((int)cols[r * k + j] + id0);
    }
}

}  // namespace dsv41_sampling

// Split stats: same (m, s) as stats_kernel, sum in fixed chunk order; max, sum per chunk, fin adds chunks in order.
namespace dsv41_sampling_split {

constexpr int NT = 256;

__device__ __forceinline__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) {
        const float u = __shfl_xor_sync(0xFFFFFFFFu, v, o);
        v = u > v ? u : v;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < NT / 32 ? red[lane] : -__int_as_float(0x7F800000);
    for (int o = 16; o > 0; o >>= 1) {
        const float u = __shfl_xor_sync(0xFFFFFFFFu, v, o);
        v = u > v ? u : v;
    }
    return v;                                                  // valid in warp 0
}

// the row's max from its chunk maxima, every thread
__device__ __forceinline__ float row_max(const float* pm, int chunks) {
    float m = -__int_as_float(0x7F800000);
    for (int j = 0; j < chunks; ++j) m = pm[j] > m ? pm[j] : m;
    return m;
}

}  // namespace dsv41_sampling_split

extern "C" __global__ void __launch_bounds__(dsv41_sampling_split::NT)
dsv41_sampling_stats_max(const float* __restrict__ lg, long long ld, int C, int chunk, float* __restrict__ pmax) {
    using namespace dsv41_sampling_split;
    __shared__ float red[NT / 32];
    const float* row = lg + (long long)blockIdx.y * ld;
    const int c0 = blockIdx.x * chunk, c1 = min(C, c0 + chunk);
    float m = -__int_as_float(0x7F800000);
    for (int c = c0 + threadIdx.x; c < c1; c += NT) {
        const float v = row[c];
        m = v > m ? v : m;
    }
    m = block_max(m, red);
    if (threadIdx.x == 0) pmax[(long long)blockIdx.y * gridDim.x + blockIdx.x] = m;
}

extern "C" __global__ void __launch_bounds__(dsv41_sampling_split::NT)
dsv41_sampling_stats_sum(const float* __restrict__ lg, long long ld, int C, int chunk, double t,
                         const float* __restrict__ pmax, double* __restrict__ psum) {
    using namespace dsv41_sampling_split;
    __shared__ double red[NT / 32];
    const int chunks = gridDim.x;
    const float* row = lg + (long long)blockIdx.y * ld;
    const double mt = (double)row_max(pmax + (long long)blockIdx.y * chunks, chunks) / t;
    const int c0 = blockIdx.x * chunk, c1 = min(C, c0 + chunk);
    double s = 0.0;
    for (int c = c0 + threadIdx.x; c < c1; c += NT) s += exp((double)row[c] / t - mt);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        double v = 0.0;
        for (int w = 0; w < NT / 32; ++w) v += red[w];
        psum[(long long)blockIdx.y * chunks + blockIdx.x] = v;
    }
}

extern "C" __global__ void __launch_bounds__(32)
dsv41_sampling_stats_fin(const float* __restrict__ pmax, const double* __restrict__ psum, int chunks,
                         double* __restrict__ out) {
    using namespace dsv41_sampling_split;
    if (threadIdx.x != 0) return;
    const long long i = blockIdx.x;
    double s = 0.0;
    for (int j = 0; j < chunks; ++j) s += psum[i * chunks + j];
    out[2 * i] = (double)row_max(pmax + i * chunks, chunks);
    out[2 * i + 1] = s;
}

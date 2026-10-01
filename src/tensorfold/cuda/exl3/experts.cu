// EXL3 routed experts, any codebook and a width per expert: fixed-order splits, slots and butterflies, no atomics; 4-bit mcg matches GLM's kernel bit for bit.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "experts_grouped.cuh"

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Grouping: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last, and the tile
// list: (expert's place << 16) | 16-row tile for every non-empty tile, in place order, so the grouped kernels launch one
// program a tile in use. Three launches, none holding the picks in shared memory: per-expert counts (integer atomics,
// exact in any order), one block's scan over the experts for places and tiles, then a warp an expert that compacts its
// picks in row order with ballots.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;
constexpr int FILL_WARPS = 4;

__global__ void group_count_kernel(const int* __restrict__ pick, int n, int E, int* __restrict__ counts) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int e = pick[i];
        if (e >= 0 && e < E) atomicAdd(counts + e, 1);
    }
}

__global__ void __launch_bounds__(GROUP_THREADS) group_scan_kernel(const int* __restrict__ counts,
                                                                   int* __restrict__ uids, int* __restrict__ ucount,
                                                                   int* __restrict__ tiles, int* __restrict__ tcount,
                                                                   int* __restrict__ place_of, int E, int maxm) {
    __shared__ int warp_tot[GROUP_THREADS / 32];
    __shared__ int warp_tiles[GROUP_THREADS / 32];
    int cnt[GROUP_PER_THREAD];
    int used = 0, ntile = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        const int c = e < E ? counts[e] : 0;
        cnt[q] = c;
        used += c > 0;
        ntile += (min(c, maxm) + 15) / 16;
    }
    // exclusive scans of `used` and of the tiles over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used, tinc = ntile;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        int tv = __shfl_up_sync(0xffffffffu, tinc, o);
        if (lane >= o) { inc += v; tinc += tv; }
    }
    if (lane == 31) { warp_tot[warp] = inc; warp_tiles[warp] = tinc; }
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane], tv = warp_tiles[lane];
        int s = v, ts = tv;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            int tx = __shfl_up_sync(0xffffffffu, ts, o);
            if (lane >= o) { s += x; ts += tx; }
        }
        warp_tot[lane] = s - v;                                   // exclusive per warp
        warp_tiles[lane] = ts - tv;
        if (lane == 31) { ucount[0] = s; tcount[0] = ts; }
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
    int tile = warp_tiles[warp] + tinc - ntile;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        if (e >= E) continue;
        if (cnt[q] == 0) {
            place_of[e] = -1;
            continue;
        }
        uids[place] = e;
        place_of[e] = place;
        for (int m = 0; m < (min(cnt[q], maxm) + 15) / 16; ++m) tiles[tile++] = (place << 16) | m;
        ++place;
    }
}

// A warp an expert: its picks' positions in row order (a ballot a 32 picks), the first maxm, then -1 to maxm.
__global__ void __launch_bounds__(FILL_WARPS * 32) group_fill_kernel(const int* __restrict__ pick, int n, int slots,
                                                                     const int* __restrict__ place_of,
                                                                     int* __restrict__ members, int E, int maxm) {
    const int e = blockIdx.x * FILL_WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (e >= E) return;
    const int place = place_of[e];
    if (place < 0) return;
    int* row = members + (size_t)place * maxm;
    int j = 0;
    for (int base = 0; base < n && j < maxm; base += 32) {
        const int i = base + lane;
        const bool hit = i < n && pick[i] == e;
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        const int k = j + __popc(m & ((1u << lane) - 1u));
        if (hit && k < maxm) row[k] = (i / slots) * 32 + (i % slots);
        j += __popc(m);
    }
    for (int k = min(j, maxm) + lane; k < maxm; k += 32) row[k] = -1;
}

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: GLM's bf16 roundings, 1: fp32), then Xd = fp16((act * suh_d) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, float limit, int act_mode) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): Y = (splits summed in order) @ H * svh_d, fp32.
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int E) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

// out[r][d] = sum over slots in order of wts[r][k] * y[r * slots + k][d] (fp32, fma chain from 0).
__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                               int D, int slots) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = fmaf(wts[r * slots + k], y[((size_t)r * slots + k) * D + d], acc);
    out[(size_t)r * D + d] = acc;
}

// down_epilogue_kernel then combine_kernel in one launch, the same arithmetic in the same order (the same bits).
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, float* __restrict__ y,
                                    const float* __restrict__ wts, float* __restrict__ out, int P, int D, int SK,
                                    int E, int slots) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = y[(size_t)p * D + n + j];
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(size_t)r * D + n + j] = acc[j];
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

namespace tf_exl3x {
extern template void grouped_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void dequant_launch<0>(const uint32_t*, half*, int, int, int, cudaStream_t);
extern template void dequant_launch<1>(const uint32_t*, half*, int, int, int, cudaStream_t);
extern template void dequant_launch<2>(const uint32_t*, half*, int, int, int, cudaStream_t);
}  // namespace tf_exl3x

void exl3x_grouped_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                        const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
                        const at::Tensor& members, const at::Tensor& tiles, const at::Tensor& tcount, at::Tensor& Z,
                        int64_t mats, int64_t K, int64_t N, int64_t P,
                        int64_t SK, int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t lo,
                        int64_t hi) {
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.tiles = tiles.data_ptr<int>();
    a.tcount = tcount.data_ptr<int>();
    a.tile_max = (int)tiles.numel();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = (int)nt; a.warps = (int)warps; a.pf = (int)pf; a.lo = (int)lo; a.hi = (int)hi;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3x::grouped_launch<0>(a, stream);
    else if (cb == 1) tf_exl3x::grouped_launch<1>(a, stream);
    else if (cb == 2) tf_exl3x::grouped_launch<2>(a, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_dequant_cuda(const at::Tensor& T, at::Tensor& out, int64_t K, int64_t N, int64_t k2, int64_t cb) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto t = reinterpret_cast<const uint32_t*>(T.data_ptr());
    auto o = reinterpret_cast<half*>(out.data_ptr());
    if (cb == 0) tf_exl3x::dequant_launch<0>(t, o, (int)K, (int)N, (int)k2, stream);
    else if (cb == 1) tf_exl3x::dequant_launch<1>(t, o, (int)K, (int)N, (int)k2, stream);
    else tf_exl3x::dequant_launch<2>(t, o, (int)K, (int)N, (int)k2, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_group_cuda(const at::Tensor& pick, at::Tensor& uids, at::Tensor& ucount, at::Tensor& members,
                      at::Tensor& tiles, at::Tensor& tcount, at::Tensor& counts, at::Tensor& place_of, int64_t R,
                      int64_t slots, int64_t E) {
    TORCH_CHECK(E <= GROUP_THREADS * GROUP_PER_THREAD, "too many experts for the grouping kernel");
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int n = (int)(R * slots);
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int>(), 0, (size_t)E * sizeof(int), stream));
    group_count_kernel<<<std::max(1, std::min((n + 255) / 256, 1024)), 256, 0, stream>>>(pick.data_ptr<int>(), n,
                                                                                         (int)E, counts.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    group_scan_kernel<<<1, GROUP_THREADS, 0, stream>>>(counts.data_ptr<int>(), uids.data_ptr<int>(),
                                                       ucount.data_ptr<int>(), tiles.data_ptr<int>(),
                                                       tcount.data_ptr<int>(), place_of.data_ptr<int>(), (int)E,
                                                       (int)members.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    group_fill_kernel<<<(unsigned)((E + FILL_WARPS - 1) / FILL_WARPS), FILL_WARPS * 32, 0, stream>>>(
        pick.data_ptr<int>(), n, (int)slots, place_of.data_ptr<int>(), members.data_ptr<int>(), (int)E,
        (int)members.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_rot_in_cuda(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
                       const at::Tensor& suh1, at::Tensor& out0, at::Tensor& out1, int64_t rows, int64_t K,
                       int64_t slots, int64_t E) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(K / 128), 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto s0 = reinterpret_cast<const half*>(suh0.data_ptr());
    auto s1 = reinterpret_cast<const half*>(suh1.data_ptr());
    auto o0 = reinterpret_cast<half*>(out0.data_ptr());
    auto o1 = reinterpret_cast<half*>(out1.data_ptr());
    if (x.scalar_type() == at::kBFloat16)
        rot_in_kernel<__nv_bfloat16><<<grid, 32, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                                              (int)x_stride, pick.data_ptr<int>(), s0, s1, o0, o1,
                                                              (int)K, (int)slots, (int)E);
    else
        rot_in_kernel<half><<<grid, 32, 0, stream>>>(reinterpret_cast<const half*>(x.data_ptr()), (int)x_stride,
                                                     pick.data_ptr<int>(), s0, s1, o0, o1, (int)K, (int)slots,
                                                     (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_gateup_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g,
                                const at::Tensor& svh_u, const at::Tensor& suh_d, at::Tensor& xd, int64_t rows,
                                int64_t P, int64_t N, int64_t SK, int64_t slots, int64_t E, double limit,
                                int64_t act_mode) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(N / 128));
    gateup_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_g.data_ptr()),
        reinterpret_cast<const half*>(svh_u.data_ptr()), reinterpret_cast<const half*>(suh_d.data_ptr()),
        reinterpret_cast<half*>(xd.data_ptr()), (int)P, (int)N, (int)SK, (int)E, (float)limit, (int)act_mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_down_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                              int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(D / 128));
    down_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
        y.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_combine_cuda(const at::Tensor& y, const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t D,
                        int64_t slots) {
    dim3 grid((unsigned)rows, (unsigned)((D + 255) / 256));
    combine_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<float>(), wts.data_ptr<float>(),
                                                                        out.data_ptr<float>(), (int)D, (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_down_combine_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                             const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t P, int64_t D, int64_t SK,
                             int64_t slots, int64_t E) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    dim3 grid((unsigned)rows, (unsigned)(D / 128));
    down_combine_kernel<<<grid, (unsigned)(32 * slots), 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
        y.data_ptr<float>(), wts.data_ptr<float>(), out.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E,
        (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

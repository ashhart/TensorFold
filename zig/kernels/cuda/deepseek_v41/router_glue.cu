// Decode router glue, arithmetic of exl3/experts.cu (blob 8ce66ca83bce); -O3, fmad on, ftz off; R <= 64, K == 5120.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv41_router_glue {
constexpr int THREADS = 1024;
constexpr int WARPS = THREADS / 32;
constexpr int MAX_E = 512;
constexpr int MAX_PICKS = 64 * 9;
constexpr float HAD_SCALE = 0.08838834764831845f;

// Verbatim arithmetic from tf_exl3_experts::fwht128 at the source above.
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

// Integer-equal to upstream group_kernel (histogram, id scan, earlier equal picks; duplicates past maxm truncated).
__device__ __forceinline__ void group_rows(const int* pick, int* uids, int* ucount,
                                          int* members, int R, int slots, int E, int maxm) {
    __shared__ int picks[MAX_PICKS], count[MAX_E], place[MAX_E], warp_tot[WARPS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = R * slots;
    if (tid < n) picks[tid] = pick[tid];
    if (tid < E) count[tid] = 0;
    __syncthreads();
    if (tid < n) {
        const int e = picks[tid];
        if (e >= 0 && e < E) atomicAdd(count + e, 1);
    }
    __syncthreads();
    const int used = tid < E && count[tid] > 0;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    if (used) {
        const int p = warp_tot[warp] + inc - used;
        uids[p] = tid;
        place[tid] = p;
    }
    __syncthreads();
    if (tid < n) {
        const int e = picks[tid];
        if (e >= 0 && e < E) {
            int rank = 0;
            for (int q = 0; q < tid; ++q) rank += picks[q] == e;
            if (rank < maxm) members[(size_t)place[e] * maxm + rank] = (tid / slots) * 32 + tid % slots;
        }
    }
    if (used)
        for (int j = count[tid]; j < maxm; ++j) members[(size_t)place[tid] * maxm + j] = -1;
}
}  // namespace dsv41_router_glue

// One CTA, block 1024, no dynamic shared memory. Same argument order as group.
extern "C" __global__ void __launch_bounds__(1024) router_group(
    const int* pick, int* uids, int* ucount, int* members, int R, int slots, int E, int maxm) {
    dsv41_router_glue::group_rows(pick, uids, ucount, members, R, slots, E, maxm);
}

// grid.x = 1 + ceil(R slots (K / 128) 2 / 32), block 1024: CTA 0 groups, other warps rotate one (pick, block, mat).
extern "C" __global__ void __launch_bounds__(1024) router_group_rot(
    const uint16_t* x, int x_stride, const int* pick, const half* suh0, const half* suh1,
    half* out0, half* out1, int* uids, int* ucount, int* members,
    int R, int K, int slots, int E, int maxm) {
    using namespace dsv41_router_glue;
    if (blockIdx.x == 0) {
        group_rows(pick, uids, ucount, members, R, slots, E, maxm);
        return;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tile = (blockIdx.x - 1) * WARPS + warp;
    const int P = R * slots, blocks = K / 128;
    if (tile >= P * blocks * 2) return;  // uniform within each warp
    const int p = tile % P, blk = (tile / P) % blocks, mat = tile / (P * blocks);
    const int row = p / slots, e = pick[p];
    if (e < 0 || e >= E) return;        // invalid picks leave outputs untouched
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = reinterpret_cast<const __nv_bfloat16*>(x)
        + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

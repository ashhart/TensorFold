// EXL3 routed experts: grouping, grouped GEMMs, epilogues; port of tensorfold's experts.cu (ExLlamaV3, Turboderp).

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_decode.cuh"
#include "exl3_plan.cuh"
#include "exl3_grouped.cuh"

// ---------------------------------------------------------------------------------------------------------------

// sm_120 and sm_121 read a warp's next k tile ahead, bit-equal (contract_exl3_grouped.zig); older SMs keep main's loop.
#if __CUDA_ARCH__ >= 1200
constexpr int GROUPED_LOAD = tf_exl3::AHEAD;
#else
constexpr int GROUPED_LOAD = tf_exl3::PLAIN;
#endif

extern "C" __global__ void tf_exl3_grouped_0(tf_exl3::GroupedArgs a) {  // 3inst, any width
    tf_exl3::grouped_body<0, 8, 4, GROUPED_LOAD>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}

extern "C" __global__ void tf_exl3_grouped_1(tf_exl3::GroupedArgs a) {  // mcg, any width
    tf_exl3::grouped_body<1, 8, 4, GROUPED_LOAD>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}

extern "C" __global__ void tf_exl3_grouped_2(tf_exl3::GroupedArgs a) {  // mul1, any width (RED-SNOW's experts)
    tf_exl3::grouped_body<2, 8, 4, GROUPED_LOAD>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}

namespace tf_exl3 {

// Grouping in one block: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

struct GroupArgs {
    const int* pick; // int32 [rows * slots]
    int* uids; // int32 [nexp_max]
    int* ucount; // int32 [1]
    int* members; // int32 [nexp_max, maxm]
    int rows, slots, experts, maxm;
};

extern "C" __global__ void __launch_bounds__(GROUP_THREADS) tf_exl3_group(const GroupArgs a) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    const int n = a.rows * a.slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = a.pick[i];
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < a.experts)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    // exclusive scan of `used` over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v; // exclusive per warp
        if (lane == 31) a.ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        a.uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < a.maxm; ++i)
            if (sh_pick[i] == e) a.members[place * a.maxm + j++] = (i / a.slots) * 32 + (i % a.slots);
        for (; j < a.maxm; ++j) a.members[place * a.maxm + j] = -1;
        ++place;
    }
}

// Program (slot, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
struct RotInArgs {
    const void* x; // bf16 or fp16 [rows, x_stride]
    int x_stride;
    int x_dtype; // 0: fp16, 1: bf16 (exl3.zig DType)
    const int* pick; // int32 [rows * slots]
    const half* suh0; // fp16 [E, K], gate
    const half* suh1; // fp16 [E, K], up
    half* out0; // fp16 [P, K]
    half* out1;
    int k, slots, experts;
};

extern "C" __global__ void tf_exl3_experts_rot_in(const RotInArgs a) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int e = a.pick[p];
    if (e < 0 || e >= a.experts) return;
    const int lane = threadIdx.x;
    const int row = p / a.slots;
    const half* suh = (mat ? a.suh1 : a.suh0) + (size_t)e * a.k + blk * 128 + 4 * lane;
    const uint16_t* xr = reinterpret_cast<const uint16_t*>(a.x) + (size_t)row * a.x_stride + blk * 128 + 4 * lane;
    float v[4];
    if (a.x_dtype == 1) {
        const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(xr);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xv[j]) * __half2float(suh[j]);
    } else {
        const half* xv = reinterpret_cast<const half*>(xr);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __half2float(xv[j]) * __half2float(suh[j]);
    }
    fwht128(v, lane);
    half* o = (mat ? a.out1 : a.out0) + (size_t)p * a.k + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (slot, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: bf16, 1: fp32), then Xd.
struct GateUpArgs {
    const float* z; // [(mat * sk + split) * P + p] * N + col
    const int* pick; // int32 [P]
    const half* svh_g; // fp16 [E, N]
    const half* svh_u; // fp16 [E, N]
    const half* suh_d; // fp16 [E, N]
    half* xd; // fp16 [P, N]
    int p, n, sk, experts;
    float limit;
    int act_mode;
};

extern "C" __global__ void tf_exl3_gateup_epilogue(const GateUpArgs a) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = a.pick[p];
    if (e < 0 || e >= a.experts) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < a.sk; ++s) {
            sg += a.z[((size_t)(0 * a.sk + s) * a.p + p) * a.n + n + j];
            su += a.z[((size_t)(1 * a.sk + s) * a.p + p) * a.n + n + j];
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
        if (a.act_mode == 0) {
            float gg = bf16r(gv[j] * HAD_SCALE * __half2float(a.svh_g[(size_t)e * a.n + n + j]));
            float uu = bf16r(uv[j] * HAD_SCALE * __half2float(a.svh_u[(size_t)e * a.n + n + j]));
            if (a.limit > 0) { gg = upTo(gg, a.limit); uu = within(uu, a.limit); }
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = gv[j] * HAD_SCALE * __half2float(a.svh_g[(size_t)e * a.n + n + j]);
            float uu = uv[j] * HAD_SCALE * __half2float(a.svh_u[(size_t)e * a.n + n + j]);
            if (a.limit > 0) { gg = upTo(gg, a.limit); uu = within(uu, a.limit); }
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(a.suh_d[(size_t)e * a.n + n + j]);
    }
    fwht128(v, lane);
    half* o = a.xd + (size_t)p * a.n + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the width): down epilogue and combine in one launch, the same bits as the two.
struct DownArgs {
    const float* z; // [split * P + p] * D + col
    const int* pick; // int32 [P]
    const half* svh_d; // fp16 [E, D]
    float* y; // fp32 [P, D]
    const float* wts; // fp32 [rows, slots]
    float* out; // fp32 [rows, D]
    int p, d, sk, experts, slots;
};

extern "C" __global__ void tf_exl3_down_combine(const DownArgs a) {
    __shared__ float4 part[32][32]; // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * a.slots + k;
    const int e = a.pick[p];
    float o[4];
    if (e >= 0 && e < a.experts) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < a.sk; ++q) s += a.z[((size_t)q * a.p + p) * a.d + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(a.svh_d[(size_t)e * a.d + n + j]);
            a.y[(size_t)p * a.d + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = a.y[(size_t)p * a.d + n + j];
    }
    if (!a.wts) return; // runSlots reads the same post-rotation per-slot FP32 outputs without reducing them.
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < a.slots; ++q) {
        const float w = a.wts[r * a.slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) a.out[(size_t)r * a.d + n + j] = acc[j];
}

}  // namespace tf_exl3

extern "C" __global__ void tf_exl3_plan_rotate(const PlanStage a) {
    const int it = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    if (it >= a.counts[0]) return;
    const int e = a.items[3 * it], first = a.items[3 * it + 1], count = a.items[3 * it + 2];
    if (e == a.skip) return;
    const int lane = threadIdx.x;
    const half* su = (mat ? a.su1 : a.su0) + (size_t)e * a.k + blk * 128 + 4 * lane;
    for (int j = 0; j < count; ++j) {
        const int p = a.members[first + j], row = a.slots ? p / a.slots : p;
        const __nv_bfloat16* x = a.x + (size_t)row * a.stride + blk * 128 + 4 * lane;
        float v[4];
        for (int q = 0; q < 4; ++q) v[q] = __bfloat162float(x[q]) * __half2float(su[q]);
        tf_exl3::fwht128(v, lane);
        half* out = (mat ? a.x1 : a.x0) + (size_t)p * a.k + blk * 128 + 4 * lane;
        for (int q = 0; q < 4; ++q) out[q] = __float2half_rn(v[q] * tf_exl3::HAD_SCALE);
    }
}

extern "C" __global__ void tf_exl3_plan_finish(const PlanStage a) { tf_exl3::plan_finish<tf_exl3::FINISH_SERVED>(a); }

// Transposed fp32 prompt accumulators in the shared EXL3 module; decode kernels above stay unchanged.
#include "exl3_prompt.cuh"
using namespace tf_exl3_prompt;
extern "C" __global__ void tf_exl3_prompt_rotate(const __nv_bfloat16* __restrict__ x, int x_stride, const half* __restrict__ suh,
        half* __restrict__ out, int K) {
    const int row = blockIdx.x, blk = blockIdx.y;
    const int lane = threadIdx.x;
    const half* sh = suh + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(sh[j]);
    fwht128(v, lane);
    half* o = out + (size_t)row * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_gu(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode) {

    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];   // one buffer for either instantiation
    __shared__ int rows_sh[TPM];
    if (xdiv > 1) tgateup<TPM>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, smem_raw, rows_sh);
    else tgateup<2 * TPM>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, 1, limit, act_mode, smem_raw, rows_sh);
}

// Block (256 model-width columns, item): y[row][slot][:] = the down epilogue of the item's pairs (this rank's share).
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_down(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int ys) {

    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];
    __shared__ int rows_sh[TPM];
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    tmain<1, TPM>(Xd, Xd, Td, Td, rows_sh, cnt, e, K, D >> 4, cb, 1, reinterpret_cast<half*>(smem_raw), acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();
        tspill(Ed, DN_ES, warp * TMT * 16, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const int n = cb * 256 + blk * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
                fwht128(v, lane);
                float* o = y + (size_t)p * D + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            }
        }
    }
}

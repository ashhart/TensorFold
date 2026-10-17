// EXL3 routed experts: grouping, grouped GEMMs, epilogues; port of tensorfold's experts.cu (ExLlamaV3, Turboderp).

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_decode.cuh"

namespace tf_exl3 {

// One projection's grouped arguments; the fields must stay in launch.zig add order.
struct GroupedArgs {
    const half* x0; // fp16 [P, K], the rotated gate (or down) inputs
    const half* x1; // fp16 [P, K], the rotated up inputs
    const long long* tp0; // int64 [E], one trellis pointer per expert
    const long long* tp1;
    const int* k2_0; // int32 [E], half-bits a value
    const int* k2_1;
    const int* uids; // int32 [nexp_max], distinct routed experts in id order
    const int* ucount; // int32 [1]
    const int* members; // int32 [nexp_max, maxm], row * 32 + slot, -1 after the last member
    float* z; // fp32 partials, [(mat * sk + split) * P + p] * N + col
    int k, n, p, sk, maxm, slots, nexp_max, mats, ns;
};

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix into acc: per-lane loads and the shared decode (same bits).
template <int CB, int K2, int NT>
__device__ __forceinline__ void warp_tiles(const uint32_t* T, int ntiles, int kt0, int nkt, int nt0, const half* x0,
                                           const half* x1, bool ok0, bool ok1, int lane, float (&acc)[NT][2][4]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    const size_t kstride = (size_t)ntiles * TW;
    for (int i = 0; i < nkt; ++i) {
        const uint32_t* t0 = T + (size_t)(kt0 + i) * kstride + (size_t)nt0 * TW;
        uint32_t w[NT][LW];
#pragma unroll
        for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(t0 + j * TW, lane, w[j]);
        const int k = (kt0 + i) * 16;
        const uint32_t a[4] = {ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u,
                               ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u,
                               ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u,
                               ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t b0[2], b1[2];
            decode_lane<K2, CB>(w[j], lane, b0, b1);
            mma16816(acc[j][0], a, b0);
            mma16816(acc[j][1], a, b1);
        }
    }
}

// Program (expert u, n block, split, member tile): 16 members times W_q over the split's K range, summed in warp order.
template <int CB, int NT, int W>
__device__ __forceinline__ void grouped_body(const GroupedArgs a, int u, int nblock, int zz) {
    if (u >= a.ucount[0]) return;
    const int MT = (a.maxm + 15) / 16;
    const int mtile = zz % MT;
    const int split = (zz / MT) % a.sk;
    const int mat = zz / MT / a.sk;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? a.tp1[e] : a.tp0[e]);
    const int k2 = mat ? a.k2_1[e] : a.k2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = a.ns > 0 ? a.ns : (a.n >> 4); // row stride of the trellis, in tiles

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = m < a.maxm ? a.members[u * a.maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * a.slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return; // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * a.k + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * a.k + 2 * t;
    const int per_warp = (a.k >> 4) / a.sk / W;
    const int kt0 = split * (per_warp * W) + warp * per_warp;
    const int nt0 = nblock * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    __shared__ __align__(16) float red[W][16][NT * 16];
    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                   \
    case K2_:                                                                                                \
        warp_tiles<CB, K2_, NT>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);          \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        a.z[(((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col] = s;
    }
}

}  // namespace tf_exl3

// ---------------------------------------------------------------------------------------------------------------

extern "C" __global__ void tf_exl3_grouped_0(tf_exl3::GroupedArgs a) {  // 3inst, any width
    tf_exl3::grouped_body<0, 8, 4>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}

extern "C" __global__ void tf_exl3_grouped_1(tf_exl3::GroupedArgs a) {  // mcg, any width
    tf_exl3::grouped_body<1, 8, 4>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}

extern "C" __global__ void tf_exl3_grouped_2(tf_exl3::GroupedArgs a) {  // mul1, any width (RED-SNOW's experts)
    tf_exl3::grouped_body<2, 8, 4>(a, blockIdx.x, blockIdx.y, blockIdx.z);
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

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

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
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(a.svh_g[(size_t)e * a.n + n + j])), a.limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(a.svh_u[(size_t)e * a.n + n + j])),
                                    -a.limit), a.limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(a.svh_g[(size_t)e * a.n + n + j]), a.limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(a.svh_u[(size_t)e * a.n + n + j]), -a.limit),
                             a.limit);
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

// Copy of src/tensorfold/families/deepseek_v41/cuda/router_gemv.cu (blob 5841ced53bf5): sass_check.sh
#define DSV41_RG_NO_TORCH



// DeepSeek-V4.1 MoE router, one GEMV + top-K launch: each logit is one fp32 FMA chain + xor butterfly, any tiling
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace dsv41_rg {

constexpr int NV = 20;              // 16-byte vectors a lane a gate row: D = 32 x 8 x NV = 5,120
constexpr int D = 32 * 8 * NV;
constexpr int JC = 2;               // vectors a lane a chunk
constexpr int NC = NV / JC;         // chunks of x a row tile
constexpr int CK = 32 * 8 * JC;     // k a chunk (512)
constexpr int EPL = 16;             // experts a lane in the selection: E <= 512
constexpr int KMAX = 8;             // top-k at most
static_assert(NV % JC == 0, "whole chunks");

__device__ __forceinline__ uint4 ldg_stream(const uint16_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// 8 bf16 (k ascending: the low half of each word first) -> fp32, exactly
__device__ __forceinline__ void widen8(const uint4 v, float (&f)[8]) {
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xffff0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xffff0000u);
    f[4] = __uint_as_float(v.z << 16); f[5] = __uint_as_float(v.z & 0xffff0000u);
    f[6] = __uint_as_float(v.w << 16); f[7] = __uint_as_float(v.w & 0xffff0000u);
}

// the reference router's safe softplus, operation for operation (no contraction: the intrinsics are never fused)
__device__ __forceinline__ float softplus(float z) {
    const float e = expf(fminf(z, 20.0f));
    if (z > 20.0f) return z;
    if (z < -10.0f) return __fsub_rn(e, __fmul_rn(0.5f, __fmul_rn(e, e)));
    return logf(__fadd_rn(1.0f, e));
}

// TF_DSV41_RG_PRUNE: the top-p prune folded into the selection, its operations in order on the row's K weights
struct Prune {
    float min_w, topp, scale;
    int min_k, orig, drop, dup, on;
};

__device__ __forceinline__ void prune_row(const Prune& P, int K, float (&w)[KMAX], int (&p)[KMAX]) {
    float total = 0.0f, f[KMAX], excl[KMAX];
    int rank[KMAX];
    bool keep[KMAX];
#pragma unroll
    for (int i = 0; i < KMAX; ++i) if (i < K) total = __fadd_rn(total, w[i]);
#pragma unroll
    for (int i = 0; i < KMAX; ++i) if (i < K) f[i] = __fdiv_rn(w[i], total);
#pragma unroll
    for (int i = 0; i < KMAX; ++i) {
        rank[i] = 0;
#pragma unroll
        for (int j = 0; j < KMAX; ++j)
            if (i < K && j < K) rank[i] += (f[j] > f[i]) || (f[j] == f[i] && j < i);
    }
    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < KMAX; ++r)
#pragma unroll
        for (int i = 0; i < KMAX; ++i)
            if (r < K && i < K && rank[i] == r) { excl[i] = acc; acc = __fadd_rn(acc, f[i]); }
    int nk = 0, top = 0;
    float ks = 0.0f;
#pragma unroll
    for (int i = 0; i < KMAX; ++i) {
        if (i >= K) break;
        keep[i] = ((excl[i] < P.topp) && (f[i] >= P.min_w)) || (rank[i] < P.min_k);
        nk += keep[i];
        if (rank[i] == 0) top = p[i];
        ks = __fadd_rn(ks, keep[i] ? w[i] : 0.0f);
    }
    if (nk >= K) return;
#pragma unroll
    for (int i = 0; i < KMAX; ++i) {
        if (i >= K) break;
        w[i] = keep[i] ? (P.orig ? w[i] : __fmul_rn(P.scale, __fdiv_rn(w[i], ks))) : 0.0f;
        if (!keep[i]) p[i] = P.dup ? top : P.drop;
    }
}

// Prune pass of one selected row (lane 0); noinline so its arrays never share a live range with select_row's
__device__ __noinline__ void prune_pass(const Prune& P, int K, int kit, int* pick, float* wts, int* psh) {
    float wv[KMAX];
    int ip[KMAX];
#pragma unroll
    for (int t = 0; t < KMAX; ++t) {
        wv[t] = t < K ? wts[t] : 0.0f;
        ip[t] = t < K ? pick[t] : 0;
    }
    prune_row(P, K, wv, ip);
#pragma unroll
    for (int t = 0; t < KMAX; ++t) {
        if (t >= K) break;
        pick[t] = ip[t];
        wts[t] = kit ? __half2float(__float2half_rn(wv[t])) : wv[t];
        psh[t] = ip[t];
    }
}

// One row's selection by one warp (lane l holds experts l + 32 i).
__device__ __forceinline__ void select_row(const float* lg, const float* __restrict__ bias, int E, int K, int slots,
                                           float scale, int kit, int* pick, float* wts, int* psh, int lane,
                                           const Prune& P) {
    float s[EPL], ch[EPL];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
        const int e = lane + 32 * i;
        if (e < E) {
            s[i] = __fsqrt_rn(softplus(__ldcg(lg + e)));
            ch[i] = __fadd_rn(s[i], bias[e]);
        } else {
            s[i] = 0.0f;
            ch[i] = -INFINITY;
        }
    }
    float sp[KMAX];
    int ip[KMAX];
    float total = 0.0f;
#pragma unroll
    for (int t = 0; t < KMAX; ++t) {
        if (t >= K) break;
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < EPL; ++i)               // ids ascend with i: strict > keeps the lower id on a tie
            if (ch[i] > bv) { bv = ch[i]; bi = lane + 32 * i; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {          // (score desc, id asc): a total order, every lane ends equal
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int i = 0; i < EPL; ++i)
            if (lane + 32 * i == bi) { mine = s[i]; ch[i] = -INFINITY; }
        sp[t] = __shfl_sync(0xffffffffu, mine, bi & 31);
        ip[t] = bi;
        total = __fadd_rn(total, sp[t]);
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < KMAX; ++t) {
            if (t >= K) break;
            float w = __fmul_rn(scale, __fdiv_rn(sp[t], total));
            if (kit && !P.on) w = __half2float(__float2half_rn(w));   // pick.kit_weights (with prune: after it)
            pick[t] = ip[t];
            wts[t] = w;
            psh[t] = ip[t];
        }
        if (slots > K) { pick[K] = E; wts[K] = 1.0f; psh[K] = E; }
    }
}

struct Args {
    const uint16_t* x;      // [R, x_stride] bf16, rows 16-byte aligned
    long long xs;
    const uint16_t* w;      // [E, D] bf16
    const float* bias;      // [E]
    int* pick;              // [R, slots]
    float* wts;             // [R, slots]
    float* lg;              // [R, E] fp32 (scratch, or the caller's logits)
    int* cnt;               // [>= grid.y] int32, 0 between launches
    int R, E, K, slots, select;
    float scale;
    int kit;                // the routed weights rounded to fp16 (pick.kit_weights, folded)
    int* gid;               // group != 0: upstream's group kernel folded (one row tile only): distinct ids (< GE)
    int* gcnt;              //   in id order, their count, members [., maxm] row * 32 + slot in row order, -1 after
    int* gmem;
    int GE, maxm, group;
    Prune prune;            // the top-p prune rule folded into the selection (on = 0: off)
};

constexpr int GE_MAX = 512;         // table entries the folded grouping takes
constexpr int PICKS_MAX = 16 * (KMAX + 1);
constexpr int TAIL_INTS = PICKS_MAX;                            // the tail's picks (both kernels)
constexpr int GROUP_INTS = TAIL_INTS + 2 * GE_MAX + PICKS_MAX + 8;  // + the grouping's (the narrow kernel)

// Upstream's group_kernel (exl3/experts.cu) on the R <= 16 rows of one row tile, by the selecting CTA; integers only
template <int NW>
__device__ __forceinline__ void group_rows(const Args& a, const int* psh, int* sh, int tid) {
    constexpr int T = NW * 32;
    int* gc = sh;                                     // the main loop's shared memory, reused
    int* gpl = gc + GE_MAX;
    int* cpl = gpl + GE_MAX;
    int* wtot = cpl + PICKS_MAX;
    const int n = a.R * a.slots, GE = a.GE, lane = tid & 31, warp = tid >> 5;
    for (int e = tid; e < GE; e += T) gc[e] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += T) {
        const int e = psh[i];
        if (e >= 0 && e < GE) atomicAdd(gc + e, 1);
    }
    __syncthreads();
    const int per = (GE + T - 1) / T, lo = min(GE, tid * per), hi = min(GE, lo + per);
    int used = 0;
    for (int e = lo; e < hi; ++e) used += gc[e] > 0;
    int inc = used;                                   // inclusive scan in the warp, then over warps
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) wtot[warp] = inc;
    __syncthreads();
    int base = 0, total = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        base += w < warp ? wtot[w] : 0;
        total += wtot[w];
    }
    int place = base + inc - used;
    for (int e = lo; e < hi; ++e)
        if (gc[e] > 0) {
            a.gid[place] = e;
            gpl[e] = place;
            cpl[place] = gc[e];
            ++place;
        }
    if (tid == 0) a.gcnt[0] = total;
    __syncthreads();
    for (int i = tid; i < n; i += T) {               // rank of pick i among its expert's picks: row order
        const int e = psh[i];
        if (e < 0 || e >= GE) continue;
        int rank = 0;
        for (int q = 0; q < i; ++q) rank += psh[q] == e;
        if (rank < a.maxm) a.gmem[(size_t)gpl[e] * a.maxm + rank] = (i / a.slots) * 32 + i % a.slots;
    }
    for (int q = tid; q < total * a.maxm; q += T)
        if (q % a.maxm >= cpl[q / a.maxm]) a.gmem[q] = -1;
}

// Fused tail: the row tile's last CTA selects its rows, groups them (GROUP), resets the counter; sh: TAIL_INTS ints
template <int NW, bool GROUP>
__device__ __forceinline__ void tail(const Args& a, int r0, int nr, int tid, int* sh) {
    __shared__ int last_sh;
    int* psh = sh;
    const int lane = tid & 31, warp = tid >> 5;
    __threadfence();                                 // this CTA's logits visible device-wide before its arrival
    __syncthreads();                                 // (also: every thread is past the main loop's shared memory)
    if (tid == 0) last_sh = atomicAdd(a.cnt + blockIdx.y, 1) == (int)gridDim.x - 1;
    __syncthreads();
    if (!last_sh) return;
    __threadfence();
    for (int r = warp; r < nr; r += NW)
        select_row(a.lg + (size_t)(r0 + r) * a.E, a.bias, a.E, a.K, a.slots, a.scale, a.kit,
                   a.pick + (size_t)(r0 + r) * a.slots, a.wts + (size_t)(r0 + r) * a.slots, psh + r * a.slots, lane,
                   a.prune);
    if (a.prune.on && lane == 0)                    // the rule, then the kit rounding (each lane 0: its own rows)
        for (int r = warp; r < nr; r += NW)
            prune_pass(a.prune, a.K, a.kit, a.pick + (size_t)(r0 + r) * a.slots, a.wts + (size_t)(r0 + r) * a.slots,
                       psh + r * a.slots);
    if (GROUP && a.group) {
        __syncthreads();
        group_rows<NW>(a, psh, sh + TAIL_INTS, tid);
    }
    if (tid == 0) a.cnt[blockIdx.y] = 0;             // self-cleaning
}

// RT rows a tile (1 .. 16), NW warps a CTA, EW experts a warp.
template <int RT, int NW, int EW>
__global__ void __launch_bounds__(NW * 32) gemv_kernel(const Args a) {
    constexpr int PD = 8 / EW;                      // vectors in flight a lane, per expert
    constexpr int XU = RT * CK / 8;                 // uint4 of x a chunk
    constexpr int XT = (XU + NW * 32 - 1) / (NW * 32);
    static_assert(PD < NV, "ring shorter than a row");
    __shared__ __align__(16) uint16_t xsh[2][RT][CK];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.y * RT;
    const int nr = min(RT, a.R - r0);
    const int e0 = (blockIdx.x * NW + warp) * EW;

    auto wload = [&](int j, uint4 (&dst)[EW]) {
#pragma unroll
        for (int e = 0; e < EW; ++e) {
            if (e0 + e < a.E) dst[e] = ldg_stream(a.w + (size_t)(e0 + e) * D + (size_t)(j * 32 + lane) * 8);
            else dst[e] = make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto xload = [&](int c, uint4 (&xr)[XT]) {
#pragma unroll
        for (int q = 0; q < XT; ++q) {
            const int u = tid + NW * 32 * q, row = u / (CK / 8), col = (u % (CK / 8)) * 8;
            xr[q] = make_uint4(0u, 0u, 0u, 0u);
            if (u < XU && row < nr)
                xr[q] = __ldg(reinterpret_cast<const uint4*>(a.x + (size_t)(r0 + row) * a.xs + c * CK + col));
        }
    };
    auto xstore = [&](int buf, const uint4 (&xr)[XT]) {
#pragma unroll
        for (int q = 0; q < XT; ++q) {
            const int u = tid + NW * 32 * q, row = u / (CK / 8), col = (u % (CK / 8)) * 8;
            if (u < XU) *reinterpret_cast<uint4*>(&xsh[buf][row][col]) = xr[q];
        }
    };

    float acc[EW][RT];
#pragma unroll
    for (int e = 0; e < EW; ++e)
#pragma unroll
        for (int r = 0; r < RT; ++r) acc[e][r] = 0.0f;
    uint4 ring[PD][EW];
#pragma unroll
    for (int j = 0; j < PD; ++j) wload(j, ring[j]);
    uint4 xr[XT];
    xload(0, xr);
    xstore(0, xr);
    __syncthreads();

#pragma unroll
    for (int j = 0; j < NV; ++j) {
        const int c = j / JC, jj = j % JC;
        if (jj == 0 && c + 1 < NC) xload(c + 1, xr);           // next chunk's x: L2, consumed at the chunk's end
        float wf[EW][8];
#pragma unroll
        for (int e = 0; e < EW; ++e) widen8(ring[j % PD][e], wf[e]);
        if (j + PD < NV) wload(j + PD, ring[j % PD]);           // the slot refilled PD steps ahead
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            float xf[8];
            widen8(*reinterpret_cast<const uint4*>(&xsh[c & 1][r][(jj * 32 + lane) * 8]), xf);
#pragma unroll
            for (int e = 0; e < EW; ++e)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[e][r] = __fmaf_rn(xf[i], wf[e][i], acc[e][r]);
        }
        if (jj == JC - 1) {
            if (c + 1 < NC) xstore((c + 1) & 1, xr);           // the buffer chunk c - 1 read (all past its barrier)
            __syncthreads();
        }
    }

#pragma unroll
    for (int e = 0; e < EW; ++e)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            float v = acc[e][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
            if (lane == r && r < nr && e0 + e < a.E) a.lg[(size_t)(r0 + r) * a.E + e0 + e] = v;
        }
    static_assert(sizeof(xsh) >= TAIL_INTS * sizeof(int), "the tail's picks fit the staged x");
    if (a.select) tail<NW, false>(a, r0, nr, tid, reinterpret_cast<int*>(&xsh[0][0][0]));
}

#include "router_narrow.cuh"

template <int RT, int NW>
cudaError_t launch_narrow(const Args& a, cudaStream_t stream) {
    static const bool carved = [] {                // the most shared memory a SM (8 / NW CTAs resident at once); a
        cudaFuncSetAttribute(narrow_kernel<RT, NW>,  // preference only: an error here is cleared, never fatal
                             cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);
        (void)cudaGetLastError();
        return true;
    }();
    (void)carved;
    const dim3 grid((unsigned)((a.E + NW - 1) / NW), (unsigned)((a.R + RT - 1) / RT));
    narrow_kernel<RT, NW><<<grid, NW * 32, 0, stream>>>(a);
    return cudaGetLastError();
}

template <int NW>
cudaError_t by_rt_narrow(const Args& a, int rt, cudaStream_t stream) {
    switch (rt) {
    case 1: return launch_narrow<1, NW>(a, stream);
    case 2: return launch_narrow<2, NW>(a, stream);
    case 4: return launch_narrow<4, NW>(a, stream);
    case 8: return launch_narrow<8, NW>(a, stream);
    case 16: return launch_narrow<16, NW>(a, stream);
    default: return cudaErrorInvalidValue;
    }
}

template <int RT, int NW, int EW>
cudaError_t launch(const Args& a, cudaStream_t stream) {
    const dim3 grid((unsigned)((a.E + NW * EW - 1) / (NW * EW)), (unsigned)((a.R + RT - 1) / RT));
    gemv_kernel<RT, NW, EW><<<grid, NW * 32, 0, stream>>>(a);
    return cudaGetLastError();
}

template <int NW, int EW>
cudaError_t by_rt(const Args& a, int rt, cudaStream_t stream) {
    switch (rt) {
    case 1: return launch<1, NW, EW>(a, stream);
    case 2: return launch<2, NW, EW>(a, stream);
    case 4: return launch<4, NW, EW>(a, stream);
    case 8: return launch<8, NW, EW>(a, stream);
    case 16: return launch<16, NW, EW>(a, stream);
    default: return cudaErrorInvalidValue;
    }
}

// rt rows a tile (1, 2, 4, 8, 16); (nw, ew) in {4, 8} x {1, 2}, or ew == 0: narrow, nw in {1, 2, 4, 8}
inline cudaError_t dispatch(const Args& a, int rt, int nw, int ew, cudaStream_t stream) {
    if (ew == 0) {
        switch (nw) {
        case 1: return by_rt_narrow<1>(a, rt, stream);
        case 2: return by_rt_narrow<2>(a, rt, stream);
        case 4: return by_rt_narrow<4>(a, rt, stream);
        case 8: return by_rt_narrow<8>(a, rt, stream);
        default: return cudaErrorInvalidValue;
        }
    }
    if (nw == 8 && ew == 1) return by_rt<8, 1>(a, rt, stream);
    if (nw == 4 && ew == 1) return by_rt<4, 1>(a, rt, stream);
    if (nw == 8 && ew == 2) return by_rt<8, 2>(a, rt, stream);
    if (nw == 4 && ew == 2) return by_rt<4, 2>(a, rt, stream);
    return cudaErrorInvalidValue;
}

}  // namespace dsv41_rg

#ifndef DSV41_RG_NO_TORCH       // the compile test builds the kernels alone (nvcc, no torch headers)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

void dsv41_rg_route_cuda(const at::Tensor& X, const at::Tensor& W, const at::Tensor& B, at::Tensor& pick,
                         at::Tensor& wts, at::Tensor& LG, at::Tensor& cnt, int64_t K, int64_t slots, double scale,
                         int64_t rt, int64_t nw, int64_t ew, bool select, bool kit, at::Tensor& gid, at::Tensor& gcnt,
                         at::Tensor& gmem, int64_t GE, bool group, const std::vector<double>& prune) {
    using namespace dsv41_rg;
    TORCH_CHECK(prune.empty() || prune.size() == 7, "router_gemv: prune = [min_w, topp, scale, min_k, orig, drop, dup]");
    const int64_t R = X.size(0), E = W.size(0);
    TORCH_CHECK(X.size(1) == D && W.size(1) == D, "router_gemv: D must be ", D);
    TORCH_CHECK(X.stride(1) == 1 && X.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(X.data_ptr()) % 16 == 0,
                "router_gemv: x rows must be unit-stride and 16-byte aligned");
    TORCH_CHECK(E >= 1 && E <= 32 * EPL && K >= 1 && K <= KMAX && K <= E && slots >= K && slots <= K + 1,
                "router_gemv: E <= ", 32 * EPL, ", 1 <= K <= ", KMAX, ", slots K or K + 1");
    TORCH_CHECK(LG.numel() >= R * E, "router_gemv: logits buffer too small");
    TORCH_CHECK(!select || cnt.numel() >= (R + rt - 1) / rt, "router_gemv: counters too few for ", R, " rows");
    const bool wide = (nw == 4 || nw == 8) && (ew == 1 || ew == 2);
    const bool narrow = ew == 0 && (nw == 1 || nw == 2 || nw == 4 || nw == 8);
    TORCH_CHECK((rt == 1 || rt == 2 || rt == 4 || rt == 8 || rt == 16) && (wide || narrow),
                "router_gemv: no instance (rt, nw, ew) = (", rt, ", ", nw, ", ", ew, ")");
    TORCH_CHECK(!group || (select && narrow && R <= rt && GE >= 1 && GE <= GE_MAX && gid.numel() >= std::min(R * slots, GE)
                           && gcnt.numel() >= 1 && gmem.dim() == 2 && gmem.size(0) >= std::min(R * slots, GE)
                           && gmem.size(1) >= 1),
                "router_gemv: the folded grouping takes one row tile, GE <= ", GE_MAX, " and ids / members sized");
    if (R == 0) return;
    Args a;
    a.x = reinterpret_cast<const uint16_t*>(X.data_ptr());
    a.xs = X.stride(0);
    a.w = reinterpret_cast<const uint16_t*>(W.data_ptr());
    a.bias = B.data_ptr<float>();
    a.pick = pick.data_ptr<int>();
    a.wts = wts.data_ptr<float>();
    a.lg = LG.data_ptr<float>();
    a.cnt = cnt.data_ptr<int>();
    a.R = (int)R; a.E = (int)E; a.K = (int)K; a.slots = (int)slots; a.select = select ? 1 : 0;
    a.scale = (float)scale;
    a.kit = kit ? 1 : 0;
    a.group = group ? 1 : 0;
    a.gid = group ? gid.data_ptr<int>() : nullptr;
    a.gcnt = group ? gcnt.data_ptr<int>() : nullptr;
    a.gmem = group ? gmem.data_ptr<int>() : nullptr;
    a.GE = (int)GE;
    a.maxm = group ? (int)gmem.size(1) : 0;          // upstream's Scratch.window: members [maxu, R]
    a.prune = Prune{};
    if (!prune.empty()) {                             // fp32 thresholds: the rounding of Triton's float arguments
        a.prune.min_w = (float)prune[0]; a.prune.topp = (float)prune[1]; a.prune.scale = (float)prune[2];
        a.prune.min_k = (int)prune[3]; a.prune.orig = (int)prune[4]; a.prune.drop = (int)prune[5];
        a.prune.dup = (int)prune[6]; a.prune.on = 1;
    }
    C10_CUDA_CHECK(dispatch(a, (int)rt, (int)nw, (int)ew, at::cuda::getCurrentCUDAStream()));
}
#endif

// The instantiations dispatch() launches: gemv_kernel<RT, NW, EW> and narrow_kernel<RT, NW>.
template __global__ void dsv41_rg::gemv_kernel<1, 8, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<1, 4, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<1, 8, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<1, 4, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<1, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<1, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<1, 4>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<1, 8>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<2, 8, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<2, 4, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<2, 8, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<2, 4, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<2, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<2, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<2, 4>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<2, 8>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<4, 8, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<4, 4, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<4, 8, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<4, 4, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<4, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<4, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<4, 4>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<4, 8>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<8, 8, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<8, 4, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<8, 8, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<8, 4, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<8, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<8, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<8, 4>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<8, 8>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<16, 8, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<16, 4, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<16, 8, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::gemv_kernel<16, 4, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<16, 1>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<16, 2>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<16, 4>(const dsv41_rg::Args);
template __global__ void dsv41_rg::narrow_kernel<16, 8>(const dsv41_rg::Args);

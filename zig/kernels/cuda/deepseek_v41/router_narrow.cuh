// router_gemv.cu's narrow_kernel and its cp.async helpers, inside namespace dsv41_rg
#pragma once
// x through L1 (ld.global.nc), volatile so it stays in the step loop (a plain __ldg was hoisted, spilling registers)
__device__ __forceinline__ uint4 ldg_keep(const uint16_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
// n a constant once the step loop is unrolled: the switch folds to one wait_group
__device__ __forceinline__ void cp_wait_n(int n) {
    switch (n) {
#define DSV41_RG_W(k) case k: cp_wait<k>(); break;
    DSV41_RG_W(0) DSV41_RG_W(1) DSV41_RG_W(2) DSV41_RG_W(3) DSV41_RG_W(4) DSV41_RG_W(5) DSV41_RG_W(6) DSV41_RG_W(7)
    DSV41_RG_W(8) DSV41_RG_W(9) DSV41_RG_W(10) DSV41_RG_W(11) DSV41_RG_W(12) DSV41_RG_W(13) DSV41_RG_W(14)
    DSV41_RG_W(15) DSV41_RG_W(16) DSV41_RG_W(17) DSV41_RG_W(18) DSV41_RG_W(19)
#undef DSV41_RG_W
    default: cp_wait<0>(); break;
    }
}

// Narrow kernel (TF_DSV41_RG_NARROW = NW, rows <= 16): gemv_kernel's chains and bits; gate via cp.async, x via L1
template <int RT, int NW>
__global__ void __launch_bounds__(NW * 32) narrow_kernel(const Args a) {
    constexpr int PD = NW <= 4 && RT <= 4 ? NV : NV / 2;   // gate vectors in flight a lane
    constexpr int XP = RT <= 4 ? 2 : 0;             // steps of x in flight (0: loaded at use, L1)
    constexpr int TS = (GROUP_INTS * 4 + NW * 512 - 1) / (NW * 512);   // slots the tail needs
    __shared__ __align__(16) uint4 wsh[NW][PD > TS ? PD : TS][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.y * RT;
    const int nr = min(RT, a.R - r0);
    const int e = blockIdx.x * NW + warp;
    const bool live = e < a.E;
    const uint16_t* wrow = a.w + (size_t)(live ? e : 0) * D + (size_t)lane * 8;

    auto wload = [&](int j) {                       // a dead warp's slots are never read: committed empty
        if (live) cp_async16(&wsh[warp][j % PD][lane], wrow + (size_t)j * 256);
        cp_commit();
    };
    auto xload = [&](int j, uint4 (&dst)[RT]) {
#pragma unroll
        for (int r = 0; r < RT; ++r)
            dst[r] = (live && r < nr)
                ? ldg_keep(a.x + (size_t)(r0 + r) * a.xs + (size_t)(j * 32 + lane) * 8)
                : make_uint4(0u, 0u, 0u, 0u);
    };

#pragma unroll
    for (int j = 0; j < PD; ++j) wload(j);
    uint4 xr[XP > 0 ? XP : 1][RT];
#pragma unroll
    for (int j = 0; j < XP; ++j) xload(j, xr[j]);
    float acc[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) acc[r] = 0.0f;

#pragma unroll
    for (int j = 0; j < NV; ++j) {
        // groups committed before step j: min(NV, PD + j); group j done = at most that - j - 1 pending
        cp_wait_n(min(NV - j - 1, PD - 1));
        float wf[8];
        widen8(wsh[warp][j % PD][lane], wf);
        uint4 xc[RT];
        if constexpr (XP == 0) {
            xload(j, xc);
        } else {
#pragma unroll
            for (int r = 0; r < RT; ++r) xc[r] = xr[j % XP][r];
            if (j + XP < NV) xload(j + XP, xr[j % XP]);
        }
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            float xf[8];
            widen8(xc[r], xf);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r] = __fmaf_rn(xf[i], wf[i], acc[r]);
        }
        if (j + PD < NV) wload(j + PD);             // the slot just read (its value is in wf: used above)
    }

#pragma unroll
    for (int r = 0; r < RT; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == r && r < nr && live) a.lg[(size_t)(r0 + r) * a.E + e] = v;
    }
    if (a.select) tail<NW, true>(a, r0, nr, tid, reinterpret_cast<int*>(&wsh[0][0][0]));
}

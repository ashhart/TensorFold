// EXL3 routed experts in prompt windows: each weight tile decoded once for up to 64 pairs, every pair's bits kept.
// Program (n block, plan item, matrix): its warps decode the block's tiles into shared memory a chunk at a time, each
// runs 16 pairs over the whole K with grouped_kernel's arithmetic (the same m16n8k16 chains over the same K segments,
// segments added in warp order, splits added from 0.f), so no split partials reach memory. A arrives by cp.async and
// leaves by ldmatrix (the same halves, padding rows zero).
#pragma once

#include "experts_grouped.cuh"

namespace tf_exl3x {

constexpr int PROMPT_WARPS = 4;                        // member tiles an item holds (16 pairs a warp): 64-pair items
constexpr float HAD_SCALE_P = 0.08838834764831845f;   // 1 / sqrt(128), as experts.cu's HAD_SCALE

struct PromptArgs {
    const half* x0;
    const half* x1;
    const int64_t* tp0;
    const int64_t* tp1;
    const int* k2_0;
    const int* k2_1;
    const int* items;
    const int* counts;
    const int* members;
    float* z;                // [mats, P, N] final sums
    int K, N, P, E;
    int seg, wps;            // k tiles a segment (grouped_kernel's per_warp), segments a split (its warps)
    int items_max, mats, nt;
};

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr));
}

// This warp's 16 rows x CH k tiles of A into its stage (rows of CH * 32 bytes, 16-byte chunks XOR-swizzled by row).
template <int CH>
__device__ __forceinline__ void stage_a(uint32_t abuf, const half* __restrict__ X, const int* rows16, int K, int kt0,
                                        int KT, int lane) {
    constexpr int CPR = CH * 2;                            // 16-byte chunks a row
#pragma unroll
    for (int i = 0; i < (16 * CPR) / 32; ++i) {
        const int idx = lane + 32 * i, r = idx / CPR, c = idx % CPR;
        const int row = rows16[r];
        const int kt = kt0 + c / 2;
        const bool ok = row >= 0 && kt < KT;
        const half* src = X + (size_t)(ok ? row : 0) * K + (size_t)(ok ? kt0 * 16 + c * 8 : 0);
        cp_async16(abuf + r * (CPR * 16) + ((c ^ (r & 7)) % CPR) * 16, src, ok);
    }
}

// TOT: also keep the splits' running sum (gate/up's SK splits); without it the result is the first split's sum
// (one split: down), which the caller adds to 0.f as the epilogue does.
template <int CB, int K2, int NT, int CH, bool TOT>
__device__ __forceinline__ void prompt_body(const uint32_t* __restrict__ T, const half* __restrict__ X,
                                            const int* rows16, int K, int N, int nt0, int seg, int wps, bool active,
                                            int warp, int lane, uint4* sb, uint32_t abase, float (&res)[NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    constexpr int DPW = NT / PROMPT_WARPS;              // n tiles this warp decodes a k tile
    constexpr int ABYTES = 16 * CH * 32;                // a warp's A stage
    static_assert(NT % PROMPT_WARPS == 0, "NT must be a multiple of the warps");
    const LaneMap<K2> map(lane);
    const int KT = K >> 4, NTILES = N >> 4;
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + (size_t)(nt0 + warp * DPW) * TW + lane;
    const uint32_t abuf0 = abase + warp * 2 * ABYTES;

    float acc[NT][2][4], ts[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = ts[i][h][c] = res[i][h][c] = 0.f;

    uint32_t wn[CH][DPW][LW];
#pragma unroll
    for (int kc = 0; kc < CH; ++kc)
        if (kc < KT)
#pragma unroll
            for (int j = 0; j < DPW; ++j) load_words<K2>(wn[kc][j], tp + (size_t)kc * kstride + j * TW, lane);
    if (active) stage_a<CH>(abuf0, X, rows16, K, 0, KT, lane);
    cp_async_commit();

    int sleft = seg, g = 0;                              // k tiles left in this segment, the segment
    const int chunks = (KT + CH - 1) / CH;
    for (int c = 0; c < chunks; ++c) {
        uint4* buf = sb + (size_t)(c & 1) * CH * NT * 32;
        // decode this chunk's tiles of this warp's n tiles into shared memory
#pragma unroll
        for (int kc = 0; kc < CH; ++kc)
            if (c * CH + kc < KT)
#pragma unroll
                for (int j = 0; j < DPW; ++j) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(wn[kc][j], map, lane, b0, b1);
                    buf[(kc * NT + warp * DPW + j) * 32 + lane] = make_uint4(b0[0], b0[1], b1[0], b1[1]);
                }
        // the next chunk's words and A rows in flight while this one runs
#pragma unroll
        for (int kc = 0; kc < CH; ++kc) {
            const int kt = (c + 1) * CH + kc;
            if (kt < KT)
#pragma unroll
                for (int j = 0; j < DPW; ++j) load_words<K2>(wn[kc][j], tp + (size_t)kt * kstride + j * TW, lane);
        }
        __syncwarp();                                    // every lane is done reading the stage it overwrites
        if (active && c + 1 < chunks)
            stage_a<CH>(abuf0 + ((c + 1) & 1) * ABYTES, X, rows16, K, (c + 1) * CH, KT, lane);
        cp_async_commit();
        cp_async_wait<1>();                              // this chunk's A rows have landed
        __syncthreads();
        if (active) {
            const uint32_t abuf = abuf0 + (c & 1) * ABYTES;
            const int ar = lane & 15;
#pragma unroll
            for (int kc = 0; kc < CH; ++kc) {
                const int kt = c * CH + kc;
                if (kt < KT) {
                    uint32_t a[4];
                    const int ch = kc * 2 + (lane >> 4);
                    ldmatrix_x4(a, abuf + ar * (CH * 32) + ((ch ^ (ar & 7)) % (CH * 2)) * 16);
#pragma unroll
                    for (int i = 0; i < NT; ++i) {
                        const uint4 b = buf[(kc * NT + i) * 32 + lane];
                        const uint32_t b0[2] = {b.x, b.y}, b1[2] = {b.z, b.w};
                        mma16816(acc[i][0], a, b0);
                        mma16816(acc[i][1], a, b1);
                    }
                    if (--sleft == 0) {                  // a segment ends: grouped_kernel's warp, then split sums
                        const int wi = g % wps;
#pragma unroll
                        for (int i = 0; i < NT; ++i)
#pragma unroll
                            for (int h = 0; h < 2; ++h)
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    ts[i][h][q] = wi == 0 ? acc[i][h][q] : ts[i][h][q] + acc[i][h][q];
                                    acc[i][h][q] = 0.f;
                                    if constexpr (TOT) {
                                        if (wi == wps - 1) res[i][h][q] += ts[i][h][q];
                                    }
                                }
                        sleft = seg;
                        ++g;
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    if constexpr (!TOT) {
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < 4; ++q) res[i][h][q] = ts[i][h][q];
    }
}

// k tiles a stage: B (2 * CH * NT tiles of 512 bytes) and A (4 warps x 2 x 16 rows x CH * 32 bytes)
template <int NT> __host__ __device__ constexpr int prompt_ch() { return NT == 8 ? 2 : 4; }

// Program (n block, plan item, matrix): the item's (expert, first member, count <= 64) pairs, warp w member tile w.
// One instance a width (K2): a layer's launch runs one per width it holds, each program skipping other widths' experts,
// so no instance carries the registers of the widest decode.
template <int CB, int K2, int NT>
__global__ void __launch_bounds__(PROMPT_WARPS * 32) prompt_kernel(PromptArgs a) {
    constexpr int CH = prompt_ch<NT>();
    const int item = blockIdx.y;
    if (item >= a.counts[0]) return;
    const int e = a.items[3 * item], first = a.items[3 * item + 1], cnt = a.items[3 * item + 2];
    if (e < 0 || e >= a.E) return;                        // a skipped pick's item
    const int mat = blockIdx.z;
    if ((mat ? a.k2_1[e] : a.k2_0[e]) != K2) return;
    const half* X = mat ? a.x1 : a.x0;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? a.tp1[e] : a.tp0[e]);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int nt0 = blockIdx.x * NT;

    __shared__ uint4 sb[2 * CH * NT * 32];
    __shared__ __align__(128) uint4 sa[PROMPT_WARPS * 2 * 16 * CH * 2];
    __shared__ int rows_sh[PROMPT_WARPS * 16];
    for (int i = threadIdx.x; i < PROMPT_WARPS * 16; i += PROMPT_WARPS * 32)
        rows_sh[i] = i < cnt ? a.members[first + i] : -1;
    __syncthreads();
    const bool active = warp * 16 < cnt;
    const int r0 = rows_sh[warp * 16 + g], r1 = rows_sh[warp * 16 + g + 8];

    float tot[NT][2][4];
    prompt_body<CB, K2, NT, CH, true>(T, X, rows_sh + warp * 16, a.K, a.N, nt0, a.seg, a.wps, active, warp,
                                            lane, sb, (uint32_t)__cvta_generic_to_shared(sa), tot);
    if (!active) return;
    float* zm = a.z + (size_t)mat * a.P * a.N;
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = (nt0 + i) * 16 + h * 8 + 2 * t;
            if (r0 >= 0)
                *reinterpret_cast<float2*>(zm + (size_t)r0 * a.N + col) = make_float2(tot[i][h][0], tot[i][h][1]);
            if (r1 >= 0)
                *reinterpret_cast<float2*>(zm + (size_t)r1 * a.N + col) = make_float2(tot[i][h][2], tot[i][h][3]);
        }
}

#define TF_EXL3P_WIDTHS(X_) \
    X_(2) X_(3) X_(4) X_(5) X_(6) X_(7) X_(8) X_(9) X_(10) X_(11) X_(12) X_(13) X_(14) X_(15) X_(16)

// widths: a bit per K2 the layer's experts hold (bit k2)
template <int CB>
void prompt_launch(const PromptArgs& a, int widths, cudaStream_t stream) {
    if (a.items_max < 1) return;
    TORCH_CHECK(a.nt == 4, "prompt experts: n tiles a program must be 4, not ", a.nt);   // 8 would spill
    dim3 grid((unsigned)(a.N / (16 * a.nt)), (unsigned)a.items_max, (unsigned)a.mats);
#define TF_PW(K2_) if (widths & (1 << K2_)) prompt_kernel<CB, K2_, 4><<<grid, PROMPT_WARPS * 32, 0, stream>>>(a);
    TF_EXL3P_WIDTHS(TF_PW)
#undef TF_PW
}

// ---------------------------------------------------------------------------------------------------------------
// Down projection with its epilogue: a program owns one 128-column Hadamard block (8 n tiles), so after the K loop each
// pair's 128 sums go through down_epilogue_kernel's arithmetic (s = 0; s += sum; fwht128; * HAD_SCALE * svh_d) and
// are stored as fp32 or rounded once to bf16 (as a copy_ of the fp32 y would): no Z, no fp32 y. One split only.

__device__ __forceinline__ void fwht128_p(float (&v)[4], int lane) {      // experts.cu's fwht128, the same order
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

struct DownArgs {
    const half* x;
    const int64_t* tp;
    const int* k2;
    const half* svh;         // [E, N]
    const int* items;
    const int* counts;
    const int* members;
    void* y;                 // [P, N] fp32 or bf16
    int K, N, P, E, seg, wps, items_max, bf16;
};

template <int CB, int K2>
__global__ void __launch_bounds__(PROMPT_WARPS * 32) prompt_down_kernel(DownArgs a) {
    constexpr int NT = 8, CH = prompt_ch<8>();
    const int item = blockIdx.y;
    if (item >= a.counts[0]) return;
    const int e = a.items[3 * item], first = a.items[3 * item + 1], cnt = a.items[3 * item + 2];
    if (e < 0 || e >= a.E) return;
    if (a.k2[e] != K2) return;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(a.tp[e]);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int nt0 = blockIdx.x * NT;

    // B stages (16 KB) and A stages (8 KB), then 4 warps x 16 rows x 128 fp32 (32 KB) for the epilogue
    __shared__ __align__(128) uint4 sm[2048];
    uint4* sb = sm;
    const uint32_t sa = (uint32_t)__cvta_generic_to_shared(sm + 2 * CH * NT * 32);
    __shared__ int rows_sh[PROMPT_WARPS * 16];
    for (int i = threadIdx.x; i < PROMPT_WARPS * 16; i += PROMPT_WARPS * 32)
        rows_sh[i] = i < cnt ? a.members[first + i] : -1;
    __syncthreads();
    const bool active = warp * 16 < cnt;

    float ts[NT][2][4];
    prompt_body<CB, K2, NT, CH, false>(T, a.x, rows_sh + warp * 16, a.K, a.N, nt0, a.seg, a.wps, active, warp,
                                          lane, sb, sa, ts);
    __syncthreads();                                       // every warp is done with the stages
    if (!active) return;
    float* st = reinterpret_cast<float*>(sm) + warp * 16 * 128;
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            *reinterpret_cast<float2*>(st + g * 128 + col) = make_float2(ts[i][h][0], ts[i][h][1]);
            *reinterpret_cast<float2*>(st + (g + 8) * 128 + col) = make_float2(ts[i][h][2], ts[i][h][3]);
        }
    __syncwarp();
    const int n = blockIdx.x * 128 + 4 * lane;
    const half* sv = a.svh + (size_t)e * a.N + n;
    const float s0 = __half2float(sv[0]), s1 = __half2float(sv[1]), s2 = __half2float(sv[2]), s3 = __half2float(sv[3]);
    for (int r = 0; r < 16; ++r) {
        const int p = rows_sh[warp * 16 + r];
        if (p < 0) break;                                  // members come first
        const float4 u = *reinterpret_cast<const float4*>(st + r * 128 + 4 * lane);
        float v[4];
        v[0] = 0.f; v[0] += u.x;                           // the epilogue's s = 0; s += Z (one split)
        v[1] = 0.f; v[1] += u.y;
        v[2] = 0.f; v[2] += u.z;
        v[3] = 0.f; v[3] += u.w;
        fwht128_p(v, lane);
        const float o0 = v[0] * HAD_SCALE_P * s0, o1 = v[1] * HAD_SCALE_P * s1;
        const float o2 = v[2] * HAD_SCALE_P * s2, o3 = v[3] * HAD_SCALE_P * s3;
        if (a.bf16) {
            __nv_bfloat162 lo2 = __floats2bfloat162_rn(o0, o1), hi2 = __floats2bfloat162_rn(o2, o3);
            uint2 w;
            w.x = *reinterpret_cast<uint32_t*>(&lo2);
            w.y = *reinterpret_cast<uint32_t*>(&hi2);
            *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(a.y) + (size_t)p * a.N + n) = w;
        } else {
            float* yp = reinterpret_cast<float*>(a.y) + (size_t)p * a.N + n;
            *reinterpret_cast<float4*>(yp) = make_float4(o0, o1, o2, o3);
        }
    }
}

template <int CB>
void prompt_down_launch(const DownArgs& a, int widths, cudaStream_t stream) {
    if (a.items_max < 1) return;
    dim3 grid((unsigned)(a.N / 128), (unsigned)a.items_max, 1);
#define TF_PD(K2_) if (widths & (1 << K2_)) prompt_down_kernel<CB, K2_><<<grid, PROMPT_WARPS * 32, 0, stream>>>(a);
    TF_EXL3P_WIDTHS(TF_PD)
#undef TF_PD
}

}  // namespace tf_exl3x

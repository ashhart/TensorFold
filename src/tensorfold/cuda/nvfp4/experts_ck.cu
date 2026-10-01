// Grouped NVFP4 experts in the checkpoint's own math, on the shared plan: each (row, slot) pair's NVFP4 row times its
// expert's NVFP4 weights (lane4.cu's words and block scales, one projection's experts stacked) on the block-scaled FP4
// mma, one fp32 chain over K a pair, so a pair's bits never depend on the other pairs. gate|up hands SiLU(gate) * up
// to down as NVFP4 rows under the expert's own down input scale (nvfp4q.cuh, as the prompt GEMM's epilogue does).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "mma4.cuh"
#include "nvfp4q.cuh"

namespace {

using namespace mma4;

constexpr int WARPS = 4;
constexpr int NJ = 4;                   // n8 tiles a warp: half a stored 64-column tile, 32 columns
constexpr int DEPTH = 4;                // K steps in flight a warp: the next step loads while this one multiplies

// NVFP4 rows: codes [*, K/2] (input 2j in byte j's low nibble), scales [K/64, mpad, 4]. slots > 0: pair p reads row
// p / slots (rows are tokens); 0: row p (rows are pairs).
struct Rows {
    const uint8_t* codes;
    const uint8_t* scales;
    int mpad;
    int slots;
};

// One projection of every expert, lane-major within each half of a stored 64-column tile: words
// [E * N/64, K/64, 2, 32, 4] uint2 (half, lane, n8 tile) and block scales [E * N/64, K/64, 2, 8, 4] uint32 (half,
// column in the n8 tile, n8 tile), so a lane's step is two 16-byte loads and one; and each expert's output factor
// (input scale x weight scale, fp32).
struct Mat {
    const uint4* w;
    const uint4* s;
    const float* alpha;
};

__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return __ldg(reinterpret_cast<const uint32_t*>(p)); }

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {          // weights stream through once: no L1 line
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

template <int NM>
struct Step {
    uint32_t a[4], sa;                  // rows (g, g + 8): this lane's codes; its row's four scales
    uint4 b[NM][NJ / 2];                // each matrix's n8 tiles, two a load
    uint4 sb[NM];                       // their columns' four scales, a tile a word
};

// Step kg of rows (r0, r1) and matrix tile ``tile``'s columns [32 half, 32 half + 32).
template <int NM>
__device__ __forceinline__ void load(Step<NM>& st, const Mat (&mats)[NM], size_t tile, int half, int KG, int kg,
                                     const uint8_t* p0, const uint8_t* p1, const uint8_t* srow, int mpad, bool v0,
                                     bool v1, bool vs, int lane) {
    const int g = lane >> 2, t = lane & 3;
    st.a[0] = v0 ? ld32(p0 + kg * 32 + 4 * t) : 0u;
    st.a[2] = v0 ? ld32(p0 + kg * 32 + 16 + 4 * t) : 0u;
    st.a[1] = v1 ? ld32(p1 + kg * 32 + 4 * t) : 0u;
    st.a[3] = v1 ? ld32(p1 + kg * 32 + 16 + 4 * t) : 0u;
    // lanes t = 0 (2) give row g's four scales, t = 1 (3) row g + 8's; a row past the item scales to zero
    st.sa = vs ? ld32(srow + static_cast<size_t>(kg) * mpad * 4) : 0u;
    const size_t step = (tile * KG + kg) * 2 + half;                  // this half's 1,024 code and 128 scale bytes
#pragma unroll
    for (int m = 0; m < NM; ++m) {
        const uint4* b = mats[m].w + step * 64 + lane * 2;
        st.b[m][0] = ld_nc(b);
        st.b[m][1] = ld_nc(b + 1);
        st.sb[m] = ld_nc(mats[m].s + step * 8 + g);
    }
}

template <int NM>
__device__ __forceinline__ void multiply(float (&acc)[NM][NJ][4], const Step<NM>& st) {
#pragma unroll
    for (int m = 0; m < NM; ++m)
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const uint4& b = st.b[m][j >> 1];
            const uint32_t sb = j == 0 ? st.sb[m].x : j == 1 ? st.sb[m].y : j == 2 ? st.sb[m].z : st.sb[m].w;
            mma_fp4(acc[m][j], st.a, (j & 1) ? b.z : b.x, (j & 1) ? b.w : b.y, st.sa, sb);
        }
}

// acc[m][j] += rows (g, g + 8) x matrix m's n8 tile j over all of K, in K order, DEPTH steps in flight.
template <int NM>
__device__ __forceinline__ void chain(float (&acc)[NM][NJ][4], const Mat (&mats)[NM], size_t tile, int half, int KG,
                                      int K2, const Rows& x, int r0, int r1, bool v0, bool v1, int lane) {
    const int t = lane & 3;
    const uint8_t* p0 = x.codes + static_cast<size_t>(r0) * K2;
    const uint8_t* p1 = x.codes + static_cast<size_t>(r1) * K2;
    const bool hi = t & 1, vs = hi ? v1 : v0;
    const uint8_t* srow = x.scales + static_cast<size_t>(hi ? r1 : r0) * 4;
    Step<NM> st[DEPTH];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d)
        if (d < KG) load<NM>(st[d], mats, tile, half, KG, d, p0, p1, srow, x.mpad, v0, v1, vs, lane);
    for (int k0 = 0; k0 < KG; k0 += DEPTH) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            const int kg = k0 + d;
            if (kg < KG) {
                multiply<NM>(acc, st[d]);
                if (kg + DEPTH < KG)
                    load<NM>(st[d], mats, tile, half, KG, kg + DEPTH, p0, p1, srow, x.mpad, v0, v1, vs, lane);
            }
        }
    }
}

// gate|up (EPI 1) -> down's NVFP4 input rows a pair (codes [P, NI/2], scales [NI/64, ppad, 4]); down (EPI 0: fp32,
// 3: bf16) -> out [P, N]. Items of expert ``skip`` are left alone (the shared expert, run elsewhere).
template <int EPI>
__global__ void __launch_bounds__(WARPS * 32)
    experts_ck_kernel(Rows x, Mat m0, Mat m1, const float* __restrict__ qg, int KG, int NT,
                      const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
                      void* __restrict__ out, uint8_t* __restrict__ out_scales, int ppad, int N, int skip) {
    constexpr int NM = EPI == 1 ? 2 : 1;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int units = __ldg(counts) * NT * 2, K2 = KG * 32;             // a unit: an item's 32 columns
    Mat mats[NM];
    mats[0] = m0;
    if constexpr (NM == 2) mats[1] = m1;
    for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
        const int it = unit / (2 * NT), cc = unit - it * 2 * NT, cb = cc >> 1, half = cc & 1;
        const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
        if (e == skip) continue;
        const size_t tile = static_cast<size_t>(e) * NT + cb;
        for (int q0 = 0; q0 < cnt; q0 += 16) {
            const bool v0 = q0 + g < cnt, v1 = q0 + g + 8 < cnt;
            const int p0 = v0 ? __ldg(members + first + q0 + g) : 0;
            const int p1 = v1 ? __ldg(members + first + q0 + g + 8) : 0;
            const int r0 = x.slots ? p0 / x.slots : p0, r1 = x.slots ? p1 / x.slots : p1;
            float acc[NM][NJ][4];
#pragma unroll
            for (int m = 0; m < NM; ++m)
#pragma unroll
                for (int j = 0; j < NJ; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.0f;
            chain<NM>(acc, mats, tile, half, KG, K2, x, r0, r1, v0, v1, lane);
            if constexpr (EPI == 1) {
                const float ag = __ldg(m0.alpha + e), au = __ldg(m1.alpha + e), q = __ldg(qg + e);
                uint8_t* codes = reinterpret_cast<uint8_t*>(out);
                const int ni2 = NT * 32;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const bool v = h ? v1 : v0;
                    const int p = h ? p1 : p0;
                    uint32_t sw = 0;
#pragma unroll
                    for (int b = 0; b < NJ / 2; ++b) {     // 16 columns: n8 tiles 2b, 2b + 1 of this half
                        float a[4], amax = 0.0f;
#pragma unroll
                        for (int e2 = 0; e2 < 2; ++e2)
#pragma unroll
                            for (int e1 = 0; e1 < 2; ++e1) {
                                const float gv = acc[0][2 * b + e2][2 * h + e1] * ag;
                                const float uv = acc[1][2 * b + e2][2 * h + e1] * au;
                                const float s = gv / (1.0f + expf(-gv)) * uv;     // SiLU(gate) * up, fp32
                                a[2 * e2 + e1] = s;
                                amax = fmaxf(amax, fabsf(s));
                            }
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                        const nvfp4q::Scale sc = nvfp4q::block_scale(amax, q);
                        uint32_t lo = (nvfp4q::e2m1(a[0] * sc.mul) | nvfp4q::e2m1(a[1] * sc.mul) << 4) << (8 * t);
                        uint32_t hi = (nvfp4q::e2m1(a[2] * sc.mul) | nvfp4q::e2m1(a[3] * sc.mul) << 4) << (8 * t);
                        lo |= __shfl_xor_sync(0xffffffffu, lo, 1);
                        hi |= __shfl_xor_sync(0xffffffffu, hi, 1);
                        lo |= __shfl_xor_sync(0xffffffffu, lo, 2);
                        hi |= __shfl_xor_sync(0xffffffffu, hi, 2);
                        if (t == 0 && v)
                            *reinterpret_cast<uint2*>(codes + static_cast<size_t>(p) * ni2 + cb * 32 + half * 16 +
                                                      b * 8) = make_uint2(lo, hi);
                        sw |= sc.sf8 << (8 * b);
                    }
                    // this half's two scale bytes of the 64-input group: bytes 2 half, 2 half + 1
                    if (t == 0 && v)
                        *reinterpret_cast<uint16_t*>(out_scales + (static_cast<size_t>(cb) * ppad + p) * 4 +
                                                     2 * half) = static_cast<uint16_t>(sw);
                }
            } else {
                const float al = __ldg(m0.alpha + e);
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int col = cb * 64 + half * 32 + j * 8 + 2 * t;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        if (!(h ? v1 : v0)) continue;
                        const size_t at = static_cast<size_t>(h ? p1 : p0) * N + col;
                        const float y0 = acc[0][j][2 * h] * al, y1 = acc[0][j][2 * h + 1] * al;
                        if constexpr (EPI == 0) {
                            *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + at) = make_float2(y0, y1);
                        } else {
                            *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + at) =
                                __floats2bfloat162_rn(y0, y1);
                        }
                    }
                }
            }
        }
    }
}

template <int EPI>
void launch(const Rows& x, const Mat& m0, const Mat& m1, const float* qg, int KG, int NT, const at::Tensor& items,
            const at::Tensor& counts, const at::Tensor& members, void* out, uint8_t* out_scales, int ppad, int N,
            int skip, int64_t max_units) {
    static int per_sm = 0, sms = 0;
    if (per_sm == 0) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, experts_ck_kernel<EPI>, WARPS * 32, 0);
        sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        per_sm = per_sm < 1 ? 1 : per_sm;
    }
    const int64_t need = (max_units + WARPS - 1) / WARPS;
    const int grid = static_cast<int>(need < static_cast<int64_t>(per_sm) * sms ? need
                                      : static_cast<int64_t>(per_sm) * sms);
    if (grid < 1) return;
    experts_ck_kernel<EPI><<<grid, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        x, m0, m1, qg, KG, NT, items.data_ptr<int>(), counts.data_ptr<int>(), members.data_ptr<int>(), out,
        out_scales, ppad, N, skip);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Mat mat(const at::Tensor& w, const at::Tensor& s, const at::Tensor& alpha) {
    return {reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint4*>(s.data_ptr()),
            alpha.data_ptr<float>()};
}

}  // namespace

void experts_gu_ck_cuda(const at::Tensor& xc, const at::Tensor& xs, int64_t slots, const at::Tensor& wg,
                        const at::Tensor& sg, const at::Tensor& ag, const at::Tensor& wu, const at::Tensor& su,
                        const at::Tensor& au, const at::Tensor& qg, int64_t k, int64_t ni, const at::Tensor& items,
                        const at::Tensor& counts, const at::Tensor& members, at::Tensor& codes, at::Tensor& scales,
                        int64_t skip, int64_t max_units) {
    const Rows x = {xc.data_ptr<uint8_t>(), xs.data_ptr<uint8_t>(), static_cast<int>(xs.size(1)),
                    static_cast<int>(slots)};
    launch<1>(x, mat(wg, sg, ag), mat(wu, su, au), qg.data_ptr<float>(), static_cast<int>(k / 64),
              static_cast<int>(ni / 64), items, counts, members, codes.data_ptr(), scales.data_ptr<uint8_t>(),
              static_cast<int>(scales.size(1)), static_cast<int>(ni), static_cast<int>(skip), max_units);
}

void experts_down_ck_cuda(const at::Tensor& xc, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& s,
                          const at::Tensor& alpha, int64_t k, int64_t n, const at::Tensor& items,
                          const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t skip,
                          int64_t max_units) {
    const Rows x = {xc.data_ptr<uint8_t>(), xs.data_ptr<uint8_t>(), static_cast<int>(xs.size(1)), 0};
    const Mat m = mat(w, s, alpha);
    const int KG = static_cast<int>(k / 64), NT = static_cast<int>(n / 64), N = static_cast<int>(n);
    if (out.scalar_type() == at::kFloat)
        launch<0>(x, m, m, nullptr, KG, NT, items, counts, members, out.data_ptr(), nullptr, 0, N,
                  static_cast<int>(skip), max_units);
    else
        launch<3>(x, m, m, nullptr, KG, NT, items, counts, members, out.data_ptr(), nullptr, 0, N,
                  static_cast<int>(skip), max_units);
}

// Backbone bf16 projections (fp32 accumulate, ascending K16 groups) and the routed experts' block-FP8 W8A8 products.

#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

namespace drama_gemm {

constexpr int WARPS = 4;

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// e4m3 mma needs sm_89 or newer; older images trap (Model.init refuses those devices before any launch).
__device__ __forceinline__ void mma_e4m3(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// OUT 0: bf16; 2: fp32 holding the bf16-rounded value (a bf16 torch output read back as float).
template <int OUT>
__device__ __forceinline__ void store(void* y, size_t at, float v) {
    if constexpr (OUT == 0) {
        static_cast<__nv_bfloat16*>(y)[at] = __float2bfloat16_rn(v);
    } else {
        static_cast<float*>(y)[at] = __bfloat162float(__float2bfloat16_rn(v));
    }
}

// Y[m, n] = sum_k X[m, k] W[n, k]; a warp owns NT 8-column tiles of a 16 * RT-row tile; K % 32 == 0.
template <int RT, int NT, int OUT>
__device__ __forceinline__ void gemm_bf16(const __nv_bfloat16* __restrict__ X, int ldx, int M,
                                          const __nv_bfloat16* __restrict__ W, int N, int K, void* __restrict__ Y,
                                          int ldy) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int col0 = (blockIdx.x * WARPS + warp) * (8 * NT);
    if (col0 >= N) return;
    const int row0 = blockIdx.y * 16 * RT;
    using Load = std::conditional_t<OUT == 0, uint32_t, uint4>;
    const Load* xa[RT];
    const Load* xb[RT];
    bool va[RT], vb[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        const int ra = row0 + r * 16 + g, rb = ra + 8;
        va[r] = ra < M;
        vb[r] = rb < M;
        xa[r] = reinterpret_cast<const Load*>(X + static_cast<size_t>(va[r] ? ra : 0) * ldx) + (OUT == 0 ? 0 : t);
        xb[r] = reinterpret_cast<const Load*>(X + static_cast<size_t>(vb[r] ? rb : 0) * ldx) + (OUT == 0 ? 0 : t);
    }
    const Load* wp[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int n = col0 + j * 8 < N ? col0 + j * 8 + g : g;  // dead tiles read row g and store nothing
        wp[j] = reinterpret_cast<const Load*>(W + static_cast<size_t>(n) * K) + (OUT == 0 ? 0 : t);
    }
    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0.f;
    const int chunks = K / 32;
#pragma unroll 2
    for (int c = 0; c < chunks; ++c) {
        if constexpr (OUT == 0) {
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                // Each word holds k=32c+16s+2t and k+1; the second fragment is eight k positions later.
                const int lo = 16 * c + 8 * s + t;
                uint32_t b0[NT], b1[NT], a0[RT], a1[RT], a2[RT], a3[RT];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    b0[j] = __ldg(wp[j] + lo);
                    b1[j] = __ldg(wp[j] + lo + 4);
                }
#pragma unroll
                for (int r = 0; r < RT; ++r) {
                    a0[r] = va[r] ? __ldg(xa[r] + lo) : 0u;
                    a2[r] = va[r] ? __ldg(xa[r] + lo + 4) : 0u;
                    a1[r] = vb[r] ? __ldg(xb[r] + lo) : 0u;
                    a3[r] = vb[r] ? __ldg(xb[r] + lo + 4) : 0u;
                }
#pragma unroll
                for (int r = 0; r < RT; ++r)
#pragma unroll
                    for (int j = 0; j < NT; ++j)
                        mma_bf16(acc[r][j], a0[r], a1[r], a2[r], a3[r], b0[j], b1[j]);
            }
        } else {
            const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
            uint4 b[NT], a0[RT], a1[RT];
#pragma unroll
            for (int j = 0; j < NT; ++j) b[j] = ld_nc(wp[j] + 4 * c);
#pragma unroll
            for (int r = 0; r < RT; ++r) {
                a0[r] = va[r] ? __ldg(xa[r] + 4 * c) : zero;
                a1[r] = vb[r] ? __ldg(xb[r] + 4 * c) : zero;
            }
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int r = 0; r < RT; ++r)
#pragma unroll
                    for (int j = 0; j < NT; ++j)
                        mma_bf16(acc[r][j], word(a0[r], 2 * s), word(a1[r], 2 * s), word(a0[r], 2 * s + 1),
                                 word(a1[r], 2 * s + 1), word(b[j], 2 * s), word(b[j], 2 * s + 1));
        }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        const int ra = row0 + r * 16 + g, rb = ra + 8;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int n = col0 + j * 8 + 2 * t + q;
                if (n >= N) continue;
                if (ra < M) store<OUT>(Y, static_cast<size_t>(ra) * ldy + n, acc[r][j][q]);
                if (rb < M) store<OUT>(Y, static_cast<size_t>(rb) * ldy + n, acc[r][j][2 + q]);
            }
        }
    }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// One row, bit-identical to gemm_bf16<1, 1, 0>: a warp's 8 weight rows stream through a padded cp.async ring.
constexpr int GEMV_STAGES = 4;
constexpr int GEMV_CHUNK = 64;
constexpr int GEMV_ROW = GEMV_CHUNK * 2 + 16;
constexpr int GEMV_STAGE = 8 * GEMV_ROW;

__device__ __forceinline__ void gemv_bf16(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                          int N, int K, __nv_bfloat16* __restrict__ Y) {
    __shared__ __align__(16) unsigned char ring_all[WARPS * GEMV_STAGES * GEMV_STAGE];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int col0 = (blockIdx.x * WARPS + warp) * 8;
    if (col0 >= N) return;
    unsigned char* ring = ring_all + warp * GEMV_STAGES * GEMV_STAGE;
    const unsigned char* rows = reinterpret_cast<const unsigned char*>(W + static_cast<size_t>(col0) * K);
    const int chunks = K / GEMV_CHUNK;
    // copies: lane (row g, pieces 2t and 2t + 1 of the row's eight 16-byte pieces)
    auto issue = [&](int c) {
        unsigned char* dst = ring + (c % GEMV_STAGES) * GEMV_STAGE + g * GEMV_ROW + 32 * t;
        const unsigned char* src = rows + static_cast<size_t>(g) * K * 2 + static_cast<size_t>(c) * GEMV_CHUNK * 2 + 32 * t;
        cp_async16(dst, src);
        cp_async16(dst + 16, src + 16);
    };
#pragma unroll
    for (int c = 0; c < GEMV_STAGES - 1; ++c) {
        if (c < chunks) issue(c);
        asm volatile("cp.async.commit_group;\n");
    }
    const uint32_t* xw = reinterpret_cast<const uint32_t*>(X);
    const bool live = g == 0;  // the single row is A's row 0; rows 1..15 are zero as in gemm_bf16
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const unsigned char* row_at = ring + (lane & 7) * GEMV_ROW + (lane >> 3) * 16;  // ldmatrix: matrix lane / 8
    for (int c = 0; c < chunks; ++c) {
        if (c + GEMV_STAGES - 1 < chunks) issue(c + GEMV_STAGES - 1);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group %0;\n" ::"n"(GEMV_STAGES - 1));
        __syncwarp();
        const unsigned char* stage = row_at + (c % GEMV_STAGES) * GEMV_STAGE;
#pragma unroll
        for (int h = 0; h < GEMV_CHUNK / 32; ++h) {
            uint32_t b[4];
            ldmatrix_x4(b, stage + 64 * h);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int lo = (c * GEMV_CHUNK + 32 * h + 16 * s) / 2 + t;
                const uint32_t a0 = live ? __ldg(xw + lo) : 0u, a2 = live ? __ldg(xw + lo + 4) : 0u;
                mma_bf16(acc, a0, 0u, a2, 0u, b[2 * s], b[2 * s + 1]);
            }
        }
        __syncwarp();
    }
    if (!live) return;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int n = col0 + 2 * t + q;
        if (n < N) Y[n] = __float2bfloat16_rn(acc[q]);
    }
}

// Routed experts over the experts.cu plan: per 128-k block acc += dot(a_q, w_q) * a_scale * w_scale, as fused_moe.
template <int NT, int MODE>
__device__ __forceinline__ void expert_fp8(const uint8_t* __restrict__ A, int a_div, const float* __restrict__ As,
                                           const uint8_t* __restrict__ W, const float* __restrict__ Ws, int N, int K,
                                           const int* __restrict__ items, const int* __restrict__ counts,
                                           const int* __restrict__ members, const float* __restrict__ rw,
                                           __nv_bfloat16* __restrict__ Y) {
    const int item = blockIdx.x;
    if (item >= __ldg(counts)) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int col0 = (blockIdx.y * WARPS + warp) * (8 * NT);
    if (col0 >= N) return;
    const int e = __ldg(items + 3 * item), first = __ldg(items + 3 * item + 1), cnt = __ldg(items + 3 * item + 2);
    const bool v0 = g < cnt, v1 = g + 8 < cnt;
    const int p0 = v0 ? __ldg(members + first + g) : 0, p1 = v1 ? __ldg(members + first + g + 8) : 0;
    const int r0 = p0 / a_div, r1 = p1 / a_div;
    const int KB = K / 128;
    const uint4* a0p = reinterpret_cast<const uint4*>(A + static_cast<size_t>(r0) * K) + 2 * t;
    const uint4* a1p = reinterpret_cast<const uint4*>(A + static_cast<size_t>(r1) * K) + 2 * t;
    const float* s0p = As + static_cast<size_t>(r0) * KB;
    const float* s1p = As + static_cast<size_t>(r1) * KB;
    const uint8_t* we = W + static_cast<size_t>(e) * N * K;
    const uint4* wp[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) wp[j] = reinterpret_cast<const uint4*>(we + static_cast<size_t>(col0 + j * 8 + g) * K) + 2 * t;
    const float* wsp = Ws + (static_cast<size_t>(e) * (N / 128) + col0 / 128) * KB;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    for (int kb = 0; kb < KB; ++kb) {
        // 128 k a block: 32 bytes a lane (two uint4), four k32 steps on the words in pairs
        const uint4 a0l = v0 ? __ldg(a0p + 8 * kb) : zero, a0h = v0 ? __ldg(a0p + 8 * kb + 1) : zero;
        const uint4 a1l = v1 ? __ldg(a1p + 8 * kb) : zero, a1h = v1 ? __ldg(a1p + 8 * kb + 1) : zero;
        float part[NT][4];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const uint4 bl = ld_nc(wp[j] + 8 * kb), bh = ld_nc(wp[j] + 8 * kb + 1);
            part[j][0] = part[j][1] = part[j][2] = part[j][3] = 0.f;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const uint4& a0 = s < 2 ? a0l : a0h;
                const uint4& a1 = s < 2 ? a1l : a1h;
                const uint4& b = s < 2 ? bl : bh;
                const int w0 = 2 * (s & 1);
                mma_e4m3(part[j], word(a0, w0), word(a1, w0), word(a0, w0 + 1), word(a1, w0 + 1), word(b, w0),
                         word(b, w0 + 1));
            }
        }
        const float sa0 = v0 ? __ldg(s0p + kb) : 0.f, sa1 = v1 ? __ldg(s1p + kb) : 0.f;
        const float sw = __ldg(wsp + kb);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[j][0] += part[j][0] * sa0 * sw;
            acc[j][1] += part[j][1] * sa0 * sw;
            acc[j][2] += part[j][2] * sa1 * sw;
            acc[j][3] += part[j][3] * sa1 * sw;
        }
    }
    const float w0 = MODE == 1 && v0 ? __ldg(rw + p0) : 1.f;
    const float w1 = MODE == 1 && v1 ? __ldg(rw + p1) : 1.f;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int n = col0 + j * 8 + 2 * t + q;
            if (v0) Y[static_cast<size_t>(p0) * N + n] = __float2bfloat16_rn(MODE == 1 ? acc[j][q] * w0 : acc[j][q]);
            if (v1)
                Y[static_cast<size_t>(p1) * N + n] =
                    __float2bfloat16_rn(MODE == 1 ? acc[j][2 + q] * w1 : acc[j][2 + q]);
        }
    }
}

}  // namespace drama_gemm

#define DRAMA_GEMM(RT, NT, OUT)                                                                                     \
    extern "C" __global__ void __launch_bounds__(32 * drama_gemm::WARPS)                                           \
        drama_gemm_bf16_r##RT##_n##NT##_o##OUT(const __nv_bfloat16* X, int ldx, int M, const __nv_bfloat16* W,   \
                                               int N, int K, void* Y, int ldy) {                                   \
        drama_gemm::gemm_bf16<RT, NT, OUT>(X, ldx, M, W, N, K, Y, ldy);                                            \
    }
DRAMA_GEMM(1, 1, 0)
DRAMA_GEMM(2, 2, 0)
DRAMA_GEMM(4, 2, 0)
DRAMA_GEMM(1, 1, 2)  // the semantic head: the last row, bf16-rounded logits widened to fp32

extern "C" __global__ void __launch_bounds__(32 * drama_gemm::WARPS)
    drama_gemv_bf16_r1(const __nv_bfloat16* X, const __nv_bfloat16* W, int N, int K, __nv_bfloat16* Y) {
    drama_gemm::gemv_bf16(X, W, N, K, Y);
}

#define DRAMA_EXPERT(NT, MODE)                                                                                      \
    extern "C" __global__ void __launch_bounds__(32 * drama_gemm::WARPS) drama_expert_fp8_n##NT##_m##MODE(          \
        const uint8_t* A, int a_div, const float* As, const uint8_t* W, const float* Ws, int N, int K,            \
        const int* items, const int* counts, const int* members, const float* rw, __nv_bfloat16* Y) {             \
        drama_gemm::expert_fp8<NT, MODE>(A, a_div, As, W, Ws, N, K, items, counts, members, rw, Y);                \
    }
DRAMA_EXPERT(1, 0)
DRAMA_EXPERT(2, 0)
DRAMA_EXPERT(1, 1)
DRAMA_EXPERT(2, 1)

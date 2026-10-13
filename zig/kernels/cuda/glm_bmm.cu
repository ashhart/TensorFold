// GLM's BF16 products (qmm.py _bmm, glue.py router): mma.sync m16n8k16, four K=16 chunks per 64-wide step.
#include <cstdint>
#include <cuda_bf16.h>

namespace tf_glm_bmm {

using bf16 = __nv_bfloat16;

struct BmmArgs {
    const bf16* x;        // bf16 [m, k], rows `xs` apart
    long long xs;         // x's row stride in elements
    const bf16* w;        // bf16 [n, k] row-major
    void* out;            // bf16 or fp32 [m, n] when one slice
    float* part;          // fp32 [sk, m, n] partials when split
    int m, n, k, sk;
};

struct RouterArgs {
    const bf16* x;        // bf16 [m, d], rows `xs` apart
    long long xs;
    const bf16* w;        // bf16 [ne, d] row-major
    float* out;           // fp32 [m, ne] when the whole K ran in one pass
    float* part;          // fp32 [ks, m, ne] partials, one per slice
    int m, ne, d, ks;
};

__device__ __forceinline__ unsigned pack2(bf16 lo, bf16 hi) {
    return static_cast<unsigned>(__bfloat16_as_ushort(lo)) | (static_cast<unsigned>(__bfloat16_as_ushort(hi)) << 16);
}

__device__ __forceinline__ void mma_16n8k16(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One 16x16 A tile of x (row-major, `xs` elements a row): four fragment words, rows at or past `m` zeroed.
__device__ __forceinline__ void load_a16(const bf16* x, long long xs, int row, int col, int m, unsigned (&a)[4]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int r0 = lane >> 2, c0 = (lane & 3) * 2;
    const bf16* base = x + static_cast<long long>(row) * xs + col;
    const bool ok0 = row + r0 < m, ok1 = row + r0 + 8 < m;
    a[0] = ok0 ? *reinterpret_cast<const unsigned*>(base + static_cast<long long>(r0) * xs + c0) : 0u;
    a[1] = ok1 ? *reinterpret_cast<const unsigned*>(base + static_cast<long long>(r0 + 8) * xs + c0) : 0u;
    a[2] = ok0 ? *reinterpret_cast<const unsigned*>(base + static_cast<long long>(r0) * xs + c0 + 8) : 0u;
    a[3] = ok1 ? *reinterpret_cast<const unsigned*>(base + static_cast<long long>(r0 + 8) * xs + c0 + 8) : 0u;
}

// One 16x8 B tile of w [n, k] (weight rows, K contiguous): two fragment words, columns at or past `n` zeroed.
__device__ __forceinline__ void load_b16(const bf16* w, long long k_stride, int n0, int k0, int n, unsigned (&b)[2]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int kk = (lane & 3) * 2, n_row = lane >> 2;
    const bool ok = n0 + n_row < n;
    const bf16* base = w + static_cast<long long>(n0 + n_row) * k_stride + k0 + kk;
    b[0] = ok ? *reinterpret_cast<const unsigned*>(base) : 0u;
    b[1] = ok ? *reinterpret_cast<const unsigned*>(base + 8) : 0u;
}

// A warp's 16x8 result tile: `row` is where it lands in the output, `mrow` the unmasked input row.
__device__ __forceinline__ void store_tile(void* out, bool f32_out, long long out_stride, int row, int mrow, int col,
                                           int m, int n, const float (&d)[4]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int r0 = lane >> 2, c0 = (lane & 3) * 2;
    const bool ok0 = mrow + r0 < m, ok1 = mrow + r0 + 8 < m;
    const bool k0 = col + c0 < n, k1 = col + c0 + 1 < n;
    if (f32_out) {
        float* base = static_cast<float*>(out) + static_cast<long long>(row) * out_stride + col;
        if (ok0 && k0) base[static_cast<long long>(r0) * out_stride + c0] = d[0];
        if (ok0 && k1) base[static_cast<long long>(r0) * out_stride + c0 + 1] = d[1];
        if (ok1 && k0) base[static_cast<long long>(r0 + 8) * out_stride + c0] = d[2];
        if (ok1 && k1) base[static_cast<long long>(r0 + 8) * out_stride + c0 + 1] = d[3];
    } else {
        bf16* base = static_cast<bf16*>(out) + static_cast<long long>(row) * out_stride + col;
        if (ok0 && k0 && k1) {
            *reinterpret_cast<unsigned*>(base + static_cast<long long>(r0) * out_stride + c0) =
                pack2(__float2bfloat16_rn(d[0]), __float2bfloat16_rn(d[1]));
        }
        if (ok1 && k0 && k1) {
            *reinterpret_cast<unsigned*>(base + static_cast<long long>(r0 + 8) * out_stride + c0) =
                pack2(__float2bfloat16_rn(d[2]), __float2bfloat16_rn(d[3]));
        }
    }
}

// A packed w's 16x8 B tiles of one 64-wide K step: lane's (kk, half) words, two 16-byte loads (fragment-order layout).
__device__ __forceinline__ void load_b_packed(const bf16* w, int k, int n0, int k0, int n, unsigned (&b)[8]) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (n0 >= n) {
#pragma unroll
        for (int i = 0; i < 8; i++) b[i] = 0u;
        return;
    }
    const uint4* p = reinterpret_cast<const uint4*>(w) + (static_cast<long long>(n0 >> 3) * (k >> 6) + (k0 >> 6)) * 64 + lane * 2;
    const uint4 u0 = __ldg(p), u1 = __ldg(p + 1);
    b[0] = u0.x; b[1] = u0.y; b[2] = u0.z; b[3] = u0.w;
    b[4] = u1.x; b[5] = u1.y; b[6] = u1.z; b[7] = u1.w;
}

// The bf16 matmul: y = x [m, k] through w [n, k], 64-wide K steps in order, `sk` slices; PK: w in fragment order.
template <int BM, int WN, bool F32_OUT, bool PK>
__device__ __forceinline__ void bmm_body(const BmmArgs a) {
    constexpr int nt = 64 / (WN * 8);                      // 16x8 B tiles per warp
    constexpr int mt = BM / 16;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int per = a.k / a.sk;                            // one slice's K
    const int col0 = blockIdx.y * 64 + warp * (64 / WN);
    const int row0 = blockIdx.x * BM;
    const int k0_base = blockIdx.z * per;

    float acc[mt][nt][4];
#pragma unroll
    for (int r = 0; r < mt; r++)
#pragma unroll
        for (int c = 0; c < nt; c++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[r][c][i] = 0.0f;

    for (int k0 = k0_base; k0 < k0_base + per; k0 += 64) {
        unsigned bp[PK ? nt : 1][8];
        if constexpr (PK) {
#pragma unroll
            for (int c = 0; c < nt; c++) load_b_packed(a.w, a.k, col0 + c * 8, k0, a.n, bp[c]);
        }
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int kcol = k0 + kk * 16;
            unsigned av[mt][4];
#pragma unroll
            for (int r = 0; r < mt; r++) load_a16(a.x, a.xs, row0 + r * 16, kcol, a.m, av[r]);
#pragma unroll
            for (int c = 0; c < nt; c++) {
                unsigned b[2];
                if constexpr (PK) {
                    b[0] = bp[c][2 * kk];
                    b[1] = bp[c][2 * kk + 1];
                } else {
                    load_b16(a.w, a.k, col0 + c * 8, kcol, a.n, b);
                }
#pragma unroll
                for (int r = 0; r < mt; r++) mma_16n8k16(acc[r][c], av[r], b);
            }
        }
    }

    if (a.sk > 1) {                                        // fp32 partials, slice s at rows [s*m, s*m + m)
#pragma unroll
        for (int r = 0; r < mt; r++)
#pragma unroll
            for (int c = 0; c < nt; c++)
                store_tile(a.part, true, a.n, blockIdx.z * a.m + row0 + r * 16, row0 + r * 16, col0 + c * 8, a.m, a.n,
                           acc[r][c]);
    } else {
#pragma unroll
        for (int r = 0; r < mt; r++)
#pragma unroll
            for (int c = 0; c < nt; c++)
                store_tile(a.out, F32_OUT, a.n, row0 + r * 16, row0 + r * 16, col0 + c * 8, a.m, a.n, acc[r][c]);
    }
}

// _bmm's split partials added in slice order, then one bf16 rounding (or fp32 through).
__device__ __forceinline__ void reduce_body(const float* part, void* out, int total, int sk, int f32) {
    const int i = blockIdx.x * 1024 + static_cast<int>(threadIdx.x);
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < sk; s++) acc += part[s * total + i];
    if (f32) static_cast<float*>(out)[i] = acc;
    else static_cast<bf16*>(out)[i] = __float2bfloat16_rn(acc);
}

// glue._router: OUT[m, e] fp32 = x[m] . w[e] over the whole K, 64-wide steps in order (the unsliced choice).
template <int BM>
__device__ __forceinline__ void router_body(const RouterArgs a) {
    constexpr int mt = BM / 16;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int col0 = blockIdx.y * 64 + warp * 16;
    const int row0 = blockIdx.x * BM;

    float acc[mt][2][4];
#pragma unroll
    for (int r = 0; r < mt; r++)
#pragma unroll
        for (int i = 0; i < 8; i++) acc[r][i >> 2][i & 3] = 0.0f;
    for (int k0 = 0; k0 < a.d; k0 += 64) {
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            unsigned av[mt][4];
#pragma unroll
            for (int r = 0; r < mt; r++) load_a16(a.x, a.xs, row0 + r * 16, k0 + kk * 16, a.m, av[r]);
            unsigned b[2];
            load_b16(a.w, a.d, col0, k0 + kk * 16, a.ne, b);
#pragma unroll
            for (int r = 0; r < mt; r++) mma_16n8k16(acc[r][0], av[r], b);
            load_b16(a.w, a.d, col0 + 8, k0 + kk * 16, a.ne, b);
#pragma unroll
            for (int r = 0; r < mt; r++) mma_16n8k16(acc[r][1], av[r], b);
        }
    }
#pragma unroll
    for (int r = 0; r < mt; r++)
#pragma unroll
        for (int c = 0; c < 2; c++)
            store_tile(a.out, true, a.ne, row0 + r * 16, row0 + r * 16, col0 + c * 8, a.m, a.ne, acc[r][c]);
}

// glue._router_part: the same over one of `ks` equal K slices; PART[s, m, e] fp32.
template <int BM>
__device__ __forceinline__ void router_part_body(const RouterArgs a) {
    constexpr int mt = BM / 16;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int col0 = blockIdx.y * 32 + warp * 8;
    const int row0 = blockIdx.x * BM;
    const int per = a.d / a.ks;
    const int k0_base = blockIdx.z * per;

    float acc[mt][4];
#pragma unroll
    for (int r = 0; r < mt; r++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[r][i] = 0.0f;
    for (int k0 = k0_base; k0 < k0_base + per; k0 += 64) {
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            unsigned av[mt][4];
#pragma unroll
            for (int r = 0; r < mt; r++) load_a16(a.x, a.xs, row0 + r * 16, k0 + kk * 16, a.m, av[r]);
            unsigned b[2];
            load_b16(a.w, a.d, col0, k0 + kk * 16, a.ne, b);
#pragma unroll
            for (int r = 0; r < mt; r++) mma_16n8k16(acc[r], av[r], b);
        }
    }
#pragma unroll
    for (int r = 0; r < mt; r++)
        store_tile(a.part, true, a.ne, blockIdx.z * a.m + row0 + r * 16, row0 + r * 16, col0, a.m, a.ne, acc[r]);
}

// glue._router_sum: the `ks` slice partials added in slice order.
__device__ __forceinline__ void router_sum_body(const float* part, float* out, int total, int ks) {
    const int i = blockIdx.x * 1024 + static_cast<int>(threadIdx.x);
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < ks; s++) acc += part[s * total + i];
    out[i] = acc;
}

// One extern "C" kernel per instantiation: the fatbin carries __global__ functions, the host resolves them by name.
#define TF_GLM_BMM(BM, WN, F32)                                                                              \
    extern "C" __global__ void __launch_bounds__((WN) *32)                                                   \
        tf_glm_bmm_b16_##BM##_##F32(tf_glm_bmm::BmmArgs a) {                                                 \
        tf_glm_bmm::bmm_body<BM, WN, F32, false>(a);                                                         \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__((WN) *32)                                                   \
        tf_glm_bmm_p16_##BM##_##F32(tf_glm_bmm::BmmArgs a) {                                                 \
        tf_glm_bmm::bmm_body<BM, WN, F32, true>(a);                                                          \
    }
TF_GLM_BMM(16, 4, 0)
TF_GLM_BMM(16, 4, 1)
TF_GLM_BMM(32, 4, 0)
TF_GLM_BMM(32, 4, 1)
TF_GLM_BMM(64, 4, 0)
TF_GLM_BMM(64, 4, 1)
TF_GLM_BMM(128, 8, 0)
TF_GLM_BMM(128, 8, 1)

#define TF_GLM_ROUTER(BM)                                                                                    \
    extern "C" __global__ void __launch_bounds__(128) tf_glm_router_##BM(tf_glm_bmm::RouterArgs a) {         \
        tf_glm_bmm::router_body<BM>(a);                                                                      \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) tf_glm_router_part_##BM(tf_glm_bmm::RouterArgs a) {    \
        tf_glm_bmm::router_part_body<BM>(a);                                                                 \
    }
TF_GLM_ROUTER(16)
TF_GLM_ROUTER(32)
TF_GLM_ROUTER(64)
TF_GLM_ROUTER(128)

extern "C" __global__ void tf_glm_bmm_reduce(const float* part, void* out, int total, int sk, int f32) {
    tf_glm_bmm::reduce_body(part, out, total, sk, f32);
}

extern "C" __global__ void tf_glm_router_sum(const float* part, float* out, int total, int ks) {
    tf_glm_bmm::router_sum_body(part, out, total, ks);
}

}  // namespace tf_glm_bmm

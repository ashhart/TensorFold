// Living Weights on the 27B: the pocket beside each dense MLP, its input gradients and group-64 4-bit products.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"
#include "common/act.cuh"

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 b16(float v) { return __float2bfloat16(v); }
__device__ __forceinline__ float wide(float v) { return v; }
__device__ __forceinline__ float wide(bf16 v) { return __bfloat162float(v); }

using tf_common::warp_sum;
using tf_common::block_sum_tree;

using tf_common::sigmoidf_;
using tf_common::silu;

using tf_common::silu_grad;

// ---- 4-bit products, any group --------------------------------------------------------------------------------------

// C [M, N] (+)= A [M, K] W for a 4-bit MLX [K, N] W: a linear's input gradient, split K partials in C + z M N.
template <typename TA>
__device__ __forceinline__ void gemm_q4(const TA* __restrict__ A, const unsigned* __restrict__ W, const bf16* __restrict__ S,
                                        const bf16* __restrict__ B, float* __restrict__ C, int M, int N, int K, int lda, int ldc,
                                        int accumulate, int kchunk, int group) {
    __shared__ __align__(16) bf16 As[64][40];
    __shared__ __align__(16) bf16 Bs[64][40];  // W's tile transposed: [n][k]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int kb = blockIdx.z * kchunk, ke = min(K, kb + kchunk);
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const int words = N / 8, groups = N / group;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    const bf16 zero = b16(0.0f);
    const bool wide_a = sizeof(TA) == 2 && (lda & 7) == 0 && ((size_t)A & 15) == 0;
    for (int k0 = kb; k0 < ke; k0 += 32) {
        for (int e = tid; e < 64 * 4; e += 128) {
            const int r = e >> 2, c = (e & 3) * 8, gm = m0 + r, gk = k0 + c;
            if (wide_a && gm < M && gk + 8 <= ke) {
                *reinterpret_cast<uint4*>(&As[r][c]) = *reinterpret_cast<const uint4*>(A + (size_t)gm * lda + gk);
            } else {
                for (int i = 0; i < 8; i++) As[r][c + i] = (gm < M && gk + i < ke) ? b16(wide(A[(size_t)gm * lda + gk + i])) : zero;
            }
        }
        for (int e = tid; e < 32 * 8; e += 128) {
            const int r = e >> 3, c = e & 7, gk = k0 + r, gn = n0 + c * 8;
            if (gk < ke && gn < N) {
                const unsigned q = W[(size_t)gk * words + gn / 8];
                const float s = f32(S[(size_t)gk * groups + gn / group]), b = f32(B[(size_t)gk * groups + gn / group]);
#pragma unroll
                for (int i = 0; i < 8; i++) Bs[c * 8 + i][r] = b16(fmaf(s, (float)((q >> (4 * i)) & 0xfu), b));
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) Bs[c * 8 + i][r] = zero;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            unsigned af[2][4], bq[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int r = wm + i * 16 + (lane >> 2), c = kk + (lane & 3) * 2;
                af[i][0] = *reinterpret_cast<const unsigned*>(&As[r][c]);
                af[i][1] = *reinterpret_cast<const unsigned*>(&As[r + 8][c]);
                af[i][2] = *reinterpret_cast<const unsigned*>(&As[r][c + 8]);
                af[i][3] = *reinterpret_cast<const unsigned*>(&As[r + 8][c + 8]);
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = wn + j * 8 + (lane >> 2), k = kk + (lane & 3) * 2;
                bq[j][0] = *reinterpret_cast<const unsigned*>(&Bs[n][k]);
                bq[j][1] = *reinterpret_cast<const unsigned*>(&Bs[n][k + 8]);
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bq[j][0]), "r"(bq[j][1]));
        }
        __syncthreads();
    }
    const bool split = gridDim.z > 1;
    float* out0 = split ? C + (size_t)blockIdx.z * M * N : C;
    const int ld = split ? N : ldc;
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = m0 + wm + i * 16 + (lane >> 2) + h * 8, c = n0 + wn + j * 8 + (lane & 3) * 2;
                if (r >= M) continue;
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    if (c + e >= N) continue;
                    float* out = out0 + (size_t)r * ld + c + e;
                    const float v = acc[i][j][h * 2 + e];
                    *out = (accumulate && !split) ? *out + v : v;
                }
            }
}

extern "C" __global__ void __launch_bounds__(128) tf_q27_gemm_q4(const float* A, const unsigned* W, const bf16* S, const bf16* B,
                                                                float* C, int M, int N, int K, int lda, int ldc, int accumulate,
                                                                int kchunk, int group) {
    gemm_q4<float>(A, W, S, B, C, M, N, K, lda, ldc, accumulate, kchunk, group);
}

// tf_q27_gemm_q4 with bf16 A: the head's gradient from tf_train_softmax's bf16 rows.
extern "C" __global__ void __launch_bounds__(128) tf_q27_gemm_q4_bf(const bf16* A, const unsigned* W, const bf16* S, const bf16* B,
                                                                   float* C, int M, int N, int K, int lda, int ldc, int accumulate,
                                                                   int kchunk, int group) {
    gemm_q4<bf16>(A, W, S, B, C, M, N, K, lda, ldc, accumulate, kchunk, group);
}

// ---- 4-bit products, wide tiles.

__device__ __forceinline__ unsigned smem_at(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ld4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_at(p)));
}

__device__ __forceinline__ void ld4t(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_at(p)));
}

// gemm_q4's products bit for bit.
template <typename TA>
__device__ __forceinline__ void gemm_q4w(const TA* __restrict__ A, const unsigned* __restrict__ W, const bf16* __restrict__ S,
                                         const bf16* __restrict__ B, float* __restrict__ C, int M, int N, int K, int lda, int ldc,
                                         int accumulate, int kchunk, int group, const int* __restrict__ rows) {
    constexpr int BM = 128, BN = 128, BK = 32, AS = BK + 8, BS = BN + 8;
    constexpr int AV = sizeof(TA) == 4 ? 4 : 2;  // A's 16-byte loads a thread
    __shared__ __align__(16) bf16 As[2][BM][AS];
    __shared__ __align__(16) bf16 Bs[2][BK][BS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int kb = blockIdx.z * kchunk, ke = min(K, kb + kchunk);
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
    const int words = N / 8, groups = N / group;
    // this thread's share of a slab: A's 16-byte pieces, W's two words and their scales and biases
    const TA* arow[AV];
    int acol[AV], ar[AV];
#pragma unroll
    for (int v = 0; v < AV; v++) {
        const int f = tid + 256 * v;
        ar[v] = sizeof(TA) == 4 ? f >> 3 : f >> 2;
        acol[v] = sizeof(TA) == 4 ? (f & 7) * 4 : (f & 3) * 8;
        const int gm = m0 + ar[v];
        arow[v] = gm < M ? A + (size_t)(rows ? rows[gm] : gm) * lda : nullptr;
    }
    const int kr = tid >> 3, nw = (tid & 7) * 2;
    const int g0 = (n0 + nw * 8) / group, g1 = (n0 + nw * 8 + 8) / group;
    uint4 ra[AV];
    uint2 rw;
    float rs0, rb0, rs1, rb1;
    auto fetch = [&](int k0) {
#pragma unroll
        for (int v = 0; v < AV; v++) ra[v] = arow[v] ? *reinterpret_cast<const uint4*>(arow[v] + k0 + acol[v]) : make_uint4(0, 0, 0, 0);
        const size_t gk = (size_t)(k0 + kr);
        rw = *reinterpret_cast<const uint2*>(W + gk * words + n0 / 8 + nw);
        rs0 = f32(S[gk * groups + g0]), rb0 = f32(B[gk * groups + g0]);
        rs1 = f32(S[gk * groups + g1]), rb1 = f32(B[gk * groups + g1]);
    };
    auto place = [&](int buf) {
#pragma unroll
        for (int v = 0; v < AV; v++) {
            if (sizeof(TA) == 4) {
                const float4 x = *reinterpret_cast<const float4*>(&ra[v]);
                __nv_bfloat162 lo = __floats2bfloat162_rn(x.x, x.y), hi = __floats2bfloat162_rn(x.z, x.w);
                uint2 p;
                p.x = *reinterpret_cast<unsigned*>(&lo);
                p.y = *reinterpret_cast<unsigned*>(&hi);
                *reinterpret_cast<uint2*>(&As[buf][ar[v]][acol[v]]) = p;
            } else {
                *reinterpret_cast<uint4*>(&As[buf][ar[v]][acol[v]]) = ra[v];
            }
        }
        const unsigned wv[2] = {rw.x, rw.y};
        const float sv[2] = {rs0, rs1}, bv[2] = {rb0, rb1};
#pragma unroll
        for (int e = 0; e < 2; e++) {
            __align__(16) bf16 o[8];
#pragma unroll
            for (int i = 0; i < 8; i++) o[i] = b16(fmaf(sv[e], (float)((wv[e] >> (4 * i)) & 0xfu), bv[e]));
            *reinterpret_cast<uint4*>(&Bs[buf][kr][nw * 8 + e * 8]) = *reinterpret_cast<const uint4*>(o);
        }
    };
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    if (kb < ke) fetch(kb);
    int buf = 0;
    for (int k0 = kb; k0 < ke; k0 += BK, buf ^= 1) {
        place(buf);
        __syncthreads();
        if (k0 + BK < ke) fetch(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned af[4][4], bq[4][2];
#pragma unroll
            for (int i = 0; i < 4; i++) ld4(af[i], &As[buf][wm + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8][kk + (lane >> 4) * 8]);
#pragma unroll
            for (int jp = 0; jp < 2; jp++) {
                unsigned r[4];
                ld4t(r, &Bs[buf][kk + (lane & 7) + ((lane >> 3) & 1) * 8][wn + (jp * 2 + (lane >> 4)) * 8]);
                bq[2 * jp][0] = r[0], bq[2 * jp][1] = r[1], bq[2 * jp + 1][0] = r[2], bq[2 * jp + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bq[j][0]), "r"(bq[j][1]));
        }
    }
    const bool split = gridDim.z > 1;
    float* out0 = split ? C + (size_t)blockIdx.z * M * N : C;
    const int ld = split ? N : ldc;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = m0 + wm + i * 16 + (lane >> 2) + h * 8;
            if (r >= M) continue;
            const int rr = (rows && !split) ? rows[r] : r;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c = n0 + wn + j * 8 + (lane & 3) * 2;
                float2* out = reinterpret_cast<float2*>(out0 + (size_t)rr * ld + c);
                float2 v = make_float2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
                if (accumulate && !split) {
                    const float2 o = *out;
                    v.x = o.x + v.x;
                    v.y = o.y + v.y;
                }
                *out = v;
            }
        }
}

extern "C" __global__ void __launch_bounds__(256, 2) tf_q27_gemm_q4w(const float* A, const unsigned* W, const bf16* S, const bf16* B,
                                                                 float* C, int M, int N, int K, int lda, int ldc, int accumulate,
                                                                 int kchunk, int group, const int* rows) {
    gemm_q4w<float>(A, W, S, B, C, M, N, K, lda, ldc, accumulate, kchunk, group, rows);
}

extern "C" __global__ void __launch_bounds__(256, 2) tf_q27_gemm_q4w_bf(const bf16* A, const unsigned* W, const bf16* S, const bf16* B,
                                                                    float* C, int M, int N, int K, int lda, int ldc, int accumulate,
                                                                    int kchunk, int group, const int* rows) {
    gemm_q4w<bf16>(A, W, S, B, C, M, N, K, lda, ldc, accumulate, kchunk, group, rows);
}

// ---- the dense MLP --------------------------------------------------------------------------------------------------

// act = silu(gate) up undone from dact: dgate = dact up silu'(gate), dup = dact silu(gate), n values.
extern "C" __global__ void tf_q27_swiglu_back(const bf16* gate, const bf16* up, const float* dact, float* dgate, float* dup,
                                              long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const float g = f32(gate[i]), u = f32(up[i]), d = dact[i];
    dgate[i] = d * u * silu_grad(g);
    dup[i] = d * silu(g);
}

// ---- the pocket's neurons (SwiGLU, beside the dense MLP) ------------------------------------------------------------

#define POCKET_MAX 128

// Neuron j's dot products on row y (d values): z = (gate_j . y, up_j . y) and, when g is set, e = down_j . g.
__device__ __forceinline__ void pocket_dots(const bf16* y, const float* g, int d, const float* gate, const float* up,
                                            const float* down, int j, float& zg, float& zu, float& e) {
    const int lane = threadIdx.x & 31;
    const float* gj = gate + (size_t)j * d;
    const float* uj = up + (size_t)j * d;
    const float* dj = down + (size_t)j * d;
    float a = 0, b = 0, c = 0;
    for (int i = lane; i < d; i += 32) {
        const float v = f32(y[i]);
        a += gj[i] * v;
        b += uj[i] * v;
        if (g) c += dj[i] * g[i];
    }
    zg = warp_sum(a), zu = warp_sum(b), e = g ? warp_sum(c) : 0.0f;
}

// Neuron j's scale on row r: the row's own (rs[r], a pass's sequence scale) from neuron `from` on, else 1.
__device__ __forceinline__ float pocket_scale(const float* rs, int from, int j, int r) {
    return (rs != nullptr && j >= from) ? rs[r] : 1.0f;
}

// out[r] = bf16(out[r] + sum_j f_j silu(gate_j . y_r) (up_j . y_r) down_j); grid rows, block 256.
extern "C" __global__ void __launch_bounds__(256) tf_q27_pocket_rows(const bf16* y, int d, const float* gate, const float* up,
                                                                    const float* down, int n, bf16* out, const float* rs, int from) {
    __shared__ float act[POCKET_MAX];
    const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const bf16* yr = y + (size_t)r * d;
    for (int j = w; j < n; j += 8) {
        float zg, zu, e;
        pocket_dots(yr, nullptr, d, gate, up, down, j, zg, zu, e);
        if (lane == 0) act[j] = pocket_scale(rs, from, j, r) * silu(zg) * zu;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < d; k += 256) {
        float v = 0;
        for (int j = 0; j < n; j++) v += act[j] * down[(size_t)j * d + k];
        bf16* o = out + (size_t)r * d + k;
        *o = b16(f32(*o) + v);
    }
}

// The pocket undone on row r: dx[r] += sum_j f_j (down_j . g) (silu'(zg) zu gate_j + silu(zg) up_j).
extern "C" __global__ void __launch_bounds__(256) tf_q27_pocket_back(const bf16* y, int d, const float* gate, const float* up,
                                                                    const float* down, int n, const float* g, float* dx,
                                                                    const float* rs, int from) {
    __shared__ float cg[POCKET_MAX], cu[POCKET_MAX];
    const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const bf16* yr = y + (size_t)r * d;
    const float* gr = g + (size_t)r * d;
    for (int j = w; j < n; j += 8) {
        float zg, zu, e;
        pocket_dots(yr, gr, d, gate, up, down, j, zg, zu, e);
        if (lane == 0) {
            const float f = pocket_scale(rs, from, j, r);
            cg[j] = f * e * silu_grad(zg) * zu;
            cu[j] = f * e * silu(zg);
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < d; k += 256) {
        float v = 0;
        for (int j = 0; j < n; j++) v += cg[j] * gate[(size_t)j * d + k] + cu[j] * up[(size_t)j * d + k];
        dx[(size_t)r * d + k] += v;
    }
}

// grad[j] += w act_j push[rows[j]]: neuron j's output vector's gradient (no row when negative).
extern "C" __global__ void __launch_bounds__(256) tf_q27_pocket_grad(const bf16* y, int d, const float* gate, const float* up,
                                                                    int n, const float* push, const int* rows, float w,
                                                                    float* grad) {
    __shared__ float scratch[32];
    const int j = blockIdx.x, t = threadIdx.x;
    if (j >= n) return;
    const int r = rows[j];
    if (r < 0) return;
    const bf16* yr = y + (size_t)r * d;
    float a = 0, b = 0;
    for (int i = t; i < d; i += 256) {
        const float v = f32(yr[i]);
        a += gate[(size_t)j * d + i] * v;
        b += up[(size_t)j * d + i] * v;
    }
    a = block_sum_tree(a, scratch, 8);
    b = block_sum_tree(b, scratch, 8);
    const float k = w * silu(a) * b;
    for (int i = t; i < d; i += 256) grad[(size_t)j * d + i] += k * push[(size_t)r * d + i];
}

// ---- a pass's keeps -------------------------------------------------------------------------------------------------

// dst[r, col0 + c] = src[r, c] for c below w: a projection's bf16 rows into their columns of a wider kept row.
extern "C" __global__ void tf_q27_cols(const bf16* src, int w, bf16* dst, int ld, int col0, int rows) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)rows * w) return;
    const long long r = i / w, c = i % w;
    dst[r * ld + col0 + c] = src[i];
}

// y[i] = bf16(y[i] + x[i]) for n values: a nudge onto its row of the MLP's bf16 output.
extern "C" __global__ void tf_q27_add_bf16(bf16* y, const float* x, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = b16(f32(y[i]) + x[i]);
}

// x[r] = 0 for each row r of a pass the backward no longer needs (mask[r] == 0), ld values a row; grid rows.
extern "C" __global__ void tf_q27_zero_rows(float* x, int ld, const unsigned char* mask) {
    const int r = blockIdx.x;
    if (mask[r]) return;
    for (int i = threadIdx.x; i < ld; i += 256) x[(size_t)r * ld + i] = 0.0f;
}

// ---- a pocket neuron's input weights fitted on the device.

// Kinds as pocket_fit.Kind numbers them.
#define FIT_FACT 0
#define FIT_HELD 1
#define FIT_STEADY 2
#define FIT_OTHER 3

// The eight lanes' sum as the host's @reduce(.Add) forms it: ORDER 0 in order, 1 as halves, quarters, pairs.
template <int ORDER>
__device__ __forceinline__ float fit_reduce8(float v) {
    const int lane = threadIdx.x & 31;
    if (ORDER == 0) {
        float s = __shfl_sync(0xffffffffu, v, 0);
        for (int j = 1; j < 8; j++) s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, j));
        return s;
    }
    // (v0 + v4, v1 + v5, v2 + v6, v3 + v7), then (w0 + w2, w1 + w3), then x0 + x1
    const float w = __fadd_rn(v, __shfl_sync(0xffffffffu, v, (lane + 4) & 31));
    const float x = __fadd_rn(w, __shfl_sync(0xffffffffu, w, (lane + 2) & 31));
    const float y = __fadd_rn(x, __shfl_sync(0xffffffffu, x, (lane + 1) & 31));
    return __shfl_sync(0xffffffffu, y, 0);
}

// A step's dot products: c[l][i] the row's hinge coefficient (0 on the right side, held and free rows).
template <int ORDER>
__device__ __forceinline__ void fit_dots(const bf16* __restrict__ X, int m, int d, const float* __restrict__ U,
                                         const unsigned char* __restrict__ kinds, const float* __restrict__ invs, float below,
                                         float* __restrict__ C, int* __restrict__ wrong) {
    extern __shared__ float us[];
    const int l = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int k = threadIdx.x; k < d; k += 256) us[k] = U[(size_t)l * d + k];
    __syncthreads();
    const int i = blockIdx.x * 8 + warp;
    if (i >= m) return;
    const int kind = kinds[(size_t)l * m + i];
    float* c = C + (size_t)l * m + i;
    if (kind == FIT_HELD || kind > FIT_OTHER) {
        if (lane == 0) *c = 0.0f;
        return;
    }
    const bf16* x = X + ((size_t)l * m + i) * d;
    float acc = 0.0f;
    for (int t = lane; t < d; t += 32) acc = __fadd_rn(acc, __fmul_rn(us[t], f32(x[t])));
    // lane j < 8: (acc[j] + acc[j + 8]) + (acc[j + 16] + acc[j + 24]).
    const float a8 = __shfl_down_sync(0xffffffffu, acc, 8), a16 = __shfl_down_sync(0xffffffffu, acc, 16),
                a24 = __shfl_down_sync(0xffffffffu, acc, 24);
    const float v = __fadd_rn(__fadd_rn(acc, a8), __fadd_rn(a16, a24));
    const float z = __fmul_rn(fit_reduce8<ORDER>(v), invs[l]);
    if (lane != 0) return;
    float ci = 0.0f;
    if (kind == FIT_FACT) {
        if (z < 1.0f) {
            ci = __fmul_rn(-2.0f, __fsub_rn(1.0f, z));
            atomicAdd(&wrong[l * 4 + 0], 1);
        }
    } else if (z > -below) {
        ci = __fmul_rn(2.0f, __fadd_rn(z, below));
        atomicAdd(&wrong[l * 4 + (kind == FIT_STEADY ? 1 : 2)], 1);
    }
    *c = ci;
}

extern "C" __global__ void __launch_bounds__(256) tf_q27_fit_dots(const bf16* X, int m, int d, const float* U, const unsigned char* kinds,
                                                                 const float* invs, float below, float* C, int* wrong) {
    fit_dots<0>(X, m, d, U, kinds, invs, below, C, wrong);
}

extern "C" __global__ void __launch_bounds__(256) tf_q27_fit_dots_tree(const bf16* X, int m, int d, const float* U,
                                                                      const unsigned char* kinds, const float* invs, float below,
                                                                      float* C, int* wrong) {
    fit_dots<1>(X, m, d, U, kinds, invs, below, C, wrong);
}

// A step's gradient and Adam: each group's wrong rows summed in row order (y + c x, as axpyRow).
extern "C" __global__ void __launch_bounds__(256) tf_q27_fit_step(const bf16* __restrict__ X, int m, int d,
                                                                 const unsigned char* __restrict__ kinds, const float* __restrict__ C,
                                                                 const int* __restrict__ wrong, const float* __restrict__ invs,
                                                                 float* __restrict__ U, float* __restrict__ M, float* __restrict__ Vv,
                                                                 float rate, float c1, float c2) {
    __shared__ float cs[1024];
    __shared__ unsigned char ks[1024];
    const int l = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
    float p[3] = {0.0f, 0.0f, 0.0f};
    for (int r0 = 0; r0 < m; r0 += 1024) {
        const int n = min(1024, m - r0);
        __syncthreads();
        for (int r = threadIdx.x; r < n; r += 256) {
            cs[r] = C[(size_t)l * m + r0 + r];
            ks[r] = kinds[(size_t)l * m + r0 + r];
        }
        __syncthreads();
        if (col >= d) continue;
        for (int r = 0; r < n; r++) {
            const float c = cs[r];
            if (c == 0.0f) continue;
            const float x = f32(X[((size_t)l * m + r0 + r) * d + col]);
            const int at = ks[r] == FIT_FACT ? 0 : (ks[r] == FIT_STEADY ? 1 : 2);
            p[at] = __fadd_rn(p[at], __fmul_rn(c, x));
        }
    }
    if (col >= d) return;
    const float inv = invs[l];
    float g = 0.0f;
#pragma unroll
    for (int at = 0; at < 3; at++) g = __fadd_rn(g, __fmul_rn(__fdiv_rn(inv, (float)max(wrong[l * 4 + at], 1)), p[at]));
    const size_t k = (size_t)l * d + col;
    const float ui = U[k];
    const float gg = __fadd_rn(g, __fmul_rn(2e-4f, ui));
    const float mi = __fadd_rn(__fmul_rn(0.9f, M[k]), __fmul_rn(0.1f, gg));
    const float vi = __fadd_rn(__fmul_rn(0.999f, Vv[k]), __fmul_rn(__fmul_rn(0.001f, gg), gg));
    M[k] = mi;
    Vv[k] = vi;
    const float st = __fdiv_rn(__fmul_rn(rate, __fdiv_rn(mi, c1)), __fadd_rn(__fsqrt_rn(__fdiv_rn(vi, c2)), 1e-8f));
    U[k] = __fsub_rn(ui, st);
}

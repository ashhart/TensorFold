// Input gradients for MLX affine 4-bit (group 32) models: linears, routed SwiGLU experts and softmax top-k routing.

// fp32 gradients from bf16 kept activations, as train.cu; experts read MLX stacks or experts_pack.cu's packed table.

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

// ---- products -------------------------------------------------------------------------------------------------------

// C [M, N] (+)= A [M, K] W, W 4-bit [K, N]: tf_train_gemm's tiles, A and W in bf16; split z writes partial C + z M N.
template <typename TA>
__device__ __forceinline__ void gemm_q4(const TA* __restrict__ A, const unsigned* __restrict__ W, const bf16* __restrict__ S,
                                        const bf16* __restrict__ B, float* __restrict__ C, int M, int N, int K, int lda, int ldc,
                                        int accumulate, int kchunk) {
    __shared__ __align__(16) bf16 As[64][40];
    __shared__ __align__(16) bf16 Bs[64][40];  // W's tile transposed: [n][k]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int kb = blockIdx.z * kchunk, ke = min(K, kb + kchunk);
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const int words = N / 8, groups = N / 32;
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
                const float s = f32(S[(size_t)gk * groups + gn / 32]), b = f32(B[(size_t)gk * groups + gn / 32]);
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

extern "C" __global__ void __launch_bounds__(128) tf_q4_back(const float* A, const unsigned* W, const bf16* S, const bf16* B,
                                                               float* C, int M, int N, int K, int lda, int ldc, int accumulate,
                                                               int kchunk) {
    gemm_q4<float>(A, W, S, B, C, M, N, K, lda, ldc, accumulate, kchunk);
}

// tf_q4_back with bf16 A (a softmax's bf16 gradient rows).
extern "C" __global__ void __launch_bounds__(128) tf_q4_back_bf(const bf16* A, const unsigned* W, const bf16* S, const bf16* B,
                                                                  float* C, int M, int N, int K, int lda, int ldc, int accumulate,
                                                                  int kchunk) {
    gemm_q4<bf16>(A, W, S, B, C, M, N, K, lda, ldc, accumulate, kchunk);
}

// C [M, N] (rows ldc apart) (+)= the split products' partials [splits, M, N], summed in split order.
extern "C" __global__ void tf_q4_sum_parts(const float* parts, int splits, int M, int N, float* C, int ldc, int accumulate) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * N) return;
    float s = 0;
    for (int z = 0; z < splits; z++) s += parts[(size_t)z * M * N + i];
    float* out = C + (size_t)(i / N) * ldc + i % N;
    *out = accumulate ? *out + s : s;
}

// ---- routed experts -------------------------------------------------------------------------------------------------

// Expert e's 4-bit matrix: e below `routed` from the stack w [routed, n, k], e == routed the shared expert's w2.
__device__ __forceinline__ void expert_mat(int e, int routed, size_t per_w, size_t per_s, const unsigned*& w, const bf16*& s,
                                           const bf16*& b, const unsigned* w2, const bf16* s2, const bf16* b2) {
    if (e == routed && w2 != nullptr) {
        w = w2, s = s2, b = b2;
    } else {
        w += (size_t)e * per_w, s += (size_t)e * per_s, b += (size_t)e * per_s;
    }
}

// y[s] (+)= x[s] w[e] (fp32) for each sorted pair s, x[order[s] / slots] if order is set; the shared expert: `routed`.
extern "C" __global__ void __launch_bounds__(64) tf_q4_experts_mm(const float* x, int ldx, const int* order, int slots,
                                                                 const unsigned* w, const bf16* scales, const bf16* biases,
                                                                 const unsigned* w2, const bf16* s2, const bf16* b2,
                                                                 const unsigned* tiles, const int* offsets, float* y, int pairs,
                                                                 int n_in, int k_out, int routed, int accumulate) {
    __shared__ float xs[16][33];
    __shared__ __align__(16) float ws[32][64];
    const int e = (int)tiles[2 * blockIdx.y], r0 = (int)tiles[2 * blockIdx.y + 1];
    if (e > routed) return;
    const int last = e < routed ? offsets[e + 1] : pairs;
    const int k0 = blockIdx.x * 64, words = k_out / 8, groups = k_out / 32, t = threadIdx.x, tr = t / 16, tc = t % 16;
    expert_mat(e, routed, (size_t)n_in * words, (size_t)n_in * groups, w, scales, biases, w2, s2, b2);
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (int n0 = 0; n0 < n_in; n0 += 32) {
        // rows past the expert's last read its last row again, and are dropped
        for (int i = t; i < 16 * 32; i += 64) {
            const int s = min(r0 + i / 32, last - 1), row = order ? order[s] / slots : s;
            xs[i / 32][i % 32] = x[(size_t)row * ldx + n0 + i % 32];
        }
        for (int i = t; i < 32 * 8; i += 64) {
            const size_t n = (size_t)n0 + i / 8;
            const unsigned q = w[n * words + k0 / 8 + i % 8];
            const int g = (k0 + (i % 8) * 8) / 32;
            const float sc = f32(scales[n * groups + g]), bi = f32(biases[n * groups + g]);
            for (int j = 0; j < 8; j++) ws[i / 8][(i % 8) * 8 + j] = fmaf(sc, (float)((q >> (4 * j)) & 0xfu), bi);
        }
        __syncthreads();
        for (int nn = 0; nn < 32; nn++) {
            const float4 b = *reinterpret_cast<const float4*>(&ws[nn][tc * 4]);
            for (int i = 0; i < 4; i++) {
                const float a = xs[tr * 4 + i][nn];
                acc[i][0] = fmaf(a, b.x, acc[i][0]);
                acc[i][1] = fmaf(a, b.y, acc[i][1]);
                acc[i][2] = fmaf(a, b.z, acc[i][2]);
                acc[i][3] = fmaf(a, b.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        const int r = r0 + tr * 4 + i;
        if (r >= last) continue;
        float4* out = reinterpret_cast<float4*>(y + (size_t)r * k_out + k0 + tc * 4);
        float4 v = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        if (accumulate) {
            const float4 o = *out;
            v.x += o.x, v.y += o.y, v.z += o.z, v.w += o.w;
        }
        *out = v;
    }
}

// y[s, col0 + n] = w[e][n] . x[order[s] / slots] (fp32): each sorted pair's gate or up rows again, from bf16 x.
extern "C" __global__ void __launch_bounds__(64) tf_q4_experts_nt(const bf16* x, int ldx, const int* order, int slots,
                                                                 const unsigned* w, const bf16* scales, const bf16* biases,
                                                                 const unsigned* w2, const bf16* s2, const bf16* b2,
                                                                 const unsigned* tiles, const int* offsets, float* y, int ldy,
                                                                 int col0, int pairs, int k_in, int n_out, int routed) {
    __shared__ float xs[16][33];
    __shared__ __align__(16) float ws[32][68];  // [k][n]
    const int e = (int)tiles[2 * blockIdx.y], r0 = (int)tiles[2 * blockIdx.y + 1];
    if (e > routed) return;
    const int last = e < routed ? offsets[e + 1] : pairs;
    const int n0 = blockIdx.x * 64, words = k_in / 8, groups = k_in / 32, t = threadIdx.x, tr = t / 16, tc = t % 16;
    expert_mat(e, routed, (size_t)n_out * words, (size_t)n_out * groups, w, scales, biases, w2, s2, b2);
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (int k0 = 0; k0 < k_in; k0 += 32) {
        for (int i = t; i < 16 * 32; i += 64) {
            const int s = min(r0 + i / 32, last - 1);
            xs[i / 32][i % 32] = f32(x[(size_t)(order[s] / slots) * ldx + k0 + i % 32]);
        }
        for (int i = t; i < 64 * 4; i += 64) {
            const int nn = i / 4, c = i % 4;
            const size_t n = (size_t)n0 + nn;
            const unsigned q = w[n * words + k0 / 8 + c];
            const float sc = f32(scales[n * groups + k0 / 32]), bi = f32(biases[n * groups + k0 / 32]);
            for (int j = 0; j < 8; j++) ws[c * 8 + j][nn] = fmaf(sc, (float)((q >> (4 * j)) & 0xfu), bi);
        }
        __syncthreads();
        for (int kk = 0; kk < 32; kk++) {
            const float4 b = *reinterpret_cast<const float4*>(&ws[kk][tc * 4]);
            for (int i = 0; i < 4; i++) {
                const float a = xs[tr * 4 + i][kk];
                acc[i][0] = fmaf(a, b.x, acc[i][0]);
                acc[i][1] = fmaf(a, b.y, acc[i][1]);
                acc[i][2] = fmaf(a, b.z, acc[i][2]);
                acc[i][3] = fmaf(a, b.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        const int r = r0 + tr * 4 + i;
        if (r < last) *reinterpret_cast<float4*>(y + (size_t)r * ldy + col0 + n0 + tc * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

// The packed table's block of expert e (experts_pack.cu, group 32: 128 code words, then 32 of scales and biases).
__device__ __forceinline__ const unsigned* packed_block(const unsigned* w, int stack, int half, size_t e, int R, int K, int r,
                                                        int k) {
    const size_t b = (e * (size_t)(R / 32) + (size_t)(r / 32)) * (size_t)(K / 32) + (size_t)(k / 32);
    return w + (b * stack + half) * 160;
}

// Row rr's word kk (codes 8 kk .. 8 kk + 7 of the block's 32 inputs) in MLX's order: code j at bits 4 j.
__device__ __forceinline__ unsigned packed_word(const unsigned* blk, int rr, int kk) {
    const int i = rr * 4 + kk;
    const unsigned p = blk[(i % 32) * 4 + i / 32];
    unsigned out = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) out |= (((p >> (4 * j)) & 0xfu) << (8 * j)) | (((p >> (16 + 4 * j)) & 0xfu) << (8 * j + 4));
    return out;
}

// Row rr's scale and bias in its block.
__device__ __forceinline__ void packed_sb(const unsigned* blk, int rr, float& s, float& b) {
    const int t = rr / 8, p = (rr % 8) / 2, hi = rr % 2;
    const unsigned ws = blk[128 + p * 8 + t], wb = blk[128 + p * 8 + 4 + t];
    s = __uint_as_float((hi ? ws >> 16 : ws & 0xffffu) << 16);
    b = __uint_as_float((hi ? wb >> 16 : wb & 0xffffu) << 16);
}

// tf_q4_experts_nt over the packed table w [E, n_out, k_in] (`stack` tables side by side, `half` picks one).
extern "C" __global__ void __launch_bounds__(64) tf_q4_experts_nt_packed(const bf16* x, int ldx, const int* order, int slots,
                                                                        const unsigned* w, int stack, int half,
                                                                        const unsigned* tiles, const int* offsets, float* y, int ldy,
                                                                        int col0, int pairs, int k_in, int n_out, int routed) {
    __shared__ float xs[16][33];
    __shared__ __align__(16) float ws[32][68];  // [k][n]
    const int e = (int)tiles[2 * blockIdx.y], r0 = (int)tiles[2 * blockIdx.y + 1];
    if (e > routed) return;
    const int last = e < routed ? offsets[e + 1] : pairs;
    const int n0 = blockIdx.x * 64, t = threadIdx.x, tr = t / 16, tc = t % 16;
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (int k0 = 0; k0 < k_in; k0 += 32) {
        for (int i = t; i < 16 * 32; i += 64) {
            const int s = min(r0 + i / 32, last - 1);
            xs[i / 32][i % 32] = f32(x[(size_t)(order[s] / slots) * ldx + k0 + i % 32]);
        }
        for (int i = t; i < 64 * 4; i += 64) {
            const int nn = i / 4, c = i % 4, n = n0 + nn;
            const unsigned* blk = packed_block(w, stack, half, e, n_out, k_in, n, k0);
            const unsigned q = packed_word(blk, n % 32, c);
            float sc, bi;
            packed_sb(blk, n % 32, sc, bi);
            for (int j = 0; j < 8; j++) ws[c * 8 + j][nn] = fmaf(sc, (float)((q >> (4 * j)) & 0xfu), bi);
        }
        __syncthreads();
        for (int kk = 0; kk < 32; kk++) {
            const float4 b = *reinterpret_cast<const float4*>(&ws[kk][tc * 4]);
            for (int i = 0; i < 4; i++) {
                const float a = xs[tr * 4 + i][kk];
                acc[i][0] = fmaf(a, b.x, acc[i][0]);
                acc[i][1] = fmaf(a, b.y, acc[i][1]);
                acc[i][2] = fmaf(a, b.z, acc[i][2]);
                acc[i][3] = fmaf(a, b.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        const int r = r0 + tr * 4 + i;
        if (r < last) *reinterpret_cast<float4*>(y + (size_t)r * ldy + col0 + n0 + tc * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

// tf_q4_experts_mm over the packed table w [E, n_in, k_out].
extern "C" __global__ void __launch_bounds__(64) tf_q4_experts_mm_packed(const float* x, int ldx, const int* order, int slots,
                                                                        const unsigned* w, int stack, int half,
                                                                        const unsigned* tiles, const int* offsets, float* y, int pairs,
                                                                        int n_in, int k_out, int routed, int accumulate) {
    __shared__ float xs[16][33];
    __shared__ __align__(16) float ws[32][64];
    const int e = (int)tiles[2 * blockIdx.y], r0 = (int)tiles[2 * blockIdx.y + 1];
    if (e > routed) return;
    const int last = e < routed ? offsets[e + 1] : pairs;
    const int k0 = blockIdx.x * 64, t = threadIdx.x, tr = t / 16, tc = t % 16;
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (int n0 = 0; n0 < n_in; n0 += 32) {
        for (int i = t; i < 16 * 32; i += 64) {
            const int s = min(r0 + i / 32, last - 1), row = order ? order[s] / slots : s;
            xs[i / 32][i % 32] = x[(size_t)row * ldx + n0 + i % 32];
        }
        for (int i = t; i < 32 * 8; i += 64) {
            const int n = n0 + i / 8, kw = i % 8, k = k0 + kw * 8;
            const unsigned* blk = packed_block(w, stack, half, e, n_in, k_out, n, k);
            const unsigned q = packed_word(blk, n % 32, (k % 32) / 8);
            float sc, bi;
            packed_sb(blk, n % 32, sc, bi);
            for (int j = 0; j < 8; j++) ws[i / 8][kw * 8 + j] = fmaf(sc, (float)((q >> (4 * j)) & 0xfu), bi);
        }
        __syncthreads();
        for (int nn = 0; nn < 32; nn++) {
            const float4 b = *reinterpret_cast<const float4*>(&ws[nn][tc * 4]);
            for (int i = 0; i < 4; i++) {
                const float a = xs[tr * 4 + i][nn];
                acc[i][0] = fmaf(a, b.x, acc[i][0]);
                acc[i][1] = fmaf(a, b.y, acc[i][1]);
                acc[i][2] = fmaf(a, b.z, acc[i][2]);
                acc[i][3] = fmaf(a, b.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        const int r = r0 + tr * 4 + i;
        if (r >= last) continue;
        float4* out = reinterpret_cast<float4*>(y + (size_t)r * k_out + k0 + tc * 4);
        float4 v = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        if (accumulate) {
            const float4 o = *out;
            v.x += o.x, v.y += o.y, v.z += o.z, v.w += o.w;
        }
        *out = v;
    }
}

// Pair s's SwiGLU undone from dact: a[p] = dact . act (dL/d its weight), dgu = wt[p] (dact u silu'(g), dact silu(g)).
extern "C" __global__ void __launch_bounds__(128) tf_q4_swiglu_back(const float* gu, const float* dact, const int* order,
                                                                   const float* wts, float* a, float* dgu, int width) {
    __shared__ float scratch[32];
    const int s = blockIdx.x, t = threadIdx.x, p = order[s];
    const float wt = wts[p];
    const float* g = gu + (size_t)s * 2 * width;
    const float* da = dact + (size_t)s * width;
    float* out = dgu + (size_t)s * 2 * width;
    float dot = 0;
    for (int i = t; i < width; i += 128) {
        const float gv = g[i], uv = g[width + i], d = da[i], sg = silu(gv);
        dot += d * sg * uv;
        out[i] = wt * d * uv * silu_grad(gv);
        out[width + i] = wt * d * sg;
    }
    dot = block_sum_tree(dot, scratch, 4);
    if (t == 0) a[p] = dot;
}

// ---- routing --------------------------------------------------------------------------------------------------------

// dx[r] += the top_k softmax weights' and the shared gate's (sigmoid of logit `shared`) gradients through the router.
extern "C" __global__ void __launch_bounds__(256) tf_q4_route_back(const float* logits, int nl, const int* ids, int slots,
                                                                  const float* a, const bf16* router, float* dx, int top_k,
                                                                  int shared, int dim, float* dlog) {
    __shared__ float dl[17];
    __shared__ int at[17];
    const int r = blockIdx.x, t = threadIdx.x;
    if (t == 0) {
        const float* l = logits + (size_t)r * nl;
        float top = -INFINITY;
        for (int k = 0; k < top_k; k++) top = fmaxf(top, l[ids[r * slots + k]]);
        float e[16], total = 0;
        for (int k = 0; k < top_k; k++) {
            e[k] = expf(l[ids[r * slots + k]] - top);
            total += e[k];
        }
        float mix = 0;
        for (int k = 0; k < top_k; k++) mix += e[k] / total * a[r * slots + k];
        for (int k = 0; k < top_k; k++) {
            dl[k] = e[k] / total * (a[r * slots + k] - mix);
            at[k] = ids[r * slots + k];
        }
        const float sg = sigmoidf_(l[shared]);
        dl[top_k] = a[r * slots + top_k] * sg * (1.0f - sg);
        at[top_k] = shared;
        if (dlog)
            for (int k = 0; k <= top_k; k++) dlog[(size_t)r * slots + k] = dl[k];
    }
    __syncthreads();
    for (int i = t; i < dim; i += 256) {
        float s = 0;
        for (int k = 0; k <= top_k; k++) s += dl[k] * f32(router[(size_t)at[k] * dim + i]);
        dx[(size_t)r * dim + i] += s;
    }
}

// y[i] += x[i] for n floats.
extern "C" __global__ void tf_q4_axpy(float* y, const float* x, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// Learning in the weights on CUDA: train.metal's kernels one for one (a simdgroup a warp), on this engine's layouts.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 b16(float v) { return __float2bfloat16(v); }

using tf_common::warp_sum;
using tf_common::warp_max;
using tf_common::block_sum_tree;
using tf_common::block_max_tree;

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

__device__ __forceinline__ float silu_grad(float v) {
    const float s = 1.0f / (1.0f + expf(-v));
    return s * (1.0f + v * (1.0f - s));
}

__device__ __forceinline__ float softplus(float v) { return v > 20.0f ? v : log1pf(expf(v)); }

// ---- the loss -------------------------------------------------------------------------------------------------------

// A row's softmax over bf16 logits: stats = (loss, target's probability), then in place (p - onehot) weight[r].
extern "C" __global__ void __launch_bounds__(1024) tf_train_softmax(bf16* logits, const unsigned* targets, const float* weights,
                                                                    float2* stats, int vocab) {
    __shared__ float scratch[32];
    const int r = blockIdx.x, t = threadIdx.x;
    bf16* l = logits + (size_t)r * vocab;
    float m = -INFINITY;
    for (int v = t; v < vocab; v += 1024) m = fmaxf(m, f32(l[v]));
    m = block_max_tree(m, scratch, 32);
    float s = 0;
    for (int v = t; v < vocab; v += 1024) s += expf(f32(l[v]) - m);
    s = block_sum_tree(s, scratch, 32);
    const unsigned target = targets[r];
    const float lt = f32(l[target]);
    if (t == 0) stats[r] = make_float2(logf(s) + m - lt, expf(lt - m) / s);
    __syncthreads();
    const float w = weights[r];
    for (int v = t; v < vocab; v += 1024) l[v] = b16((expf(f32(l[v]) - m) / s - (v == (int)target ? 1.0f : 0.0f)) * w);
}

// g[r] += s w dx - s^3 h (w dx . h) / dim: a row's input RMS norm undone onto the residual's gradient; 256 a row.
extern "C" __global__ void __launch_bounds__(256) tf_train_rms_back(const bf16* h, const bf16* w, const float* dx, float* g, int dim,
                                                                   float eps) {
    __shared__ float scratch[32];
    const int r = blockIdx.x, t = threadIdx.x;
    const bf16* hr = h + (size_t)r * dim;
    const float* dr = dx + (size_t)r * dim;
    float sq = 0, dot = 0;
    for (int j = t; j < dim; j += 256) {
        const float v = f32(hr[j]);
        sq += v * v;
        dot += f32(w[j]) * dr[j] * v;
    }
    sq = block_sum_tree(sq, scratch, 8);
    dot = block_sum_tree(dot, scratch, 8);
    const float s = rsqrtf(sq / (float)dim + eps), k = s * s * s * dot / (float)dim;
    for (int j = t; j < dim; j += 256) g[(size_t)r * dim + j] += s * f32(w[j]) * dr[j] - k * f32(hr[j]);
}

// ---- the low-rank change --------------------------------------------------------------------------------------------

// xa[r, q] = x[r] . a[q] and xn[r] = |x[r]|^2 (x rows `x_stride` apart); a warp a (q, r), grid (32 rank / 256, rows).
extern "C" __global__ void __launch_bounds__(256) tf_train_lora_in(const bf16* x, long long x_stride, const float* a, float* xa,
                                                                  int rows, int n_in, int rank, float* xn) {
    const int q = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31, r = blockIdx.y;
    if (q >= rank || r >= rows) return;
    const bf16* xr = x + (size_t)r * x_stride;
    float s = 0, n = 0;
    for (int i = lane; i < n_in; i += 32) {
        const float v = f32(xr[i]);
        s += v * a[(size_t)q * n_in + i];
        n += v * v;
    }
    s = warp_sum(s);
    n = warp_sum(n);
    if (lane == 0) xa[(size_t)r * rank + q] = s;
    if (lane == 0 && q == 0) xn[r] = n;
}

// A pass's rows by stream: each segment's first row, rows, its decision row (an index in the pass) and its latch slot.
struct Route {
    unsigned count, known, whole, pad1; // known: every latch was set before the pass (a training example); whole: see the gate
    int seg[8][4];
};

// Block b acts on row r after its stream's decision row when its lesson chose to; the decision row adds to its score.
extern "C" __global__ void tf_train_gate(float* xa, const float* xn, const float* tau, float* gates, int rows, int rank, int blocks,
                                         int stride, float unit, Route route, const unsigned* latch, float* score) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (b >= blocks || r >= rows) return;
    int rel = -1;
    unsigned slot = 0;
    bool inside = false, settled = false;
    for (unsigned s = 0; s < route.count; s++) {
        const int* g = route.seg[s];
        if (r >= g[0] && r < g[0] + g[1]) {
            inside = true;
            rel = r - g[2];
            slot = (unsigned)g[3];
            settled = route.known != 0 || g[2] < g[0];
        }
    }
    float* row = xa + (size_t)r * rank + b * 16;
    bool open = false;
    if (route.whole != 0) {
        // lessons that edit the weights: every row, a training pass's flagged lessons, an inference pass's all
        open = inside && (route.known == 0 || latch[slot * stride + b] != 0);
    } else if (rel == 0) {
        // the decision row: this layer's part of the router score (its first direction's weighed cosine less tau) only
        const float n = unit * xn[r];
        if (n > 0) score[slot * stride + b] += row[0] * rsqrtf(n) - tau[b];
    } else if (rel > 0 && settled) {
        open = latch[slot * stride + b] != 0;
    }
    gates[(size_t)r * blocks + b] = open ? 1.0f : 0.0f;
    if (!open)
        for (int q = 0; q < 16; q++) row[q] = 0;
}

// A stream's lessons at its decision row: cleared before a pass; after it, those whose score reached `need` (0: never).
extern "C" __global__ void tf_lesson_latch(float* score, const float* need, unsigned* latch, int slot, int blocks, int stride, int set) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int b = 0; b < blocks; b++) {
        latch[slot * stride + b] = set && need[b] > 0 && score[slot * stride + b] >= need[b] ? 1u : 0u;
        if (!set) score[slot * stride + b] = 0;
    }
}

// y[r, j] += scale xa[r] . b[:, j] (y rows `y_stride` apart, bf16 or fp32); grid (out / 256, rows).
extern "C" __global__ void __launch_bounds__(256) tf_train_lora_out(const float* xa, const float* b, void* y, long long y_stride,
                                                                   int y_f32, int rows, int n_out, int rank, float scale) {
    const int j = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (j >= n_out || r >= rows) return;
    float s = 0;
    for (int q = 0; q < rank; q++) s += xa[(size_t)r * rank + q] * b[(size_t)q * n_out + j];
    const size_t at = (size_t)r * y_stride + j;
    if (y_f32)
        static_cast<float*>(y)[at] += scale * s;
    else
        static_cast<bf16*>(y)[at] = b16(f32(static_cast<bf16*>(y)[at]) + scale * s);
}

// p[r, j] = f[j] . x[r] / |x[r]|: a row's input, made unit, along each candidate direction; a warp a (j, r).
extern "C" __global__ void __launch_bounds__(256) tf_train_project(const bf16* x, long long x_stride, const float* f, float* p,
                                                                  int rows, int n_in, int k) {
    const int j = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31, r = blockIdx.y;
    if (j >= k || r >= rows) return;
    const bf16* xr = x + (size_t)r * x_stride;
    float s = 0, n = 0;
    for (int i = lane; i < n_in; i += 32) {
        const float v = f32(xr[i]);
        s += v * f[(size_t)j * n_in + i];
        n += v * v;
    }
    s = warp_sum(s);
    n = warp_sum(n);
    if (lane == 0) p[(size_t)r * k + j] = n > 0 ? s * rsqrtf(n) : 0;
}

// db[q, j] += scale sum_r xa[r, first + q] g[r, j] for the open block's ranks; grid (out / 256, rank - first).
extern "C" __global__ void __launch_bounds__(256) tf_train_lora_db(const float* xa, const float* g, float* db, int rows, int n_out,
                                                                  int rank, int first, float scale) {
    const int j = blockIdx.x * 256 + threadIdx.x, q = blockIdx.y;
    if (j >= n_out || first + q >= rank) return;
    float s = 0;
    for (int r = 0; r < rows; r++) s += xa[(size_t)r * rank + first + q] * g[(size_t)r * n_out + j];
    db[(size_t)q * n_out + j] += scale * s;
}

// dxa[r, q] = scale g[r] . b[q] where q's block is open on row r, else 0; a warp a (q, r).
extern "C" __global__ void __launch_bounds__(256) tf_train_lora_dxa(const float* g, const float* b, float* dxa, int rows, int n_out,
                                                                   int rank, float scale, const float* gates) {
    const int q = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31, r = blockIdx.y;
    if (q >= rank || r >= rows) return;
    float s = 0;
    for (int j = lane; j < n_out; j += 32) s += g[(size_t)r * n_out + j] * b[(size_t)q * n_out + j];
    s = warp_sum(s);
    if (lane == 0) dxa[(size_t)r * rank + q] = scale * s * gates[(size_t)r * (rank / 16) + q / 16];
}

// dx[r, i] += sum_q dxa[r, q] a[q, i]: the change's part of its site's input gradient; grid (in / 256, rows).
extern "C" __global__ void __launch_bounds__(256) tf_train_lora_dx(const float* dxa, const float* a, float* dx, int rows, int n_in,
                                                                  int rank) {
    const int i = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (i >= n_in || r >= rows) return;
    float s = 0;
    for (int q = 0; q < rank; q++) s += dxa[(size_t)r * rank + q] * a[(size_t)q * n_in + i];
    dx[(size_t)r * n_in + i] += s;
}

// A sign from a row's place and a sketch column, +1 or -1 with even odds (train.metal's coin).
__device__ __forceinline__ float coin(unsigned row, unsigned j, unsigned seed) {
    unsigned h = row * 0x9E3779B9u + j * 0x7FEB352Du + seed;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

// y[j, i] += w sum_r coin(first + r, j) x[r, i]: a site's input rows into a random sketch; grid (in / 256, k).
extern "C" __global__ void __launch_bounds__(256) tf_train_sketch(const bf16* x, long long x_stride, float* y, int rows, int n_in,
                                                                 int k, unsigned first, unsigned seed, float w) {
    const int i = blockIdx.x * 256 + threadIdx.x, j = blockIdx.y;
    if (i >= n_in || j >= k) return;
    float s = 0;
    for (int r = 0; r < rows; r++) s += coin(first + r, j, seed) * f32(x[(size_t)r * x_stride + i]);
    y[(size_t)j * n_in + i] += w * s;
}

// One Adam step on a change's factors; the gradient is then cleared.
extern "C" __global__ void tf_train_adam(float* p, float* g, float* m, float* v, long long n, float lr, float b1, float b2,
                                         float eps, float c1, float c2) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = g[i];
    const float mi = b1 * m[i] + (1 - b1) * gi, vi = b2 * v[i] + (1 - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    p[i] -= lr * (mi * c1) / (sqrtf(vi * c2) + eps);
    g[i] = 0;
}

// ---- products and small moves ---------------------------------------------------------------------------------------

// C [M, N] (+)= A [M, K] B [K, N], bf16 row-major in, fp32 out: 64x64 tiles, four warps of 32x32, mma m16n8k16.
extern "C" __global__ void __launch_bounds__(128) tf_train_gemm(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                                               float* __restrict__ C, int M, int N, int K, int lda, int ldb,
                                                               int ldc, int accumulate) {
    __shared__ __align__(16) bf16 As[64][40];
    __shared__ __align__(16) bf16 Bs[64][40];  // B's tile transposed: [n][k]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    const bf16 zero = b16(0.0f);
    // eight values a load where a row's eight lie inside the matrix and the strides keep 16-byte alignment
    const bool wide_a = (lda & 7) == 0 && ((size_t)A & 15) == 0, wide_b = (ldb & 7) == 0 && ((size_t)B & 15) == 0;
    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int e = tid; e < 64 * 4; e += 128) {
            const int r = e >> 2, c = (e & 3) * 8, gm = m0 + r, gk = k0 + c;
            if (wide_a && gm < M && gk + 8 <= K) {
                *reinterpret_cast<uint4*>(&As[r][c]) = *reinterpret_cast<const uint4*>(A + (size_t)gm * lda + gk);
            } else {
                for (int i = 0; i < 8; i++) As[r][c + i] = (gm < M && gk + i < K) ? A[(size_t)gm * lda + gk + i] : zero;
            }
        }
        for (int e = tid; e < 32 * 8; e += 128) {
            const int r = e >> 3, c = (e & 7) * 8, gk = k0 + r, gn = n0 + c;
            if (wide_b && gk < K && gn + 8 <= N) {
                const uint4 v = *reinterpret_cast<const uint4*>(B + (size_t)gk * ldb + gn);
                const bf16* h = reinterpret_cast<const bf16*>(&v);
                for (int i = 0; i < 8; i++) Bs[c + i][r] = h[i];
            } else {
                for (int i = 0; i < 8; i++) Bs[c + i][r] = (gk < K && gn + i < N) ? B[(size_t)gk * ldb + gn + i] : zero;
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
                    float* out = C + (size_t)r * ldc + c + e;
                    const float v = acc[i][j][h * 2 + e];
                    *out = accumulate ? *out + v : v;
                }
            }
}

// A 4-bit [N, K] matrix (MLX's layout: 8 codes a word, a bf16 scale and bias every 64) as bf16 [N, K], in order.
extern "C" __global__ void tf_train_dequant(const unsigned* w, const bf16* scales, const bf16* biases, bf16* out, long long count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned q = (w[i / 8] >> (4 * (i % 8))) & 0xfu;
    out[i] = b16(f32(scales[i / 64]) * (float)q + f32(biases[i / 64]));
}

extern "C" __global__ void tf_train_zero(float* x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0;
}

extern "C" __global__ void tf_train_widen(const bf16* x, float* out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f32(x[i]);
}

extern "C" __global__ void tf_train_narrow(const float* x, bf16* out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = b16(x[i]);
}

extern "C" __global__ void tf_train_add(const float* x, float* y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// dst row r = src row (idx ? idx[r] : r), `width` bf16 each, rows their strides apart; grid (width / 256, rows).
extern "C" __global__ void tf_train_rows(const bf16* src, long long src_stride, const int* idx, bf16* dst, long long dst_stride,
                                         int width, int rows) {
    const int i = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (i >= width || r >= rows) return;
    const long long from = idx ? idx[r] : r;
    dst[(size_t)r * dst_stride + i] = src[(size_t)from * src_stride + i];
}

// out[r] = wt[r * wt_stride + wt_at] g[r]: a row's gradient at its slot's routing weight; grid (dim / 256, rows).
extern "C" __global__ void tf_train_scale_rows(const float* g, const float* wt, int wt_stride, int wt_at, float* out, int dim) {
    const int i = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (i >= dim) return;
    out[(size_t)r * dim + i] = wt[(size_t)r * wt_stride + wt_at] * g[(size_t)r * dim + i];
}

// dx[r, i] += sum_j g[r, j] rest[j, i]: a projection's bf16 rest [n, width] carried back to its input.
extern "C" __global__ void __launch_bounds__(256) tf_train_rest_back(const float* g, const bf16* rest, float* dx, long long dx_stride,
                                                                    int rows, int n, int width) {
    const int i = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (i >= width || r >= rows) return;
    float s = 0;
    for (int j = 0; j < n; j++) s += g[(size_t)r * n + j] * f32(rest[(size_t)j * width + i]);
    dx[(size_t)r * dx_stride + i] += s;
}

// ---- the MoE layer --------------------------------------------------------------------------------------------------

// du = da 2 relu(u) from the squared ReLU's output a = relu(u)^2, which this engine keeps: relu(u) = sqrt(a).
extern "C" __global__ void tf_train_relu2_back(const bf16* a, const float* da, float* du, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) du[i] = da[i] * 2.0f * sqrtf(fmaxf(f32(a[i]), 0.0f));
}

// The routed pairs (slots below top_k) sorted by expert, stable: order (slot to pair), inv and offsets; 1024 experts.
extern "C" __global__ void __launch_bounds__(256) tf_train_sort_pairs(const int* ids, int rows, int slots, int top_k, int experts,
                                                                     int* order, int* inv, int* offsets) {
    __shared__ int count[1024];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        int n = 0;
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < top_k; j++) n += ids[r * slots + j] == e;
        count[e] = n;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int at = 0;
        for (int x = 0; x < experts; x++) {
            const int c = count[x];
            count[x] = at;
            at += c;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        int at = count[e];
        offsets[e] = at;
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < top_k; j++)
                if (ids[r * slots + j] == e) {
                    order[at] = r * slots + j;
                    inv[r * top_k + j] = at;
                    at++;
                }
    }
}

// logits[r, e] = x[r] . gate[e] in fp32: the router's scores again, for the routing weights' gradient; a warp a (e, r).
extern "C" __global__ void __launch_bounds__(256) tf_train_logits(const bf16* x, const bf16* gate, float* logits, int rows,
                                                                 int experts, int dim) {
    const int e = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31, r = blockIdx.y;
    if (e >= experts || r >= rows) return;
    float s = 0;
    for (int i = lane; i < dim; i += 32) s += f32(x[(size_t)r * dim + i]) * f32(gate[(size_t)e * dim + i]);
    s = warp_sum(s);
    if (lane == 0) logits[(size_t)r * experts + e] = s;
}

// dy[s] = wt[p] g[p / slots] at sorted slot s of pair p = order[s]: the routed experts' output gradients.
extern "C" __global__ void tf_train_pairs_in(const int* order, const float* wt, const float* g, float* dy, int dim, int slots) {
    const int j = blockIdx.x * 256 + threadIdx.x, s = blockIdx.y;
    if (j >= dim) return;
    const int p = order[s];
    dy[(size_t)s * dim + j] = wt[p] * g[(size_t)(p / slots) * dim + j];
}

// dx[t] += its top_k pairs' input gradients, summed in pair order through inv; grid (dim / 256, rows).
extern "C" __global__ void tf_train_pairs_out(const int* inv, const float* dxp, float* dx, int dim, int top_k) {
    const int j = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (j >= dim) return;
    float s = 0;
    for (int k = 0; k < top_k; k++) s += dxp[(size_t)inv[t * top_k + k] * dim + j];
    dx[(size_t)t * dim + j] += s;
}

// dx[r] += sum_j dz_j gate[e_j]: weights s_j / sum s (s = sigmoid(logit)) back to the router's input; ys unweighted.
extern "C" __global__ void __launch_bounds__(256) tf_train_route_back(const float* logits, const int* ids, int slots, const bf16* ys,
                                                                     const float* g, const bf16* gate, float* dx, int experts,
                                                                     int top_k, int dim, float scale) {
    __shared__ float scratch[32];
    __shared__ float dz[16];
    const int r = blockIdx.x, t = threadIdx.x;
    float a[16];
    for (int j = 0; j < top_k; j++) {
        float s = 0;
        for (int i = t; i < dim; i += 256) s += g[(size_t)r * dim + i] * f32(ys[((size_t)r * slots + j) * dim + i]);
        a[j] = block_sum_tree(s, scratch, 8);
    }
    if (t == 0) {
        float p[16];
        float total = 0;
        for (int j = 0; j < top_k; j++) {
            const float z = logits[(size_t)r * experts + ids[r * slots + j]];
            p[j] = 1.0f / (1.0f + expf(-z));
            total += p[j];
        }
        total += 1e-20f;
        float mix = 0;
        for (int j = 0; j < top_k; j++) mix += a[j] * p[j] / total;
        for (int j = 0; j < top_k; j++) dz[j] = scale / total * (a[j] - mix) * p[j] * (1 - p[j]);
    }
    __syncthreads();
    for (int i = t; i < dim; i += 256) {
        float s = 0;
        for (int j = 0; j < top_k; j++) s += dz[j] * f32(gate[(size_t)ids[r * slots + j] * dim + i]);
        dx[(size_t)r * dim + i] += s;
    }
}

// Each expert's pairs cut into tiles of 16: tiles[2t, 2t + 1] = (expert, first pair), then (experts, 0) up to `cap`.
extern "C" __global__ void tf_train_experts_tiles(const int* offsets, unsigned* tiles, int pairs, int cap, int experts) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int t = 0;
    for (int e = 0; e < experts; e++) {
        const int first = offsets[e], last = e + 1 < experts ? offsets[e + 1] : pairs;
        for (int r = first; r < last && t < cap; r += 16, t++) {
            tiles[2 * t] = e;
            tiles[2 * t + 1] = r;
        }
    }
    for (; t < cap; t++) {
        tiles[2 * t] = experts;
        tiles[2 * t + 1] = 0;
    }
}

// y[s] = x[s] w[e] (f32), tiles of 16 pairs by 64 columns: w dequantized once a tile, each row's sum over n in order.
extern "C" __global__ void __launch_bounds__(64) tf_train_experts_mm(const float* x, const unsigned* w, const bf16* scales,
                                                                      const bf16* biases, const unsigned* tiles, const int* offsets,
                                                                      float* y, int pairs, int n_in, int k_out, int experts) {
    __shared__ float xs[16][33];
    __shared__ __align__(16) float ws[32][64];
    const int e = (int)tiles[2 * blockIdx.y], r0 = (int)tiles[2 * blockIdx.y + 1];
    if (e >= experts) return;
    const int last = e + 1 < experts ? offsets[e + 1] : pairs;
    const int k0 = blockIdx.x * 64, words = k_out / 8, groups = k_out / 64, t = threadIdx.x, tr = t / 16, tc = t % 16;
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (int n0 = 0; n0 < n_in; n0 += 32) {
        // rows past the expert's last read its last row again, and are dropped
        for (int i = t; i < 16 * 32; i += 64) xs[i / 32][i % 32] = x[(size_t)min(r0 + i / 32, last - 1) * n_in + n0 + i % 32];
        for (int i = t; i < 32 * 8; i += 64) {
            const size_t n = (size_t)e * n_in + n0 + i / 8;
            const unsigned q = w[n * words + k0 / 8 + i % 8];
            const float sc = f32(scales[n * groups + k0 / 64]), bi = f32(biases[n * groups + k0 / 64]);
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
        if (r < last) *reinterpret_cast<float4*>(y + (size_t)r * k_out + k0 + tc * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

// nemotron_ops' tf_pocket_rows over a pass's rows, the neurons from `from` on times their row's scale; grid rows.
extern "C" __global__ void __launch_bounds__(256) tf_train_pocket_rows(const bf16* __restrict__ y, int d, const float* __restrict__ up,
                                                                      const float* __restrict__ down, int n, void* __restrict__ out,
                                                                      int out_f32, int row_mul, int row_add, const float* scale,
                                                                      int from) {
    __shared__ float act[64];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const bf16* yr = y + static_cast<size_t>(blockIdx.x) * d;
    for (int j = warp; j < n; j += 8) {
        const float* uj = up + static_cast<size_t>(j) * d;
        float s = 0.f;
        for (int i = lane; i < d; i += 32) s += uj[i] * __bfloat162float(yr[i]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) {
            const float z = fmaxf(s, 0.f);
            float a = z * z;
            if (j >= from) a *= scale[blockIdx.x];
            act[j] = a;
        }
    }
    __syncthreads();
    const size_t row = static_cast<size_t>(blockIdx.x) * row_mul + row_add;
    for (int k = threadIdx.x; k < d; k += 256) {
        float s = 0.f;
        for (int j = 0; j < n; ++j) s += act[j] * down[static_cast<size_t>(j) * d + k];
        if (out_f32) {
            static_cast<float*>(out)[row * d + k] += s;
        } else {
            __nv_bfloat16* o = static_cast<__nv_bfloat16*>(out) + row * d + k;
            *o = __float2bfloat16(__bfloat162float(*o) + s);
        }
    }
}

#include "train_mixers.cuh"

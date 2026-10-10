// Learning in the weights on CUDA: train.metal's kernels one for one (a simdgroup a warp), on this engine's layouts.

#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 b16(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// The sum over a block of `groups` warps, the same in every thread.
__device__ __forceinline__ float block_sum(float v, float* scratch, int groups) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    v = warp_sum(lane < groups ? scratch[lane] : 0.0f);
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_max(float v, float* scratch, int groups) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    v = warp_max(lane < groups ? scratch[lane] : -INFINITY);
    __syncthreads();
    return v;
}

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
    m = block_max(m, scratch, 32);
    float s = 0;
    for (int v = t; v < vocab; v += 1024) s += expf(f32(l[v]) - m);
    s = block_sum(s, scratch, 32);
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
    sq = block_sum(sq, scratch, 8);
    dot = block_sum(dot, scratch, 8);
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

// Block b opens on row r if the row's cosine with b's first direction reaches tau, else its xa and gate are 0.
extern "C" __global__ void tf_train_gate(float* xa, const float* xn, const float* tau, float* gates, int rows, int rank, int blocks,
                                         float unit) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (b >= blocks || r >= rows) return;
    float* row = xa + (size_t)r * rank + b * 16;
    const float n = unit * xn[r];
    const bool open = n > 0 && row[0] >= tau[b] * sqrtf(n);
    gates[(size_t)r * blocks + b] = open ? 1.0f : 0.0f;
    if (!open)
        for (int q = 0; q < 16; q++) row[q] = 0;
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
    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int e = tid; e < 64 * 32; e += 128) {
            const int r = e >> 5, c = e & 31, gm = m0 + r, gk = k0 + c;
            As[r][c] = (gm < M && gk < K) ? A[(size_t)gm * lda + gk] : zero;
        }
        for (int e = tid; e < 32 * 64; e += 128) {
            const int r = e >> 6, c = e & 63, gk = k0 + r, gn = n0 + c;
            Bs[c][r] = (gk < K && gn < N) ? B[(size_t)gk * ldb + gn] : zero;
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

// The routed pairs (slots below top_k) sorted by expert, stable: order (slot to pair), inv and offsets.
extern "C" __global__ void __launch_bounds__(256) tf_train_sort_pairs(const int* ids, int rows, int slots, int top_k, int experts,
                                                                     int* order, int* inv, int* offsets) {
    __shared__ int count[256];
    const int e = threadIdx.x;
    int n = 0;
    if (e < experts)
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < top_k; j++) n += ids[r * slots + j] == e;
    count[e] = n;
    __syncthreads();
    if (e == 0) {
        int at = 0;
        for (int x = 0; x < experts; x++) {
            const int c = count[x];
            count[x] = at;
            at += c;
        }
    }
    __syncthreads();
    if (e >= experts) return;
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

// dx[r] += sum_j dz_j gate[e_j]: weights scale s_j / sum s (s = sigmoid(logit)) back to the router's input; ys unweighted.
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
        a[j] = block_sum(s, scratch, 8);
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

// part[slice][s] = x[s] w[e] on a slice of w's rows (4-bit [E][n_in][k_out]), a thread 8 columns; grid slices * E.
extern "C" __global__ void tf_train_experts_part(const float* x, const unsigned* w, const bf16* scales, const bf16* biases,
                                                 const int* offsets, float* part, int rows, int n_in, int k_out, int experts,
                                                 int slices) {
    const int words = k_out / 8, word = threadIdx.x, e = blockIdx.x / slices, slice = blockIdx.x % slices;
    if (word >= words || e >= experts) return;
    const int first = offsets[e], last = e + 1 < experts ? offsets[e + 1] : rows;
    if (first >= last) return;
    const int span = n_in / slices, n0 = slice * span, n1 = n0 + span;
    const int groups = k_out / 64;
    const unsigned* we = w + (size_t)e * n_in * words + word;
    const bf16* se = scales + (size_t)e * n_in * groups + word / 8;
    const bf16* be = biases + (size_t)e * n_in * groups + word / 8;
    for (int r0 = first; r0 < last; r0 += 4) {
        const float4* xr[4];
        for (int r = 0; r < 4; r++) xr[r] = reinterpret_cast<const float4*>(x + (size_t)min(r0 + r, last - 1) * n_in);
        float acc[4][8];
        for (int r = 0; r < 4; r++)
            for (int j = 0; j < 8; j++) acc[r][j] = 0;
        for (int n = n0; n < n1; n += 4) {
            unsigned q[4];
            float sc[4], bi[4];
            float4 xv[4];
            for (int u = 0; u < 4; u++) {
                q[u] = we[(size_t)(n + u) * words];
                sc[u] = f32(se[(size_t)(n + u) * groups]);
                bi[u] = f32(be[(size_t)(n + u) * groups]);
            }
            for (int r = 0; r < 4; r++) xv[r] = xr[r][n / 4];
            for (int u = 0; u < 4; u++) {
                float v[8];
                for (int j = 0; j < 8; j++) v[j] = fmaf(sc[u], (float)((q[u] >> (4 * j)) & 0xfu), bi[u]);
                for (int r = 0; r < 4; r++) {
                    const float xs = u == 0 ? xv[r].x : u == 1 ? xv[r].y : u == 2 ? xv[r].z : xv[r].w;
                    for (int j = 0; j < 8; j++) acc[r][j] = fmaf(xs, v[j], acc[r][j]);
                }
            }
        }
        for (int r = 0; r < min(4, last - r0); r++)
            for (int j = 0; j < 8; j++) part[((size_t)slice * rows + r0 + r) * k_out + word * 8 + j] = acc[r][j];
    }
}

// y[s, k] = the slices' partial sums in slice order; grid (k_out / 256, rows).
extern "C" __global__ void tf_train_experts_sum(const float* part, float* y, int rows, int k_out, int slices) {
    const int k = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (k >= k_out || r >= rows) return;
    float sum = 0;
    for (int s = 0; s < slices; s++) sum += part[((size_t)s * rows + r) * k_out + k];
    y[(size_t)r * k_out + k] = sum;
}

#include "train_mixers.cuh"

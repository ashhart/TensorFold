// Learning in the weights on CUDA: a low-rank change at each layer's output, the loss, and products run backward.

#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

namespace {

constexpr int BLOCK = 16;  // ranks a lesson adds at each layer

__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float bits16(uint32_t h) { return __uint_as_float(h << 16); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// The sum over a block of `warps` warps, the same in every thread.
__device__ __forceinline__ float block_sum(float v, float* scratch, int warps) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    v = warp_sum(lane < warps ? scratch[lane] : 0.0f);
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_max(float v, float* scratch, int warps) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    v = warp_max(lane < warps ? scratch[lane] : -INFINITY);
    __syncthreads();
    return v;
}

// Block b of a change is open on a row whose cosine with its first direction reaches tau[b].
__device__ __forceinline__ bool opens(float xa0, float xn, float tau, float unit) {
    const float n = unit * xn;
    return n > 0.0f && xa0 >= tau * sqrtf(n);
}

}  // namespace

// xa[r, q] = x[r] . a[q] for the 16 ranks of block blockIdx.x, rows of tile blockIdx.y; block 0 also xn[r] = |x[r]|^2.
extern "C" __global__ void __launch_bounds__(256) tf_slide_xa(const bf16* __restrict__ x, int x_stride, const float* __restrict__ a,
                                                              const uint32_t* __restrict__ rank, float* __restrict__ xa,
                                                              float* __restrict__ xn, int rows, int in, int max_rank) {
    constexpr int CH = 256;  // inputs of a staged in shared memory at a time
    __shared__ float as[BLOCK][CH];
    const int b = blockIdx.x, r0 = blockIdx.y * 16, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (b * BLOCK >= static_cast<int>(*rank)) return;
    float acc[2][BLOCK], sq[2] = {0.0f, 0.0f};
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int q = 0; q < BLOCK; ++q) acc[h][q] = 0.0f;
    for (int i0 = 0; i0 < in; i0 += CH) {
        const int n = min(CH, in - i0);
        __syncthreads();
        for (int c = threadIdx.x; c < BLOCK * CH; c += 256) {
            const int q = c / CH, i = c % CH;
            as[q][i] = i < n ? a[static_cast<size_t>(b * BLOCK + q) * in + i0 + i] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = r0 + warp + 8 * h;
            if (r >= rows) continue;
            const bf16* xr = x + static_cast<size_t>(r) * x_stride + i0;
            for (int i = lane; i < n; i += 32) {
                const float v = bf(xr[i]);
                sq[h] += v * v;
#pragma unroll
                for (int q = 0; q < BLOCK; ++q) acc[h][q] += v * as[q][i];
            }
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = r0 + warp + 8 * h;
        if (r >= rows) continue;
#pragma unroll
        for (int q = 0; q < BLOCK; ++q) {
            const float s = warp_sum(acc[h][q]);
            if (lane == 0) xa[static_cast<size_t>(r) * max_rank + b * BLOCK + q] = s;
        }
        const float s = warp_sum(sq[h]);
        if (lane == 0 && b == 0) xn[r] = s;
    }
}

// y[r] += scale sum_q xa[r, q] b[q] over the blocks open on row r; y row r at (r * row_mul + row_add) * y_stride.
extern "C" __global__ void __launch_bounds__(256) tf_slide_out(const float* __restrict__ xa, const float* __restrict__ xn,
                                                               const float* __restrict__ tau, const float* __restrict__ b,
                                                               const uint32_t* __restrict__ rank, void* __restrict__ y, int y_f32,
                                                               int y_stride, int row_mul, int row_add, int rows, int out, float scale,
                                                               float unit, int max_rank) {
    extern __shared__ float xs[];  // [16][rank]: the tile's rows of xa, closed blocks zeroed
    const int r0 = blockIdx.y * 16, j = blockIdx.x * 256 + threadIdx.x;
    const int rk = static_cast<int>(*rank), tile = min(16, rows - r0);
    if (rk == 0) return;
    for (int c = threadIdx.x; c < tile * rk; c += 256) {
        const int r = c / rk, q = c % rk;
        const float* row = xa + static_cast<size_t>(r0 + r) * max_rank;
        xs[r * rk + q] = opens(row[q / BLOCK * BLOCK], xn[r0 + r], tau[q / BLOCK], unit) ? row[q] : 0.0f;
    }
    __syncthreads();
    if (j >= out) return;
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;
    for (int q = 0; q < rk; ++q) {
        const float bq = b[static_cast<size_t>(q) * out + j];
#pragma unroll
        for (int r = 0; r < 16; ++r)
            if (r < tile) acc[r] += xs[r * rk + q] * bq;
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r >= tile) continue;
        const size_t at = static_cast<size_t>((r0 + r) * row_mul + row_add) * y_stride + j;
        if (y_f32) {
            static_cast<float*>(y)[at] += scale * acc[r];
        } else {
            bf16* yb = static_cast<bf16*>(y);
            yb[at] = __float2bfloat16_rn(bf(yb[at]) + scale * acc[r]);
        }
    }
}

// A window's xa (rows <= 16): a warp a (rank q, row r), x[r] . a[q] over every input; q = 0's warps also xn[r].
extern "C" __global__ void __launch_bounds__(256) tf_slide_xa_rows(const bf16* __restrict__ x, int x_stride, const float* __restrict__ a,
                                                                   const uint32_t* __restrict__ rank, float* __restrict__ xa,
                                                                   float* __restrict__ xn, int in, int max_rank) {
    const int q = blockIdx.x * 8 + (threadIdx.x >> 5), r = blockIdx.y, lane = threadIdx.x & 31;
    if (q >= static_cast<int>(*rank)) return;
    const bf16* xr = x + static_cast<size_t>(r) * x_stride;
    const float* aq = a + static_cast<size_t>(q) * in;
    float s = 0.0f, n = 0.0f;
    for (int i = lane; i < in; i += 32) {
        const float v = bf(xr[i]);
        s += v * aq[i];
        n += v * v;
    }
    s = warp_sum(s);
    n = warp_sum(n);
    if (lane == 0) xa[static_cast<size_t>(r) * max_rank + q] = s;
    if (lane == 0 && q == 0) xn[r] = n;
}

// A window's y += scale xa b (rows <= 16): a lane a column, the ranks shared over 8 warps, summed in shared memory.
extern "C" __global__ void __launch_bounds__(256) tf_slide_out_rows(const float* __restrict__ xa, const float* __restrict__ xn,
                                                                    const float* __restrict__ tau, const float* __restrict__ b,
                                                                    const uint32_t* __restrict__ rank, void* __restrict__ y, int y_f32,
                                                                    int y_stride, int row_mul, int row_add, int rows, int out, float scale,
                                                                    float unit, int max_rank) {
    __shared__ float part[8][16][32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, j = blockIdx.x * 32 + lane;
    const int rk = static_cast<int>(*rank);
    if (rk == 0) return;
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;
    if (j < out) {
        for (int b0 = 0; b0 < rk; b0 += BLOCK) {
            bool open[16];
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const float* row = xa + static_cast<size_t>(r) * max_rank;
                open[r] = r < rows && opens(row[b0], xn[r], tau[b0 / BLOCK], unit);
            }
            for (int q = b0 + w; q < b0 + BLOCK; q += 8) {
                const float bq = b[static_cast<size_t>(q) * out + j];
#pragma unroll
                for (int r = 0; r < 16; ++r)
                    if (open[r]) acc[r] += xa[static_cast<size_t>(r) * max_rank + q] * bq;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) part[w][r][lane] = acc[r];
    __syncthreads();
    for (int c = threadIdx.x; c < rows * 32; c += 256) {
        const int r = c / 32, l = c % 32, jj = blockIdx.x * 32 + l;
        if (jj >= out) continue;
        float s = 0.0f;
#pragma unroll
        for (int ww = 0; ww < 8; ++ww) s += part[ww][r][l];
        const size_t at = static_cast<size_t>(r * row_mul + row_add) * y_stride + jj;
        if (y_f32) {
            static_cast<float*>(y)[at] += scale * s;
        } else {
            bf16* yb = static_cast<bf16*>(y);
            yb[at] = __float2bfloat16_rn(bf(yb[at]) + scale * s);
        }
    }
}

// Each row's closed blocks zeroed in xa, and gates[r, b] = 1 where block b is open on row r (else 0).
extern "C" __global__ void tf_train_gate(float* __restrict__ xa, const float* __restrict__ xn, const float* __restrict__ tau,
                                         float* __restrict__ gates, int rows, int rank, int max_rank, float unit) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y, blocks = rank / BLOCK;
    if (b >= blocks || r >= rows) return;
    float* row = xa + static_cast<size_t>(r) * max_rank + b * BLOCK;
    const bool open = opens(row[0], xn[r], tau[b], unit);
    gates[static_cast<size_t>(r) * (max_rank / BLOCK) + b] = open ? 1.0f : 0.0f;
    if (!open)
        for (int q = 0; q < BLOCK; ++q) row[q] = 0.0f;
}

// db[q, j] += scale sum_r xa[r, first + q] g[r, j]: the open block's gradient; grid (out / 256, 16).
extern "C" __global__ void tf_train_lora_db(const float* __restrict__ xa, const float* __restrict__ g, float* __restrict__ db,
                                            int rows, int out, int first, int max_rank, float scale) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, q = blockIdx.y;
    if (j >= out) return;
    float s = 0.0f;
    for (int r = 0; r < rows; ++r) s += xa[static_cast<size_t>(r) * max_rank + first + q] * g[static_cast<size_t>(r) * out + j];
    db[static_cast<size_t>(q) * out + j] += scale * s;
}

// dxa[r, q] = scale (g[r] . b[q]) where q's block is open on row r, else 0; a warp a (q, r).
extern "C" __global__ void tf_train_lora_dxa(const float* __restrict__ g, const float* __restrict__ b, const float* __restrict__ gates,
                                             float* __restrict__ dxa, int rows, int out, int rank, int max_rank, float scale) {
    const int q = (blockIdx.x * blockDim.x + threadIdx.x) / 32, r = blockIdx.y, lane = threadIdx.x & 31;
    if (q >= rank || r >= rows) return;
    float s = 0.0f;
    for (int j = lane; j < out; j += 32) s += g[static_cast<size_t>(r) * out + j] * b[static_cast<size_t>(q) * out + j];
    s = warp_sum(s);
    if (lane == 0) dxa[static_cast<size_t>(r) * max_rank + q] = scale * s * gates[static_cast<size_t>(r) * (max_rank / BLOCK) + q / BLOCK];
}

// dx[r, i] += sum_q dxa[r, q] a[q, i]: the change's part of its site's input gradient; dx row r at r * dx_stride.
extern "C" __global__ void tf_train_lora_dx(const float* __restrict__ dxa, const float* __restrict__ a, float* __restrict__ dx,
                                            int dx_stride, int rows, int in, int rank, int max_rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= in || r >= rows) return;
    float s = 0.0f;
    for (int q = 0; q < rank; ++q) s += dxa[static_cast<size_t>(r) * max_rank + q] * a[static_cast<size_t>(q) * in + i];
    dx[static_cast<size_t>(r) * dx_stride + i] += s;
}

// p[r, j] = f[j] . x[r] / |x[r]|: a site's input row, made unit, along candidate direction j; a warp a (j, r).
extern "C" __global__ void tf_train_project(const bf16* __restrict__ x, int x_stride, const float* __restrict__ f, float* __restrict__ p,
                                            int rows, int in, int k) {
    const int j = (blockIdx.x * blockDim.x + threadIdx.x) / 32, r = blockIdx.y, lane = threadIdx.x & 31;
    if (j >= k || r >= rows) return;
    float s = 0.0f, n = 0.0f;
    for (int i = lane; i < in; i += 32) {
        const float v = bf(x[static_cast<size_t>(r) * x_stride + i]);
        s += v * f[static_cast<size_t>(j) * in + i];
        n += v * v;
    }
    s = warp_sum(s);
    n = warp_sum(n);
    if (lane == 0) p[static_cast<size_t>(r) * k + j] = n > 0.0f ? s * rsqrtf(n) : 0.0f;
}

// A sign from a row's place and a sketch column, +1 or -1 with even odds (the Metal learner's hash).
__device__ __forceinline__ float coin(uint32_t row, uint32_t j, uint32_t seed) {
    uint32_t h = row * 0x9E3779B9u + j * 0x7FEB352Du + seed;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

// y[j, i] += w sum_r coin(first + r, j) x[r, i]: a site's input rows into a random sketch [k, in].
extern "C" __global__ void tf_train_sketch(const bf16* __restrict__ x, int x_stride, float* __restrict__ y, int rows, int in, int k,
                                           uint32_t first, uint32_t seed, float w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
    if (i >= in || j >= k) return;
    float s = 0.0f;
    for (int r = 0; r < rows; ++r) s += coin(first + r, j, seed) * bf(x[static_cast<size_t>(r) * x_stride + i]);
    y[static_cast<size_t>(j) * in + i] += w * s;
}

// A row's softmax over bf16 logits: stats = (loss, target's probability), then in place (p - onehot) weight[r].
extern "C" __global__ void __launch_bounds__(1024) tf_train_softmax(bf16* __restrict__ logits, const uint32_t* __restrict__ targets,
                                                                    const float* __restrict__ weights, float2* __restrict__ stats, int vocab) {
    __shared__ float scratch[32];
    const int r = blockIdx.x, t = threadIdx.x;
    bf16* l = logits + static_cast<size_t>(r) * vocab;
    float m = -INFINITY;
    for (int v = t; v < vocab; v += 1024) m = fmaxf(m, bf(l[v]));
    m = block_max(m, scratch, 32);
    float s = 0.0f;
    for (int v = t; v < vocab; v += 1024) s += expf(bf(l[v]) - m);
    s = block_sum(s, scratch, 32);
    const uint32_t target = targets[r];
    const float lt = bf(l[target]);
    if (t == 0) stats[r] = make_float2(logf(s) + m - lt, expf(lt - m) / s);
    __syncthreads();
    const float w = weights[r];
    for (int v = t; v < vocab; v += 1024) l[v] = __float2bfloat16_rn((expf(bf(l[v]) - m) / s - (v == static_cast<int>(target) ? 1.0f : 0.0f)) * w);
}

// g[r] += s w dx - s^3 h (w dx . h) / dim: a row's input RMS norm undone onto the residual's gradient; 256 a row.
extern "C" __global__ void __launch_bounds__(256) tf_train_rms_back(const bf16* __restrict__ h, const bf16* __restrict__ w,
                                                                   const float* __restrict__ dx, float* __restrict__ g, int dim, float eps) {
    __shared__ float scratch[32];
    const int r = blockIdx.x;
    const bf16* hr = h + static_cast<size_t>(r) * dim;
    const float* dr = dx + static_cast<size_t>(r) * dim;
    float sq = 0.0f, dot = 0.0f;
    for (int j = threadIdx.x; j < dim; j += 256) {
        const float v = bf(hr[j]);
        sq += v * v;
        dot += bf(w[j]) * dr[j] * v;
    }
    sq = block_sum(sq, scratch, 8);
    dot = block_sum(dot, scratch, 8);
    const float s = rsqrtf(sq / static_cast<float>(dim) + eps), k = s * s * s * dot / static_cast<float>(dim);
    for (int j = threadIdx.x; j < dim; j += 256) g[static_cast<size_t>(r) * dim + j] += s * bf(w[j]) * dr[j] - k * bf(hr[j]);
}

// A tiled 4-bit projection (tf_pack_dense's words, (kg, npad) scales and biases) as bf16 [k, n]: out[i, j] = W[j, i].
extern "C" __global__ void tf_train_dequant_t(const uint32_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                              const uint16_t* __restrict__ biases, bf16* __restrict__ out, int n, int k, int npad) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int64_t>(n) * k) return;
    const int row = static_cast<int>(idx % n), input = static_cast<int>(idx / n), kg = k / 64;
    const int t = row / 64, j = (row % 64) / 8, g = input / 64, v = (input % 64) / 32, i32 = input % 32;
    const int lane = (row % 8) * 4 + ((i32 >> 1) & 3), p = (i32 & 1) * 4 + (i32 >> 3);
    const int64_t word = ((static_cast<int64_t>(t) * kg + g) << 9) | (j << 6) | (lane << 1) | v;
    const float q = static_cast<float>((w[word] >> (4 * p)) & 0xFu);
    const int64_t sb = static_cast<int64_t>(g) * npad + row;
    out[idx] = __float2bfloat16_rn(bits16(scales[sb]) * q + bits16(biases[sb]));
}

extern "C" __global__ void tf_train_narrow(const float* __restrict__ x, bf16* __restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(x[i]);
}

extern "C" __global__ void tf_train_widen(const bf16* __restrict__ x, float* __restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bf(x[i]);
}

namespace {

__device__ __forceinline__ uint32_t smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem(p)));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

}  // namespace

// C [m, n] += A [m, k] B [n, k]^T: bf16 on the tensor cores, fp32 out; k a multiple of 32, a slice per blockIdx.z.
extern "C" __global__ void __launch_bounds__(128) tf_train_gemm(const bf16* __restrict__ A, int lda, const bf16* __restrict__ B, int ldb,
                                                               float* __restrict__ C, int ldc, int m, int n, int k, int split) {
    constexpr int BM = 64, BN = 64, BK = 32, PAD = 40;
    __shared__ __align__(16) bf16 as[BM][PAD];
    __shared__ __align__(16) bf16 bs[BN][PAD];
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const int per = (k / BK + split - 1) / split * BK, k_lo = blockIdx.z * per, k_hi = min(k, k_lo + per);
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.0f;
    for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = tid + 128 * h, row = c >> 2, col = (c & 3) * 8;
            uint4 va = make_uint4(0, 0, 0, 0), vb = make_uint4(0, 0, 0, 0);
            if (m0 + row < m) va = *reinterpret_cast<const uint4*>(A + static_cast<size_t>(m0 + row) * lda + k0 + col);
            if (n0 + row < n) vb = *reinterpret_cast<const uint4*>(B + static_cast<size_t>(n0 + row) * ldb + k0 + col);
            *reinterpret_cast<uint4*>(&as[row][col]) = va;
            *reinterpret_cast<uint4*>(&bs[row][col]) = vb;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t fa[2][4], fb[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) ldsm4(fa[i], &as[wm + 16 * i + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
            for (int p = 0; p < 2; ++p) ldsm4(fb[p], &bs[wn + 16 * p + (lane >> 4) * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma16816(acc[i][j], fa[i], fb[j >> 1][(j & 1) * 2], fb[j >> 1][(j & 1) * 2 + 1]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm + 16 * i + (lane >> 2) + 8 * h, col = n0 + wn + 8 * j + (lane & 3) * 2;
                if (row >= m) continue;
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    if (col + c < n) atomicAdd(C + static_cast<size_t>(row) * ldc + col + c, acc[i][j][2 * h + c]);
            }
}

// dy[p, j] = wt[p] g[p / slots, j]: each expert pair's output gradient, its routing weight (shared halves: 1) in.
extern "C" __global__ void tf_train_pairs_in(const float* __restrict__ wt, const float* __restrict__ g, float* __restrict__ dy,
                                             int slots, int dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, p = blockIdx.y;
    if (j >= dim) return;
    dy[static_cast<size_t>(p) * dim + j] = wt[p] * g[static_cast<size_t>(p / slots) * dim + j];
}

// out[p, i] = sum_j x[p, j] W_e[j, i] for each plan item's members, W_e expert e of a packed table [n rows, k inputs].
extern "C" __global__ void __launch_bounds__(256) tf_train_experts_back(const float* __restrict__ x, int x_stride,
                                                                       const uint32_t* __restrict__ w, const int* __restrict__ items,
                                                                       const int* __restrict__ counts, const int* __restrict__ members,
                                                                       float* __restrict__ out, int out_stride, int kg, int nb) {
    __shared__ float ws[32][65];  // the item's expert at one 32-row block and one 64-input group
    __shared__ float xs[64][33];  // the members' x at those 32 rows
    const int it = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    if (it >= counts[0]) return;
    const int e = items[3 * it], first = items[3 * it + 1], cnt = min(items[3 * it + 2], 64);
    const int i = tid & 63, lane4 = tid >> 6;
    float acc[16];
#pragma unroll
    for (int a = 0; a < 16; ++a) acc[a] = 0.0f;
    // the stored word this thread decodes: (t, j, r, q) placed as tf_experts_pack places it
    const int f = (tid / 128) * 128 + (tid % 4) * 32 + (tid % 128) / 4;
    const int q = f % 4, r8 = (f / 4) % 8, jj = (f / 32) % 2, t = f / 64;
    const int row = t * 8 + r8, col = (q * 2 + jj) * 8;
    for (int b = 0; b < nb; ++b) {
        const uint32_t* blk = w + ((static_cast<size_t>(e) * nb + b) * kg + g) * 288;
        const uint32_t word = blk[tid];
        const uint32_t sw = blk[256 + (r8 / 2) * 8 + t], bw = blk[256 + (r8 / 2) * 8 + 4 + t];
        const float sc = bits16(r8 % 2 ? sw >> 16 : sw & 0xFFFFu), bi = bits16(r8 % 2 ? bw >> 16 : bw & 0xFFFFu);
#pragma unroll
        for (int mm = 0; mm < 8; ++mm) {
            // tf_experts_pack's shuffle: the even nibbles low, the odd high
            const int pos = (mm % 2 == 0) ? mm / 2 : 4 + mm / 2;
            ws[row][col + mm] = sc * static_cast<float>((word >> (4 * pos)) & 0xFu) + bi;
        }
        for (int c = tid; c < cnt * 32; c += 256) {
            const int pp = c / 32, rr = c % 32;
            xs[pp][rr] = x[static_cast<size_t>(members[first + pp]) * x_stride + b * 32 + rr];
        }
        __syncthreads();
#pragma unroll
        for (int a = 0; a < 16; ++a) {
            const int pp = lane4 + 4 * a;
            if (pp >= cnt) break;
            float s = 0.0f;
#pragma unroll 8
            for (int rr = 0; rr < 32; ++rr) s += xs[pp][rr] * ws[rr][i];
            acc[a] += s;
        }
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < 16; ++a) {
        const int pp = lane4 + 4 * a;
        if (pp >= cnt) break;
        out[static_cast<size_t>(members[first + pp]) * out_stride + g * 64 + i] = acc[a];
    }
}

// du = da 2 relu(u) from the stored relu(u)^2 (bf16): da 2 sqrt(act).
extern "C" __global__ void tf_train_relu2_back(const bf16* __restrict__ act, const float* __restrict__ da, float* __restrict__ du,
                                               int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) du[i] = da[i] * 2.0f * sqrtf(fmaxf(bf(act[i]), 0.0f));
}

// dx[r, j] += the input gradients of row r's `slots` pairs, in slot order.
extern "C" __global__ void tf_train_pairs_out(const float* __restrict__ dxp, float* __restrict__ dx, int slots, int dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (j >= dim) return;
    float s = 0.0f;
    for (int k = 0; k < slots; ++k) s += dxp[(static_cast<size_t>(r) * slots + k) * dim + j];
    dx[static_cast<size_t>(r) * dim + j] += s;
}

// dx[r] += sum_j dz_j gate[e_j]: routing weights scale s_j / sum s (s = sigmoid(logit)) back into the router's input.
extern "C" __global__ void __launch_bounds__(256) tf_train_route_back(const float* __restrict__ part, int sk, const int* __restrict__ ids,
                                                                     const bf16* __restrict__ ys, const float* __restrict__ g,
                                                                     const bf16* __restrict__ gate, float* __restrict__ dx, int rows, int experts,
                                                                     int top_k, int slots, int dim, float scale) {
    __shared__ float scratch[32];
    __shared__ float dz[8];
    __shared__ int ex[8];
    const int r = blockIdx.x, t = threadIdx.x;
    float a[8];
    for (int j = 0; j < top_k; ++j) {
        float s = 0.0f;
        for (int i = t; i < dim; i += 256) s += g[static_cast<size_t>(r) * dim + i] * bf(ys[(static_cast<size_t>(r) * slots + j) * dim + i]);
        a[j] = block_sum(s, scratch, 8);
    }
    if (t == 0) {
        float p[8], total = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            const int e = ids[r * slots + j];
            float z = 0.0f;
            for (int s = 0; s < sk; ++s) z += part[(static_cast<size_t>(s) * rows + r) * experts + e];
            p[j] = 1.0f / (1.0f + expf(-z));
            total += p[j];
            ex[j] = e;
        }
        total += 1e-20f;
        float mix = 0.0f;
        for (int j = 0; j < top_k; ++j) mix += a[j] * p[j] / total;
        for (int j = 0; j < top_k; ++j) dz[j] = scale / total * (a[j] - mix) * p[j] * (1.0f - p[j]);
    }
    __syncthreads();
    for (int i = t; i < dim; i += 256) {
        float s = 0.0f;
        for (int j = 0; j < top_k; ++j) s += dz[j] * bf(gate[static_cast<size_t>(ex[j]) * dim + i]);
        dx[static_cast<size_t>(r) * dim + i] += s;
    }
}

// dx[r, i] += sum_j g[r, j] rest[j, i]: a projection's bf16 rest [n, width] carried back; dx row r at r * dx_stride.
extern "C" __global__ void tf_train_rest_back(const float* __restrict__ g, const bf16* __restrict__ rest, float* __restrict__ dx,
                                              int dx_stride, int rows, int n, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= width || r >= rows) return;
    float s = 0.0f;
    for (int j = 0; j < n; ++j) s += g[static_cast<size_t>(r) * n + j] * bf(rest[static_cast<size_t>(j) * width + i]);
    dx[static_cast<size_t>(r) * dx_stride + i] += s;
}

// Adam on a change's factors (lr, beta1, beta2, eps; c1, c2 the bias corrections); the gradient is then cleared.
extern "C" __global__ void tf_train_adam(float* __restrict__ p, float* __restrict__ g, float* __restrict__ m, float* __restrict__ v, int n,
                                         float lr, float b1, float b2, float eps, float c1, float c2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = g[i];
    const float mi = b1 * m[i] + (1.0f - b1) * gi, vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    p[i] -= lr * (mi * c1) / (sqrtf(vi * c2) + eps);
    g[i] = 0.0f;
}

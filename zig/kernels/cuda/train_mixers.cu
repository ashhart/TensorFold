// Gradients through Nemotron's Mamba-2 and attention mixers from position 0, in f32, on CUDA's layouts.

#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

// A Mamba layer's shape: rows, heads, head dim, groups, state, inner width, conv width, in-projection width.
struct Shape {
    int rows, heads, dh, groups, n, inner, conv, proj;
    float lo, hi;  // the dt clamp (time_step_limit)
};

// Attention's shape: rows, heads, kv heads, head dim, the qkv row width; scale 1/sqrt(head dim).
struct Heads {
    int rows, heads, kv_heads, dim, nqkv;
    float scale;
};

namespace {

constexpr int CK = 16;  // scan steps between kept states

__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

__device__ __forceinline__ float silu_grad(float v) {
    const float s = 1.0f / (1.0f + expf(-v));
    return s * (1.0f + v * (1.0f - s));
}

// The forward's dt: softplus(raw + bias) within the clamp, as scan_rows.cu computes it.
__device__ __forceinline__ float dt_of(float v, float lo, float hi) {
    return fminf(fmaxf(fmaxf(v, 0.0f) + logf(1.0f + expf(-fabsf(v))), lo), hi);
}

// Sum over the 16 lanes holding one state row (lanes 0..15 or 16..31 of a warp).
__device__ __forceinline__ float row_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

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

__device__ __forceinline__ float all_sum(float v, float* scratch, int warps) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    v = warp_sum(lane < warps ? scratch[lane] : 0.0f);
    __syncthreads();
    return v;
}

// Sum over all 64 state rows for each of n state columns: lanes l, l+16 first, then the 32 warps in turn.
__device__ __forceinline__ void column_sums(const float (&v)[8], float* scratch, int n, float* out) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, t = threadIdx.x;
    float s8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s8[i] = v[i] + __shfl_down_sync(0xffffffffu, v[i], 16);
    if (lane < 16)
#pragma unroll
        for (int i = 0; i < 8; ++i) scratch[w * n + lane * 8 + i] = s8[i];
    __syncthreads();
    if (t < n) {
        float s = 0.0f;
        for (int g = 0; g < 32; ++g) s += scratch[g * n + t];
        out[t] = s;
    }
    __syncthreads();
}

}  // namespace

// dt[t, h] = softplus(raw + bias) within the clamp, from the in-projection's last columns; grid (heads, rows).
extern "C" __global__ void tf_train_dt(const bf16* __restrict__ proj, const float* __restrict__ bias, float* __restrict__ dt, Shape s) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (h >= s.heads || t >= s.rows) return;
    dt[t * s.heads + h] = dt_of(bf(proj[static_cast<size_t>(t) * s.proj + s.inner + s.conv + h]) + bias[h], s.lo, s.hi);
}

// The scan for one head, a block of 1024 (64 rows x 16 lanes of 8 columns): y with the D skip, and kept states.
extern "C" __global__ void __launch_bounds__(1024) tf_train_ssm_fwd(const bf16* __restrict__ act, const float* __restrict__ dt,
                                                                    const float* __restrict__ a_neg, const float* __restrict__ d_skip,
                                                                    float* __restrict__ y, float* __restrict__ ckpt, Shape s) {
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31;
    const int p = t / 16, n0 = (t % 16) * 8, g = h / (s.heads / s.groups), size = s.dh * s.n;
    const int b0 = s.inner + g * s.n, c0 = s.inner + s.groups * s.n + g * s.n;
    float st[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float* mine = ckpt + static_cast<size_t>(h) * ((s.rows + CK - 1) / CK + 1) * size;
    for (int i = 0; i < 8; ++i) mine[p * s.n + n0 + i] = 0.0f;
    for (int r = 0; r < s.rows; ++r) {
        const bf16* row = act + static_cast<size_t>(r) * s.conv;
        const float d = dt[r * s.heads + h], a = expf(d * a_neg[h]), x = bf(row[h * s.dh + p]);
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            st[i] = a * st[i] + d * x * bf(row[b0 + n0 + i]);
            acc += st[i] * bf(row[c0 + n0 + i]);
        }
        acc = row_sum(acc);
        if (lane % 16 == 0) y[static_cast<size_t>(r) * s.inner + h * s.dh + p] = acc + d_skip[h] * x;
        if ((r + 1) % CK == 0)
#pragma unroll
            for (int i = 0; i < 8; ++i) mine[((r + 1) / CK) * size + p * s.n + n0 + i] = st[i];
    }
}

// The scan backward for one head: dx (with the D skip) into dact, per-head dB and dC partials, and ddt.
extern "C" __global__ void __launch_bounds__(1024) tf_train_ssm_back(const bf16* __restrict__ act, const float* __restrict__ dt,
                                                                     const float* __restrict__ a_neg, const float* __restrict__ d_skip,
                                                                     const float* __restrict__ dy, const float* __restrict__ ckpt,
                                                                     float* __restrict__ states, float* __restrict__ dact,
                                                                     float* __restrict__ db_part, float* __restrict__ dc_part,
                                                                     float* __restrict__ ddt, Shape s) {
    __shared__ float scratch[32 * 128];
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31;
    const int p = t / 16, n0 = (t % 16) * 8, g = h / (s.heads / s.groups), size = s.dh * s.n;
    const int b0 = s.inner + g * s.n, c0 = s.inner + s.groups * s.n + g * s.n;
    const float* mine = ckpt + static_cast<size_t>(h) * ((s.rows + CK - 1) / CK + 1) * size;
    float* chunk = states + static_cast<size_t>(h) * CK * size;
    float ds[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int c = (s.rows - 1) / CK; c >= 0; c--) {
        const int first = c * CK, last = min(first + CK, s.rows);
        float st[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) st[i] = mine[c * size + p * s.n + n0 + i];
        for (int r = first; r < last; r++) {
            const bf16* row = act + static_cast<size_t>(r) * s.conv;
            const float d = dt[r * s.heads + h], a = expf(d * a_neg[h]), x = bf(row[h * s.dh + p]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                st[i] = a * st[i] + d * x * bf(row[b0 + n0 + i]);
                chunk[(r - first) * size + p * s.n + n0 + i] = st[i];
            }
        }
        for (int r = last - 1; r >= first; r--) {
            const bf16* row = act + static_cast<size_t>(r) * s.conv;
            const float d = dt[r * s.heads + h], a = expf(d * a_neg[h]), x = bf(row[h * s.dh + p]);
            const float gy = dy[static_cast<size_t>(r) * s.inner + h * s.dh + p];
            float cur[8], prev[8], v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                cur[i] = chunk[(r - first) * size + p * s.n + n0 + i];
                prev[i] = r > first ? chunk[(r - first - 1) * size + p * s.n + n0 + i] : mine[c * size + p * s.n + n0 + i];
                ds[i] += gy * bf(row[c0 + n0 + i]);
                v[i] = gy * cur[i];
            }
            column_sums(v, scratch, s.n, dc_part + (static_cast<size_t>(r) * s.heads + h) * s.n);
            float dx = 0.0f, dd = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float bn = bf(row[b0 + n0 + i]);
                dx += ds[i] * bn;
                dd += ds[i] * (x * bn + a_neg[h] * a * prev[i]);
                v[i] = d * ds[i] * x;
            }
            column_sums(v, scratch, s.n, db_part + (static_cast<size_t>(r) * s.heads + h) * s.n);
            dx = row_sum(dx);
            if (lane % 16 == 0) dact[static_cast<size_t>(r) * s.conv + h * s.dh + p] = d * dx + d_skip[h] * gy;
            dd = all_sum(dd, scratch, 32);
            if (t == 0) ddt[r * s.heads + h] = dd;
#pragma unroll
            for (int i = 0; i < 8; ++i) ds[i] *= a;
        }
    }
}

// dact's B and C columns: each group's per-head partials summed in head order; grid (n, groups, rows).
extern "C" __global__ void tf_train_ssm_bc(const float* __restrict__ db_part, const float* __restrict__ dc_part, float* __restrict__ dact,
                                           Shape s) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, g = blockIdx.y, r = blockIdx.z, per = s.heads / s.groups;
    if (n >= s.n || g >= s.groups || r >= s.rows) return;
    float b = 0.0f, c = 0.0f;
    for (int h = g * per; h < (g + 1) * per; h++) {
        b += db_part[(static_cast<size_t>(r) * s.heads + h) * s.n + n];
        c += dc_part[(static_cast<size_t>(r) * s.heads + h) * s.n + n];
    }
    dact[static_cast<size_t>(r) * s.conv + s.inner + g * s.n + n] = b;
    dact[static_cast<size_t>(r) * s.conv + s.inner + s.groups * s.n + g * s.n + n] = c;
}

// The gate and grouped RMS norm undone: dn (at the norm's output) into the SSM output's dy and dproj's z columns.
extern "C" __global__ void __launch_bounds__(256) tf_train_gate_back(const float* __restrict__ y, const bf16* __restrict__ proj,
                                                                    const bf16* __restrict__ w, const float* __restrict__ dn,
                                                                    float* __restrict__ dy, float* __restrict__ dproj, Shape s, float eps) {
    __shared__ float scratch[32];
    const int t = threadIdx.x, g = blockIdx.x, r = blockIdx.y, width = s.inner / s.groups, j0 = g * width;
    float sq = 0.0f, dot = 0.0f;
    for (int j = j0 + t; j < j0 + width; j += 256) {
        const float gated = y[static_cast<size_t>(r) * s.inner + j] * silu(bf(proj[static_cast<size_t>(r) * s.proj + j]));
        sq += gated * gated;
        dot += bf(w[j]) * dn[static_cast<size_t>(r) * s.inner + j] * gated;
    }
    sq = all_sum(sq, scratch, 8);
    dot = all_sum(dot, scratch, 8);
    const float inv = rsqrtf(sq / static_cast<float>(width) + eps), k = inv * inv * inv * dot / static_cast<float>(width);
    for (int j = j0 + t; j < j0 + width; j += 256) {
        const float z = bf(proj[static_cast<size_t>(r) * s.proj + j]), yv = y[static_cast<size_t>(r) * s.inner + j];
        const float gated = yv * silu(z), dg = inv * bf(w[j]) * dn[static_cast<size_t>(r) * s.inner + j] - k * gated;
        dy[static_cast<size_t>(r) * s.inner + j] = dg * silu(z);
        dproj[static_cast<size_t>(r) * s.proj + j] = dg * yv * silu_grad(z);
    }
}

// The depthwise causal conv and its SiLU undone into dproj's xBC columns; the conv rebuilt from proj (fresh state).
extern "C" __global__ void tf_train_conv_back(const float* __restrict__ dact, const bf16* __restrict__ proj, const float* __restrict__ cw,
                                              const float* __restrict__ cb, float* __restrict__ dproj, Shape s) {
    constexpr int TAPS = 4;
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c >= s.conv || r >= s.rows) return;
    float acc = 0.0f;
    for (int j = 0; j < TAPS; j++) {
        const int at = r + TAPS - 1 - j;
        if (at >= s.rows) continue;
        float pre = cb[c];
        for (int jj = 0; jj < TAPS; jj++) {
            const int src = at - (TAPS - 1) + jj;
            if (src >= 0) pre = __fmaf_rn(cw[jj * s.conv + c], bf(proj[static_cast<size_t>(src) * s.proj + s.inner + c]), pre);
        }
        acc += cw[j * s.conv + c] * dact[static_cast<size_t>(at) * s.conv + c] * silu_grad(rbf(pre));
    }
    dproj[static_cast<size_t>(r) * s.proj + s.inner + c] = acc;
}

// d(raw dt) = ddt sigmoid(raw + bias) into dproj's last columns (0 where the clamp held dt); grid (heads, rows).
extern "C" __global__ void tf_train_dt_back(const bf16* __restrict__ proj, const float* __restrict__ bias, const float* __restrict__ ddt,
                                            float* __restrict__ dproj, Shape s) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (h >= s.heads || r >= s.rows) return;
    const size_t at = static_cast<size_t>(r) * s.proj + s.inner + s.conv + h;
    const float v = bf(proj[at]) + bias[h];
    const float sp = fmaxf(v, 0.0f) + logf(1.0f + expf(-fabsf(v)));
    const bool held = sp < s.lo || sp > s.hi;
    dproj[at] = held ? 0.0f : ddt[r * s.heads + h] / (1.0f + expf(-v));
}

// One query row of one head: its probabilities and their gradient kept, and dq into dqkv; 256 threads, one a key row.
extern "C" __global__ void __launch_bounds__(256) tf_train_attn_q(const bf16* __restrict__ qkv, const float* __restrict__ dout,
                                                                 float* __restrict__ probs, float* __restrict__ dscores,
                                                                 float* __restrict__ dqkv, Heads a) {
    __shared__ float qi[256], gi[256], ds[256], scratch[32];
    const int t = threadIdx.x, h = blockIdx.x, i = blockIdx.y, kv = h / (a.heads / a.kv_heads), lane = t & 31, w = t >> 5;
    const int qd = a.heads * a.dim, kvd = a.kv_heads * a.dim;
    const bf16* k = qkv + qd + kv * a.dim;
    const bf16* v = qkv + qd + kvd + kv * a.dim;
    for (int d = t; d < a.dim; d += 256) {
        qi[d] = bf(qkv[static_cast<size_t>(i) * a.nqkv + h * a.dim + d]);
        gi[d] = dout[static_cast<size_t>(i) * qd + h * a.dim + d];
    }
    __syncthreads();
    const bool live = t <= i && t < a.rows;
    float score = -INFINITY, dp = 0.0f;
    if (live) {
        float sc = 0.0f, dv = 0.0f;
        for (int d = 0; d < a.dim; d++) {
            sc += qi[d] * bf(k[static_cast<size_t>(t) * a.nqkv + d]);
            dv += gi[d] * bf(v[static_cast<size_t>(t) * a.nqkv + d]);
        }
        score = sc * a.scale;
        dp = dv;
    }
    float top = warp_max(score);
    if (lane == 0) scratch[w] = top;
    __syncthreads();
    top = warp_max(lane < 8 ? scratch[lane] : -INFINITY);
    __syncthreads();
    const float e = live ? expf(score - top) : 0.0f;
    const float total = all_sum(e, scratch, 8);
    const float pr = e / total;
    const float mix = all_sum(pr * dp, scratch, 8);
    const float g = pr * (dp - mix);
    ds[t] = g;
    if (t < a.rows) {
        probs[(static_cast<size_t>(h) * a.rows + i) * a.rows + t] = pr;
        dscores[(static_cast<size_t>(h) * a.rows + i) * a.rows + t] = g;
    }
    __syncthreads();
    for (int d = t; d < a.dim; d += 256) {
        float acc = 0.0f;
        for (int j = 0; j <= i; j++) acc += ds[j] * bf(k[static_cast<size_t>(j) * a.nqkv + d]);
        dqkv[static_cast<size_t>(i) * a.nqkv + h * a.dim + d] = acc * a.scale;
    }
}

// One key row of one kv head: dk and dv into dqkv, summed over the heads that read it and the query rows after it.
extern "C" __global__ void tf_train_attn_kv(const bf16* __restrict__ qkv, const float* __restrict__ dout, const float* __restrict__ probs,
                                            const float* __restrict__ dscores, float* __restrict__ dqkv, Heads a) {
    const int d = threadIdx.x, kv = blockIdx.x, j = blockIdx.y, per = a.heads / a.kv_heads;
    const int qd = a.heads * a.dim, kvd = a.kv_heads * a.dim;
    if (d >= a.dim) return;
    float gk = 0.0f, gv = 0.0f;
    for (int h = kv * per; h < (kv + 1) * per; h++)
        for (int i = j; i < a.rows; i++) {
            const size_t at = (static_cast<size_t>(h) * a.rows + i) * a.rows + j;
            gk += dscores[at] * bf(qkv[static_cast<size_t>(i) * a.nqkv + h * a.dim + d]);
            gv += probs[at] * dout[static_cast<size_t>(i) * qd + h * a.dim + d];
        }
    dqkv[static_cast<size_t>(j) * a.nqkv + qd + kv * a.dim + d] = gk * a.scale;
    dqkv[static_cast<size_t>(j) * a.nqkv + qd + kvd + kv * a.dim + d] = gv;
}

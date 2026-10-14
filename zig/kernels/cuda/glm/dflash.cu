// DFlash2 block drafter kernels (our own, after z-lab/dflash's reference, MIT): the target's layer taps, the drafter's
// per-head norms with rotary positions, its grouped two-tap dynamic convolutions, SwiGLU, and its attention over a
// stream's context ring (the target's features, a 2048-token window) and the 8-row block (bidirectional).
// Layout: hidden 4096, 32 query heads and 8 key/value heads of 128, a block of 8 rows a stream.

#include <cuda_bf16.h>
#include <stdint.h>

namespace {
constexpr int D = 4096;
constexpr int HD = 128;
constexpr int KVH = 8;
constexpr int QH = 32;
constexpr int KVW = KVH * HD;  // 1024
constexpr int WIN = 2048;      // the sliding window (and the ring's slots)
constexpr int BLOCK = 8;
constexpr int GROUP = 16;  // channels sharing one dynamic tap weight
constexpr int GROUPS = D / GROUP;
constexpr int CH = 16;  // key chunks of the chunked attention (df_attn_part: a block a kv head and chunk)

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

// The ring of one layer: K then V, [WIN][KVW] each, layers one after another.
__device__ __forceinline__ __nv_bfloat16* ringK(uint64_t base, int layer) {
    return reinterpret_cast<__nv_bfloat16*>(base) + (long long)layer * 2 * WIN * KVW;
}
__device__ __forceinline__ __nv_bfloat16* ringV(uint64_t base, int layer) { return ringK(base, layer) + (long long)WIN * KVW; }

// Qwen3's per-head RMSNorm then rotate-half RoPE of one head (64 threads: dims d and d + 64), in place or to out.
__device__ __forceinline__ void normRope(const __nv_bfloat16* x, const __nv_bfloat16* w, float eps, long long pos, float theta,
                                         __nv_bfloat16* out, float* red) {
    const int d = threadIdx.x;  // 0..63
    const float a = bf(x[d]), b = bf(x[d + 64]);
    float s = a * a + b * b;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((d & 31) == 0) red[d >> 5] = s;
    __syncthreads();
    const float rinv = rsqrtf((red[0] + red[1]) / HD + eps);
    __syncthreads();
    const float na = bf(tobf(bf(w[d]) * bf(tobf(a * rinv))));
    const float nb = bf(tobf(bf(w[d + 64]) * bf(tobf(b * rinv))));
    const float inv = powf(theta, -(float)(2 * d) / HD);
    const float ang = (float)pos * inv;
    const float c = cosf(ang), sn = sinf(ang);
    out[d] = tobf(na * c - nb * sn);
    out[d + 64] = tobf(nb * c + na * sn);
}
}  // namespace

// The completed residual of a target layer, its 4 streams' mean, into column block ``slot`` of out [R][5 D].
// grid (R, D / 1024), 256 threads.
// mode 1 (development, TF_GLM_DFLASH2_TAP=s0): stream 0 alone.
extern "C" __global__ void df_tap(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int slot, int mode) {
    const int r = blockIdx.x;
    const __nv_bfloat16* xr = x + (long long)r * 4 * D;
    __nv_bfloat16* o = out + (long long)r * 5 * D + (long long)slot * D;
    for (int d = blockIdx.y * 1024 + threadIdx.x; d < (blockIdx.y + 1) * 1024; d += blockDim.x) {
        if (mode == 1) {
            o[d] = xr[d];
            continue;
        }
        const float a = ((bf(xr[d]) + bf(xr[D + d])) + bf(xr[2 * D + d])) + bf(xr[3 * D + d]);
        o[d] = tobf(a / 4.0f);
    }
}

// The block's queries [R][QH*HD] (from the fused qkv rows, stride ldq) normed and rotated at pos[r] into q, and its
// keys likewise into kb, values copied into vb ([R][KVW] each). grid (R, QH + KVH), 64 threads.
extern "C" __global__ void df_block_qkv(const __nv_bfloat16* __restrict__ qkv, int ldq, const __nv_bfloat16* __restrict__ qn,
                                        const __nv_bfloat16* __restrict__ kn, const int* __restrict__ pos, float eps, float theta,
                                        __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kb, __nv_bfloat16* __restrict__ vb) {
    __shared__ float red[2];
    const int r = blockIdx.x, h = blockIdx.y;
    const __nv_bfloat16* row = qkv + (long long)r * ldq;
    if (h < QH) {
        normRope(row + h * HD, qn, eps, pos[r], theta, q + (long long)r * QH * HD + h * HD, red);
    } else {
        const int kh = h - QH;
        normRope(row + QH * HD + kh * HD, kn, eps, pos[r], theta, kb + (long long)r * KVW + kh * HD, red);
        const __nv_bfloat16* v = row + QH * HD + KVW + kh * HD;
        __nv_bfloat16* vo = vb + (long long)r * KVW + kh * HD;
        vo[threadIdx.x] = v[threadIdx.x];
        vo[threadIdx.x + 64] = v[threadIdx.x + 64];
    }
}

// Context rows' keys and values ([R][2 KVW]: k | v) normed, rotated at pos[r] and written into each row's stream
// ring (rings[r]) at slot pos % WIN of ``layer``. grid (R, KVH), 64 threads.
extern "C" __global__ void df_ctx_kv(const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ kn,
                                     const long long* __restrict__ pos, const uint64_t* __restrict__ rings, int layer, float eps,
                                     float theta) {
    __shared__ float red[2];
    const int r = blockIdx.x, kh = blockIdx.y;
    const long long p = pos[r];
    const int slot = (int)(p % WIN);
    const __nv_bfloat16* row = kv + (long long)r * 2 * KVW;
    normRope(row + kh * HD, kn, eps, p, theta, ringK(rings[r], layer) + (long long)slot * KVW + kh * HD, red);
    __nv_bfloat16* vo = ringV(rings[r], layer) + (long long)slot * KVW + kh * HD;
    vo[threadIdx.x] = row[KVW + kh * HD + threadIdx.x];
    vo[threadIdx.x + 64] = row[KVW + kh * HD + threadIdx.x + 64];
}

// The grouped two-tap dynamic causal convolution along each stream's block of 8 rows:
//   y[t, c] = sum over o in {0, 1} of (base[o][c] + dyn[t][which][o][c / 16]) * z[t - o][c]   (z[-1] = 0)
// dyn [R][2 (pre, post)][2 (tap)][GROUPS]; base [2 (tap)][D] (the module's pre or post kernel). add: x += y, else
// out = y. grid (R, D / 1024), 256 threads.
extern "C" __global__ void df_conv(const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ dyn, int which,
                                   const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out, int add) {
    const int r = blockIdx.x, t = r % BLOCK;
    const __nv_bfloat16* dr = dyn + (long long)r * 4 * GROUPS + which * 2 * GROUPS;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x) {
        const int g = c / GROUP;
        float y = (bf(base[c]) + bf(dr[g])) * bf(z[(long long)r * D + c]);
        if (t > 0) y += (bf(base[D + c]) + bf(dr[GROUPS + g])) * bf(z[(long long)(r - 1) * D + c]);
        __nv_bfloat16* o = out + (long long)r * D + c;
        *o = add ? tobf(bf(*o) + bf(tobf(y))) : tobf(y);
    }
}

// SwiGLU over [gate | up] rows (width F each): bf16(bf16(silu(g)) * u). grid (R, F / 1024), 256 threads.
extern "C" __global__ void df_swiglu(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ out, int F) {
    const int r = blockIdx.x;
    const __nv_bfloat16* g = gu + (long long)r * 2 * F;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024 && c < F; c += blockDim.x) {
        const float x = bf(g[c]);
        const float s = bf(tobf(x / (1.0f + expf(-x))));
        out[(long long)r * F + c] = tobf(s * bf(g[F + c]));
    }
}

// Attention of each stream's block (8 rows, 4 query heads a key/value head) over its ring's context keys in the
// window, positions max(from, p - 2047) .. start - 1 for a row at p, and the block's 8 keys (every one visible):
// grid (S, KVH), 256 threads (a warp a block row, its 4 heads; a lane 4 of the 128 dims).
// q [S*8][QH*HD], kb / vb [S*8][KVW], o [S*8][QH*HD]; starts / froms the streams' block and context starts.
extern "C" __global__ void __launch_bounds__(256) df_attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
                                                          const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings,
                                                          const long long* __restrict__ starts, const long long* __restrict__ froms,
                                                          int layer, float scale, __nv_bfloat16* __restrict__ o) {
    const int s = blockIdx.x, kh = blockIdx.y;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = s * BLOCK + j;
    const long long start = starts[s], p = start + j;
    long long lo = p - (WIN - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    float qv[4][4], m[4], l[4], acc[4][4];
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer);
    const __nv_bfloat16* V = ringV(rings[s], layer);
    const long long n_ctx = start > lo ? start - lo : 0;
    for (long long n = 0; n < n_ctx + BLOCK; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % WIN);
            kr = K + (long long)slot * KVW + kh * HD + lane * 4;
            vr = V + (long long)slot * KVW + kh * HD + lane * 4;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * KVW + kh * HD + lane * 4;
            vr = vb + rb * KVW + kh * HD + lane * 4;
        }
        float kf[4], vf[4];
        for (int i = 0; i < 4; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
    for (int h = 0; h < 4; ++h) {
        __nv_bfloat16* oh = o + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        const float inv = 1.0f / l[h];
        for (int i = 0; i < 4; ++i) oh[i] = tobf(acc[h][i] * inv);
    }
}

// df_attn in chunks: block (s, kh, c) takes the c-th 1/CH of each row's keys (its context, then the block's 8) and
// writes the row's 4 heads' running max, sum and unnormalized output (pm, pl [R][QH][CH]; pacc [R][QH][CH][HD], fp32);
// df_attn_merge combines the chunks. The serial walk over 2,048 keys made df_attn latency-bound; this one runs
// CH blocks a kv head. grid (S, KVH, CH), 256 threads (a warp a block row, as df_attn).
extern "C" __global__ void __launch_bounds__(256) df_attn_part(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
        const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings, const long long* __restrict__ starts,
        const long long* __restrict__ froms, int layer, float scale, float* __restrict__ pm, float* __restrict__ pl,
        float* __restrict__ pacc) {
    const int s = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = s * BLOCK + j;
    const long long start = starts[s], p = start + j;
    long long lo = p - (WIN - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    const long long n_ctx = start > lo ? start - lo : 0;
    const long long T = n_ctx + BLOCK, L = (T + CH - 1) / CH;
    const long long n0 = (long long)c * L, n1 = n0 + L < T ? n0 + L : T;
    float qv[4][4], m[4], l[4], acc[4][4];
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer);
    const __nv_bfloat16* V = ringV(rings[s], layer);
    for (long long n = n0; n < n1; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % WIN);
            kr = K + (long long)slot * KVW + kh * HD + lane * 4;
            vr = V + (long long)slot * KVW + kh * HD + lane * 4;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * KVW + kh * HD + lane * 4;
            vr = vb + rb * KVW + kh * HD + lane * 4;
        }
        float kf[4], vf[4];
        for (int i = 0; i < 4; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
    for (int h = 0; h < 4; ++h) {
        const long long at = ((long long)r * QH + kh * 4 + h) * CH + c;
        float* pa = pacc + at * HD + lane * 4;
        for (int i = 0; i < 4; ++i) pa[i] = acc[h][i];
        if (lane == 0) {
            pm[at] = m[h];   // -inf with l = 0 for a chunk past the row's keys: the merge weighs it 0
            pl[at] = l[h];
        }
    }
}

// The chunks' partials of row r, head h into o[r][h] (bf16). grid (R, QH), 128 threads (a thread a dim).
extern "C" __global__ void df_attn_merge(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                         __nv_bfloat16* __restrict__ o) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long long base = ((long long)r * QH + h) * CH;
    float m = -INFINITY;
    for (int c = 0; c < CH; ++c) m = fmaxf(m, pm[base + c]);
    float l = 0.f, a = 0.f;
    for (int c = 0; c < CH; ++c) {
        const float w = __expf(pm[base + c] - m);
        l += pl[base + c] * w;
        a += pacc[(base + c) * HD + d] * w;
    }
    o[(long long)r * QH * HD + h * HD + d] = tobf(a / l);
}

// x += y (bf16 rows of D). grid (R, D / 1024), 256 threads.
extern "C" __global__ void df_add(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y) {
    const long long r = blockIdx.x;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x)
        x[r * D + c] = tobf(bf(x[r * D + c]) + bf(y[r * D + c]));
}

// out = bf16(g rank 0 + g rank 1): both ranks' gathered fp32 partials [2][R][D] of a split linear (as the engine's
// glm_residual_add reads them). grid (R, D / 1024), 256 threads.
extern "C" __global__ void df_sum2(const float* __restrict__ g, __nv_bfloat16* __restrict__ out, int R) {
    const long long r = blockIdx.x;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x) {
        const long long i = r * D + c;
        out[i] = tobf(g[i] + g[(long long)R * D + i]);
    }
}

// A rank's 16 largest logits of each head row (bf16 [R][cols]), as global ids and fp32 values, in two stages:
// df_top16_part: each 4096-column chunk's 16 (grid (R, chunks), 256 threads); df_top16_merge: the chunks' into
// out [R][32] (16 ids, then 16 values' bits; grid R, 256 threads). Ties keep the lower id.
namespace {
constexpr int CHUNK_COLS = 4096;
constexpr int TOP = 16;

__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

// The block's best (value, index) of sv/si [n] (256 threads), the winner's slot cleared to -inf.
__device__ void takeBest(float* sv, int* si, int n, float* rv, int* ri, float* outv, int* outi) {
    float bv = -INFINITY;
    int bi = 0x7fffffff, bs = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        if (better(sv[j], si[j], bv, bi)) { bv = sv[j]; bi = si[j]; bs = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o), os = __shfl_xor_sync(0xffffffffu, bs, o);
        if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bs = os; }
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) { rv[w] = bv; ri[w] = bi; ri[8 + w] = bs; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int best = 0;
        for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
            if (better(rv[k], ri[k], rv[best], ri[best])) best = k;
        *outv = rv[best];
        *outi = ri[best];
        if (ri[8 + best] >= 0) sv[ri[8 + best]] = -INFINITY;
    }
    __syncthreads();
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256) df_top16_part(const __nv_bfloat16* __restrict__ logits, int cols, int offset,
                                                                float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float sv[CHUNK_COLS];
    __shared__ int si[CHUNK_COLS];
    __shared__ float rv[8];
    __shared__ int ri[16];
    const int r = blockIdx.x, c0 = blockIdx.y * CHUNK_COLS;
    const int n = min(CHUNK_COLS, cols - c0);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = bf(logits[(long long)r * cols + c0 + j]);
        si[j] = offset + c0 + j;
    }
    __syncthreads();
    const long long base = ((long long)r * gridDim.y + blockIdx.y) * TOP;
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, pv + base + t, pi + base + t);
}

extern "C" __global__ void __launch_bounds__(256) df_top16_merge(const float* __restrict__ pv, const int* __restrict__ pi, int chunks,
                                                                 int* __restrict__ out) {
    __shared__ float sv[64 * TOP];
    __shared__ int si[64 * TOP];
    __shared__ float rv[8];
    __shared__ int ri[16];
    __shared__ float ov[TOP];
    __shared__ int oi[TOP];
    const int r = blockIdx.x, n = chunks * TOP;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = pv[(long long)r * n + j];
        si[j] = pi[(long long)r * n + j];
    }
    __syncthreads();
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, ov + t, oi + t);
    if (threadIdx.x < TOP) {
        out[r * 2 * TOP + threadIdx.x] = oi[threadIdx.x];
        out[r * 2 * TOP + TOP + threadIdx.x] = __float_as_int(ov[threadIdx.x]);
    }
}

// GLM-5.3-Flash on CUDA: MLA attention on the 512-wide latent cache, ours. A program is one row's 32 heads over one
// chunk of 512 keys (online softmax in 16-key tiles, tensor-core scores and outputs), and the chunks are merged in
// order; nothing a program computes depends on another row, so a row has the same bits alone, in a decode window, in
// sixteen streams' windows or in a prompt chunk. Every row names its own cache (``lc``, the latent rows of its
// stream) and position (``pos``), so one launch serves any mix of streams.

#include <cuda_bf16.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

constexpr int H = 32, LW = 512, CH = 512, KT = 16, THREADS = 256;
constexpr int DENSE_LIMIT = 2051;   // index_topk + pool - 1: rows from here attend to their selected tokens alone
constexpr int ROWB = LW * 2;        // a latent row's bytes
constexpr int Q_BYTES = H * ROWB;
constexpr int KV_BYTES = KT * ROWB;
constexpr int SMEM = Q_BYTES + 2 * KV_BYTES + 2 * H * KT * 4 + H * KT * 2 + 3 * H * 4;

// 16-byte chunk c of a latent row r sits at c ^ (r & 7): ldmatrix's 8 rows land in 8 bank groups
__device__ __forceinline__ int lsw(int r, int c) { return (c & ~7) | ((c ^ r) & 7); }

template <bool SPARSE>
__device__ __forceinline__ void chunks(const __nv_bfloat16* __restrict__ qa, const unsigned long long* __restrict__ lc,
        const int* __restrict__ pos, const int* __restrict__ tok, const int* __restrict__ cnt, float* __restrict__ po,
        float* __restrict__ pm, float* __restrict__ pl, int R, float scale) {
    extern __shared__ __align__(128) unsigned char sm[];
    unsigned char* sq = sm;
    unsigned char* skv = sm + Q_BYTES;
    float* ssc = reinterpret_cast<float*>(skv + 2 * KV_BYTES);      // [2 k halves][H][KT] score partials
    __nv_bfloat16* sp = reinterpret_cast<__nv_bfloat16*>(ssc + 2 * H * KT);   // [H][KT] probabilities
    float* salpha = reinterpret_cast<float*>(sp + H * KT);
    float* sm_m = salpha + H;
    float* sm_l = sm_m + H;
    const int r = blockIdx.x, c = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const __nv_bfloat16* cache = reinterpret_cast<const __nv_bfloat16*>(lc[r]);
    // the keys this program reads: [lo, hi) of the chunk's positions (dense) or of the row's token list (sparse)
    int lo = c * CH, hi;
    if (SPARSE) {
        hi = min(lo + CH, cnt[r]);
    } else {
        const int q = pos[r];
        hi = q >= DENSE_LIMIT ? lo : min(lo + CH, q + 1);
    }
    if (tid < H) {
        sm_m[tid] = -INFINITY;
        sm_l[tid] = 0.0f;
    }
    float o[16][4];   // warp: heads 16 (warp & 1) .. +16, latent 128 (warp >> 1) .. +128
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.0f;
    const int ntiles = hi > lo ? (hi - lo + KT - 1) / KT : 0;
    auto key_of = [&](int idx) -> long long { return SPARSE ? (long long)tok[(long long)r * 2051 + idx] : (long long)idx; };
    auto load_kv = [&](int buf, int t) {
        unsigned char* dst = skv + buf * KV_BYTES;
        for (int p = tid; p < KT * (ROWB / 16); p += THREADS) {
            const int kr = p / (ROWB / 16), ch = p % (ROWB / 16);
            const int idx = lo + t * KT + kr;
            const bool ok = idx < hi;
            const long long key = ok ? key_of(idx) : 0;
            cp16z(dst + kr * ROWB + lsw(kr, ch) * 16, cache + key * LW + ch * 8, ok);
        }
    };
    if (ntiles > 0) {
        for (int p = tid; p < H * (ROWB / 16); p += THREADS) {
            const int h = p / (ROWB / 16), ch = p % (ROWB / 16);
            cp16(sq + h * ROWB + lsw(h, ch) * 16, qa + ((long long)r * H + h) * LW + ch * 8);
        }
        load_kv(0, 0);
    }
    commit();
    const int wm = warp & 1, wn = warp >> 1;
    for (int t = 0; t < ntiles; ++t) {
        if (t + 1 < ntiles) load_kv((t + 1) & 1, t + 1);
        commit();
        wait<1>();
        __syncthreads();
        const unsigned char* kv = skv + (t & 1) * KV_BYTES;
        {   // scores: warp (heads 16 (warp & 1), keys 8 ((warp >> 1) & 1), latent half warp >> 2)
            const int mi = warp & 1, ni = (warp >> 1) & 1, kh = warp >> 2;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
            for (int ks = 0; ks < 16; ++ks) {
                const int kc = kh * 32 + ks * 2;              // the k-step's first 16-byte chunk
                uint32_t a[4];
                const int ar = mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ldmatrix4(a, sq + ar * ROWB + lsw(ar, kc + (lane >> 4)) * 16);
                uint32_t b[2];
                const int br = ni * 8 + (lane & 7);
                ldmatrix2(b, kv + br * ROWB + lsw(br, kc + ((lane >> 3) & 1)) * 16);
                mma(d, a, b[0], b[1]);
            }
            const int g = lane >> 2, tq = lane & 3;
            float* dst = ssc + kh * H * KT;
            dst[(mi * 16 + g) * KT + ni * 8 + 2 * tq] = d[0];
            dst[(mi * 16 + g) * KT + ni * 8 + 2 * tq + 1] = d[1];
            dst[(mi * 16 + g + 8) * KT + ni * 8 + 2 * tq] = d[2];
            dst[(mi * 16 + g + 8) * KT + ni * 8 + 2 * tq + 1] = d[3];
        }
        __syncthreads();
        {   // the online softmax: 8 threads a head, 2 keys each
            const int h = tid >> 3, j = (tid & 7) * 2;
            const int base = lo + t * KT;
            float s[2];
            bool valid[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                valid[u] = base + j + u < hi;
                s[u] = valid[u] ? (ssc[h * KT + j + u] + ssc[H * KT + h * KT + j + u]) * scale : -INFINITY;
            }
            float tm = fmaxf(s[0], s[1]);
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, off));
            const float m = sm_m[h];
            const bool active = tm != -INFINITY;
            const float nm = active ? fmaxf(m, tm) : m;
            const float alpha = active ? (m == -INFINITY ? 0.0f : expf(m - nm)) : 1.0f;
            float p[2], ps = 0.0f;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                p[u] = (valid[u] && active) ? expf(s[u] - nm) : 0.0f;
                sp[h * KT + j + u] = __float2bfloat16_rn(p[u]);
            }
            ps = p[0] + p[1];
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off);
            __syncwarp();
            if ((tid & 7) == 0) {
                salpha[h] = alpha;
                sm_m[h] = nm;
                sm_l[h] = sm_l[h] * alpha + ps;
            }
        }
        __syncthreads();
        {   // o = o * alpha + p . kv
            const int g = lane >> 2;
            const float a0 = salpha[wm * 16 + g], a1 = salpha[wm * 16 + g + 8];
            uint32_t a[4];
            const int ar = wm * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            ldmatrix4(a, reinterpret_cast<const unsigned char*>(sp) + ar * (KT * 2) + (lane >> 4) * 16);
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                // two n8 tiles of latent columns: keys 0-7 / 8-15 of each, transposed into B fragments
                uint32_t b[4];
                const int kr = (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = wn * 16 + j + (lane >> 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"(smem(kv + kr * ROWB + lsw(kr, ch) * 16)));
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    o[j][e] *= (e < 2 ? a0 : a1);
                    o[j + 1][e] *= (e < 2 ? a0 : a1);
                }
                mma(o[j], a, b[0], b[1]);
                mma(o[j + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    wait<0>();
    // the partials: o [H][LW], m, l of (chunk, row)
    const long long base = ((long long)c * R + r) * H;
    {
        const int g = lane >> 2, tq = lane & 3;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int col = wn * 128 + j * 8 + 2 * tq;
            const int h0 = wm * 16 + g;
            *reinterpret_cast<float2*>(po + (base + h0) * LW + col) = make_float2(o[j][0], o[j][1]);
            *reinterpret_cast<float2*>(po + (base + h0 + 8) * LW + col) = make_float2(o[j][2], o[j][3]);
        }
    }
    __syncthreads();
    if (tid < H) {
        pm[base + tid] = sm_m[tid];
        pl[base + tid] = sm_l[tid];
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(THREADS) glm_attn_dense(const __nv_bfloat16* __restrict__ qa,
        const unsigned long long* __restrict__ lc, const int* __restrict__ pos, float* __restrict__ po, float* __restrict__ pm,
        float* __restrict__ pl, int R, float scale) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    chunks<false>(qa, lc, pos, nullptr, nullptr, po, pm, pl, R, scale);
}

extern "C" __global__ void __launch_bounds__(THREADS) glm_attn_sparse(const __nv_bfloat16* __restrict__ qa,
        const unsigned long long* __restrict__ lc, const int* __restrict__ tok, const int* __restrict__ cnt, float* __restrict__ po,
        float* __restrict__ pm, float* __restrict__ pl, int R, float scale) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    chunks<true>(qa, lc, nullptr, tok, cnt, po, pm, pl, R, scale);
}

// the shared size the chunk kernels take (the host sets it as their dynamic shared memory)
extern "C" __global__ void glm_attn_smem(int* out) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
 out[0] = SMEM; }

// the chunk partials of row r, head h in chunk order -> out [R, H, LW] bf16. dense rows past the limit and sparse
// rows with no tokens are left alone (the other path writes them). grid (R, H), 128 threads.
extern "C" __global__ void glm_attn_merge(const float* __restrict__ po, const float* __restrict__ pm, const float* __restrict__ pl,
        __nv_bfloat16* __restrict__ out, const int* __restrict__ pos, const int* __restrict__ cnt, int R, int nch, int sparse) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int r = blockIdx.x, h = blockIdx.y;
    if (sparse ? cnt[r] == 0 : pos[r] >= DENSE_LIMIT) return;
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c = 0; c < nch; ++c) {
        const long long base = ((long long)c * R + r) * H + h;
        const float cm = pm[base], cl = pl[base];
        const bool active = cl > 0.0f;
        const float nm = active ? fmaxf(m, cm) : m;
        const float a = active ? (m == -INFINITY ? 0.0f : expf(m - nm)) : 1.0f;
        const float b = active ? expf(cm - nm) : 0.0f;
        const float4 co = *reinterpret_cast<const float4*>(po + base * LW + threadIdx.x * 4);
        acc[0] = acc[0] * a + co.x * b;
        acc[1] = acc[1] * a + co.y * b;
        acc[2] = acc[2] * a + co.z * b;
        acc[3] = acc[3] * a + co.w * b;
        l = l * a + cl * b;
        m = nm;
    }
    __nv_bfloat16* dst = out + ((long long)r * H + h) * LW + threadIdx.x * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[e] = __float2bfloat16_rn(acc[e] / l);
}

// the window rows' latents into their caches: row r's 512 values at slot pos[r] of lc[r]
extern "C" __global__ void glm_lat_write(const __nv_bfloat16* __restrict__ lat, long long lat_stride,
                                         const unsigned long long* __restrict__ lc, const int* __restrict__ pos) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int r = blockIdx.x;
    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(lc[r]) + (long long)pos[r] * LW;
    const uint4* src = reinterpret_cast<const uint4*>(lat + r * lat_stride);
    for (int i = threadIdx.x; i < LW / 8; i += blockDim.x) reinterpret_cast<uint4*>(dst)[i] = src[i];
}

// absorb: qa[r, h, n] = sum_k q[r, h, k] wk[h, k, n] (fp32 in k order, then bf16); q [R, H, 256], wk [H, 256, 512].
// grid (H, LW / 64, ceil(R / 16)), 64 threads: the weight block once in shared memory, each row summed alone.
extern "C" __global__ void glm_absorb_q(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ wk,
                                        __nv_bfloat16* __restrict__ qa, int R) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    constexpr int D = 256;
    __shared__ __nv_bfloat16 w[D][64];
    __shared__ float qs[D];
    const int h = blockIdx.x, n0 = blockIdx.y * 64, n = threadIdx.x;
    for (int k = 0; k < D; ++k) w[k][n] = wk[((long long)h * D + k) * LW + n0 + n];
    const int r1 = min(R, (int)(blockIdx.z + 1) * 16);
    for (int r = blockIdx.z * 16; r < r1; ++r) {
        __syncthreads();
        for (int k = n; k < D; k += 64) qs[k] = __bfloat162float(q[((long long)r * H + h) * D + k]);
        __syncthreads();
        float acc = 0.0f;
        for (int k = 0; k < D; ++k) acc = fmaf(qs[k], __bfloat162float(w[k][n]), acc);
        qa[((long long)r * H + h) * LW + n0 + n] = __float2bfloat16_rn(acc);
    }
}

// expand: out[r, h, n] = sum_k ol[r, h, k] wv[h, n, k]; ol [R, H, 512], wv [H, 256, 512] -> out [R, H, 256].
// grid (H, 256 / 8, ceil(R / 16)), 256 threads: a warp an output, lanes over k in order, then a xor tree.
extern "C" __global__ void glm_expand_v(const __nv_bfloat16* __restrict__ ol, const __nv_bfloat16* __restrict__ wv,
                                        __nv_bfloat16* __restrict__ out, int R) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    constexpr int DV = 256;
    const int h = blockIdx.x, lane = threadIdx.x & 31, n = blockIdx.y * 8 + (threadIdx.x >> 5);
    float w[16];
    const __nv_bfloat16* wr = wv + ((long long)h * DV + n) * LW;
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = __bfloat162float(wr[i * 32 + lane]);
    const int r1 = min(R, (int)(blockIdx.z + 1) * 16);
    for (int r = blockIdx.z * 16; r < r1; ++r) {
        const __nv_bfloat16* x = ol + ((long long)r * H + h) * LW;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc = fmaf(w[i], __bfloat162float(x[i * 32 + lane]), acc);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) out[((long long)r * H + h) * DV + n] = __float2bfloat16_rn(acc);
    }
}

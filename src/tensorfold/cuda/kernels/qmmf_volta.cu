// sm_70 NVFP4 / FP8 / 16-bit matmuls on qmm_volta's mma.m8n8k4 skeleton: exact fp16 weights, one fp32 factor a column.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int THREADS = 128;    // 4 warps

constexpr int COLS = 128;       // weight columns a block: 4 warps x 4 quadpairs x 8

enum Fmt { FP4 = 0, FP8 = 1, F16 = 2 };

template <int FMT> struct Words { static constexpr int n = FMT == FP4 ? 8 : FMT == FP8 ? 16 : 32; };

__device__ __forceinline__ void mma884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// One e4m3 byte -> the exact fp16 value, in both halves.
__device__ __forceinline__ __half2 e4m3_h2(unsigned b) {
    const unsigned short hb = (unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));   // value * 2^-8, exact
    const __half h = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));               // * 256, exact
    return __halves2half2(h, h);
}

// Eight e2m1 codes of one word (code j at nibble j) -> fp16 code * 2^-14 * scale, inputs in order, two a half2.
__device__ __forceinline__ int4 fp4x8(unsigned q, __half2 s2) {
    constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
    unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);       // (code 0, code 4)
    unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);        // (1, 5)
    unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);        // (2, 6)
    unsigned v3 = (q & S) | ((q >> 3) & EM);               // (3, 7)
    const __half2 h0 = *reinterpret_cast<__half2*>(&v0), h1 = *reinterpret_cast<__half2*>(&v1);
    const __half2 h2 = *reinterpret_cast<__half2*>(&v2), h3 = *reinterpret_cast<__half2*>(&v3);
    __half2 o[4] = {__hmul2(__lows2half2(h0, h1), s2), __hmul2(__lows2half2(h2, h3), s2),
                    __hmul2(__highs2half2(h0, h1), s2), __hmul2(__highs2half2(h2, h3), s2)};
    return *reinterpret_cast<int4*>(o);
}

// Two words of e4m3 bytes (input order) -> eight fp16 values * 2^-8, exact.
__device__ __forceinline__ unsigned e4m3_pair(unsigned w) {            // bytes 0, 1 of w -> (lo half, hi half)
    return ((w << 8) & 0x8000u) | ((w << 7) & 0x3F80u) | ((w << 16) & 0x80000000u) | ((w << 15) & 0x3F800000u);
}
__device__ __forceinline__ int4 fp8x8(unsigned a, unsigned b) {
    return make_int4((int)e4m3_pair(a), (int)e4m3_pair(a >> 16), (int)e4m3_pair(b), (int)e4m3_pair(b >> 16));
}

// One 64-input group of one column in registers.
template <int FMT> struct Group {
    int4 w[Words<FMT>::n / 4];
    unsigned s;
};

template <int FMT>
__device__ __forceinline__ void load_group(Group<FMT>& gr, const int4* wp, const unsigned* sp, bool ok) {
#pragma unroll
    for (int i = 0; i < Words<FMT>::n / 4; ++i) gr.w[i] = ok ? __ldcs(wp + i) : make_int4(0, 0, 0, 0);
    if (FMT == FP4) gr.s = ok ? __ldg(sp) : 0u;
}

// Inputs [8 wi, 8 wi + 8) of the group as four fp16 pairs.
template <int FMT>
__device__ __forceinline__ int4 expand(const Group<FMT>& gr, int wi, const __half2 (&s2)[4]) {
    const unsigned* u = reinterpret_cast<const unsigned*>(gr.w);
    if (FMT == FP4) return fp4x8(u[wi], s2[wi >> 1]);
    if (FMT == FP8) return fp8x8(u[2 * wi], u[2 * wi + 1]);
    return gr.w[wi];
}

// Rows staged AHEAD groups ahead in shared memory; with ``cnt`` the last K slice adds every slice in slice order.
constexpr int AHEAD = 3, STAGES = AHEAD + 1;
template <int FMT, int MT, int NACC>
__global__ void __launch_bounds__(THREADS) qmmf884s_kernel(
        const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W,
        const unsigned* __restrict__ S, const float* __restrict__ ALPHA, void* __restrict__ OUT,
        float* __restrict__ PART, int* __restrict__ CNT, int M, int N, int K, int gper, int sk, bool f32) {
    constexpr int WPG = Words<FMT>::n;
    __shared__ __align__(16) __half sa[STAGES][MT][512];         // [stage][tile][8 k-steps x 8 rows x 8 inputs]
    __shared__ int last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int q = (lane >> 2) & 3, idx = (lane & 3) + 4 * (lane >= 16);
    const int s = blockIdx.y, m0 = blockIdx.z * 8 * MT;
    const int g0 = s * gper, g1 = g0 + gper;
    const int KG = (K + 63) / 64, K8 = K / 8;                       // a 32-input tail group is zero past K
    const int cbase = blockIdx.x * COLS + warp * 32 + q * 8;
    const int col = cbase + idx;
    const bool nok = col < N;
    const int c32 = nok ? col : 0;
    const long cell0 = (long)(c32 >> 5) * KG * 32 + (c32 & 31);
    const int4* wp = reinterpret_cast<const int4*>(W + cell0 * WPG);
    const unsigned* sp = S + (FMT == FP4 ? cell0 : 0);
    constexpr long WSTEP = 32 * WPG / 4;
    int tiles = 0;                                                  // the block's row tiles that hold rows
#pragma unroll
    for (int t = 0; t < MT; ++t) tiles += m0 + t * 8 < M;

    // staging: tile t of group g is 1 KB at X[(m0/8 + t)][g*8 .. g*8+8][8][8]; 64 threads a tile, one int4 each
    constexpr int PER = (MT * 64 + THREADS - 1) / THREADS;          // int4s a thread stages a group
    auto fetch = [&](int g, int4 (&r)[PER]) {
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = threadIdx.x + j * THREADS, t = e >> 6, c = e & 63;
            r[j] = (e < MT * 64 && t < tiles && g < g1 && g * 8 + (c >> 3) < K8)
                ? __ldg(reinterpret_cast<const int4*>(X + ((long)((m0 >> 3) + t) * K8 + g * 8) * 64) + c)
                : make_int4(0, 0, 0, 0);
        }
    };
    auto store = [&](int g, const int4 (&r)[PER]) {
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = threadIdx.x + j * THREADS, t = e >> 6, c = e & 63;
            if (e < MT * 64) reinterpret_cast<int4*>(sa[g % STAGES][t])[c] = r[j];
        }
    };
    {
        int4 r[PER];
#pragma unroll
        for (int d = 0; d < AHEAD; ++d) { fetch(g0 + d, r); store(d, r); }
    }
    __syncthreads();

    float acc[MT][NACC][8];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int a = 0; a < NACC; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[t][a][i] = 0.f;

    Group<FMT> cur, nxt;                                            // the weights stream one group ahead
    load_group<FMT>(cur, wp + g0 * WSTEP, sp + (long)g0 * 32, nok);
    for (int g = g0; g < g1; ++g) {
        const int i = g - g0;
        load_group<FMT>(nxt, wp + (g + 1) * WSTEP, sp + (long)(g + 1) * 32, nok && g + 1 < g1);
        int4 ahead[PER];
        fetch(g + AHEAD, ahead);
        __half2 s2[4];
        if (FMT == FP4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) s2[j] = e4m3_h2((cur.s >> (8 * j)) & 0xFFu);
        }
        const __half* stage = &sa[i % STAGES][0][0];
        const int steps = min(8, K8 - g * 8);                       // 4 in a 32-input tail group
#pragma unroll
        for (int wi = 0; wi < 8; ++wi) {
            if (wi >= steps) continue;
            const int4 o = expand<FMT>(cur, wi, s2);
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (t >= tiles) continue;
                const uint4 a = *reinterpret_cast<const uint4*>(stage + t * 512 + wi * 64 + idx * 8);
                mma884(acc[t][(2 * wi) % NACC], a.x, a.y, (unsigned)o.x, (unsigned)o.y);
                mma884(acc[t][(2 * wi + 1) % NACC], a.z, a.w, (unsigned)o.z, (unsigned)o.w);
            }
        }
        cur = nxt;
        store(i + AHEAD, ahead);
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int a = 1; a < NACC; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[t][0][i] += acc[t][a][i];

#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = m0 + t * 8 + (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16);
            const int n = cbase + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2);
            if (m >= M || n >= N) continue;
            const float v = acc[t][0][i] * RS[m] * ALPHA[n];
            if (sk > 1) PART[((long)s * M + m) * N + n] = v;
            else if (f32) static_cast<float*>(OUT)[(long)m * N + n] = v;
            else static_cast<__nv_bfloat16*>(OUT)[(long)m * N + n] = __float2bfloat16_rn(v);
        }
    if (sk == 1 || CNT == nullptr) return;
    __threadfence();
    __syncthreads();
    const int tile_id = blockIdx.z * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) last = atomicAdd(CNT + tile_id, 1) == sk - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int e = threadIdx.x; e < 8 * MT * COLS; e += THREADS) {
        const int m = m0 + e / COLS, n = blockIdx.x * COLS + e % COLS;
        if (m >= M || n >= N) continue;
        const long at = (long)m * N + n, total = (long)M * N;
        float v[16];                                                // every slice's load in flight, then added in order
#pragma unroll
        for (int z = 0; z < 16; ++z) v[z] = z < sk ? __ldcg(PART + z * total + at) : 0.f;
        float a = v[0];
#pragma unroll
        for (int z = 1; z < 16; ++z) if (z < sk) a += v[z];
        if (f32) static_cast<float*>(OUT)[at] = a;
        else static_cast<__nv_bfloat16*>(OUT)[at] = __float2bfloat16_rn(a);
    }
    if (threadIdx.x == 0) CNT[tile_id] = 0;                         // ready for the next launch
}

// Split-K slices summed in slice order (the same order whatever the row count).
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long total, int sk, bool f32) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    float a = part[i];
    for (int s = 1; s < sk; ++s) a += part[s * total + i];
    if (f32) static_cast<float*>(out)[i] = a;
    else static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(a);
}

// A Tiled weight -> dense fp16 (N, K): the values the decode kernel expands. One thread eight inputs.
template <int FMT>
__global__ void dequant_kernel(const int* __restrict__ W, const unsigned* __restrict__ S, __half* __restrict__ out,
                               int N, int K) {
    constexpr int WPG = Words<FMT>::n;
    const int k8 = K / 8, kg = (K + 63) / 64;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;      // n * k8 + piece
    if (idx >= (long)N * k8) return;
    const long n = idx / k8;
    const int piece = idx % k8, g = piece >> 3, wi = piece & 7;
    const long cell = ((n >> 5) * kg + g) * 32 + (n & 31);
    const unsigned* u = reinterpret_cast<const unsigned*>(W) + cell * WPG;
    int4 o;
    if (FMT == FP4) o = fp4x8(u[wi], e4m3_h2((S[cell] >> (8 * (wi >> 1))) & 0xFFu));
    else if (FMT == FP8) o = fp8x8(u[2 * wi], u[2 * wi + 1]);
    else o = reinterpret_cast<const int4*>(u)[wi];
    reinterpret_cast<int4*>(out)[idx] = o;
}

// fp32 sums (M, N) of scaled rows -> y * rs[row] * alpha[col], bf16 or fp32 in place (the decode epilogue's order).
__global__ void unscale2_kernel(const float* __restrict__ y, const float* __restrict__ rs,
                                const float* __restrict__ alpha, void* __restrict__ out, long N, long total, bool f32) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = y[i] * rs[i / N] * alpha[i % N];
    if (f32) static_cast<float*>(out)[i] = v;
    else static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(v);
}

// Rows of a page-locked host table read over PCIe through its mapped device address, copied exactly.
__global__ void gather_rows_kernel(const long* __restrict__ ids, const uint4* __restrict__ table, uint4* __restrict__ out,
                                   int row16) {
    const long id = ids[blockIdx.x];
    for (int c = threadIdx.x; c < row16; c += blockDim.x) out[(long)blockIdx.x * row16 + c] = table[id * row16 + c];
}

// Routed NVFP4 experts: items of up to eight (row, slot) pairs of one expert on the decode mma chain.

constexpr int EX_ROWS = 8;           // pairs an item

// Pairs grouped by expert in one block (members[] ids, items (expert, first, count)); picks past ``experts`` left out.
__global__ void plan_kernel(const int* __restrict__ picks, int pairs, int experts, int* __restrict__ members,
                            int* __restrict__ items, int* __restrict__ counts) {
    extern __shared__ int sh[];
    int* hist = sh;
    int* offs = sh + experts;
    int* ioffs = sh + 2 * experts;
    int* fill = sh + 3 * experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { hist[e] = 0; fill[e] = 0; }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x)
        if (picks[p] < experts) atomicAdd(hist + picks[p], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, io = 0;
        for (int e = 0; e < experts; ++e) {
            offs[e] = o; ioffs[e] = io;
            o += hist[e]; io += (hist[e] + EX_ROWS - 1) / EX_ROWS;
        }
        counts[0] = io;
        counts[1] = o;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = picks[p];
        if (e < experts) members[offs[e] + atomicAdd(fill + e, 1)] = p;
    }
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
        for (int j = 0; j * EX_ROWS < hist[e]; ++j) {
            int* it = items + 3 * (ioffs[e] + j);
            it[0] = e; it[1] = offs[e] + j * EX_ROWS; it[2] = min(EX_ROWS, hist[e] - j * EX_ROWS);
        }
}

// EPI 2: SwiGLU (MM = 2) -> bf16 act [pairs, N]; 0: fp32 [pairs, N]; 3: bf16 [pairs, N].
template <int EPI>
__global__ void __launch_bounds__(THREADS) experts884_kernel(
        const __half* __restrict__ X, const float* __restrict__ RS, const int* __restrict__ W,
        const unsigned* __restrict__ S, const float* __restrict__ ALPHA, const int* __restrict__ members,
        const int* __restrict__ items, const int* __restrict__ counts, int slots, void* __restrict__ OUT,
        float* __restrict__ PART, int* __restrict__ CNT, int pairs_cap, int N, int K, int gper, int sk, float limit) {
    constexpr int MM = EPI == 2 ? 2 : 1;
    const int item = blockIdx.x;
    if (item >= counts[0]) return;
    __shared__ __align__(16) __half sa[STAGES][512];
    __shared__ int src[EX_ROWS], pid[EX_ROWS];
    __shared__ float rsr[EX_ROWS];
    __shared__ int last;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < EX_ROWS) {
        const int p = threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
        pid[threadIdx.x] = p;
        src[threadIdx.x] = p < 0 ? -1 : (slots > 0 ? p / slots : p);
        rsr[threadIdx.x] = p < 0 ? 0.f : RS[slots > 0 ? p / slots : p];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int q = (lane >> 2) & 3, idx = (lane & 3) + 4 * (lane >= 16);
    const int s = blockIdx.z;
    const int g0 = s * gper, g1 = g0 + gper;
    const int KF = K / 64, KT = (K % 64) / 32, N32 = N / 32;           // whole groups, a 32-input tail group or none
    const int cbase = blockIdx.y * COLS + warp * 32 + q * 8;
    const int col = cbase + idx;
    const bool nok = col < N;
    const int c32 = nok ? col : 0, cc = c32 & 31;
    const long tile = (long)e * N32 + (c32 >> 5);                                   // (expert, column tile)
    const int* wt = W + tile * (32L * MM * (8 * KF + 4 * KT));                    // whole groups, then the tail
    const unsigned* st = S + tile * (32L * MM * KF + 16L * MM * KT);               // the tail's as 16-bit pairs
    auto load = [&](Group<FP4>& gr, int g, int m, bool ok) {
        if (g < KF) {
            load_group<FP4>(gr, reinterpret_cast<const int4*>(wt + ((long)g * 32 + cc) * MM * 8 + m * 8),
                            st + ((long)g * 32 + cc) * MM + m, ok);
        } else {                                                                    // the tail: four words a column
            gr.w[0] = ok ? __ldcs(reinterpret_cast<const int4*>(wt + 32L * MM * 8 * KF + (cc * MM + m) * 4))
                         : make_int4(0, 0, 0, 0);
            gr.w[1] = make_int4(0, 0, 0, 0);
            gr.s = ok ? (unsigned)__ldg(reinterpret_cast<const unsigned short*>(st + 32L * MM * KF) + cc * MM + m) : 0u;
        }
    };

    // staging: thread c < 64 copies k-step c / 8 of pair row c % 8 (eight inputs) into fragment order
    auto fetch = [&](int g, int4& r) {
        const int c = threadIdx.x, row = c & 7, ks = c >> 3;
        r = (c < 64 && g < g1 && src[row] >= 0 && g * 64 + ks * 8 < K)
            ? __ldg(reinterpret_cast<const int4*>(X + (long)src[row] * K + g * 64 + ks * 8))
            : make_int4(0, 0, 0, 0);
    };
    auto store = [&](int i, const int4& r) {
        const int c = threadIdx.x;
        if (c < 64) reinterpret_cast<int4*>(sa[i % STAGES])[(c >> 3) * 8 + (c & 7)] = r;
    };
    {
        int4 r;
#pragma unroll
        for (int d = 0; d < AHEAD; ++d) { fetch(g0 + d, r); store(d, r); }
    }
    __syncthreads();

    float acc[MM][2][8];
#pragma unroll
    for (int m = 0; m < MM; ++m)
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[m][a][i] = 0.f;
    Group<FP4> cur[MM], nxt[MM];
#pragma unroll
    for (int m = 0; m < MM; ++m) load(cur[m], g0, m, nok);
    for (int g = g0; g < g1; ++g) {
        const int i = g - g0;
#pragma unroll
        for (int m = 0; m < MM; ++m) load(nxt[m], g + 1, m, nok && g + 1 < g1);
        int4 ahead;
        fetch(g + AHEAD, ahead);
        const __half* stage = sa[i % STAGES];
        const int steps = g < KF ? 8 : 4;
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            __half2 s2[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) s2[j] = e4m3_h2((cur[m].s >> (8 * j)) & 0xFFu);
#pragma unroll
            for (int wi = 0; wi < 8; ++wi) {
                if (wi >= steps) continue;
                const int4 o = expand<FP4>(cur[m], wi, s2);
                const uint4 a = *reinterpret_cast<const uint4*>(stage + wi * 64 + idx * 8);
                mma884(acc[m][0], a.x, a.y, (unsigned)o.x, (unsigned)o.y);
                mma884(acc[m][1], a.z, a.w, (unsigned)o.z, (unsigned)o.w);
            }
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) cur[m] = nxt[m];
        store(i + AHEAD, ahead);
        __syncthreads();
    }
    // per element: (row, column) of the mma fragment, both matrices' sums in the same thread
    auto finish = [&](int row, int n, const float (&v)[MM]) {
        const int p = pid[row];
        if (EPI == 2) {
            float gte = v[0], up = v[1];
            if (limit > 0.f) { gte = fminf(gte, limit); up = fminf(fmaxf(up, -limit), limit); }
            const float a = gte / (1.f + expf(-gte)) * up;
            static_cast<__nv_bfloat16*>(OUT)[(long)p * N + n] = __float2bfloat16_rn(a);
        } else if (EPI == 0) {
            static_cast<float*>(OUT)[(long)p * N + n] = v[0];
        } else {
            static_cast<__nv_bfloat16*>(OUT)[(long)p * N + n] = __float2bfloat16_rn(v[0]);
        }
    };
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16);
        const int n = cbase + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2);
        if (row >= cnt || n >= N) continue;
        float v[MM];
#pragma unroll
        for (int m = 0; m < MM; ++m) v[m] = (acc[m][0][i] + acc[m][1][i]) * rsr[row] * ALPHA[e * MM + m];
        if (sk > 1) {
#pragma unroll
            for (int m = 0; m < MM; ++m) PART[(((long)s * pairs_cap + pid[row]) * MM + m) * N + n] = v[m];
        } else {
            finish(row, n, v);
        }
    }
    if (sk == 1) return;
    __threadfence();
    __syncthreads();
    const int tile_id = item * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) last = atomicAdd(CNT + tile_id, 1) == sk - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int t = threadIdx.x; t < EX_ROWS * COLS; t += THREADS) {
        const int row = t / COLS, n = blockIdx.y * COLS + t % COLS;
        if (row >= cnt || n >= N) continue;
        float v[MM];
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            float parts[4];
#pragma unroll
            for (int z = 0; z < 4; ++z)
                parts[z] = z < sk ? __ldcg(PART + (((long)z * pairs_cap + pid[row]) * MM + m) * N + n) : 0.f;
            float a = parts[0];
#pragma unroll
            for (int z = 1; z < 4; ++z) if (z < sk) a += parts[z];
            v[m] = a;
        }
        finish(row, n, v);
    }
    if (threadIdx.x == 0) CNT[tile_id] = 0;
}

}  // namespace

// ``x16`` from ``prep884`` (any row count); ``cnt`` int32 zeros a (column block, row block), or empty to reduce apart.
void qmmf884s(torch::Tensor x16, torch::Tensor rs, torch::Tensor w, torch::Tensor s, torch::Tensor alpha,
              torch::Tensor out, c10::optional<torch::Tensor> part, torch::Tensor cnt, int64_t fmt, int64_t sk,
              int64_t mt, bool f32, int64_t n_cols) {
    TORCH_CHECK(x16.dim() == 4 && x16.size(2) == 8 && x16.size(3) == 8, "qmmf884s: rows from prep884 in fragment order");
    const int M = rs.size(0), K = x16.size(1) * 8, N = n_cols;
    TORCH_CHECK(x16.is_contiguous() && w.is_contiguous() && s.is_contiguous() && out.is_contiguous() &&
                alpha.is_contiguous() && alpha.scalar_type() == at::kFloat && alpha.numel() >= N, "qmmf884s: tensors");
    const int wpg = fmt == FP4 ? 8 : fmt == FP8 ? 16 : 32;
    const int kg = (K + 63) / 64;
    TORCH_CHECK(K % 32 == 0 && kg % sk == 0 && w.numel() >= (long)((N + 31) / 32) * 32 * kg * wpg,
                "qmmf884s: bad K, K split or weight size");
    if (M == 0) return;
    const auto stream = at::cuda::getCurrentCUDAStream();
    float* p = nullptr;
    if (sk > 1) {
        TORCH_CHECK(part.has_value() && part->numel() >= sk * (int64_t)M * N, "qmmf884s: a K split needs its partials");
        p = part->data_ptr<float>();
    }
    const dim3 grid((N + COLS - 1) / COLS, sk, (M + 8 * mt - 1) / (8 * mt));
    int* c = nullptr;
    if (sk > 1 && cnt.numel() > 0) {
        TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= (long)grid.x * grid.z, "qmmf884s: too few counters");
        c = cnt.data_ptr<int>();
    }
    const int gper = kg / sk;
    const auto X = reinterpret_cast<const __half*>(x16.data_ptr<at::Half>());
    const auto Sp = reinterpret_cast<const unsigned*>(s.data_ptr<int>());
#define QS1(F, MTV, NA) qmmf884s_kernel<F, MTV, NA><<<grid, THREADS, 0, stream>>>( \
        X, rs.data_ptr<float>(), w.data_ptr<int>(), Sp, alpha.data_ptr<float>(), out.data_ptr(), p, c, M, N, K, gper, sk, f32)
#define QS(F, MTV) QS1(F, MTV, 2)
#define QS_MT(F) if (mt == 1) QS(F, 1); else if (mt == 2) QS(F, 2); else if (mt == 4) QS(F, 4); \
                 else TORCH_CHECK(false, "qmmf884s: mt is 1, 2 or 4");
    if (fmt == FP4) { QS_MT(FP4) } else if (fmt == FP8) { QS_MT(FP8) } else if (fmt == F16) { QS_MT(F16) }
    else TORCH_CHECK(false, "qmmf884s: format ", fmt);
#undef QS_MT
#undef QS
#undef QS1
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (sk > 1 && c == nullptr) {
        const long total = (long)M * N;
        reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(p, out.data_ptr(), total, sk, f32);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// ``table`` a page-locked host tensor (rows, D) of 16-bit values; ``ids`` int64 on the GPU -> (n, D) on the GPU.
torch::Tensor gather_host_rows(torch::Tensor table, torch::Tensor ids) {
    TORCH_CHECK(!table.is_cuda() && table.is_pinned() && table.is_contiguous() && table.dim() == 2 &&
                table.element_size() == 2 && (table.size(1) * 2) % 16 == 0, "gather_host_rows: a pinned (rows, D) table");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kLong && ids.dim() == 1, "gather_host_rows: int64 ids on the GPU");
    auto out = torch::empty({ids.size(0), table.size(1)}, ids.options().dtype(table.scalar_type()));
    if (ids.size(0) == 0) return out;
    void* dev = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&dev, table.data_ptr(), 0));
    const int row16 = (int)(table.size(1) * 2 / 16);
    gather_rows_kernel<<<(unsigned)ids.size(0), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<long>(), reinterpret_cast<const uint4*>(dev), reinterpret_cast<uint4*>(out.data_ptr()), row16);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void plan_experts(torch::Tensor picks, int64_t experts, torch::Tensor members, torch::Tensor items, torch::Tensor counts) {
    TORCH_CHECK(picks.is_cuda() && picks.scalar_type() == at::kInt && picks.is_contiguous(), "plan_experts: int32 picks");
    TORCH_CHECK(experts <= 2048, "plan_experts: at most 2048 experts");
    const int pairs = picks.numel();
    const size_t smem = 4 * experts * sizeof(int);
    plan_kernel<<<1, 1024, smem, at::cuda::getCurrentCUDAStream()>>>(picks.data_ptr<int>(), pairs, (int)experts,
        members.data_ptr<int>(), items.data_ptr<int>(), counts.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ``x16``, ``rs`` from ``qmm_volta.prep``; ``max_items`` the grid's items, ``counts[0]`` of them real.
void experts884(int64_t epi, torch::Tensor x16, torch::Tensor rs, torch::Tensor w, torch::Tensor s, torch::Tensor alpha,
                torch::Tensor members, torch::Tensor items, torch::Tensor counts, int64_t slots, torch::Tensor out,
                torch::Tensor part, torch::Tensor cnt, int64_t n, int64_t sk, int64_t max_items, double limit) {
    TORCH_CHECK(x16.dim() == 2 && x16.is_contiguous() && x16.scalar_type() == at::kHalf, "experts884: fp16 rows");
    const int K = x16.size(1);
    TORCH_CHECK(K % 32 == 0 && ((K + 63) / 64) % sk == 0 && sk <= 4 && n % 32 == 0, "experts884: shape or K split");
    if (max_items == 0) return;
    const dim3 grid((unsigned)max_items, (unsigned)((n + COLS - 1) / COLS), (unsigned)sk);
    TORCH_CHECK(sk == 1 || cnt.numel() >= (long)grid.x * grid.y, "experts884: too few counters");
    const int pairs_cap = sk > 1 ? (int)(part.numel() / (sk * (epi == 2 ? 2 : 1) * n)) : 0;
    const auto st = at::cuda::getCurrentCUDAStream();
#define EX(E) experts884_kernel<E><<<grid, THREADS, 0, st>>>(reinterpret_cast<const __half*>(x16.data_ptr<at::Half>()), \
        rs.data_ptr<float>(), w.data_ptr<int>(), reinterpret_cast<const unsigned*>(s.data_ptr<int>()), alpha.data_ptr<float>(), \
        members.data_ptr<int>(), items.data_ptr<int>(), counts.data_ptr<int>(), (int)slots, out.data_ptr(), \
        sk > 1 ? part.data_ptr<float>() : nullptr, sk > 1 ? cnt.data_ptr<int>() : nullptr, pairs_cap, (int)n, K, \
        (K + 63) / 64 / (int)sk, (int)sk, (float)limit)
    if (epi == 2) EX(2); else if (epi == 0) EX(0); else EX(3);
#undef EX
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor dequantf(torch::Tensor w, torch::Tensor s, int64_t fmt, int64_t n, int64_t k) {
    TORCH_CHECK(w.is_contiguous() && s.is_contiguous(), "dequantf: a Tiled weight");
    auto out = torch::empty({n, k}, w.options().dtype(at::kHalf));
    const long pieces = n * (k / 8);
    if (pieces == 0) return out;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto Sp = reinterpret_cast<const unsigned*>(s.data_ptr<int>());
    const auto O = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    const int blocks = (int)((pieces + 255) / 256);
    if (fmt == FP4) dequant_kernel<FP4><<<blocks, 256, 0, stream>>>(w.data_ptr<int>(), Sp, O, (int)n, (int)k);
    else if (fmt == FP8) dequant_kernel<FP8><<<blocks, 256, 0, stream>>>(w.data_ptr<int>(), Sp, O, (int)n, (int)k);
    else dequant_kernel<F16><<<blocks, 256, 0, stream>>>(w.data_ptr<int>(), Sp, O, (int)n, (int)k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor unscale2(torch::Tensor y, torch::Tensor rs, torch::Tensor alpha, bool f32) {
    TORCH_CHECK(y.is_contiguous() && y.scalar_type() == at::kFloat && y.dim() == 2, "unscale2: (M, N) fp32");
    auto out = f32 ? y : torch::empty_like(y, y.options().dtype(at::kBFloat16));
    const long total = y.numel();
    if (total == 0) return out;
    unscale2_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<float>(), rs.data_ptr<float>(), alpha.data_ptr<float>(), out.data_ptr(), y.size(1), total, f32);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qmmf884s", &qmmf884s, "the staged decode matmul: rows through shared memory, split K added in-kernel");
    m.def("gather_host_rows", &gather_host_rows, "rows of a page-locked host table, read over PCIe from the GPU");
    m.def("plan_experts", &plan_experts, "(row, slot) pairs grouped by expert into items of up to eight");
    m.def("experts884", &experts884, "routed NVFP4 experts on mma.m8n8k4: gate|up with SwiGLU, or down");
    m.def("dequantf", &dequantf, "a Tiled NVFP4 / FP8 / 16-bit weight -> dense fp16 (N, K), the decode kernel's values");
    m.def("unscale2", &unscale2, "fp32 sums -> sums * row scale * column factor, bf16 or fp32 in place");
}

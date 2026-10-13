// GLM-5.3-Flash's DSA sparse indexer (sparse.py): bit-exact via inline PTX ops, m16n8k16 dots, PTX sum orders.

#include <cstdint>
#include <cuda_bf16.h>

namespace tf_glm_sparse {

using bf16 = __nv_bfloat16;

struct IndexArgs {
    const bf16* kr;       // bf16 [R, D] row-major (k_stride elements a row)
    long long k_stride;
    const float* gr;      // fp32 [R, D]
    const unsigned short* lnw; // fp16 [D] (the checkpoint's k_norm rows)
    const unsigned short* lnb; // fp16 [D]
    bf16* ik;             // bf16 [cap, D]
    bf16* ig;             // bf16 [cap, D]
    int pos;
    float eps;
    const int* posp;      // the position on the device (a captured decode step), else null and pos
};

struct PoolArgs {
    const bf16* ik;       // bf16 [cap, D]
    const bf16* ig;       // bf16 [cap, D]
    const bf16* ape;      // bf16 [4, D]
    bf16* pk;             // bf16 [cap / 4 + 2, D]
    int pos;
    int r;
    const int* posp;      // as IndexArgs
};

struct ScoresArgs {
    const bf16* qi;       // bf16 [R, H * D] row-major (the block's rows)
    long long qi_stride;  // qi's row stride in elements (160: D + H)
    const bf16* w;        // bf16 [R, H] row-major (w_stride elements a row)
    long long w_stride;
    const bf16* pk;       // bf16 [np, D]
    float* out;           // fp32 [R, np]
    int pos;
    int np;
    float scale;
    float wscale;
    const int* posp;      // as IndexArgs
};

struct SelectArgs {
    const float* s;       // fp32 [R, np]
    long long s_stride;   // np
    long long out_stride; // K
    long long* out;       // int64 pools [R, K]
    int np;
    int pos;
    int k;
    int visible;          // 1: limit = min(np, max(k, (pos + r + 1) / 4)); 0: limit = np
    const int* posp;      // as IndexArgs
};

struct TokensArgs {
    const long long* pools; // int64 [R, K]
    int* tokens;            // int32 [R, 512 * 4 + 3]
    int* counts;            // int32 [R]
    int pos;
    const int* posp;        // as IndexArgs
};

constexpr int D = 128;                 // index key dims
constexpr int H = 32;                  // index heads (the serve model's index_heads)
constexpr int BP = 64;                 // pools a scores block
constexpr int L = 1;                   // pool blocks a scores program takes (speed only: a block's sums are its own)
constexpr int LOG2E_BITS = 0x3FB8AA3B; // float log2(e)'s bits, as Triton's exp lowering multiplies by

__device__ __forceinline__ unsigned pack2(bf16 lo, bf16 hi) {
    return static_cast<unsigned>(__bfloat16_as_ushort(lo)) | (static_cast<unsigned>(__bfloat16_as_ushort(hi)) << 16);
}

__device__ __forceinline__ float rcp_approx(float x) {
    float r;
    asm("rcp.approx.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

// Triton's exp: ex2.approx(x * log2e), but past -126 the input is halved and the result squared, keeping subnormals.
__device__ __forceinline__ float exp_approx(float x) {
    float t = __fmul_rn(x, __int_as_float(LOG2E_BITS));
    const bool sub = t < -126.0f;
    if (sub) t = __fmul_rn(t, 0.5f);
    float r;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(r) : "f"(t));
    if (sub) r = __fmul_rn(r, r);
    return r;
}

__device__ __forceinline__ float sqrt_approx_ftz(float x) {
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_16n8k16(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A float32 as a uint32 whose unsigned order is the score's order (-0 counted as +0), as sparse.py's _order_key.
__device__ __forceinline__ unsigned order_key(float s) {
    const int bits = __float_as_int(__fadd_rn(s, 0.0f));
    return static_cast<unsigned>(bits ^ ((bits >> 31) | static_cast<int>(0x80000000)));
}

// The xor butterfly 16, 8, 4, 2, 1 over a warp's lanes (Triton's shfl.bfly + add stages).
__device__ __forceinline__ float warp_bfly(float v) {
    const unsigned all = 0xffffffffu;
    v = __fadd_rn(v, __shfl_xor_sync(all, v, 16));
    v = __fadd_rn(v, __shfl_xor_sync(all, v, 8));
    v = __fadd_rn(v, __shfl_xor_sync(all, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(all, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(all, v, 1));
    return v;
}

// One warp: the lane's four elements ((x0 + x1) + (x2 + x3)), then the full butterfly.
__device__ __forceinline__ float warp_sum128(float x0, float x1, float x2, float x3) {
    return warp_bfly(__fadd_rn(__fadd_rn(x0, x1), __fadd_rn(x2, x3))); // Triton's f32x2 pairing
}

// Row r: LayerNorm(kr) -> bf16 into ik[pos + r]; the gate row -> bf16 into ig[pos + r]. One warp per row.
extern "C" __global__ void tf_glm_index_write(IndexArgs a) {
    const int r = blockIdx.x;
    const int lane = threadIdx.x;
    const bf16* x_in = a.kr + static_cast<long long>(r) * a.k_stride + lane * 4;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; i++) x[i] = __bfloat162float(x_in[i]);
    const float mean = __fmul_rn(warp_sum128(x[0], x[1], x[2], x[3]), 0.0078125f); // Triton's FMUL by 1/128, not div
    float xc[4];
#pragma unroll
    for (int i = 0; i < 4; i++) xc[i] = __fsub_rn(x[i], mean);
    // Triton's variance sums (x0^2 + x2^2) + (x1^2 + x3^2), each first square fused into the FFMA
    const float s02 = __fmaf_rn(xc[0], xc[0], __fmul_rn(xc[2], xc[2]));
    const float s13 = __fmaf_rn(xc[1], xc[1], __fmul_rn(xc[3], xc[3]));
    const float var_eps = __fmaf_rn(warp_bfly(__fadd_rn(s02, s13)), 0.0078125f, a.eps);
    const float d = sqrt_approx_ftz(var_eps);
    const unsigned short* w = a.lnw + lane * 4;
    const unsigned short* b = a.lnb + lane * 4;
    const float wf[4] = {__half2float(*reinterpret_cast<const __half*>(w)), __half2float(*reinterpret_cast<const __half*>(w + 1)),
                         __half2float(*reinterpret_cast<const __half*>(w + 2)), __half2float(*reinterpret_cast<const __half*>(w + 3))};
    const float bf4[4] = {__half2float(*reinterpret_cast<const __half*>(b)), __half2float(*reinterpret_cast<const __half*>(b + 1)),
                          __half2float(*reinterpret_cast<const __half*>(b + 2)), __half2float(*reinterpret_cast<const __half*>(b + 3))};
    unsigned y[2], gv[2];
    float rcp_d;
    asm("rcp.approx.f32 %0, %1;" : "=f"(rcp_d) : "f"(d)); // Triton's div.full lowers to RCP + FMUL (no refinement)
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float t = __fmul_rn(xc[i], rcp_d);
        reinterpret_cast<bf16*>(y)[i] = __float2bfloat16_rn(__fmaf_rn(t, wf[i], bf4[i]));
        reinterpret_cast<bf16*>(gv)[i] = __float2bfloat16_rn(a.gr[static_cast<long long>(r) * D + lane * 4 + i]);
    }
    const long long p = static_cast<long long>(a.posp ? *a.posp : a.pos) + r;
    *reinterpret_cast<unsigned*>(a.ik + p * D + lane * 4) = y[0];
    *reinterpret_cast<unsigned*>(a.ik + p * D + lane * 4 + 2) = y[1];
    *reinterpret_cast<unsigned*>(a.ig + p * D + lane * 4) = gv[0];
    *reinterpret_cast<unsigned*>(a.ig + p * D + lane * 4 + 2) = gv[1];
}

// Program i: pool p = pos / 4 + i if it is complete within the window. One warp, four dims a lane.
extern "C" __global__ void tf_glm_pool_keys(PoolArgs a) {
    const int i = blockIdx.x;
    const long long pos = a.posp ? *a.posp : a.pos;
    const long long p = pos / 4 + i;
    if (4 * p + 3 > pos + a.r - 1) return;
    const int lane = threadIdx.x;
    const int d = lane * 4;
    float l[4][4], e[4][4]; // pool member j, lane dim t
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const bf16* g = a.ig + (4 * p + j) * D + d;
        const bf16* ap = a.ape + j * D + d;
#pragma unroll
        for (int t = 0; t < 4; t++)
            l[j][t] = __fadd_rn(__bfloat162float(g[t]), __bfloat162float(ap[t]));
    }
#pragma unroll
    for (int t = 0; t < 4; t++) {
        const float m = fmaxf(fmaxf(l[0][t], l[1][t]), fmaxf(l[2][t], l[3][t])); // exact: any order
#pragma unroll
        for (int j = 0; j < 4; j++) e[j][t] = exp_approx(__fsub_rn(l[j][t], m));
    }
    bf16 prod[4][4]; // pool member j, lane dim t
#pragma unroll
    for (int t = 0; t < 4; t++) {
        // Triton's s = ((e0 + e1) + e2) + e3 and pk = ((t0 + t1) + t2) + t3: strictly sequential, not a balanced tree
        float s = e[0][t];
        s = __fadd_rn(s, e[1][t]);
        s = __fadd_rn(s, e[2][t]);
        s = __fadd_rn(s, e[3][t]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            // Triton's div.full lowers to RCP + FMUL (the divisor is a sum of exps in [1, 4]: no denormal fixups)
            const float q = __bfloat162float(__float2bfloat16_rn(__fmul_rn(e[j][t], rcp_approx(s))));
            const bf16* k = a.ik + (4 * p + j) * D + d;
            prod[j][t] = __float2bfloat16_rn(__fmul_rn(q, __bfloat162float(k[t])));
        }
    }
#pragma unroll
    for (int t = 0; t < 4; t++) {
        float sum = __bfloat162float(prod[0][t]);
        sum = __fadd_rn(sum, __bfloat162float(prod[1][t]));
        sum = __fadd_rn(sum, __bfloat162float(prod[2][t]));
        sum = __fadd_rn(sum, __bfloat162float(prod[3][t]));
        reinterpret_cast<bf16*>(a.pk + p * D + d)[t] = __float2bfloat16_rn(sum);
    }
}

// Scores: s_p = sum_h w_h relu(scale*qi_h.pool_p), -inf past visible; warp w owns cols 16w..16w+15, K ascending chunks.
extern "C" __global__ void __launch_bounds__(128) tf_glm_scores(ScoresArgs a) {
    __shared__ bf16 q_sh[H][D];
    __shared__ float sc_sh[BP];
    __shared__ float w_sh[H];
    const int r = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < H * D; i += 128) { // q row r, all 32 heads (H == 32 == HP, no padding rows)
        const int hh = i / D, dd = i % D;
        q_sh[hh][dd] = a.qi[static_cast<long long>(r) * a.qi_stride + hh * D + dd];
    }
    if (tid < H) w_sh[tid] = __bfloat162float(a.w[static_cast<long long>(r) * a.w_stride + tid]) * a.wscale;
    __syncthreads();

    const int g4 = lane >> 2;      // the fragment's first row
    const int t2 = (lane & 3) * 2; // the fragment's first col
    const int npool = ((a.posp ? *a.posp : a.pos) + r + 1) / 4;
    const float neg_inf = -__int_as_float(0x7F800000);
    for (int j = 0; j < L; j++) {
        const int pb = g * L + j;
        const int p0 = pb * BP;
        if (p0 >= a.np) break;
        if (p0 >= npool) { // whole block past the row's last visible pool: -inf without dots
            for (int e = tid; e < BP && p0 + e < a.np; e += 128)
                a.out[static_cast<long long>(r) * a.np + p0 + e] = neg_inf;
            continue;
        }
        // dots: 8 ascending K-chunks, 2 m-tiles, 2 n-tiles per warp
        float c[2][2][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) c[mt][nt][i] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                unsigned av[4];
                const int row0 = g4 + 16 * mt;
                const int col0 = 16 * ks + t2;
                av[0] = *reinterpret_cast<const unsigned*>(&q_sh[row0][col0]);
                av[1] = *reinterpret_cast<const unsigned*>(&q_sh[row0 + 8][col0]);
                av[2] = *reinterpret_cast<const unsigned*>(&q_sh[row0][col0 + 8]);
                av[3] = *reinterpret_cast<const unsigned*>(&q_sh[row0 + 8][col0 + 8]);
#pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    const int pool = p0 + 16 * warp + (lane >> 2) + 8 * nt;
                    unsigned bv[2];
                    if (pool < npool) {
                        const bf16* base = a.pk + static_cast<long long>(pool) * D + 16 * ks + (lane & 3) * 2;
                        bv[0] = *reinterpret_cast<const unsigned*>(base);
                        bv[1] = *reinterpret_cast<const unsigned*>(base + 8);
                    } else {
                        bv[0] = 0u;
                        bv[1] = 0u;
                    }
                    mma_16n8k16(c[mt][nt], av, bv);
                }
            }
        }
        // per col: fma(w_g, relu_g, w_g+8*relu_g+8) + fma(w_g+16, relu_g+16, w_g+24*relu_g+24); ptxas contracts it
        float v[4]; // col slots 2t2, 2t2+1, 2t2+8, 2t2+9 within the warp's 16 cols
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const float r0 = fmaxf(__fmul_rn(c[0][nt][half], a.scale), 0.0f);
                const float r1 = fmaxf(__fmul_rn(c[0][nt][2 + half], a.scale), 0.0f);
                const float r2 = fmaxf(__fmul_rn(c[1][nt][half], a.scale), 0.0f);
                const float r3 = fmaxf(__fmul_rn(c[1][nt][2 + half], a.scale), 0.0f);
                const float p1 = __fmaf_rn(w_sh[g4], r0, __fmul_rn(w_sh[g4 + 8], r1));
                const float p2 = __fmaf_rn(w_sh[g4 + 16], r2, __fmul_rn(w_sh[g4 + 24], r3));
                v[2 * nt + half] = __fadd_rn(p1, p2);
            }
        // butterfly: lane l's col value gains lanes l^16, l^8, l^4 (same col slot) in that order
        const unsigned all = 0xffffffffu;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i] = __fadd_rn(v[i], __shfl_xor_sync(all, v[i], 16));
            v[i] = __fadd_rn(v[i], __shfl_xor_sync(all, v[i], 8));
            v[i] = __fadd_rn(v[i], __shfl_xor_sync(all, v[i], 4));
        }
        // lanes 0..3 hold complete sums for cols 16w + 2s + {0,1,8,9}: stage them, then thread e stores col e
        if (lane < 4) {
            sc_sh[16 * warp + 2 * lane] = v[0];
            sc_sh[16 * warp + 2 * lane + 1] = v[1];
            sc_sh[16 * warp + 2 * lane + 8] = v[2];
            sc_sh[16 * warp + 2 * lane + 9] = v[3];
        }
        __syncthreads();
        const int e0 = p0 + tid;
        if (tid < BP && e0 < a.np)
            a.out[static_cast<long long>(r) * a.np + e0] = e0 < npool ? sc_sh[tid] : neg_inf;
        __syncthreads();
    }
}

// Block-wide exclusive prefix sum of the threads' counts, in thread order (each thread owns 8 consecutive elements).
__device__ __forceinline__ int block_prefix(int v, int* warp_totals, int* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int p = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, p, off);
        if (lane >= off) p += u;
    }
    if (lane == 31) warp_totals[warp] = p;
    __syncthreads();
    if (threadIdx.x == 0) {
        int run = 0;
        for (int w = 0; w < 4; w++) {
            const int t = warp_totals[w];
            warp_totals[w] = run;
            run += t;
        }
        *total = run;
    }
    __syncthreads();
    return warp_totals[warp] + (lane == 0 ? 0 : p - v);
}

// Select: a radix select (8 bits/pass) finds the K-th best key, then one pass writes pools above it plus lowest ties.
extern "C" __global__ void __launch_bounds__(128) tf_glm_select_rows(SelectArgs a) {
    __shared__ int hist[256];
    __shared__ int warp_totals[4];
    __shared__ int total;
    const int r = blockIdx.x;
    const long long row = static_cast<long long>(r) * a.s_stride;
    const int limit = a.visible ? min(a.np, max(a.k, ((a.posp ? *a.posp : a.pos) + r + 1) / 4)) : a.np;
    unsigned prefix = 0, fixed = 0;
    int need = a.k;
#pragma unroll 1
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int c = 0; c < limit; c += 1024) {
            const int i = c + threadIdx.x * 8;
            for (int t = 0; t < 8; t++) {
                const int idx = i + t;
                if (idx < limit) {
                    const unsigned u = order_key(a.s[row + idx]);
                    if ((u & fixed) == prefix) atomicAdd(&hist[(u >> shift) & 0xFF], 1);
                }
            }
        }
        __syncthreads();
        int total_hist = 0;
        for (int d = 0; d < 256; d++) total_hist += hist[d];
        int excl = 0, digit = 0, above = 0;
        for (int d = 0; d < 256; d++) {
            const int h = hist[d];
            if (total_hist - excl >= need) {
                digit = d;
                above = total_hist - excl - h; // bins strictly above digit
            }
            excl += h;
        }
        need -= above;
        prefix |= static_cast<unsigned>(digit) << shift;
        fixed |= 0xFFu << shift;
        __syncthreads();
    }
    // write pass: pools above the K-th key, then the lowest ties, ascending; a block-exclusive scan places them
    int written = 0, equal_seen = 0;
    for (int c = 0; c < limit; c += 1024) {
        const int i = c + threadIdx.x * 8;
        int eq8 = 0;
        for (int t = 0; t < 8; t++) {
            const int idx = i + t;
            if (idx < limit && order_key(a.s[row + idx]) == prefix) eq8++;
        }
        const int eq_prefix = block_prefix(eq8, warp_totals, &total);
        const int eq_tot = total;
        __syncthreads();
        // every element above the K-th key is taken plus lowest ties by global tie rank; positions count all takes
        int take8 = 0, eq_before = 0;
        for (int t = 0; t < 8; t++) {
            const int idx = i + t;
            if (idx < limit) {
                const unsigned u = order_key(a.s[row + idx]);
                if (u > prefix || (u == prefix && equal_seen + eq_prefix + eq_before < need)) take8++;
                if (u == prefix) eq_before++;
            }
        }
        const int take_prefix = block_prefix(take8, warp_totals, &total);
        const int take_tot = total;
        __syncthreads();
        int run = written + take_prefix;
        eq_before = 0;
        for (int t = 0; t < 8; t++) {
            const int idx = i + t;
            if (idx < limit) {
                const unsigned u = order_key(a.s[row + idx]);
                if (u > prefix || (u == prefix && equal_seen + eq_prefix + eq_before < need)) {
                    a.out[static_cast<long long>(r) * a.out_stride + run] = idx;
                    run++;
                }
                if (u == prefix) eq_before++;
            }
        }
        __syncthreads();
        written += take_tot;
        equal_seen += eq_tot;
    }
}

// Tokens/counts: each pool's four tokens ascending, then the incomplete last pool's; counts past the dense limit.
extern "C" __global__ void tf_glm_tokens_counts(TokensArgs a) {
    const int r = blockIdx.x;
    const int t = threadIdx.x; // 512 threads, four tokens a thread
    const long long q = static_cast<long long>(a.posp ? *a.posp : a.pos) + r;
    const long long npool = (q + 1) / 4;
    const long long p = a.pools[static_cast<long long>(r) * 512 + t]; // every row's 512 selected pools' tokens
    for (int j = 0; j < 4; j++)
        a.tokens[static_cast<long long>(r) * 2051 + t * 4 + j] = static_cast<int>(p * 4 + j);
    if (t < 3) {
        const long long tail = npool * 4 + t;
        a.tokens[static_cast<long long>(r) * 2051 + 2048 + t] = tail <= q ? static_cast<int>(tail) : -1;
    }
    if (t == 0) {
        int tail_ok = 0;
        for (int j = 0; j < 3; j++)
            if (npool * 4 + j <= q) tail_ok++;
        a.counts[r] = npool > 512 ? 2048 + tail_ok : 0;
    }
}
} // namespace tf_glm_sparse

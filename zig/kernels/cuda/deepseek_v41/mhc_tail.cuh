// mhc_cuda.cu's parallel tail (TF_DSV41_MHC_TAIL=1): coef/norm items and tail(), inside namespace dsv41_mhc
#pragma once
// ---- parallel tail (TF_DSV41_MHC_TAIL=1): live CTAs claim finish()'s items from a counter, same chains, same bits
constexpr int F2C = NT * 8;                       // columns a normed-input item
constexpr int F2N = D / F2C;                      // items a row
static_assert(D % F2C == 0 && NE * NQ * 16 <= 2 * STAGE4 * 16, "tail items");

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float rn_of(float ssc, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ssc, 1.0f * D), eps)));
}

// row r's 25 sums over its 160 entries and the square sum over stream 0's 40 (finish()'s chains), then coefs()
__device__ void coef_item(const Args& a, int r, float4* st, float* mx, int t) {
    for (int i = t; i < NE * NQ; i += NT)
        cp16(st + i, a.part + (static_cast<long long>(r) * NE + i / NQ) * PW + (i % NQ) * 4);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    if (t < 26) {
        const float* sf = reinterpret_cast<const float*>(st);
        const int n = t == 25 ? NB : NE;
        float acc = 0.0f;
#pragma unroll 8
        for (int e = 0; e < n; ++e) acc = __fadd_rn(acc, sf[e * (NQ * 4) + t]);
        mx[t] = t == 25 ? rn_of(acc, a.eps) : acc;
    }
    __syncthreads();
    if (t < 32) coefs_of(a, mx, r, t < 16, t & 15);
}

// normed input of row r, columns [F2C q, F2C q + F2C): bf16((c * rn) * w)
__device__ void norm_item(const Args& a, int r, int q, float* buf, int t) {
    const int d0 = q * F2C + t * 8;
    const bool nwbf = a.nwbf16 != 0;
    float c[8], w[8], o[8];
    widen8(__ldcg(reinterpret_cast<const uint4*>(a.c + static_cast<long long>(r) * D + d0)), c);
    load8f(static_cast<const char*>(a.nw) + d0 * (nwbf ? 2 : 4), nwbf, w);
    if (t < NB) buf[t] = __ldcg(a.part + (static_cast<long long>(r) * NE + t) * PW + 25);
    __syncthreads();
    if (t == 0) {
        float ssc = 0.0f;
#pragma unroll 8
        for (int e = 0; e < NB; ++e) ssc = __fadd_rn(ssc, buf[e]);
        buf[NB] = rn_of(ssc, a.eps);
    }
    __syncthreads();
    const float rn = buf[NB];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __fmul_rn(__fmul_rn(c[k], rn), w[k]);
    *reinterpret_cast<uint4*>(a.out + static_cast<long long>(r) * D + d0) = pack8(o);
}

template <bool MIX>
__device__ void tail(const Args& a, float* rowv, float4* st, int t) {
    __shared__ int live, item;
    __shared__ float nbuf[NB + 1];                           // a normed-input item's 40 square sums + rn
    const int total = static_cast<int>(gridDim.x * gridDim.y), R = a.R;
    if (t == 0) {
        int n = atomicAdd(a.cnt, 1) + 1;
        if (n < total && a.spin > 0) {
            const long long t0 = clock64();
            while ((n = ld_acquire(a.cnt)) < total && clock64() - t0 < a.spin) __nanosleep(32);
        }
        live = n >= total;
    }
    __syncthreads();
    if (live) {
        __threadfence();
        const int nc = MIX && !a.defer ? R : 0, nitems = nc + R * F2N;
        for (;;) {
            if (t == 0) item = atomicAdd(a.cnt + 1, 1);
            __syncthreads();
            const int it = item;
            if (it >= nitems) break;
            if (it < nc) {
                coef_item(a, it, st, rowv, t);
            } else {
                const int f = it - nc;
                norm_item(a, f / F2N, f % F2N, nbuf, t);
            }
            __syncthreads();
        }
    }
    if (t == 0 && atomicAdd(a.cnt + 2, 1) == total - 1) {     // the last to leave: every counter back at 0
        atomicExch(a.cnt, 0);
        atomicExch(a.cnt + 1, 0);
        atomicExch(a.cnt + 2, 0);
    }
}

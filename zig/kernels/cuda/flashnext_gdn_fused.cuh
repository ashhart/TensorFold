// One kernel a value head for Flash Next's prompt GDN: fn_gdn_front's rows, fn_gdn_chain_tpr's chain and fn_gdn_norm's gated
// norm, each with its own op sequence (so its bits), the fp32 rows never leaving shared memory.
// 256 chain threads (TPR 2: two a state row) + 128 producer threads (4 warps): the producers make batch i+1's q, k, v, g and
// beta into the stage the chain reads next, and the gated norm of batch i-1 from the readouts the chain left in shared memory.
namespace gdn_fused {
using namespace fn_gdn;
constexpr int STEPS = 16, TPR = 2, CH = 256, PR = 128, THREADS = CH + PR;
constexpr int ROW = (DK + DK + DV) * 4 + 16;                                  // gdn_tpr::Stage's row: k, q, v, g, beta
constexpr int STAGE = STEPS * ROW;
constexpr int SMEM = 2 * STAGE + 2 * STEPS * DV * 4 + STEPS * 2 * DK * 4 + STEPS * 4;

__device__ __forceinline__ void pbar() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }

// conv_act for rows r0 .. r0 + n - 1 of channel c, each P / cs row loaded once (conv_act's taps, order and rounding)
__device__ __forceinline__ void conv_rows(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
                                          const __nv_bfloat16* __restrict__ cw, int r0, int n, int c, float (&out)[STEPS]) {
    float w[TAPS];
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) w[tap] = __bfloat162float(cw[c * TAPS + tap]);
    float xw[STEPS + TAPS - 1];
#pragma unroll
    for (int k = 0; k < STEPS + TAPS - 1; ++k) {
        const int at = r0 + k;  // conv_act's `at` for row r0 + j, tap k - j
        xw[k] = k < n + TAPS - 1 ? (at < TAPS - 1 ? __bfloat162float(cs[at * C + c]) : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c])) : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < STEPS; ++j) {
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < TAPS; ++tap) acc = acc + w[tap] * xw[j + tap];
        out[j] = bf(acc / (1.0f + expf(-acc)));
    }
}

// producers: rows r0 .. r0 + n - 1's q, k (normed as fn_gdn_front's warps 0/1), v, g, beta into the stage; x a [STEPS][2][DK]
__device__ __forceinline__ void front_batch(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
                                            const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
                                            const float* __restrict__ dt_bias, int r0, int n, int hk, int hv, int p,
                                            unsigned char* base, float (*x)[2][DK]) {
    float o[STEPS];
    conv_rows(P, cs, cw, r0, n, hk * DK + p, o);                       // fn_gdn_front unit hk, t = p: q's channel
#pragma unroll
    for (int j = 0; j < STEPS; ++j) if (j < n) x[j][0][p] = o[j];
    conv_rows(P, cs, cw, r0, n, NK * DK + hk * DK + p, o);             // t = p + DK: k's channel
#pragma unroll
    for (int j = 0; j < STEPS; ++j) if (j < n) x[j][1][p] = o[j];
    conv_rows(P, cs, cw, r0, n, 2 * NK * DK + hv * DV + p, o);         // unit NK + hv, t = p: v
#pragma unroll
    for (int j = 0; j < STEPS; ++j) if (j < n) reinterpret_cast<float*>(base + j * ROW + 2 * DK * 4)[p] = o[j];
    if (p < n) {                                                         // unit NK + hv, t == DV: row r0 + p's g and beta
        const int r = r0 + p;
        const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
        const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
        float* gbp = reinterpret_cast<float*>(base + p * ROW + (2 * DK + DV) * 4);
        gbp[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
        gbp[1] = bf(sigmoidf_(b));
    }
    pbar();
    const int warp = p >> 5, lane = p & 31;
    for (int task = warp; task < 2 * n; task += PR / 32) {               // (row, q|k): fn_gdn_front's warp 0 (q) / 1 (k)
        const int j = task >> 1, which = task & 1;
        float v4[4], ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v4[i] = x[j][which][lane * 4 + i];
            ss = ss + v4[i] * v4[i];
        }
        ss = warp_sum(ss);
        float inv = 1.0f / sqrtf(ss + 1e-6f);
        if (which == 0) inv = inv * (1.0f / sqrtf((float)DK));
        float* dst = reinterpret_cast<float*>(base + j * ROW + (which == 0 ? DK * 4 : 0));   // stage: k first, then q
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = v4[i] * inv;
    }
}

// producers: fn_gdn_norm for n rows from row r0 (readouts ys[j][dv])
__device__ __forceinline__ void norm_rows(const float (*ys)[DV], const __nv_bfloat16* __restrict__ P,
                                          const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out,
                                          float* __restrict__ xs, int r0, int n, int hv, int p, float* rinv) {
    const int warp = p >> 5, lane = p & 31;
    for (int j = warp; j < n; j += PR / 32) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float y = ys[j][lane * 4 + i];
            ss = ss + y * y;
        }
        ss = warp_sum(ss);
        if (lane == 0) rinv[j] = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    pbar();
    float zs[STEPS];
#pragma unroll
    for (int j = 0; j < STEPS; ++j) zs[j] = j < n ? __bfloat162float(P[(size_t)(r0 + j) * PW + C + hv * DV + p]) : 0.0f;
    const float nw = __bfloat162float(norm_w[p]);
#pragma unroll
    for (int j = 0; j < STEPS; ++j) {
        if (j >= n) break;
        const int r = r0 + j;
        const float yn = bf(bf(ys[j][p] * rinv[j]) * nw);
        const float z = zs[j];
        const float o = bf(yn * sigmoidf_(z));
        out[(size_t)r * NV * DV + hv * DV + p] = __float2bfloat16_rn(o);
        const float gs = warp_sum(o);
        if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
    }
    pbar();
}

__device__ __forceinline__ void run(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
                                    const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
                                    const float* __restrict__ dt_bias, const float* state_in, float* state_out, int rows,
                                    const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out,
                                    float* __restrict__ xs, float* __restrict__ ys_out) {
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* stage = smem;                                                 // [2][STEPS] rows
    float (*ysb)[STEPS][DV] = reinterpret_cast<float (*)[STEPS][DV]>(smem + 2 * STAGE);
    float (*xb)[2][DK] = reinterpret_cast<float (*)[2][DK]>(smem + 2 * STAGE + 2 * STEPS * DV * 4);
    float* rinv = reinterpret_cast<float*>(smem + 2 * STAGE + 2 * STEPS * DV * 4 + STEPS * 2 * DK * 4);
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const bool producer = threadIdx.x >= CH;
    const int p = threadIdx.x - CH;
    const int batches = (rows + STEPS - 1) / STEPS;
    auto fill = [&](int bi) {
        const int r0 = bi * STEPS, n = min(STEPS, rows - r0);
        front_batch(P, cs, cw, a_log, dt_bias, r0, n, hk, hv, p, stage + (bi & 1) * STAGE, xb);
    };
    // the chain's registers (gdn_tpr::chain<2, 16>'s)
    constexpr int L = 32 / TPR;
    const int row = threadIdx.x / TPR, t = threadIdx.x % TPR;
    const size_t sbase = (size_t)hv * DV * DK + (size_t)row * DK;
    float s[L][4];
    if (!producer) {
#pragma unroll
        for (int l = 0; l < L; ++l)
#pragma unroll
            for (int i = 0; i < 4; ++i) s[l][i] = state_in[sbase + (l * TPR + t) * 4 + i];
    } else if (batches > 0) fill(0);
    __syncthreads();
    for (int bi = 0; bi < batches; ++bi) {
        const int r0 = bi * STEPS, n = min(STEPS, rows - r0);
        if (producer) {
            if (bi + 1 < batches) fill(bi + 1);
            if (bi >= 1) norm_rows(ysb[(bi - 1) & 1], P, norm_w, eps, out, xs, (bi - 1) * STEPS, STEPS, hv, p, rinv);
        } else {
            const unsigned char* base = stage + (bi & 1) * STAGE;
            for (int rr = 0; rr < n; ++rr) {
                const unsigned char* d = base + rr * ROW;
                const float vv = reinterpret_cast<const float*>(d + 2 * DK * 4)[row];
                const float2 gbv = *reinterpret_cast<const float2*>(d + (2 * DK + DV) * 4);
                const float g = gbv.x, beta = gbv.y;
                float x[L];
#pragma unroll
                for (int l = 0; l < L; ++l) {
                    const float4 k4 = *reinterpret_cast<const float4*>(d + (l * TPR + t) * 16);
                    const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
                    float kv = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        s[l][i] = s[l][i] * g;
                        kv = kv + s[l][i] * kk[i];
                    }
                    x[l] = kv;
                }
                const float kv = gdn_tpr::tree<TPR>(x);
                const float delta = (vv - kv) * beta;
#pragma unroll
                for (int l = 0; l < L; ++l) {
                    const float4 k4 = *reinterpret_cast<const float4*>(d + (l * TPR + t) * 16);
                    const float4 q4 = *reinterpret_cast<const float4*>(d + DK * 4 + (l * TPR + t) * 16);
                    const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
                    float o = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        s[l][i] = s[l][i] + kk[i] * delta;
                        o = o + s[l][i] * qq[i];
                    }
                    x[l] = o;
                }
                const float o = bf(gdn_tpr::tree<TPR>(x));
                if (t == 0) {
                    ysb[bi & 1][rr][row] = o;
                    if (ys_out) ys_out[((size_t)(r0 + rr) * NV + hv) * DV + row] = o;
                }
            }
        }
        __syncthreads();
    }
    if (producer) {
        if (batches > 0) norm_rows(ysb[(batches - 1) & 1], P, norm_w, eps, out, xs, (batches - 1) * STEPS, rows - (batches - 1) * STEPS, hv, p, rinv);
    } else {
#pragma unroll
        for (int l = 0; l < L; ++l)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (l * TPR + t) * 4 + i] = s[l][i];
    }
}
}  // namespace gdn_fused

extern "C" __global__ void __launch_bounds__(gdn_fused::THREADS) fn_gdn_fused(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, const float* state_in, float* state_out, int rows,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out, float* __restrict__ xs,
        float* __restrict__ ys_out) {
    gdn_fused::run(P, cs, cw, a_log, dt_bias, state_in, state_out, rows, norm_w, eps, out, xs, ys_out);
}

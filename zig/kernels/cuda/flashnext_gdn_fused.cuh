// Flash Next prompt GDN, one kernel a value head: 4 producer warps front the next batch, 8 chain warps step and norm.
namespace gdn_fused {
using namespace fn_gdn;
constexpr int STEPS = 16, TPR = 2, CH = 256, PR = 128, THREADS = CH + PR;
constexpr int ROW = (DK + DK + DV) * 4 + 16;                                  // gdn_tpr::Stage's row: k, q, v, g, beta
constexpr int STAGE = STEPS * ROW;
constexpr int SMEM = 2 * STAGE + 2 * STEPS * DV * 4 + STEPS * 2 * DK * 4 + STEPS * 4;

__device__ __forceinline__ void pbar() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }

__device__ __forceinline__ void cbar() { asm volatile("bar.sync 2, 256;\n" ::: "memory"); }

constexpr int XW = STEPS + TAPS - 1;  // a batch's conv window rows

// a batch's loads, issued together: rows r0 .. r0 + n - 1's q, k and v conv windows and row r0 + p's b and a
struct Rows {
    float xw[3][XW];
    float b, a;
};

__device__ __forceinline__ void load_rows(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs, int r0,
                                          int n, const int (&ch)[3], int hv, int p, Rows& in) {
#pragma unroll
    for (int c3 = 0; c3 < 3; ++c3)
#pragma unroll
        for (int k = 0; k < XW; ++k) {
            const int at = r0 + k, c = ch[c3];  // conv_act's `at` for row r0 + j, tap k - j
            in.xw[c3][k] = k < n + TAPS - 1 ? (at < TAPS - 1 ? __bfloat162float(cs[at * C + c]) : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c])) : 0.0f;
        }
    if (p < n) {
        const int r = r0 + p;
        in.b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
        in.a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
    }
}

// producers: the batch's q, k, v, g, beta into the stage as fn_gdn_front makes them (nea = -exp(a_log), dtb dt bias)
__device__ __forceinline__ void front_rows(const Rows& in, const float (&w)[3][TAPS], float nea, float dtb, int n, int p,
                                           unsigned char* base, float (*x)[2][DK]) {
#pragma unroll
    for (int c3 = 0; c3 < 3; ++c3)
#pragma unroll
        for (int j = 0; j < STEPS; ++j) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) acc = acc + w[c3][tap] * in.xw[c3][j + tap];
            const float o = bf(acc / (1.0f + expf(-acc)));
            if (j < n) {
                if (c3 < 2) x[j][c3][p] = o;                                              // q, k: normed below
                else reinterpret_cast<float*>(base + j * ROW + 2 * DK * 4)[p] = o;       // v
            }
        }
    if (p < n) {                                                                          // row p's g and beta
        float* gbp = reinterpret_cast<float*>(base + p * ROW + (2 * DK + DV) * 4);
        gbp[0] = expf(nea * softplusf_(in.a + dtb));
        gbp[1] = bf(sigmoidf_(in.b));
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

// the chain's 256 threads after a batch's steps: fn_gdn_norm's arithmetic for its n rows from row r0 (ys[j][dv])
__device__ __forceinline__ void norm_rows(const float (*ys)[DV], const __nv_bfloat16* __restrict__ P,
                                          const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out,
                                          float* __restrict__ xs, int r0, int n, int hv, float* rinv) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    cbar();  // every readout of the batch is in ys
    for (int j = warp; j < n; j += CH / 32) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float y = ys[j][lane * 4 + i];
            ss = ss + y * y;
        }
        ss = warp_sum(ss);
        if (lane == 0) rinv[j] = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    cbar();
    constexpr int HALF = STEPS / 2;
    const int h = tid >> 7, p = tid & 127;
    float zs[HALF];
#pragma unroll
    for (int k = 0; k < HALF; ++k) {
        const int j = h * HALF + k;
        zs[k] = j < n ? __bfloat162float(P[(size_t)(r0 + j) * PW + C + hv * DV + p]) : 0.0f;
    }
    const float nw = __bfloat162float(norm_w[p]);
#pragma unroll
    for (int k = 0; k < HALF; ++k) {
        const int j = h * HALF + k;
        if (j >= n) break;
        const int r = r0 + j;
        const float yn = bf(bf(ys[j][p] * rinv[j]) * nw);
        const float z = zs[k];
        const float o = bf(yn * sigmoidf_(z));
        out[(size_t)r * NV * DV + hv * DV + p] = __float2bfloat16_rn(o);
        const float gs = warp_sum(o);
        if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + (p >> 5)] = gs;
    }
}

__device__ __forceinline__ void run(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
                                    const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
                                    const float* __restrict__ dt_bias, const float* state_in, float* state_out, int rows,
                                    const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out,
                                    float* __restrict__ xs, float* __restrict__ ys_out, int cut0, float* cut0_state,
                                    int cut1, float* cut1_state) {
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* stage = smem;                                                 // [2][STEPS] rows
    float (*ysb)[STEPS][DV] = reinterpret_cast<float (*)[STEPS][DV]>(smem + 2 * STAGE);
    float (*xb)[2][DK] = reinterpret_cast<float (*)[2][DK]>(smem + 2 * STAGE + 2 * STEPS * DV * 4);
    float* rinv = reinterpret_cast<float*>(smem + 2 * STAGE + 2 * STEPS * DV * 4 + STEPS * 2 * DK * 4);
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const bool producer = threadIdx.x >= CH;
    const int p = threadIdx.x - CH;
    const int batches = (rows + STEPS - 1) / STEPS;
    // producers: the three channels' taps and the gate constants once, then a batch's loads together, then its front
    const int ch[3] = {hk * DK + p, NK * DK + hk * DK + p, 2 * NK * DK + hv * DV + p};
    float w[3][TAPS];
    float nea = 0.0f, dtb = 0.0f;
    auto fill = [&](int bi) {
        const int r0 = bi * STEPS, n = min(STEPS, rows - r0);
        Rows in;
        load_rows(P, cs, r0, n, ch, hv, p, in);
        front_rows(in, w, nea, dtb, n, p, stage + (bi & 1) * STAGE, xb);
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
    } else {
#pragma unroll
        for (int c3 = 0; c3 < 3; ++c3)
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) w[c3][tap] = __bfloat162float(cw[ch[c3] * TAPS + tap]);
        nea = -expf(a_log[hv]);
        dtb = dt_bias[hv];
        if (batches > 0) fill(0);
    }
    __syncthreads();
    for (int bi = 0; bi < batches; ++bi) {
        const int r0 = bi * STEPS, n = min(STEPS, rows - r0);
        if (producer) {
            if (bi + 1 < batches) fill(bi + 1);
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
                // a prompt cache mark inside the pass: the state after row cut - 1, as the final store below writes it
                const int done = r0 + rr + 1;
                if (done == cut0 || done == cut1) {
                    float* cs_out = done == cut0 ? cut0_state : cut1_state;
#pragma unroll
                    for (int l = 0; l < L; ++l)
#pragma unroll
                        for (int i = 0; i < 4; ++i) cs_out[sbase + (l * TPR + t) * 4 + i] = s[l][i];
                }
                if (t == 0) {
                    ysb[bi & 1][rr][row] = o;
                    if (ys_out) ys_out[((size_t)(r0 + rr) * NV + hv) * DV + row] = o;
                }
            }
            norm_rows(ysb[bi & 1], P, norm_w, eps, out, xs, r0, n, hv, rinv);  // on the chain: it would wait at the barrier
        }
        __syncthreads();
    }
    if (!producer) {
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
        float* __restrict__ ys_out, int cut0, float* cut0_state, int cut1, float* cut1_state) {
    gdn_fused::run(P, cs, cw, a_log, dt_bias, state_in, state_out, rows, norm_w, eps, out, xs, ys_out, cut0, cut0_state, cut1, cut1_state);
}

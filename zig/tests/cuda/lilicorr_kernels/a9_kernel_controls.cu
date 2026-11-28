// PR 5102 owner ask A9 (DT-20261127-015103, seat builder): the committed controls in

// cuda_lilicorr_test.zig recompute Zig references without launching CUDA kernels. This

// harness launches the REAL kernels from zig/kernels/cuda/qwen35_draft_lilicorr.cuh on

// the real GPU, grades three controls, and pairs with committed fail-before injectors

// (a9_inject_factor.sh / a9_inject_tail.sh / a9_inject_chain.sh) that reproduce the

// recorded defect classes against a scratch copy of the kernel header. A control must

// MATCH in the clean run and FAIL under its own injector only (RED face pinned, the

// others stay green). No repo file is modified at run time: see a9_run.sh. 

//   FACTOR control: widen -> tf_lilicorr_factor at the real head face

//     (rows 56, hidden 1024, three dot terms + bias + silu, bf16 out).

//   TAIL control: tf_lilicorr_factor_rows hd=6 rows=9 (tail warps t=6,7 alive as load

//     workers) — closed form v = t + 6r - 8.5 for EVERY row plus the committed C2 goldens.

//   CHAIN control: drafter-head chain rms -> in-proj gemv -> lat_attn -> l2norm -> dots

//     (start + pair scores), device-resident feeds, real head shapes (qkv 56x3072,

//     lattice 56, head_dim 128). The edges use the flat stride=1024 face (OUT_VEC and

//     IN_VEC alias one row-per-1024 buffer) — a legal dots face, disclosed here.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>

#include "common/reduce.cuh"

namespace tf_q35d {
__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
using tf_common::warp_sum;

// A full-block sum, same shape as the family preamble (warp reduce + one cross-warp round).
__device__ __forceinline__ float tf_lilicorr_block_sum(float v, float* red) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int w = 0; w < (int)(blockDim.x / 32); ++w) total += red[w];
        red[0] = total;
    }
    __syncthreads();
    return red[0];
}
}  // namespace tf_q35d

using namespace tf_q35d;

#include "qwen35_draft_lilicorr.cuh"

// ---------------- host helpers ----------------
static inline uint16_t f2bf_rn(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounded = u + 0x7fffu + lsb;
    return (uint16_t)(rounded >> 16);
}
static inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline float widen_host(uint16_t b) { return bf2f(b); }
static inline double rbf_host(double f) { return bf2f(f2bf_rn((float)f)); }  // host per-op bf16 round

// bf16 grid step at w (abs), with a floor so near-zero cells stay gradeable.
static inline double bf16_ulp_abs(double w) {
    double a = fabs(w);
    if (a < 6e-8) return 6e-8;
    int e; frexp(a, &e);
    return ldexp(1.0, e - 9);
}
// The control bar: <= 2 bf16 ULP of the host reference + a floor (kernel math in f32, host in f64).
static inline bool bf_bar(double got, double want) {
    return fabs(got - want) <= 2.0 * bf16_ulp_abs(want) + 1e-4;
}

static int g_fail = 0;

static uint64_t rng_s;
static void rng_seed(uint64_t s) { rng_s = s; }
static inline double rng_uniform() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (double)(rng_s >> 11) / 9007199254740992.0; }
static inline float rng_normal(double sigma) {
    double u1 = rng_uniform() + 1.0 / 4294967296.0, u2 = rng_uniform();
    return (float)(sigma * sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

#define CUDA_OK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)
#define LAUNCH_OK() CUDA_OK(cudaGetLastError())

template <typename T>
static T* dev_alloc(const std::vector<T>& host) {
    T* p = nullptr;
    CUDA_OK(cudaMalloc(&p, host.size() * sizeof(T)));
    CUDA_OK(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

// ================= FACTOR control: widen -> tf_lilicorr_factor (rows 56, hd 1024) ============
static void control_factor() {
    printf("A9-FACTOR: build+launch\n");
    const int rows = 56, hd = 1024;
    const size_t nA = (size_t)rows * hd;
    std::vector<uint16_t> H_bf(nA);
    std::vector<float> w1((size_t)hd * hd), w2((size_t)hd * hd), w3((size_t)hd * hd), b(hd), anch(hd);
    rng_seed(0xA9FAC701ull);
    for (size_t i = 0; i < nA; i++) H_bf[i] = f2bf_rn(rng_normal(0.25));
    for (size_t i = 0; i < (size_t)hd * hd; i++) { w1[i] = rng_normal(1.0 / 32.0); w2[i] = rng_normal(1.0 / 32.0); w3[i] = rng_normal(1.0 / 512.0); }
    for (int t = 0; t < hd; t++) { b[t] = -2.0f + rng_uniform() * 9.0f; anch[t] = rng_normal(0.25); }
    // W layout [t][3 slab][hd] flattened: slab0 = w1 (A term), slab1 = w2 (anchor term), slab2 = w3 (cross).
    std::vector<float> W((size_t)hd * 3 * hd);
    for (int t = 0; t < hd; t++)
        for (int s = 0; s < 3; s++) {
            const float* src = s == 0 ? w1.data() : s == 1 ? w2.data() : w3.data();
            for (int i = 0; i < hd; i++)
                W[((size_t)t * 3 + s) * hd + i] = src[(size_t)t * hd + i];
        }

    printf("FACTOR inputs done\n");
    uint16_t* d_H = dev_alloc(H_bf);
    float* d_W = dev_alloc(W);
    uint16_t* d_OUT;               CUDA_OK(cudaMalloc(&d_OUT, nA * 2));
    std::vector<float> fv(anch);
    float* d_ANCH = dev_alloc(fv);
    float* d_B = dev_alloc(b);

    // Stage 1: widen — the real widening kernel produces the factor's f32 A_IN.
    float* d_A; CUDA_OK(cudaMalloc(&d_A, nA * 4));
    tf_lilicorr_widen<<<rows, 1024>>>(reinterpret_cast<const __nv_bfloat16*>(d_H), d_A, rows * hd);
    LAUNCH_OK();
    CUDA_OK(cudaDeviceSynchronize());
    // Stage 2: the factor block, one thread per output channel.
    printf("FACTOR widen done\n");
    tf_lilicorr_factor<<<rows, 1024>>>(d_A, d_ANCH, d_W, d_B, reinterpret_cast<__nv_bfloat16*>(d_OUT), rows, hd);
    LAUNCH_OK();
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<uint16_t> out(nA);
    CUDA_OK(cudaMemcpy(out.data(), d_OUT, nA * 2, cudaMemcpyDeviceToHost));

    printf("FACTOR mirror begin\n");
    int bad = 0;
    double worst = 0.0;
    for (int r = 0; r < rows; r++) {
        if (r % 16 == 0) printf("FACTOR mirror r=%d\n", r);
        for (int t = 0; t < hd; t++) {
            double v = (double)b[t];
            for (int i = 0; i < hd; i++) {
                const double a = widen_host(H_bf[(size_t)r * hd + i]);
                v += a * (double)w1[(size_t)t * hd + i];
                v += (double)anch[i] * (double)w2[(size_t)t * hd + i];
                v += a * (double)anch[i] * (double)w3[(size_t)t * hd + i];
            }
            const double want = (double)v / (1.0 + exp(-v));
            const double got = bf2f(out[(size_t)r * hd + t]);
            worst = fmax(worst, fabs(got - want));
            if (!bf_bar(got, want)) bad++;
        }
    }
    printf("A9-FACTOR: %s cells=%d worst=%.3g\n", bad ? "FAIL" : "MATCH", rows * hd, worst);
    if (bad) g_fail++;
    cudaFree(d_H); cudaFree(d_A); cudaFree(d_W); cudaFree(d_OUT); cudaFree(d_ANCH); cudaFree(d_B);
}

// ================= TAIL control: tf_lilicorr_factor_rows hd=6 rows=9 =========================
static inline float tail_closed_form(int r, int t) { return (float)t + 6.0f * (float)r - 8.5f; }

static void control_tail() {
    printf("A9-TAIL: build+launch\n");
    const int rows = 9, hd = 6;  // hd % 8 != 0: warps t=6,7 are tail warps; rows=9 uses the
                                 // load band only threads 192..255 (the tail warps) fill at i0=0.
    std::vector<float> ain((size_t)rows * hd), anch((size_t)hd, 0.0f), W((size_t)hd * 3 * hd, 0.0f), b(hd);
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < hd; i++)
            ain[(size_t)r * hd + i] = 10.0f * r + (float)i;      // A_IN[r][j] = 10r + j
    for (int t = 0; t < hd; t++) {
        b[t] = (float)t - 10.0f;                                  // the B carry
        for (int i = 0; i < hd; i++) W[(size_t)t * 3 * hd + i] = 0.1f;  // w1 slab: 0.1; w2 = w3 = 0
    }
    float *d_A = dev_alloc(ain), *d_ANCH = dev_alloc(anch), *d_W = dev_alloc(W), *d_B = dev_alloc(b);
    std::vector<uint16_t> out((size_t)rows * hd);
    uint16_t* d_OUT; CUDA_OK(cudaMalloc(&d_OUT, out.size() * 2));

    tf_lilicorr_factor_rows<<<1, 256>>>(d_A, d_ANCH, d_W, d_B, reinterpret_cast<__nv_bfloat16*>(d_OUT), rows, hd);
    LAUNCH_OK();
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(out.data(), d_OUT, out.size() * 2, cudaMemcpyDeviceToHost));

    // The committed C2 goldens (rows 0..2, ask-5 part C cell 2) re-verified on-device.

    // NOTE (A9, 11/27): the committed cell originally pinned SIGMOID goldens here; the

    // first device run of this control REDd on them — the real kernel (and its sibling

    // tf_lilicorr_factor, and the head's Zig reference at cuda_lilicorr_head.zig:380)

    // outputs silu. The cell was cured to these silu goldens in the same leg; the RED

    // face is this control's committed injector runs.
    static const float GOLD[3][6] = {
        { -0.001729f, -0.004146f, -0.009758f, -0.022386f, -0.049441f, -0.102593f },
        { -0.189645f, -0.273638f, -0.188770f, 0.311230f, 1.226362f, 2.310355f },
        { 3.397407f, 4.450559f, 5.477614f, 6.490242f, 7.495854f, 8.498271f },
    };
    int bad = 0, bad_tail_band = 0, bad_gold = 0;
    for (int r = 0; r < rows; r++) {
        for (int t = 0; t < hd; t++) {
            const double v = (double)tail_closed_form(r, t);
            const double want = v / (1.0 + exp(-v));  // silu — the kernel's output contract
            const double got = bf2f(out[(size_t)r * hd + t]);
            if (fabs(got - want) > 2.0 * bf16_ulp_abs(want) + 1e-4) {
                bad++;
                if (r == 6 || r == 7) bad_tail_band++;
            }
            if (r < 3 && fabs(got - (double)GOLD[r][t]) > 3e-3 * fabs((double)GOLD[r][t]) + 1e-6) bad_gold++;
        }
    }
    printf("A9-TAIL: %s cells=%d red-band-fails=%d golden-rows=%s\n",
           bad ? "FAIL" : "MATCH", rows * hd, bad_tail_band, bad_gold ? "FAIL" : "MATCH");
    if (bad) g_fail++;
    if (bad_gold) g_fail++;
    cudaFree(d_A); cudaFree(d_ANCH); cudaFree(d_W); cudaFree(d_B); cudaFree(d_OUT);
}

// ================= CHAIN control: rms -> in-proj gemv -> lat_attn -> l2norm -> dots =========
static const int CH_ROWS = 56, CH_HID = 1024, CH_I3 = 3072, CH_LATTICE = 56, CH_HDIM = 128;
static const int CH_STRIDE = 1024;  // flat edges face: OUT_VEC = IN_VEC = E, row width 1024

struct ChainHost {
    std::vector<uint16_t> h0;                       // [56][1024] bf16 hidden
    std::vector<float> w_rms;                       // [1024]
    std::vector<float> w_proj, b_proj;              // [3072][1024], [3072]
    std::vector<float> bias;                        // [(h*lattice+qi)*lattice+ki] = [8*56*56]
    std::vector<float> anchor;                      // [512] (the l2norm tail row, normalized)
    std::vector<float> stale;                       // [512] parked in the tail row's upper half (S4 over-run fuel)
    std::vector<double> h1, E;                      // mirrors: widened h, normalized edges [57][1024]
    std::vector<double> start, pairst;              // mirror results
};

static void chain_mirror(ChainHost& c) {
    // rms (real op order mirrored from the kernel's arithmetic, f64 host accumulate)
    c.h1.assign((size_t)CH_ROWS * CH_HID, 0.0);
    for (int r = 0; r < CH_ROWS; r++) {
        double ssq = 0.0;
        for (int i = 0; i < CH_HID; i++) { const double x = widen_host(c.h0[(size_t)r * CH_HID + i]); ssq += x * x; }
        const double inv = 1.0 / sqrt(ssq / CH_HID + 1e-6);
        for (int i = 0; i < CH_HID; i++)
            c.h1[(size_t)r * CH_HID + i] = rbf_host((float)(widen_host(c.h0[(size_t)r * CH_HID + i]) * inv * c.w_rms[i]));
    }
    // in-proj gemv: OUT[r][t] = sum_k widen(h1[r][k]) * W[t][k] + B[t]
    std::vector<double> qkv((size_t)CH_ROWS * CH_I3, 0.0);
    for (int r = 0; r < CH_ROWS; r++)
        for (int t = 0; t < CH_I3; t++) {
            const float* w = &c.w_proj[(size_t)t * CH_HID];
            double acc = 0.0;
            for (int k = 0; k < CH_HID; k++) acc += c.h1[(size_t)r * CH_HID + k] * (double)w[k];
            qkv[(size_t)r * CH_I3 + t] = acc + (double)c.b_proj[t];
        }
    // lat_attn: 56 qi x 8 heads, head_dim 128, scores + softmax + V-mix
    std::vector<double> att((size_t)CH_ROWS * CH_HID, 0.0);
    const double scale = 1.0 / sqrt((double)CH_HDIM);
    std::vector<double> p(CH_LATTICE);
    for (int qi = 0; qi < CH_ROWS; qi++) {
        for (int h = 0; h < 8; h++) {
            double mx = -1e30, sum = 0.0;
            for (int ki = 0; ki < CH_LATTICE; ki++) {
                double acc = 0.0;
                for (int t = 0; t < CH_HDIM; t++)
                    acc += qkv[(size_t)qi * CH_I3 + h * CH_HDIM + t] * qkv[(size_t)ki * CH_I3 + CH_HID + h * CH_HDIM + t];
                p[ki] = acc * scale + (double)c.bias[((size_t)h * CH_LATTICE + qi) * CH_LATTICE + ki];
                if (p[ki] > mx) mx = p[ki];
            }
            for (int ki = 0; ki < CH_LATTICE; ki++) { p[ki] = exp(p[ki] - mx); sum += p[ki]; }
            for (int ki = 0; ki < CH_LATTICE; ki++) p[ki] /= sum;
            for (int t = 0; t < CH_HDIM; t++) {
                double acc = 0.0;
                for (int ki = 0; ki < CH_LATTICE; ki++)
                    acc += p[ki] * qkv[(size_t)ki * CH_I3 + 2 * CH_HID + h * CH_HDIM + t];
                att[(size_t)qi * CH_HID + h * CH_HDIM + t] = acc;
            }
        }
    }
    // l2norm: 56 edge rows = two 512-wide halves each; the anchor tail row = tail_width

    // 512 (lower half of a stride-1024 row; the upper half stays zero — the dots start

    // dot runs over the full 1024 and mirrors the same zeros). The buffer carries TWO

    // rows behind the tail: row 57 is deliberately loaded with stale values (the S4

    // over-run's fuel — the recorded defect read "whatever sat behind it"); row 58 pads.
    c.E.assign((size_t)(CH_ROWS + 3) * CH_STRIDE, 0.0);
    for (int r = 0; r < CH_ROWS; r++)
        for (int half = 0; half < 2; half++) {
            double ssq = 0.0;
            const int width = CH_STRIDE / 2;
            for (int i = 0; i < width; i++) { const double x = att[(size_t)r * CH_HID + half * width + i]; ssq += x * x; }
            const double inv = 1.0 / fmax(sqrt(ssq), 1e-6);
            for (int i = 0; i < width; i++)
                c.E[(size_t)r * CH_STRIDE + half * width + i] = att[(size_t)r * CH_HID + half * width + i] * inv;
        }
    {
        double ssq = 0.0;
        for (int i = 0; i < CH_STRIDE / 2; i++) ssq += (double)c.anchor[i] * (double)c.anchor[i];
        const double inv = 1.0 / fmax(sqrt(ssq), 1e-6);
        for (int i = 0; i < CH_STRIDE / 2; i++) c.E[(size_t)CH_ROWS * CH_STRIDE + i] = (double)c.anchor[i] * inv;
        // The tail row's upper half carries arbitrary live data (the S4 over-run's fuel:

        // the recorded defect read "whatever sat behind it" — here, the same row's second

        // half, which is what the flat face parks next to a 512-wide tail). The clean run

        // never touches it; the injected width reads it into the anchor's inv.
        for (int i = 0; i < CH_STRIDE / 2; i++)
            c.E[(size_t)CH_ROWS * CH_STRIDE + CH_STRIDE / 2 + i] = (double)c.stale[i];
    }
    // dots: start[b] = dot(anchor_row, E[b]); pair[k] = dot(E[s*8+i], E[(s+1)*8+j]).
    c.start.assign(8, 0.0);
    c.pairst.assign((size_t)6 * 8 * 8, 0.0);
    for (int b = 0; b < 8; b++) {
        double acc = 0.0;
        for (int i = 0; i < CH_STRIDE; i++) acc += c.E[(size_t)CH_ROWS * CH_STRIDE + i] * c.E[(size_t)b * CH_STRIDE + i];
        c.start[b] = acc;
    }
    for (int k = 0; k < 6 * 8 * 8; k++) {
        const int s = k / 64, i = (k / 8) % 8, j = k % 8;
        double acc = 0.0;
        for (int t = 0; t < CH_STRIDE; t++)
            acc += c.E[(size_t)(s * 8 + i) * CH_STRIDE + t] * c.E[(size_t)((s + 1) * 8 + j) * CH_STRIDE + t];
        c.pairst[k] = acc;
    }
}

static void chain_run_device(const ChainHost& c, float* d_start, float* d_pair) {
    // E carries the 56 att rows, then the anchor tail row at row 56 (the l2norm tail block).
    float* d_E;         CUDA_OK(cudaMalloc(&d_E, (size_t)(CH_ROWS + 3) * CH_STRIDE * 4));
    uint16_t* d_H0 = dev_alloc(c.h0);
    float* d_WRMS = dev_alloc(c.w_rms);
    float* d_WPROJ = dev_alloc(c.w_proj);
    float* d_BPROJ = dev_alloc(c.b_proj);
    float* d_BIAS = dev_alloc(c.bias);
    float* d_QKV;       CUDA_OK(cudaMalloc(&d_QKV, (size_t)CH_ROWS * CH_I3 * 4));

    // 1. rms (grid rows, block 1024)
    {
        uint16_t* d_H1; CUDA_OK(cudaMalloc(&d_H1, (size_t)CH_ROWS * CH_HID * 2));
        tf_lilicorr_rms<<<CH_ROWS, 1024>>>(reinterpret_cast<const __nv_bfloat16*>(d_H0), d_WRMS, reinterpret_cast<__nv_bfloat16*>(d_H1), CH_HID, 1e-6f);
        LAUNCH_OK();
        // 2. in-proj gemv (row-batched, bf16 in): grid 96 x block 256 -> qkv f32 [56][3072]
        tf_gemv_bf16_rows<<<(CH_I3 + 31) / 32, 256>>>(reinterpret_cast<const __nv_bfloat16*>(d_H1), d_WPROJ, d_BPROJ, d_QKV, CH_HID, CH_I3, CH_ROWS);
        LAUNCH_OK();
        CUDA_OK(cudaDeviceSynchronize());
        cudaFree(d_H1);
    }
    // 3. the E rows zero first (the lat_attn output rows land in E[0..55]; rows 56..58

    // are staged below), then lat_attn (grid 56*8=448, block 128) writes straight into E.
    CUDA_OK(cudaMemset(d_E, 0, (size_t)(CH_ROWS + 3) * CH_STRIDE * 4));
    tf_lilicorr_lat_attn<<<CH_ROWS * 8, CH_HDIM>>>(d_QKV, d_BIAS, d_E, CH_LATTICE, CH_HID, CH_HDIM);
    // 4. the anchor tail row (lower 512 = the anchor, upper 512 = parked live data) +

    // then l2norm (grid 57, block 1024, tail width 512).
    CUDA_OK(cudaMemcpy(d_E + (size_t)CH_ROWS * CH_STRIDE, c.anchor.data(), (CH_STRIDE / 2) * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_E + (size_t)CH_ROWS * CH_STRIDE + CH_STRIDE / 2, c.stale.data(), (CH_STRIDE / 2) * 4, cudaMemcpyHostToDevice));
    tf_lilicorr_l2norm<<<CH_ROWS + 1, 1024>>>(d_E, CH_ROWS, CH_STRIDE, CH_STRIDE / 2, 1e-6f);
    // 5. dots (grid 8 + 384 = 392, block 256): ANCHOR_OUT = E tail row base.
    tf_lilicorr_dots<<<392, 256>>>(d_E + (size_t)CH_ROWS * CH_STRIDE, d_E, d_E, d_start, d_pair, CH_STRIDE);
    LAUNCH_OK();
    CUDA_OK(cudaDeviceSynchronize());
    cudaFree(d_E); cudaFree(d_H0); cudaFree(d_WRMS); cudaFree(d_WPROJ); cudaFree(d_BPROJ); cudaFree(d_BIAS); cudaFree(d_QKV);
}

static void control_chain() {
    printf("A9-CHAIN: build+launch\n");
    ChainHost c;
    const size_t nH = (size_t)CH_ROWS * CH_HID;
    rng_seed(0xA9C141A0ull);
    c.h0.resize(nH);
    for (size_t i = 0; i < nH; i++) c.h0[i] = f2bf_rn(rng_normal(0.06));
    c.w_rms.resize(CH_HID);
    for (int i = 0; i < CH_HID; i++) c.w_rms[i] = 0.5f + 0.5f * rng_uniform();
    c.w_proj.resize((size_t)CH_I3 * CH_HID);
    for (size_t i = 0; i < (size_t)CH_I3 * CH_HID; i++) c.w_proj[i] = rng_normal(1.0 / 32.0);
    c.b_proj.resize(CH_I3);
    for (int i = 0; i < CH_I3; i++) c.b_proj[i] = rng_normal(0.1);
    c.bias.resize((size_t)8 * CH_LATTICE * CH_LATTICE);
    for (size_t i = 0; i < c.bias.size(); i++) c.bias[i] = rng_normal(0.5);
    c.anchor.resize(CH_STRIDE / 2);
    for (int i = 0; i < CH_STRIDE / 2; i++) c.anchor[i] = rng_normal(1.0);
    c.stale.resize(CH_STRIDE / 2);
    for (int i = 0; i < CH_STRIDE / 2; i++) c.stale[i] = 0.5f * i - 3.0f;  // the parked stale half-row
    chain_mirror(c);

    float *d_start, *d_pair, *d_start2, *d_pair2;
    CUDA_OK(cudaMalloc(&d_start, 8 * 4));   CUDA_OK(cudaMalloc(&d_pair, (size_t)384 * 4));
    CUDA_OK(cudaMalloc(&d_start2, 8 * 4));  CUDA_OK(cudaMalloc(&d_pair2, (size_t)384 * 4));
    chain_run_device(c, d_start, d_pair);
    chain_run_device(c, d_start, d_pair2);  // determinism probe: pass 2 into fresh pair buffers
    std::vector<float> gs(8), gp(384), gp2(384);
    CUDA_OK(cudaMemcpy(gs.data(), d_start, 8 * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(gp.data(), d_pair, 384 * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(gp2.data(), d_pair2, 384 * 4, cudaMemcpyDeviceToHost));

    int bad = 0, nonden = memcmp(gp.data(), gp2.data(), 384 * 4) != 0;
    double worst = 0.0;
    for (int b = 0; b < 8; b++) {
        const double want = c.start[b], got = gs[b];
        worst = fmax(worst, fabs(got - want));
        if (fabs(got - want) > 5e-3 * (fabs(want) + 0.05) + 1e-3) bad++;
    }
    for (int k = 0; k < 384; k++) {
        const double want = c.pairst[k], got = gp[k];
        worst = fmax(worst, fabs(got - want));
        if (fabs(got - want) > 5e-3 * (fabs(want) + 0.05) + 1e-3) bad++;
    }
    printf("A9-CHAIN: %s cells=%d worst=%.3g %s\n", bad || nonden ? "FAIL" : "MATCH", 8 + 384, worst,
           nonden ? "(nondeterministic)" : "");
    if (bad) g_fail++;
    if (nonden) g_fail++;
    cudaFree(d_start); cudaFree(d_pair); cudaFree(d_start2); cudaFree(d_pair2);
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 8192);
    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    printf("A9-DEVICE: %s cc%d.%d\n", prop.name, prop.major, prop.minor);
    control_factor();
    control_tail();
    control_chain();
    printf("A9-VERDICT: %s (fail=%d)\n", g_fail ? "FAIL" : "ALL-CONTROLS-MATCH", g_fail);
    return g_fail ? 1 : 0;
}

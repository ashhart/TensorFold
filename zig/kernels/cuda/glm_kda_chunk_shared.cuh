// GLM-5.3-Flash's KDA prompt chunks in chunked (WY) form: one block of 256 threads per head runs the prompt

// chunk's 32-row sub-chunks in order (sub-chunks at absolute positions), the 128 x 128 state in registers.

//

// Per row the inputs are glm_kda.cu's (conv + SiLU rounded to bf16, fp32 L2 norms, q times 128^-0.5, beta =

// bf16(sigmoid(b)), the per-channel decay g = e^l, l = lower * sigmoid(exp(A_log) * (a + dt_bias))); the delta rule

// S <- S diag(g); S <- S + beta (v - S k) k^T; o = S q, o rounded to bf16, then the gated RMSNorm. S is [value, key].

//

// A sub-chunk of rows 0..31 entering state S0, E(t -> s) = g_{t+1} ... g_s per key channel (empty: 1):

//

//     L[s, t]  = beta_s sum_c k_s k_t E(t -> s)      t < s          M[s, t] = sum_c q_s k_t E(t -> s)     t <= s

//     T        = (I + L)^-1                           Y = beta (v - (k E(0 -> row)) S0^T),   V' = T Y

//     O        = (q E(0 -> row)) S0^T + M V'           S_end = S0 diag(E(0 -> 31)) + V'^T (k E(row -> 31))

//

// Every decay is a running product of per-row decays <= 1, multiplied in row order as the serial chain decays its

// state (no exponent of a difference of cumulative sums). All products run in fp32 FMAs in a fixed order, so the

// kernel is deterministic, and a row's bits depend only on its sub-chunk's rows and the entering state: the same

// whichever prompt chunk the row came in, as long as prompt chunks start on multiples of the sub-chunk. Rows

// outside the prompt chunk (a short first or last sub-chunk) are zero rows: decay 1, no key, no value, beta 0.

//

// Layout. The state: thread (vg, kp) = (tid / 8, tid % 8) holds value rows 4 vg .. 4 vg + 3 and key quads kp + 8 i

// (i < 4): an 8-lane group reads 128 contiguous bytes of a key row per load, and products with S reduce over the 8

// lanes by a transposing shuffle tree (7 shuffles for 8 sums). Read-outs and corrections: thread (vg, kp) finishes

// value column 4 vg + kp % 4 of the rows of one parity (odd for kp < 4, even otherwise), 16 rows.

// This header: the constants, the shared-memory layout and the sub-chunk helpers; glm_kda_chunk.cu the kernel.

#include <cuda_bf16.h>
#include <cstdint>


constexpr int DK = 128, DV = 128, TAPS = 4;
constexpr int BT = 32;                  // rows of a sub-chunk
constexpr int HALO = TAPS - 1;          // conv rows before a sub-chunk
constexpr int NT = 256, NW = NT / 32;   // threads, warps
constexpr int VPT = 4;                  // value rows of the state a thread holds
constexpr int KP = NT * VPT / DV;       // threads sharing those value rows (8), each 16 keys
constexpr int NQ = DK / 4 / KP;         // key quads a thread holds (4)
constexpr int OWN = BT / 2;             // rows a thread finishes (16, one parity)
constexpr int LS = BT + 4;              // row stride of L
constexpr int PCH = 3 * DK;             // projection channels of a head (q | k | v)
constexpr int PF_A = BT * DK / 8 / NT;  // decay prefetch loads a thread issues (2)
static_assert(KP == 8 && NQ == 4 && PF_A * NT * 8 == BT * DK, "layout");

struct Smem {
    float q[BT][DK];          // q -> q E(s) -> V'
    float k[BT][DK];          // k -> k E(s) -> (q E) S0^T
    float g[BT][DK];          // decays -> Y
    union {                   // staged projection rows (conv input) -> k E(s ->), L, T, M
        __nv_bfloat16 raw_p[BT + HALO][PCH];
        struct {
            float ke[BT][DK];         // k E(s ->)
            float L[BT][LS];          // L[s][t]
            float tt[BT][BT];         // T transposed: tt[t][s] = T[s][t]
            float mt[BT][BT];         // M transposed: mt[t][s] = M[s][t]
        } x;
    } u;
    __nv_bfloat16 v[BT][DV];
    __nv_bfloat16 raw_a[BT][DK];      // staged decay rows
    __nv_bfloat16 cw[PCH][TAPS];      // this head's conv weights
    float raw_b[BT];
    float beta[BT];
    float egl[DK];            // E(31)
};

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ldb(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float fma_(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

// a / b with IEEE rounding, branch-free: the fast path of CUDA's division (a refined reciprocal, then one

// correction), which is the correctly rounded quotient whenever ``div_ok`` holds; callers take a / b for the rare

// operands outside (a huge denominator, a tiny numerator).
__device__ __forceinline__ float div_rn(float a, float b) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(b));
    const float e = __fmaf_rn(-b, r, 1.0f);
    r = __fmaf_rn(e, r, r);
    const float q = a * r;
    const float rem = __fmaf_rn(-b, q, a);
    return __fmaf_rn(rem, r, q);
}

__device__ __forceinline__ bool div_ok(float a, float b) {
    return (a == 0.0f || fabsf(a) > 1e-36f) && b >= 1.0f && b < 8e37f;
}

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

// 8 bf16 from p: one 16-byte load when aligned, else 8 loads.
__device__ __forceinline__ uint4 ld8(const __nv_bfloat16* p, bool vec) {
    if (vec) return *reinterpret_cast<const uint4*>(p);
    const unsigned short* s = reinterpret_cast<const unsigned short*>(p);
    uint4 r;
    r.x = s[0] | ((unsigned)s[1] << 16); r.y = s[2] | ((unsigned)s[3] << 16);
    r.z = s[4] | ((unsigned)s[5] << 16); r.w = s[6] | ((unsigned)s[7] << 16);
    return r;
}

// One level of a transposing shuffle tree: this lane keeps the half of a[0..N) its ``hi`` bit selects, adds the

// partner's (lane ^ mask) copy of that half, and sends the other half.
template <int N>
__device__ __forceinline__ void tree_level(float (&a)[N], int mask, bool hi) {
#pragma unroll
    for (int j = 0; j < N / 2; ++j) {
        const float send = hi ? a[j] : a[j + N / 2];
        const float keep = hi ? a[j + N / 2] : a[j];
        a[j] = keep + __shfl_xor_sync(0xffffffffu, send, mask);
    }
}

// Sums of a[0..8) over the lanes xor 4, 2, 1: lane bits (2, 1, 0) = i gets the sum of a[i].
__device__ __forceinline__ float tree8(float (&a)[8], int lane) {
    tree_level<8>(a, 4, lane & 4);
    float b[4] = {a[0], a[1], a[2], a[3]};
    tree_level<4>(b, 2, lane & 2);
    float c[2] = {b[0], b[1]};
    tree_level<2>(c, 1, lane & 1);
    return c[0];
}

// Sums of a[0..16) over the lanes xor m0 > m1 > m2 > m3: the lane whose bits (m0, m1, m2, m3) spell i gets a[i]'s.
__device__ __forceinline__ float tree16(float (&a)[16], int lane, int m0, int m1, int m2, int m3) {
    tree_level<16>(a, m0, lane & m0);
    float b[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) b[j] = a[j];
    tree_level<8>(b, m1, lane & m1);
    float c[4] = {b[0], b[1], b[2], b[3]};
    tree_level<4>(c, m2, lane & m2);
    float d[2] = {c[0], c[1]};
    tree_level<2>(d, m3, lane & m3);
    return d[0];
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src));
}

// Warm L2 with the rows of the sub-chunk whose row 0 is prompt-chunk row r0 (projection rows r0 - 3 .. r0 + 31 of

// this head's q | k | v channels, its decay rows and beta logits; not when ``next`` is false) and the output gate

// rows from gate_r0.
__device__ __forceinline__ void warm(int tid, int h, int H, int r0, bool next, int rows, int gate_r0,
                                     const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
                                     const __nv_bfloat16* __restrict__ A, int a_stride,
                                     const __nv_bfloat16* __restrict__ G, int g_stride) {
    constexpr int LP = (BT + HALO) * 3 * 2, LA = BT * 2, LB = BT, LG = BT * 2;   // 128-byte lines
    for (int x = next ? tid : tid + LP + LA + LB; x < LP + LA + LB + LG; x += NT) {
        if (x < LP) {
            const int row = x / 6, part = (x % 6) / 2, half = x % 2, r = r0 - HALO + row;
            if (r >= 0 && r < rows) prefetch_l2(P + (size_t)r * p_stride + part * H * DK + h * DK + 64 * half);
        } else if (x < LP + LA) {
            const int y = x - LP, r = r0 + y / 2;
            if (r >= 0 && r < rows) prefetch_l2(A + (size_t)r * a_stride + h * DK + 64 * (y % 2));
        } else if (x < LP + LA + LB) {
            const int r = r0 + (x - LP - LA);
            if (r >= 0 && r < rows) prefetch_l2(P + (size_t)r * p_stride + b_off + h);
        } else {
            const int y = x - LP - LA - LB, r = gate_r0 + y / 2;
            if (r >= 0 && r < rows) prefetch_l2(G + (size_t)r * g_stride + h * DV + 64 * (y % 2));
        }
    }
}

// Stage the rows of the sub-chunk whose row 0 is prompt-chunk row r0 into shared memory: projection rows r0 - 3 ..

// r0 + 31 (the conv state before row 0, zero rows outside), decay rows and the beta logits. Global to shared

// directly (cp.async) when the rows are 16-byte aligned; the caller waits (cp.async.wait_all) and syncs.
__device__ __forceinline__ void stage(Smem& sm, int tid, int h, int H, int r0, int rows,
                                      const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
                                      const __nv_bfloat16* __restrict__ A, int a_stride,
                                      const __nv_bfloat16* __restrict__ cs, bool vec_p, bool vec_a) {
    const int C = 3 * H * DK;
    for (int item = tid; item < (BT + HALO) * PCH / 8; item += NT) {
        const int row = item / (PCH / 8), q8 = item % (PCH / 8);
        const int ch = (q8 / (DK / 8)) * H * DK + h * DK + 8 * (q8 % (DK / 8));
        const int r = r0 - HALO + row;
        uint4* dst = reinterpret_cast<uint4*>(&sm.u.raw_p[row][0]) + q8;
        const __nv_bfloat16* src = nullptr;
        if (r >= 0 && r < rows) src = P + (size_t)r * p_stride + ch;
        else if (r < 0 && r >= -HALO) src = cs + (size_t)(r + HALO) * C + ch;
        if (src == nullptr) *dst = make_uint4(0, 0, 0, 0);
        else if (vec_p) cp16(dst, src);
        else *dst = ld8(src, false);
    }
    for (int item = tid; item < BT * DK / 8; item += NT) {
        const int row = item / (DK / 8), c8 = item % (DK / 8);
        const int r = r0 + row;
        uint4* dst = reinterpret_cast<uint4*>(&sm.raw_a[row][0]) + c8;
        const __nv_bfloat16* src = A + (size_t)r * a_stride + h * DK + 8 * c8;
        if (r < 0 || r >= rows) *dst = make_uint4(0, 0, 0, 0);
        else if (vec_a) cp16(dst, src);
        else *dst = ld8(src, false);
    }
    if (tid < BT) {
        const int r = r0 + tid;
        sm.raw_b[tid] = (r >= 0 && r < rows) ? ldb(P + (size_t)r * p_stride + b_off + h) : 0.0f;
    }
}

// GLM's KDA prompt recurrence in 16-row WY chunks over glm_kda.cu's prep saves; opt-in, -O3 --fmad=false, _rn ops.

// A block a head, chunks from the call's first row: calls cut at multiples of 16 rows keep one call's bits.

// Not the serial row path's bits (other sums): a prompt resumed off a multiple of 16 differs from a fresh one.

// Thread t holds S rows 4 (t / 8) .. + 3 (value), keys 4 (t % 8) + 32 e + 0..3; every sum runs in one fixed order.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
namespace {

constexpr int D = 128;    // key and value dims a head
constexpr int C = 16;     // rows a chunk
constexpr int NT = 256;   // threads a block
constexpr int PARTS = 8;  // threads sharing a value row (key parts)
constexpr int R = 4;      // value rows a thread
constexpr int KG = 4;     // float4 key groups a thread
constexpr unsigned FULL = 0xffffffffu;

// N per-lane partials summed across the warp in one fixed order; lanes 32 j / N .. 32 (j + 1) / N - 1 hold total j.
template <int N>
__device__ __forceinline__ float spread(float (&v)[N], int lane) {
#pragma unroll
    for (int m = 16, h = N / 2; m > 0; m >>= 1, h >>= 1) {
        if (h >= 1) {
            const bool up = lane & m;
#pragma unroll
            for (int i = 0; i < h; ++i) {
                const float give = up ? v[i] : v[h + i], keep = up ? v[h + i] : v[i];
                v[i] = __fadd_rn(keep, __shfl_xor_sync(FULL, give, m));
            }
        } else {
            v[0] = __fadd_rn(v[0], __shfl_xor_sync(FULL, v[0], m));
        }
    }
    return v[0];
}

struct Chunk {
    float q[C][D];  // q, then its decay from the chunk's start times q
    float k[C][D];  // k, then its decay to the chunk's end times k
    float g[C][D];  // decay factors, then (decayed q) S
    float w[C][D];  // beta (decay from the start) k, then W = T of that
    float v[C][D];  // v, then beta v, then U = T (beta v), then U - W S
    float l[C][C];  // beta_i k_i . (decay j+1 .. i) k_j below the diagonal
    float m[C][C];  // q_i . (decay j+1 .. i) k_j on and below it
    float gl[D];    // the chunk's whole decay
    float beta[C];
};

// out[i][v] (sub: minus) x_i . S row v: a thread's keys in order, 8 parts by xor butterfly (same bits in every lane).
__device__ __forceinline__ void rows_dot(const float (*x)[D], const float (&s)[R][KG][4], int p, int vg, bool sub, float (*out)[D]) {
#pragma unroll
    for (int i = 0; i < C; ++i) {
        float acc[R] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int e = 0; e < KG; ++e) {
            const float4 a = *reinterpret_cast<const float4*>(&x[i][4 * p + 32 * e]);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r] = __fmaf_rn(a.x, s[r][e][0], acc[r]);
                acc[r] = __fmaf_rn(a.y, s[r][e][1], acc[r]);
                acc[r] = __fmaf_rn(a.z, s[r][e][2], acc[r]);
                acc[r] = __fmaf_rn(a.w, s[r][e][3], acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 1; m < PARTS; m <<= 1) acc[r] = __fadd_rn(acc[r], __shfl_xor_sync(FULL, acc[r], m));
        float mine = acc[0];
#pragma unroll
        for (int r = 1; r < R; ++r) mine = p == r ? acc[r] : mine;
        if (p < R) out[i][R * vg + p] = sub ? __fsub_rn(out[i][R * vg + p], mine) : mine;
    }
}

// Block (head); q_in null: the state alone (a chunked replay of saved rows, the same state bits), y_out unwritten.
__device__ __forceinline__ void chunk_body(int H, const float* __restrict__ state_in, const float* __restrict__ q_in,
                                           const float* __restrict__ k_save, const __nv_bfloat16* __restrict__ v_save,
                                           const float* __restrict__ g_save, const float* __restrict__ b_save, int rows,
                                           __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    __shared__ __align__(16) Chunk sh;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int h = blockIdx.x;
    const bool out = q_in != nullptr;
    const int vg = t / PARTS, p = t % PARTS;
    const size_t sbase = (size_t)h * D * D + (size_t)R * vg * D + 4 * p;
    float s[R][KG][4];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < KG; ++e) {
            const float4 x = *reinterpret_cast<const float4*>(state_in + sbase + r * D + 32 * e);
            s[r][e][0] = x.x, s[r][e][1] = x.y, s[r][e][2] = x.z, s[r][e][3] = x.w;
        }
    for (int c0 = 0; c0 < rows; c0 += C) {
        const int n = min(C, rows - c0);
        constexpr int LD = C * D / NT;
        float lk[LD], lq[LD], lg[LD], lv[LD]; // every load of the chunk issued before any store
#pragma unroll
        for (int e = 0; e < LD; ++e) { // the chunk's saved rows; rows past the call: k, q, v 0, decay 1, beta 0
            const int x = t + e * NT, i = x / D;
            const size_t at = ((size_t)(c0 + i) * H + h) * D + x % D;
            lk[e] = i < n ? k_save[at] : 0.f;
            lq[e] = i < n && out ? q_in[at] : 0.f;
            lg[e] = i < n ? g_save[at] : 1.f;
            lv[e] = i < n ? __bfloat162float(v_save[at]) : 0.f;
        }
#pragma unroll
        for (int e = 0; e < LD; ++e) {
            const int i = (t + e * NT) / D, d = (t + e * NT) % D;
            sh.k[i][d] = lk[e], sh.q[i][d] = lq[e], sh.g[i][d] = lg[e], sh.v[i][d] = lv[e];
        }
        if (t < C) sh.beta[t] = t < n ? b_save[(c0 + t) * H + h] : 0.f;
        {
            const int i = t / C, j = t % C;
            if (j > i) sh.l[i][j] = 0.f, sh.m[i][j] = 0.f;
            if (j == i) sh.l[i][j] = 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) { // columns j = warp and 15 - warp: e = prod a_(j+1 .. i), a running product per channel
            const int j = jj == 0 ? warp : C - 1 - warp;
            float kj[4], ej[4], part[2 * C];
#pragma unroll
            for (int e = 0; e < 4; ++e) kj[e] = sh.k[j][lane + 32 * e], ej[e] = 1.f;
#pragma unroll
            for (int i = 0; i < C; ++i) {
                float a = 0.f, m = 0.f;
                if (i >= j) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int d = lane + 32 * e;
                        if (i > j) ej[e] = __fmul_rn(ej[e], sh.g[i][d]);
                        const float ke = __fmul_rn(kj[e], ej[e]);
                        a = __fmaf_rn(sh.k[i][d], ke, a);
                        m = __fmaf_rn(sh.q[i][d], ke, m);
                    }
                }
                part[i] = a, part[C + i] = m;
            }
            const float tot = spread<2 * C>(part, lane); // lane i: row i's a, lane 16 + i: row i's m
            const int i = lane % C;
            if (lane < C && j < i) sh.l[i][j] = __fmul_rn(sh.beta[i], tot);
            if (lane >= C && j <= i) sh.m[i][j] = tot;
        }
        __syncthreads();
        if (t < D) { // a channel's prefix products (q, beta k) and suffix products (the keys' decay to the chunk's end)
            float pre = 1.f;
#pragma unroll
            for (int i = 0; i < C; ++i) {
                pre = __fmul_rn(pre, sh.g[i][t]);
                sh.w[i][t] = __fmul_rn(sh.beta[i], __fmul_rn(pre, sh.k[i][t]));
                sh.q[i][t] = __fmul_rn(pre, sh.q[i][t]);
            }
            sh.gl[t] = pre;
            float suf = 1.f;
#pragma unroll
            for (int i = C - 1; i >= 0; --i) {
                sh.k[i][t] = __fmul_rn(sh.k[i][t], suf);
                suf = __fmul_rn(suf, sh.g[i][t]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < C; ++i) sh.v[i][t - D] = __fmul_rn(sh.beta[i], sh.v[i][t - D]);
        }
        __syncthreads();
        { // (I + L) X = R by forward substitution, a thread a column of [W | U]
            float* col = t < D ? &sh.w[0][t] : &sh.v[0][t - D];
            float x[C];
#pragma unroll
            for (int i = 0; i < C; ++i) {
                float acc = col[i * D];
#pragma unroll
                for (int j = 0; j < i; ++j) acc = __fmaf_rn(-sh.l[i][j], x[j], acc);
                x[i] = acc;
            }
#pragma unroll
            for (int i = 0; i < C; ++i) col[i * D] = x[i];
        }
        __syncthreads();
        rows_dot(sh.w, s, p, vg, true, sh.v);           // v: U - W S
        if (out) rows_dot(sh.q, s, p, vg, false, sh.g); // g: (decayed q) S
        __syncthreads();
        if (out) { // row i's read-out (decayed q) S + sum_j<=i M_ij (U - W S)_j: value row t % 128, rows 8 (t / 128) .. + 7
            const int v = t % D;
#pragma unroll
            for (int ii = 0; ii < C / 2; ++ii) {
                const int i = C / 2 * (t / D) + ii;
                float o = sh.g[i][v];
                for (int j = 0; j <= i; ++j) o = __fmaf_rn(sh.m[i][j], sh.v[j][v], o);
                if (i < n) y_out[((size_t)(c0 + i) * H + h) * D + v] = __float2bfloat16_rn(o);
            }
        }
#pragma unroll
        for (int e = 0; e < KG; ++e) { // S = S diag(chunk decay) + sum_j (U - W S)_j (decayed k_j)^T
            const int k = 4 * p + 32 * e;
            const float4 g4 = *reinterpret_cast<const float4*>(&sh.gl[k]);
            float x[R][4];
#pragma unroll
            for (int r = 0; r < R; ++r)
                x[r][0] = __fmul_rn(s[r][e][0], g4.x), x[r][1] = __fmul_rn(s[r][e][1], g4.y), x[r][2] = __fmul_rn(s[r][e][2], g4.z), x[r][3] = __fmul_rn(s[r][e][3], g4.w);
#pragma unroll
            for (int j = 0; j < C; ++j) {
                const float4 kd = *reinterpret_cast<const float4*>(&sh.k[j][k]);
                const float4 vn = *reinterpret_cast<const float4*>(&sh.v[j][R * vg]);
                const float u[R] = {vn.x, vn.y, vn.z, vn.w};
#pragma unroll
                for (int r = 0; r < R; ++r)
                    x[r][0] = __fmaf_rn(kd.x, u[r], x[r][0]), x[r][1] = __fmaf_rn(kd.y, u[r], x[r][1]), x[r][2] = __fmaf_rn(kd.z, u[r], x[r][2]), x[r][3] = __fmaf_rn(kd.w, u[r], x[r][3]);
            }
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) s[r][e][c] = x[r][c];
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < KG; ++e)
            *reinterpret_cast<float4*>(state_out + sbase + r * D + 32 * e) = make_float4(s[r][e][0], s[r][e][1], s[r][e][2], s[r][e][3]);
}

}  // namespace

// glm_kda.cu's step entry, chunked: q, k, v, decay, beta rows [rows][H][128] from prep; y bf16, the state after.
extern "C" __global__ void __launch_bounds__(NT, 2) tf_glm_kda_chunk(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    chunk_body(H, state_in, q_in, k_save, v_save, g_save, b_save, rows, y_out, state_out);
}

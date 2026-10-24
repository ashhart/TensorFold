// Attention inputs for every CUDA family (float64 references: core/qk_ref.zig): QK norm, RoPE and the cache write.

// A block a (row, head), a thread an element; a head's norm sums in one fixed order, never depending on the row count.

#include "qk_rows.cuh"

using namespace tf_qk;

// QkSrc's heads -> q, k, v (bf16); q, k normed by nw, rotate-half RoPE on `rot` dims; k, v also into kc/vc.
extern "C" __global__ void __launch_bounds__(MOST) tf_qk_rows(const QkSrc in, const QkW nw, const float* cosv,
                                                              const float* sinv, const float* inv, int rot, const int* pos,
                                                              const int* slot, const QkDst dst, __nv_bfloat16* kc, __nv_bfloat16* vc,
                                                              long long len, int h, int hk, float eps) {
    __shared__ float warps[MOST / 32];
    const int d = threadIdx.x;
    const int dim = blockDim.x;
    const int64_t r = blockIdx.x;
    const int hd = blockIdx.y;
    const __nv_bfloat16* src = head_src(in, r, hd, h, hk);
    const float x = __bfloat162float(src[d]);
    const int p = pos[r];
    const int64_t pl = p < len ? p : p % len; // a 64-bit modulo only past the cache's end
    const int64_t at = slot != nullptr ? static_cast<int64_t>(slot[r]) * len + pl : pl;
    if (hd >= h + hk) { // v: as is
        const int k = hd - h - hk;
        if (dst.v != nullptr) dst.v[r * dst.v_ld + k * dst.v_hs + d] = src[d];
        if (vc != nullptr) vc[(at * hk + k) * dim + d] = src[d];
        return;
    }
    float y = x;
    float yp = 0.f;
    const int half = rot / 2;
    const int partner = d < half ? d + half : d - half;
    if (nw.q != nullptr) {
        float part = __fmul_rn(x, x);
        for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
        if ((d & 31) == 0) warps[d >> 5] = part;
        __syncthreads();
        float s = 0.f;
        for (int w = 0; w < dim / 32; ++w) s = __fadd_rn(s, warps[w]);
        const float rstd = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(dim)), eps)));
        y = __fmul_rn(__fmul_rn(x, rstd), wv(nw, hd >= h, d));
        if (d < rot) yp = __fmul_rn(__fmul_rn(__bfloat162float(src[partner]), rstd), wv(nw, hd >= h, partner));
    } else if (d < rot) {
        yp = __bfloat162float(src[partner]);
    }
    if ((cosv != nullptr || inv != nullptr) && d < rot) {
        float c, sn;
        rope_at(cosv, sinv, inv, p, rot, d, c, sn);
        y = __fadd_rn(__fmul_rn(y, c), __fmul_rn(d < half ? -yp : yp, sn));
    }
    const __nv_bfloat16 out = __float2bfloat16_rn(y);
    if (hd < h) {
        dst.q[r * dst.q_ld + hd * dst.q_hs + d] = out;
    } else {
        const int k = hd - h;
        dst.k[r * dst.k_ld + k * dst.k_hs + d] = out;
        if (kc != nullptr) kc[(at * hk + k) * dim + d] = out;
    }
}

// tf_qk_rows at dim 128, no or full rotation: a warp a head, lane l dims l + 32 j (same fold order, partner j ^ 2).
extern "C" __global__ void __launch_bounds__(128) tf_qk_rows_w(const QkSrc in, const QkW nw,
                                                               const float* cosv, const float* sinv, const float* inv, int rot,
                                                               const int* pos, const int* slot, const QkDst dst,
                                                               __nv_bfloat16* kc, __nv_bfloat16* vc, long long len,
                                                               int h, int hk, float eps) {
    constexpr int D = 128, J = D / 32;
    const int lane = threadIdx.x & 31;
    const int64_t r = blockIdx.x;
    const int hd = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (hd >= h + (in.v != nullptr ? 2 * hk : hk)) return;
    const __nv_bfloat16* src = head_src(in, r, hd, h, hk);
    const int p = pos[r];
    __nv_bfloat16 raw[J];
#pragma unroll
    for (int j = 0; j < J; ++j) raw[j] = src[lane + 32 * j];
    // every load up front (the rotation's row waits on pos, then on memory)
    float c[J], sn[J], w[J];
    const bool qk = hd < h + hk;
#pragma unroll
    for (int j = 0; j < J; ++j) {
        c[j] = sn[j] = 0.f;
        if (qk && rot != 0 && (cosv != nullptr || inv != nullptr)) rope_at(cosv, sinv, inv, p, D, lane + 32 * j, c[j], sn[j]);
        w[j] = qk && nw.q != nullptr ? wv(nw, hd >= h, lane + 32 * j) : 0.f;
    }
    const int64_t pl = p < len ? p : p % len;
    const int64_t at = slot != nullptr ? static_cast<int64_t>(slot[r]) * len + pl : pl;
    if (hd >= h + hk) { // v: as is
        const int k = hd - h - hk;
#pragma unroll
        for (int j = 0; j < J; ++j) {
            if (dst.v != nullptr) dst.v[r * dst.v_ld + k * dst.v_hs + lane + 32 * j] = raw[j];
            if (vc != nullptr) vc[(at * hk + k) * D + lane + 32 * j] = raw[j];
        }
        return;
    }
    float x[J], y[J], yp[J];
#pragma unroll
    for (int j = 0; j < J; ++j) x[j] = __bfloat162float(raw[j]), y[j] = x[j], yp[j] = 0.f;
    if (nw.q != nullptr) {
        float part[J];
#pragma unroll
        for (int j = 0; j < J; ++j) part[j] = __fmul_rn(x[j], x[j]);
        for (int m = 16; m > 0; m >>= 1)
#pragma unroll
            for (int j = 0; j < J; ++j) part[j] = __fadd_rn(part[j], __shfl_xor_sync(0xFFFFFFFFu, part[j], m));
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < J; ++j) s = __fadd_rn(s, part[j]);
        const float rstd = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(D)), eps)));
#pragma unroll
        for (int j = 0; j < J; ++j) {
            y[j] = __fmul_rn(__fmul_rn(x[j], rstd), w[j]);
            if (rot != 0) yp[j] = __fmul_rn(__fmul_rn(x[j ^ 2], rstd), w[j ^ 2]);
        }
    } else if (rot != 0) {
#pragma unroll
        for (int j = 0; j < J; ++j) yp[j] = x[j ^ 2];
    }
    if (rot != 0 && (cosv != nullptr || inv != nullptr)) {
#pragma unroll
        for (int j = 0; j < J; ++j) y[j] = __fadd_rn(__fmul_rn(y[j], c[j]), __fmul_rn(j < 2 ? -yp[j] : yp[j], sn[j]));
    }
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const __nv_bfloat16 out = __float2bfloat16_rn(y[j]);
        if (hd < h) {
            dst.q[r * dst.q_ld + hd * dst.q_hs + lane + 32 * j] = out;
        } else {
            const int k = hd - h;
            dst.k[r * dst.k_ld + k * dst.k_hs + lane + 32 * j] = out;
            if (kc != nullptr) kc[(at * hk + k) * D + lane + 32 * j] = out;
        }
    }
}

// tf_qk_rows to head dim 512 (qk_round.cuh), and the options it alone has: a unit v norm and MLX's rounding points.
#include "qk_round.cuh"

// tf_qk_rows' arithmetic at head dims to 512; vnorm: v by a unit RMS norm, bf16(x * inv), else as is.
extern "C" __global__ void __launch_bounds__(tf_qk_round::WIDE) tf_qk_rows_once(const QkSrc in, const QkW nw, const float* cosv,
                                                                                const float* sinv, const float* inv, int rot,
                                                                                const int* pos, const int* slot, const QkDst dst,
                                                                                __nv_bfloat16* kc, __nv_bfloat16* vc, long long len,
                                                                                int h, int hk, float eps, int vnorm) {
    tf_qk_round::rows(in, nw, cosv, sinv, inv, rot, pos, slot, dst, kc, vc, len, h, hk, eps, vnorm ? tf_qk_round::VNORM : 0);
}

// As tf_qk_rows_once with MLX's rounding points: q, k = bf16(w * bf16(x * inv)) before RoPE (Metal gemma_norm_rope).
extern "C" __global__ void __launch_bounds__(tf_qk_round::WIDE) tf_qk_rows_round(const QkSrc in, const QkW nw, const float* cosv,
                                                                                 const float* sinv, const float* inv, int rot,
                                                                                 const int* pos, const int* slot, const QkDst dst,
                                                                                 __nv_bfloat16* kc, __nv_bfloat16* vc, long long len,
                                                                                 int h, int hk, float eps, int vnorm) {
    tf_qk_round::rows(in, nw, cosv, sinv, inv, rot, pos, slot, dst, kc, vc, len, h, hk, eps,
                      tf_qk_round::MLX | (vnorm ? tf_qk_round::VNORM : 0));
}

// Dory's attention inputs (qk_l2.cuh): RoPE first, then each head's L2 norm, q times a scale vector, v as is.
#include "qk_l2.cuh"

// q = bf16(qs * bf16(l2(bf16(rope(x))))), k = bf16(l2(bf16(rope(x)))), v as is; k and v also into kc/vc.
extern "C" __global__ void __launch_bounds__(tf_qk_l2::WIDE) tf_qk_rows_l2(const QkSrc in, const float* qs, const float* cosv,
                                                                           const float* sinv, const float* inv, int rot,
                                                                           const int* pos, const int* slot, const QkDst dst,
                                                                           __nv_bfloat16* kc, __nv_bfloat16* vc, long long len,
                                                                           int h, int hk, float eps) {
    tf_qk_l2::rows<tf_qk_l2::ROPE | tf_qk_l2::NORM>(in, qs, cosv, sinv, inv, rot, pos, slot, dst, kc, vc, len, h, hk, eps);
}

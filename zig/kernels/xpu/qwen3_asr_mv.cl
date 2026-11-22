// Qwen3-ASR's one-row Linear in its own image: added to qwen3_asr.cl it moved asr_conv's SIMD width and spill.
#pragma OPENCL FP_CONTRACT OFF

inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

#define GEMV_FMA(c) part = fma(xf.s##c, wf.s##c, part);

// y[o] = bf16(x . w[o]), asr_gemm's bits at m = 1: a sub-group an output, lane l the 16-k steps l, l + 16, ...
__attribute__((reqd_work_group_size(64, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void asr_gemv(__global const ushort *x, __global const ushort *w, __global ushort *y, uint n, uint k) {
    const uint o = get_group_id(0) * 4 + get_sub_group_id(), lane = get_sub_group_local_id();
    if (o >= n) return;
    __global const ushort *wr = w + (ulong)o * k;
    const uint steps = (k + 15) / 16;
    float acc = 0.0f;
    for (uint s0 = 0; s0 < steps; s0 += 16) {
        const uint st = s0 + lane, k0 = st * 16;
        float part = 0.0f;
        if (k0 + 16 <= k) {
            const float16 xf = as_float16(convert_uint16(vload16(0, x + k0)) << 16);
            const float16 wf = as_float16(convert_uint16(vload16(0, wr + k0)) << 16);
            GEMV_FMA(0) GEMV_FMA(1) GEMV_FMA(2) GEMV_FMA(3) GEMV_FMA(4) GEMV_FMA(5) GEMV_FMA(6) GEMV_FMA(7)
            GEMV_FMA(8) GEMV_FMA(9) GEMV_FMA(a) GEMV_FMA(b) GEMV_FMA(c) GEMV_FMA(d) GEMV_FMA(e) GEMV_FMA(f)
        } else if (st < steps) {
            for (uint q = 0; q < 16; q++) {
                const uint kk = k0 + q;
                part = fma(kk < k ? bf(x[kk]) : 0.0f, kk < k ? bf(wr[kk]) : 0.0f, part);
            }
        }
        const uint live = min(16u, steps - s0);
        for (uint j = 0; j < live; j++) acc += sub_group_broadcast(part, j);
    }
    if (lane == 0) y[o] = to_bf(bfr(acc));
}

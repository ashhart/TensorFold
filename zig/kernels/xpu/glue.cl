// Glue ops for the whole-model decode step.
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// Residual add in place: x = bf16(x + d), both bf16 (fp32 sum, one rounding), as the upstream h = (x + delta).to(bf16).
__kernel void add_bf16(__global ushort *x, __global const ushort *d, uint n) {
    const uint i = get_global_id(0);
    if (i < n) x[i] = to_bf(bf(x[i]) + bf(d[i]));
}

// In place: fp32 values rounded to bf16 (nearest even) and widened back, as upstream logits are bf16. NaN stays NaN.
__kernel void round_bf16_f32(__global float *x, uint n) {
    const uint i = get_global_id(0);
    if (i < n && !isnan(x[i])) x[i] = bf(to_bf(x[i]));
}

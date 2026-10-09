// y[y_off + row] = bf16(sum x[i] * w[row][i]); one work-group of 4 sub-groups a row, each sub-group a quarter of the inputs with 16-byte loads and two
// partial sums a lane, fp32 accumulation, the quarters added in order. in_dim a multiple of 512 (the 4 sub-groups x 16 lanes x 8 columns).
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

__attribute__((intel_reqd_sub_group_size(16))) __kernel void mv16x(__global const ushort *w, __global const ushort *x, __global ushort *y, uint in_dim, uint y_off,
                                                                   uint rows) {
    __local float part[4];
    const uint row = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const uint chunks = in_dim / 8, per = chunks / 4; // 8 columns a chunk
    __global const half *wr = (__global const half *)(w + (ulong)row * in_dim);
    float a0 = 0.0f, a1 = 0.0f;
    for (uint c = sg * per + lane; c < (sg + 1) * per; c += 32) {
        const float8 w0 = vload_half8(c, wr);
        const float8 x0 = as_float8(convert_uint8(vload8(c, x)) << 16);
        a0 += dot(w0.s0123, x0.s0123) + dot(w0.s4567, x0.s4567);
        if (c + 16 < (sg + 1) * per) {
            const float8 w1 = vload_half8(c + 16, wr);
            const float8 x1 = as_float8(convert_uint8(vload8(c + 16, x)) << 16);
            a1 += dot(w1.s0123, x1.s0123) + dot(w1.s4567, x1.s4567);
        }
    }
    const float s = sub_group_reduce_add(a0 + a1);
    if (lane == 0) part[sg] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0 && lane == 0) y[y_off + row] = to_bf(part[0] + part[1] + part[2] + part[3]);
}

// LFM3-VL's 8-bit embedding (MLX affine, group 32): token rows and the tied head, attn.cl's contraction by word.
#define QG8 32 // values a scale and bias cover
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// Row `row` of the table dequantized to bf16: y[j] = bf16(scale * q + bias), as embed4 rounds it.
inline void embed8_row(__global const uint *w, __global const ushort *scales, __global const ushort *biases, uint row,
                       __global ushort *y, uint dim) {
    const uint words = dim / 4;
    const uint groups = dim / QG8;
    for (uint wi = get_local_id(0); wi < words; wi += get_local_size(0)) {
        const uint g = wi / (QG8 / 4);
        const float sc = bf(scales[row * groups + g]);
        const float bi = bf(biases[row * groups + g]);
        const uint pk = w[row * words + wi];
        for (uint j = 0; j < 4; j++) y[wi * 4 + j] = to_bf(sc * (float)((pk >> (8 * j)) & 255u) + bi);
    }
}

// Prompt rows: token t's row ids[t] to y + t * dim; grid (tokens).
__kernel void embed8(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                     __global const uint *ids, __global ushort *y, uint dim) {
    const uint t = get_group_id(0);
    embed8_row(w, scales, biases, ids[t], y + t * dim, dim);
}

// One token id as a kernel argument (no host memory is read when the kernel runs).
__kernel void embed8_tok(__global const uint *w, __global const ushort *scales, __global const ushort *biases, uint row,
                         __global ushort *y, uint dim) {
    embed8_row(w, scales, biases, row, y, dim);
}

// fp32 logits y[y_off + row] = sum_i x[i] * (q * scale + bias): one 16-lane sub-group a row, 4 rows a work-group.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv8_f32(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                       __global const ushort *x, __global float *y, uint in_dim, uint x_off, uint y_off, uint rows) {
    const uint row = get_group_id(0) * 4 + get_sub_group_id();
    if (row >= rows) return;
    const uint lane = get_sub_group_local_id();
    const uint words = in_dim / 4;
    const uint groups = in_dim / QG8;
    __global const ushort *xr = x + x_off;
    float acc = 0.0f;
    for (uint wi = lane; wi < words; wi += 16) {
        const uint g = wi / (QG8 / 4);
        const float sc = bf(scales[row * groups + g]);
        const float bi = bf(biases[row * groups + g]);
        const uint pk = w[row * words + wi];
        const float4 xv = as_float4(convert_uint4(vload4(wi, xr)) << 16);
        float dot = 0.0f, sx = 0.0f;
        dot += xv.s0 * (float)(pk & 255u);          sx += xv.s0;
        dot += xv.s1 * (float)((pk >> 8) & 255u);   sx += xv.s1;
        dot += xv.s2 * (float)((pk >> 16) & 255u);  sx += xv.s2;
        dot += xv.s3 * (float)((pk >> 24) & 255u);  sx += xv.s3;
        acc += sc * dot + bi * sx;
    }
    acc = sub_group_reduce_add(acc);
    if (lane == 0) y[y_off + row] = acc;
}

// 4-bit affine (MLX layout) matrix-vector product: y[row] = sum_i x[i] * (q[row][i] * scale + bias), groups of 64.
// One 16-lane sub-group per output row; x is bf16, accumulation fp32.
inline float bf(ushort v) { return as_float((uint)v << 16); }

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qmv4(__global const uint *w, __global const ushort *scales, __global const ushort *biases,
                   __global const ushort *x, __global float *y, uint in_dim) {
    const uint row = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    const uint words = in_dim / 8;
    const uint groups = in_dim / 64;
    float acc = 0.0f;
    for (uint wi = lane; wi < words; wi += 16) {
        const uint g = wi / 8;
        const float sc = bf(scales[row * groups + g]);
        const float bi = bf(biases[row * groups + g]);
        const uint pk = w[row * words + wi];
        float dot = 0.0f, sx = 0.0f;
        for (uint j = 0; j < 8; j++) {
            const float xv = bf(x[wi * 8 + j]);
            dot += xv * (float)((pk >> (4 * j)) & 15u);
            sx += xv;
        }
        acc += sc * dot + bi * sx;
    }
    acc = sub_group_reduce_add(acc);
    if (lane == 0) y[row] = acc;
}

inline float2 sam_norm_product(float2 a, float2 b) {
    const float high = a.x * b.x;
    return sam_two_sum(high, fma(a.x, b.x, -high) + a.x * b.y + a.y * b.x + a.y * b.y);
}

inline float2 sam_norm_sum(float2 value) {
    TF_UNROLL
    for (ushort offset = 16; offset; offset >>= 1) {
        const float high = simd_shuffle_xor(value.x, offset), low = simd_shuffle_xor(value.y, offset);
        sam_pair_add(value, high);
        sam_pair_add(value, low);
    }
    return value;
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void sam_norm_compensated(const device float* x, const device float* weight, const device float* bias,
    device float* out, constant uint2& shape, constant float& eps,
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + sg, width = shape.y;
    if (row >= shape.x) return;
    const ulong base = ulong(row) * width;
    float2 sum(0);
    for (uint col = lane; col < width; col += 32) sam_pair_add(sum, x[base + col]);
    float2 mean = sam_norm_sum(sum);
    const float inverse_width = 1.0f / float(width);
    const float2 divisor(inverse_width, fma(-inverse_width, float(width), 1.0f) / float(width));
    mean = sam_norm_product(mean, divisor);
    float2 squares(0);
    for (uint col = lane; col < width; col += 32) {
        float2 delta = sam_two_sum(x[base + col], -mean.x);
        sam_pair_add(delta, -mean.y);
        const float2 square = sam_norm_product(delta, delta);
        sam_pair_add(squares, square.x);
        sam_pair_add(squares, square.y);
    }
    float2 variance = sam_norm_sum(squares);
    variance = sam_norm_product(variance, divisor);
    sam_pair_add(variance, eps);
    const float inverse = precise::rsqrt(variance.x), square = inverse * inverse;
    const float residual = fma(-variance.x, square, 1.0f) - variance.x * fma(inverse, inverse, -square) - variance.y * square;
    const float2 inv = sam_two_sum(inverse, 0.5f * inverse * residual);
    for (uint col = lane; col < width; col += 32) {
        float2 delta = sam_two_sum(x[base + col], -mean.x);
        sam_pair_add(delta, -mean.y);
        delta = sam_norm_product(delta, inv);
        if (!isfinite(weight[col]) || !isfinite(bias[col])) {
            out[base + col] = (delta.x + delta.y) * weight[col] + bias[col];
        } else {
            sam_pair_scale(delta, weight[col]);
            sam_pair_add(delta, bias[col]);
            out[base + col] = delta.x + delta.y;
        }
    }
}

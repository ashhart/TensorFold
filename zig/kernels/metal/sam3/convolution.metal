#include <metal_stdlib>
using namespace metal;

// Mirrors kernel_abi.ConvShape; its field kernel is spelled kernel_side because kernel is a Metal keyword.
struct Sam3ConvShape { uint in_h, in_w, in_c, out_c, kernel_side, out_h, out_w, stride, pad, first_row, row_count; };

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_conv_pack(const device uint *raw [[buffer(0)]], device uint *packed [[buffer(1)]],
                    constant uint3 &shape [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint outputs = shape.x, inputs = shape.y, taps = shape.z;
    if (i >= outputs * inputs * taps) return;
    const uint channel = i % inputs, tap = (i / inputs) % taps, output = i / (inputs * taps);
    packed[i] = raw[(output * inputs + channel) * taps + tap];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_deconv_pack(const device uint *raw [[buffer(0)]], device uint *packed [[buffer(1)]],
                      constant uint3 &shape [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint outputs = shape.x, inputs = shape.y, taps = shape.z;
    if (i >= taps * outputs * inputs) return;
    const uint channel = i % inputs, output = (i / inputs) % outputs, tap = i / (inputs * outputs);
    packed[i] = raw[(channel * outputs + output) * taps + tap];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_deconv_scatter(const device float *product [[buffer(0)]], const device float *bias [[buffer(1)]],
                         device float *dst [[buffer(2)]], constant Sam3ConvShape &p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.out_h * p.out_w * p.out_c) return;
    const uint output = i % p.out_c, x = (i / p.out_c) % p.out_w, y = i / (p.out_c * p.out_w);
    const uint source = ((y / 2u) * p.in_w + x / 2u) * 4u * p.out_c + (2u * (y % 2u) + x % 2u) * p.out_c + output;
    dst[i] = product[source] + bias[output];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_unfold(const device uint *src [[buffer(0)]], device uint *columns [[buffer(1)]],
                 constant Sam3ConvShape &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint width = p.kernel_side * p.kernel_side * p.in_c;
    if (i >= p.row_count * width) return;
    const uint row = i / width, column = i % width;
    const uint tap = column / p.in_c, channel = column % p.in_c;
    const uint pixel = p.first_row + row;
    const uint oy = pixel / p.out_w, ox = pixel % p.out_w;
    const int y = int(oy * p.stride + tap / p.kernel_side) - int(p.pad);
    const int x = int(ox * p.stride + tap % p.kernel_side) - int(p.pad);
    const bool inside = y >= 0 && x >= 0 && y < int(p.in_h) && x < int(p.in_w);
    columns[i] = inside ? src[(uint(y) * p.in_w + uint(x)) * p.in_c + channel] : 0u;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_depthwise(const device float *src [[buffer(0)]], const device float *weight [[buffer(1)]],
                    const device float *bias [[buffer(2)]], device float *dst [[buffer(3)]],
                    constant Sam3ConvShape &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    const uint channels = p.in_c, k = p.kernel_side;
    if (i >= p.in_h * p.in_w * channels) return;
    const uint channel = i % channels, x = (i / channels) % p.in_w, y = i / (channels * p.in_w);
    float sum = bias[channel];
    for (uint ky = 0; ky < k; ky++) {
        const int sy = int(y + ky) - int(p.pad);
        if (sy < 0 || sy >= int(p.in_h)) continue;
        for (uint kx = 0; kx < k; kx++) {
            const int sx = int(x + kx) - int(p.pad);
            if (sx < 0 || sx >= int(p.in_w)) continue;
            sum += src[(uint(sy) * p.in_w + uint(sx)) * channels + channel] * weight[(channel * k + ky) * k + kx];
        }
    }
    dst[i] = sum;
}

// One threadgroup of 256 threads per group; the tree reduction fixes the summation order.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_group_norm(const device float *src [[buffer(0)]], const device float *weight [[buffer(1)]],
                     const device float *bias [[buffer(2)]], device float *dst [[buffer(3)]],
                     constant uint3 &shape [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sums[256];
    threadgroup float squares[256];
    const uint channels = shape.y, width = channels / shape.z, count = shape.x * width;
    float sum = 0.0f, square = 0.0f;
    for (uint j = t; j < count; j += 256u) {
        const float v = src[(j / width) * channels + group * width + j % width];
        sum += v;
        square += v * v;
    }
    sums[t] = sum;
    squares[t] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_width = 128u; half_width > 0u; half_width >>= 1u) {
        if (t < half_width) {
            sums[t] += sums[t + half_width];
            squares[t] += squares[t + half_width];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = sums[0] / float(count);
    const float variance = max(0.0f, squares[0] / float(count) - mean * mean);
    const float scale = fast::rsqrt(variance + 1e-5f);
    for (uint j = t; j < count; j += 256u) {
        const uint channel = group * width + j % width;
        const uint at = (j / width) * channels + channel;
        dst[at] = (src[at] - mean) * scale * weight[channel] + bias[channel];
    }
}

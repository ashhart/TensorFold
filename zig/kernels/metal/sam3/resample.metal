#include <metal_stdlib>
using namespace metal;

#ifndef SAM3_RESAMPLE_NEAREST_MODE
#define SAM3_RESAMPLE_NEAREST_MODE 0u
#endif
// Samples of magnitude at least 2^126, infinities and NaNs take the expanded interpolation.
#ifndef SAM3_RESAMPLE_EXCEPTIONAL
#define SAM3_RESAMPLE_EXCEPTIONAL(s) any((as_type<uint4>(s) & 0x7fffffffu) >= 0x7e800000u)
#endif

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_resample(const device float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                   constant uint4 &shape [[buffer(2)]], constant uint2 &size [[buffer(3)]],
                   constant float2 &ratio [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    const uint in_h = shape.x, in_w = shape.y, channels = shape.z;
    if (i >= size.x * size.y * channels) return;
    const uint channel = i % channels, ox = (i / channels) % size.y, oy = i / (channels * size.y);
    if (shape.w == SAM3_RESAMPLE_NEAREST_MODE) {
        const uint y = oy * in_h / size.x, x = ox * in_w / size.y;
        dst[i] = src[(y * in_w + x) * channels + channel];
        return;
    }
    const float sy = max(0.0f, fma(float(oy) + 0.5f, ratio.x, -0.5f));
    const float sx = max(0.0f, fma(float(ox) + 0.5f, ratio.y, -0.5f));
    const uint y0 = min(uint(sy), in_h - 1u), x0 = min(uint(sx), in_w - 1u);
    const uint y1 = min(y0 + 1u, in_h - 1u), x1 = min(x0 + 1u, in_w - 1u);
    const float fy = sy - float(y0);
    const float fx = sx - float(x0);
    const float a = src[(y0 * in_w + x0) * channels + channel];
    const float b = src[(y0 * in_w + x1) * channels + channel];
    const float c = src[(y1 * in_w + x0) * channels + channel];
    const float d = src[(y1 * in_w + x1) * channels + channel];
    const float4 samples = float4(a, b, c, d);
    if (SAM3_RESAMPLE_EXCEPTIONAL(samples)) {
        const float top = (1.0f - fx) * a + fx * b;
        const float bottom = (1.0f - fx) * c + fx * d;
        dst[i] = (1.0f - fy) * top + fy * bottom;
        return;
    }
    dst[i] = mix(mix(a, b, fx), mix(c, d, fx), fy);
}

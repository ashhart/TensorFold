#include <metal_stdlib>
using namespace metal;

#ifndef SAM3_FIRST_CONV_PROMPTED
#define SAM3_FIRST_CONV_PROMPTED(flag) (flag != 0u)
#endif

struct Sam3Squash { uint pixels, stride, slots, sources, active, threshold, accurate; float gain, offset; };

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_mask_squash(const device float *logits [[buffer(0)]], device float *dst [[buffer(1)]],
                      constant Sam3Squash &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.pixels * p.slots) return;
    const uint pixel = i / p.slots, slot = i % p.slots;
    const uint at = pixel * p.stride + slot;
    if (((p.active >> slot) & 1u) == 0u) {
        dst[at] = 0.0f;
        return;
    }
    const float l = logits[pixel * p.sources + slot];
    float squashed;
    if (p.threshold != 0u) squashed = l > 0.0f ? 1.0f : 0.0f;
    else if (p.accurate != 0u) squashed = 1.0f / (1.0f + precise::exp(-l));
    else squashed = 1.0f / (1.0f + fast::exp(-l));
    dst[at] = squashed * p.gain + p.offset;
}

// A 3 x 3, stride 2, pad 1 convolution whose only nonzero inputs are channel 0 and, when prompted, channel 16 at 1.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_mask_first_conv(const device float *x [[buffer(0)]], const device float *w [[buffer(1)]],
                          const device float *bias [[buffer(2)]], device float *dst [[buffer(3)]],
                          constant uint &prompted [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= 576u * 576u * 16u) return;
    const uint output = i % 16u, column = (i / 16u) % 576u, row = i / (16u * 576u);
    float sum = 0.0f;
    for (uint ky = 0; ky < 3u; ky++) {
        const int y = int(2u * row + ky) - 1;
        if (y < 0 || y >= 1152) continue;
        for (uint kx = 0; kx < 3u; kx++) {
            const int x_at = int(2u * column + kx) - 1;
            if (x_at < 0 || x_at >= 1152) continue;
            const uint tap = ky * 3u + kx;
            sum = fma(x[uint(y) * 1152u + uint(x_at)], w[output * 32u * 9u + tap], sum);
            if (SAM3_FIRST_CONV_PROMPTED(prompted)) sum = sum + w[(output * 32u + 16u) * 9u + tap];
        }
    }
    dst[i] = sum + bias[output];
}

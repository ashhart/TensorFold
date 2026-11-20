#include <metal_stdlib>
using namespace metal;

struct Sam3Scale { uint count; float factor; };
struct Sam3ChannelFill { uint pixels, channels, first, count, bits; };
struct Sam3Tiles { uint channels, grid, tile, direction; };

// Tiles are enumerated column by column; rows inside a tile are row-major.
inline uint sam3_layout_grid_row(uint row, uint grid, uint tile) {
    const uint across = grid / tile, area = tile * tile;
    const uint index = row / area, local = row % area;
    const uint tx = index / across, ty = index % across;
    return (ty * tile + local / tile) * grid + tx * tile + local % tile;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_copy(const device uint *src [[buffer(0)]], device uint *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
               uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    dst[i] = src[i];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_bias_rows(device float *x [[buffer(0)]], const device float *bias [[buffer(1)]],
                    constant uint2 &shape [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= shape.x) return;
    x[i] = x[i] + bias[i % shape.y];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_columns(const device uint *src [[buffer(0)]], device uint *dst [[buffer(1)]], constant uint4 &shape [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= shape.x * shape.w) return;
    const uint row = i / shape.w, column = i % shape.w;
    dst[i] = src[row * shape.y + shape.z + column];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_transpose(const device uint *src [[buffer(0)]], device uint *dst [[buffer(1)]],
                    constant uint2 &shape [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= shape.x * shape.y) return;
    const uint row = i / shape.y, column = i % shape.y;
    dst[column * shape.x + row] = src[i];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_scale(const device float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant Sam3Scale &p [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    dst[i] = src[i] * p.factor;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_relu(const device float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
               uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    dst[i] = max(src[i], 0.0f);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_sigmoid(const device float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    dst[i] = 1.0f / (1.0f + fast::exp(-src[i]));
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_channel_fill(device float *dst [[buffer(0)]], constant Sam3ChannelFill &p [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.pixels * p.count) return;
    const uint pixel = i / p.count, bit = i % p.count;
    dst[pixel * p.channels + p.first + bit] = ((p.bits >> bit) & 1u) != 0u ? 1.0f : 0.0f;
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_tiles(const device uint *src [[buffer(0)]], device uint *dst [[buffer(1)]], constant Sam3Tiles &p [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
    if (i >= p.grid * p.grid * p.channels) return;
    const uint row = i / p.channels, channel = i % p.channels;
    const uint at = sam3_layout_grid_row(row, p.grid, p.tile) * p.channels + channel;
    if (p.direction == 0u) dst[i] = src[at];
    else dst[at] = src[i];
}

// Adjacent-pair RoPE and YaRN match DeepSeek V4.1's in-place BF16 rotation.
#include <metal_stdlib>
using namespace metal;
struct Dsv41Rope { uint rows, heads, width, rotary, stride, position, step, inverse, spectrum; };
kernel void dsv41_rope(device ushort *x [[buffer(0)]], constant Dsv41Rope &p [[buffer(1)]],
                       device const float2 *factors [[buffer(2)]],
                       uint3 i [[thread_position_in_grid]]) {
    if (i.x >= p.rotary / 2 || i.y >= p.heads || i.z >= p.rows) return;
    // The encoder binds the selected spectrum; rows are absolute source positions.
    float2 rotation = factors[(ulong(p.position) + ulong(i.z) * p.step) * (p.rotary / 2) + i.x];
    float c = rotation.x, s = p.inverse ? -rotation.y : rotation.y;
    uint at = i.z * p.stride + i.y * p.width + p.width - p.rotary + 2 * i.x;
    float a = as_type<float>(uint(x[at]) << 16), b = as_type<float>(uint(x[at + 1]) << 16);
    uint u = as_type<uint>(a * c - b * s), v = as_type<uint>(a * s + b * c);
    x[at] = ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    x[at + 1] = ushort((v + 0x7fff + ((v >> 16) & 1)) >> 16);
}

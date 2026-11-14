// Adjacent-pair RoPE and YaRN match DeepSeek V4.1's in-place BF16 rotation.
#include <metal_stdlib>
using namespace metal;
struct Dsv41Rope { uint rows, heads, width, rotary, stride, position, step, inverse, spectrum; };
static_assert(sizeof(Dsv41Rope) == 36, "Dsv41Rope must mirror rope.zig's Params");
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
// A batched decode window: row z is one slot's one row, rotated at that slot's position `pos[z]` (dsv41_rope's math).
struct Dsv41RopeRows { uint rows, heads, width, rotary, stride, inverse; };
static_assert(sizeof(Dsv41RopeRows) == 24, "Dsv41RopeRows must mirror rope.zig's RowsParams");
kernel void dsv41_rope_rows(device ushort *x [[buffer(0)]], constant Dsv41RopeRows &p [[buffer(1)]],
                            device const float2 *factors [[buffer(2)]], device const uint *pos [[buffer(3)]],
                            uint3 i [[thread_position_in_grid]]) {
    if (i.x >= p.rotary / 2 || i.y >= p.heads || i.z >= p.rows) return;
    float2 rotation = factors[ulong(pos[i.z]) * (p.rotary / 2) + i.x];
    float c = rotation.x, s = p.inverse ? -rotation.y : rotation.y;
    uint at = i.z * p.stride + i.y * p.width + p.width - p.rotary + 2 * i.x;
    float a = as_type<float>(uint(x[at]) << 16), b = as_type<float>(uint(x[at + 1]) << 16);
    uint u = as_type<uint>(a * c - b * s), v = as_type<uint>(a * s + b * c);
    x[at] = ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    x[at + 1] = ushort((v + 0x7fff + ((v >> 16) & 1)) >> 16);
}

#include <metal_stdlib>
using namespace metal;

struct Sam3Rotary { uint rows, heads, head_width, rotated_rows, positions, side; float scale; };
struct Sam3RotarySplit { uint rows, width, head_width, positions; };

// Pair j of a head turns by the x angle for j < head_width/4 and by the y angle above; tables hold (cos, sin).
[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_rotary_table(device float *table [[buffer(0)]], constant Sam3Rotary &p [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    const uint pairs = p.head_width / 2u, quarter = p.head_width / 4u;
    if (i >= p.side * p.side * pairs) return;
    const uint pair = i % pairs, position = i / pairs;
    const float u = pair < quarter ? float(position % p.side) : float(position / p.side);
    const float exponent = -float(pair % quarter) / float(quarter);
    const float theta = u * p.scale * fast::pow(10000.0f, exponent);
    table[2u * i] = fast::cos(theta);
    table[2u * i + 1u] = fast::sin(theta);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_rotate(const device float *x [[buffer(0)]], const device float *table [[buffer(1)]], device float *y [[buffer(2)]],
                 constant Sam3Rotary &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.heads * p.head_width) return;
    const uint row = i / (p.heads * p.head_width), lane = i % p.head_width;
    if (row >= p.rotated_rows) {
        y[i] = x[i];
        return;
    }
    const uint at = ((row % p.positions) * (p.head_width / 2u) + lane / 2u) * 2u;
    const float cosine = table[at], sine = table[at + 1u];
    const float turned = (lane & 1u) != 0u ? sine : -sine;
    y[i] = x[i] * cosine + x[i ^ 1u] * turned;
}

template <typename T>
inline void sam3_rotary_split(const device float *x, const device float *table, device T *q, device T *k, device T *v,
                              constant Sam3RotarySplit &p, uint i) {
    if (i >= p.rows * p.width) return;
    const uint row = i / p.width, column = i % p.width, lane = column % p.head_width;
    const uint a = row * 3u * p.width + column, b = a + p.width;
    const uint at = ((row % p.positions) * (p.head_width / 2u) + lane / 2u) * 2u;
    const float cosine = table[at], sine = table[at + 1u];
    const float turned = (lane & 1u) != 0u ? sine : -sine;
    const float query = x[a] * cosine + x[a ^ 1u] * turned;
    const float key = x[b] * cosine + x[b ^ 1u] * turned;
    q[i] = T(query);
    k[i] = T(key);
    v[i] = T(x[a + 2u * p.width]);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_rotate_split_f32(const device float *x [[buffer(0)]], const device float *table [[buffer(1)]],
                           device float *q [[buffer(2)]], device float *k [[buffer(3)]], device float *v [[buffer(4)]],
                           constant Sam3RotarySplit &p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    sam3_rotary_split<float>(x, table, q, k, v, p, i);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void sam3_rotate_split_f16(const device float *x [[buffer(0)]], const device float *table [[buffer(1)]],
                           device half *q [[buffer(2)]], device half *k [[buffer(3)]], device half *v [[buffer(4)]],
                           constant Sam3RotarySplit &p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    sam3_rotary_split<half>(x, table, q, k, v, p, i);
}

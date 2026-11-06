#pragma once

// Dense BF16 embeddings; the caller checks ids before launch. Invalid indices cannot read outside the table.
extern "C" __global__ void __launch_bounds__(256)
tf_embed_bf16(const uint16_t* table, const int* ids, uint16_t* out, int rows, int width, int vocab) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * width) return;
    const int id = ids[i / width];
    out[i] = id >= 0 && id < vocab ? table[static_cast<long long>(id) * width + i % width] : 0;
}

// Torch BF16 RoPE: cos/sin, both products, and the sum round separately. Q is widened after the last rounding.
// A cache output uses the actual device positions; the query output is [head, row, dim].
extern "C" __global__ void __launch_bounds__(256)
tf_rope_torch_bf16(const uint16_t* x, const int* positions, void* out, int out_kind, int rows,
                   int heads, int dim, float theta, int cache, int total) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * heads * dim) return;
    const int j = i % dim, h = (i / dim) % heads, r = i / (dim * heads);
    const int p = positions[r];
    if (p < 0 || (cache && p >= total)) return;
    const int half = dim / 2, f = j % half;
    const float inv = 1.0f / powf(theta, static_cast<float>(f) / half);
    const float angle = static_cast<float>(p) * inv;
    const float co = rounded(cosf(angle), 2), si = rounded(sinf(angle), 2);
    const long long base = (static_cast<long long>(r) * heads + h) * dim;
    const int partner = j < half ? j + half : j - half;
    const float a = bf16_to_float(x[base + j]);
    const float b = (j < half ? -1.0f : 1.0f) * bf16_to_float(x[base + partner]);
    const float y = rounded(rounded(a * co, 2) + rounded(b * si, 2), 2);
    const long long dst = (static_cast<long long>(h) * (cache ? total : rows) + (cache ? p : r)) * dim + j;
    store(out, out_kind, dst, y);
}

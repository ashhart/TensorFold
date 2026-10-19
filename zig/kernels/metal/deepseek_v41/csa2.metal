// Source is compiled after kv.metal, which provides exact packed-cache decoders.
struct Dsv41Attention {
    uint rows, heads, width, q_stride, position, chunk_rows, window, ring_stride;
    uint chunk_stride, main_stride, main_block, ratio, compressed_rows, topk, index_stride;
    float scale;
};
// BF16 query/key products are exact in FP32; compensate accumulation and lane reduction.
inline void dsv41_dot_add(thread float &sum, thread float &error, float value) {
    float next = sum + value;
    float recovered = next - sum;
    error += (sum - (next - recovered)) + (value - recovered);
    sum = next;
}
inline float dsv41_dot_reduce(float sum, float error) {
    for (uint offset = 16; offset != 0; offset >>= 1) {
        float other = simd_shuffle_xor(sum, offset);
        float other_error = simd_shuffle_xor(error, offset);
        dsv41_dot_add(sum, error, other);
        dsv41_dot_add(sum, error, other_error);
    }
    return sum + error;
}
// BF16 probability rounding uses fixed query-based tiles, never a chunk-dependent physical key layout.
inline uint dsv41_window_columns(constant Dsv41Attention &p, uint row) {
    return p.position + row == 0 ? 1 : p.window;
}
inline device const uchar *dsv41_attn_key(uint slot, uint row, device const uchar *ring,
    device const uchar *chunk, device const uchar *main, device const int *indices,
    constant Dsv41Attention &p, thread uint &kind) {
    uint query = p.position + row;
    uint columns = dsv41_window_columns(p, row);
    if (slot < columns) {
        uint pos;
        if (query == 0) {
            pos = 0;
        } else if (query + 1 < p.window) {
            pos = (query + 1 + slot) % p.window;
        } else {
            uint begin = query + 1 > p.window ? query + 1 - p.window : 0;
            pos = begin + slot;
        }
        if (pos > query) return nullptr;
        kind = 0;
        return pos >= p.position ? chunk + (pos - p.position) * p.chunk_stride
                                : ring + (pos % p.window) * p.ring_stride;
    }
    if (!p.ratio) return nullptr;
    int pos = indices[row * p.index_stride + slot - columns];
    if (pos < 0 || uint(pos) >= min(p.compressed_rows, (query + 1) / p.ratio)) return nullptr;
    kind = 1;
    return main + uint(pos) * p.main_stride;
}
kernel void dsv41_sparse_attention(device const ushort *q [[buffer(0)]], device const uchar *ring [[buffer(1)]],
    device const uchar *chunk [[buffer(2)]], device const uchar *main [[buffer(3)]],
    device const int *indices [[buffer(4)]], device const float *sink [[buffer(5)]],
    device ushort *out [[buffer(6)]], constant Dsv41Attention &p [[buffer(7)]],
    uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint head = g.x, row = g.y;
    if (head >= p.heads || row >= p.rows) return;
    threadgroup float scores[64];
    float query[16], acc[16];
    for (uint j = 0; j < p.width / 32; ++j) {
        query[j] = dsv41_bf(q[row * p.q_stride + head * p.width + lane + j * 32]); acc[j] = 0;
    }
    float mx = -1e30f, sum = 0;
    uint query_position = p.position + row;
    uint compressed_columns = p.ratio ? min(p.topk, min(p.compressed_rows, (query_position + 1) / p.ratio)) : 0;
    uint count = dsv41_window_columns(p, row) + compressed_columns;
    for (uint base = 0; base < count; base += 64) {
        float next_max = mx;
        for (uint j = 0; j < 64; ++j) {
            uint kind = 0;
            device const uchar *key = base + j < count ? dsv41_attn_key(base + j, row, ring, chunk, main, indices, p, kind) : nullptr;
            float dot = 0, dot_error = 0;
            if (key) for (uint d = 0; d < p.width / 32; ++d)
                dsv41_dot_add(dot, dot_error, query[d] * dsv41_unpack(key, lane + d * 32, p.width, kind ? p.main_block : 32, kind));
            dot = key ? dsv41_dot_reduce(dot, dot_error) * p.scale : -INFINITY;
            next_max = max(next_max, dot);
            if (lane == 0) scores[j] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float correction = exp(mx - next_max);
        float tile_sum = 0;
        for (uint d = 0; d < p.width / 32; ++d) acc[d] *= correction;
        for (uint j = 0; j < 64; ++j) {
            float probability = exp(scores[j] - next_max);
            tile_sum += probability;
            float rounded = dsv41_bf(dsv41_round(probability));
            uint kind = 0;
            device const uchar *key = base + j < count ? dsv41_attn_key(base + j, row, ring, chunk, main, indices, p, kind) : nullptr;
            if (key) for (uint d = 0; d < p.width / 32; ++d)
                acc[d] += rounded * dsv41_unpack(key, lane + d * 32, p.width, kind ? p.main_block : 32, kind);
        }
        sum = sum * correction + tile_sum;
        mx = next_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The source adds sink mass to the completed online denominator before the FP32 division.
    float denom = sum + exp(sink[head] - mx);
    for (uint d = 0; d < p.width / 32; ++d)
        out[row * p.q_stride + head * p.width + lane + d * 32] = dsv41_round(acc[d] / denom);
}

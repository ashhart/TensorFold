
inline float im_read(const device uchar* data, ulong i, uint kind) {
    if (kind == 1) return reinterpret_cast<const device float*>(data)[i];
    if (kind == 2) return float(reinterpret_cast<const device half*>(data)[i]);
    return float(reinterpret_cast<const device bfloat*>(data)[i]);
}

inline void im_write(device uchar* data, ulong i, uint kind, float x) {
    if (kind == 1) reinterpret_cast<device float*>(data)[i] = x;
    else if (kind == 2) reinterpret_cast<device half*>(data)[i] = half(x);
    else reinterpret_cast<device bfloat*>(data)[i] = bfloat(x);
}

[[kernel]] void im_cast(const device uchar* x [[buffer(0)]], device uchar* out [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0]) im_write(out, i, p[2], im_read(x, i, p[1]));
}

struct ImPoint {
    uint count;
    uint width;
    uint kind;
    uint operation;
};

[[kernel]] void im_point(const device uchar* a [[buffer(0)]], const device uchar* b [[buffer(1)]],
    device uchar* out [[buffer(2)]], constant ImPoint& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    const float x = im_read(a, i, p.kind);
    const float y = im_read(b, p.operation == 1 || p.operation == 3 ? i % p.width : i, p.kind);
    im_write(out, i, p.kind, p.operation == 4 ? x - y : p.operation >= 2 ? x * y : x + y);
}

struct ImNorm {
    uint width;
    uint kind;
    uint rms;
    uint has_bias;
    float eps;
};

[[max_total_threads_per_threadgroup(256)]] [[kernel]] void im_norm(
    const device uchar* x [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    const device uchar* bias [[buffer(2)]], device uchar* out [[buffer(3)]], constant ImNorm& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[33];
    const ulong base = ulong(row) * p.width;
    float mean = 0;
    if (!p.rms) {
        float sum = 0;
        for (uint d = tid; d < p.width; d += 256) sum += im_read(x, base + d, p.kind);
        mean = qwv_sum(sum, partial, 256, lane, sg) / p.width;
    }
    float squares = 0;
    for (uint d = tid; d < p.width; d += 256) {
        const float value = im_read(x, base + d, p.kind) - mean;
        squares += value * value;
    }
    const float inv = metal::precise::rsqrt(qwv_sum(squares, partial, 256, lane, sg) / p.width + p.eps);
    for (uint d = tid; d < p.width; d += 256) {
        const float value = (im_read(x, base + d, p.kind) - mean) * inv * im_read(weight, d, p.kind);
        im_write(out, base + d, p.kind, value + (p.has_bias ? im_read(bias, d, p.kind) : 0.0f));
    }
}

struct ImHeads { uint rows; uint heads; uint src_dim; uint dst_dim; uint stride; uint offset; uint kind; uint head_stride; };
[[kernel]] void im_heads(const device uchar* x [[buffer(0)]], device uchar* out [[buffer(1)]],
    constant ImHeads& p [[buffer(2)]], uint3 at [[thread_position_in_grid]]) {
    const ulong dest = (ulong(at.z) * p.heads + at.y) * p.dst_dim + at.x;
    const ulong src = ulong(at.z) * p.stride + p.offset + at.y * p.head_stride + at.x;
    im_write(out, dest, p.kind, at.x < p.src_dim ? im_read(x, src, p.kind) : 0.0f);
}

[[kernel]] void im_gelu(const device uchar* x [[buffer(0)]], device uchar* out [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    const float v = im_read(x, i, p[1]);
    const float inner = p[2] ? qwv_erf(v * 0.70710678118654752f)
                            : metal::precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v));
    im_write(out, i, p[1], 0.5f * v * (1.0f + inner));
}

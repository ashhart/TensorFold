// Quantization of the KV cache rows as they are appended: bf16 [rows][4 kv heads][256] -> records [pos][kv head][REC bytes] (see qwen_attn_long.cl):
// 8 blocks of 32 dims, each with an fp16 scale d (q8: d = amax / 127, q = round(x / d); q4 as ggml's q4_0: d = (the signed element of largest magnitude) / -8,
// n = clamp(trunc(x / d + 8.5), 0, 15), value (n - 8) d), the scales after the data. 4-bit values are packed 8 dims a dword in the pair layout: nibble p holds
// dim 2p and nibble p + 4 holds dim 2p + 1 of the dword's 8 dims, so a shift and a mask give a bf16 pair (0x4300 | n = 128 + n).
// One work-item a (row, head, block); global size rows * 32, the row's position is pos0 + row.
inline float bf(ushort v) { return as_float((uint)v << 16); }

// d rounded to fp16 (returned) and its bits
inline float to_half(float d, ushort *bits) {
    vstore_half(d, 0, (__private half *)bits);
    return vload_half(0, (__private half *)bits);
}

__kernel void kvq_quant8(__global const ushort *src, __global uchar *dst, uint pos0) {
    const uint g = get_global_id(0), r = g >> 5, h = (g >> 3) & 3, b = g & 7;
    const __global ushort *x = src + ((ulong)r * 4 + h) * 256 + b * 32;
    float v[32], amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        v[i] = bf(x[i]);
        amax = fmax(amax, fabs(v[i]));
    }
    ushort hb;
    const float d = to_half(amax / 127.0f, &hb);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    __global uchar *rec = dst + ((ulong)(pos0 + r) * 4 + h) * 272;
    for (int i = 0; i < 32; i += 4) {
        uint w = 0;
        for (int j = 0; j < 4; j++) w |= (uint)(uchar)(char)clamp((int)rint(v[i + j] * id), -127, 127) << (8 * j);
        *(__global uint *)(rec + b * 32 + i) = w;
    }
    *(__global ushort *)(rec + 256 + 2 * b) = hb;
}

__kernel void kvq_quant4(__global const ushort *src, __global uchar *dst, uint pos0) {
    const uint g = get_global_id(0), r = g >> 5, h = (g >> 3) & 3, b = g & 7;
    const __global ushort *x = src + ((ulong)r * 4 + h) * 256 + b * 32;
    float v[32], amax = 0.0f, mx = 0.0f;
    for (int i = 0; i < 32; i++) {
        v[i] = bf(x[i]);
        if (fabs(v[i]) > amax) {
            amax = fabs(v[i]);
            mx = v[i];
        }
    }
    ushort hb;
    const float d = to_half(mx / -8.0f, &hb);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    __global uchar *rec = dst + ((ulong)(pos0 + r) * 4 + h) * 144;
    for (int m = 0; m < 4; m++) {
        uint w = 0;
        for (int p = 0; p < 4; p++) {
            const uint n0 = (uint)clamp((int)(v[8 * m + 2 * p] * id + 8.5f), 0, 15);
            const uint n1 = (uint)clamp((int)(v[8 * m + 2 * p + 1] * id + 8.5f), 0, 15);
            w |= (n0 << (4 * p)) | (n1 << (4 * p + 16));
        }
        *(__global uint *)(rec + b * 16 + 4 * m) = w;
    }
    *(__global ushort *)(rec + 128 + 2 * b) = hb;
}

// Shared weight-only projection arithmetic; inputs stay BF16 and every sum stays FP32.
#ifndef TF_WEIGHT_ONLY_BODY
#define TF_WEIGHT_ONLY_BODY
struct TfWeightOnly {
    int rows, n, k, format;
    int group, scale_kind, reserved0, reserved1;
    float global_scale;
};
TF_INLINE float tf_wo_bf(unsigned short bits) { return TF_FROM_BITS(unsigned(bits) << 16); }
TF_INLINE unsigned short tf_wo_round(float x) {
    unsigned bits = TF_TO_BITS(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((bits >> 16) | 0x40u);
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
TF_INLINE float tf_wo_fp8(unsigned char code) {
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    if (exponent == 15 && mantissa == 7) return TF_FROM_BITS(0x7fc00000u);
    float v = exponent == 0 ? TF_LDEXP(float(mantissa), -9) : TF_LDEXP(float(8 + mantissa), exponent - 10);
    return (code & 128) ? -v : v;
}
TF_INLINE float tf_wo_fp4(unsigned code) {
    const unsigned magnitude = code & 7;
    float v = magnitude < 4 ? float(magnitude) * 0.5f : (magnitude == 4 ? 2.0f : magnitude == 5 ? 3.0f : magnitude == 6 ? 4.0f : 6.0f);
    return (code & 8) ? -v : v;
}
TF_INLINE unsigned short tf_wo_u16(TF_DATA const unsigned char* bytes, TF_INDEX i) {
    return (unsigned short)(unsigned(bytes[2 * i]) | (unsigned(bytes[2 * i + 1]) << 8));
}
TF_INLINE float tf_wo_meta(TF_DATA const unsigned char* bytes, TF_INDEX i, int kind) {
    if (kind == 0) return tf_wo_bf(tf_wo_u16(bytes, i));
    if (kind == 1) return TF_HALF(tf_wo_u16(bytes, i));
    unsigned bits = unsigned(bytes[4*i]) | (unsigned(bytes[4*i+1])<<8) | (unsigned(bytes[4*i+2])<<16) | (unsigned(bytes[4*i+3])<<24);
    return TF_FROM_BITS(bits);
}
TF_INLINE float tf_wo_weight(TF_DATA const unsigned char* w, TF_DATA const unsigned char* s,
                            TF_DATA const unsigned char* b, TfWeightOnly p, int col, int k) {
    const TF_INDEX at = (TF_INDEX)col * p.k + k;
    if (p.format == 0) return tf_wo_bf(tf_wo_u16(w, at));
    if (p.format == 3) return tf_wo_fp8(w[at]) * p.global_scale;
    if (p.format == 4) {
        const unsigned code = (w[at / 2] >> ((at & 1) * 4)) & 15;
        const float local_scale = tf_wo_fp8(s[(TF_INDEX)col * (p.k / 16) + k / 16]);
        return (tf_wo_fp4(code) * local_scale) * p.global_scale;
    }
    const int bits = p.format == 1 ? 4 : 8;
    const TF_INDEX byte = at * bits / 8;
    const unsigned code = (w[byte] >> ((at * bits) & 7)) & ((1u << bits) - 1);
    const TF_INDEX group = (TF_INDEX)col * (p.k / p.group) + k / p.group;
    return TF_FMA(float(code), tf_wo_meta(s, group, p.scale_kind), tf_wo_meta(b, group, p.scale_kind));
}
TF_INLINE float tf_wo_partial(TF_DATA const unsigned short* x, TF_DATA const unsigned char* w,
                             TF_DATA const unsigned char* s, TF_DATA const unsigned char* b,
                             TfWeightOnly p, int row, int col, int first, int lane, int control) {
    float sum = 0.0f;
    const int end = first + p.group < p.k ? first + p.group : p.k;
    for (int k = first + lane; k < end; k += 32) {
        if (control == 3 && k == p.k - 1) continue;
        const float a = tf_wo_bf(x[(TF_INDEX)row * p.k + k]);
        const float weight = tf_wo_weight(w, s, b, p, col, k);
        sum = TF_FMA(a, weight, sum);
        if (control == 4) sum = tf_wo_bf(tf_wo_round(sum));
    }
    return sum;
}
#endif

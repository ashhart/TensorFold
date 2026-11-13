#include <metal_stdlib>
using namespace metal;

struct SamDenseShape {
    uint rows, columns, depth;
};

inline float2 sam_two_sum(float a, float b) {
    const float sum = a + b, part = sum - a;
    return float2(sum, (a - (sum - part)) + (b - part));
}

inline void sam_pair_add(thread float2 &sum, float value) {
    const float2 joined = sam_two_sum(sum.x, value);
    sum = sam_two_sum(joined.x, sum.y + joined.y);
}

inline void sam_pair_scale(thread float2 &value, float scale) {
    const float high = value.x * scale;
    value = sam_two_sum(high, fma(value.x, scale, -high) + value.y * scale);
}

inline int sam_binary_exponent(uint bits) {
    const uint exponent = (bits >> 23) & 255u, mantissa = bits & 0x7fffffu;
    return exponent ? int(exponent) - 127 : -126 - int(clz(mantissa) - 8u);
}

inline float sam_binary_mantissa(uint bits) {
    uint mantissa = bits & 0x7fffffu;
    if ((bits & 0x7f800000u) == 0) mantissa <<= clz(mantissa) - 8u;
    return as_type<float>((bits & 0x80000000u) | 0x3f800000u | (mantissa & 0x7fffffu));
}

inline uint sam_exception(uint x, uint y) {
    const uint a = x & 0x7fffffffu, b = y & 0x7fffffffu;
    if (a < 0x7f800000u && b < 0x7f800000u) return 0;
    if (a > 0x7f800000u || b > 0x7f800000u || a == 0 || b == 0) return 4;
    return ((x ^ y) >> 31) ? 2 : 1;
}

inline uint sam_scaled_bits(float2 sum, int exponent) {
    if (sum.x == 0) return as_type<uint>(ldexp(sum.y, exponent));
    const int actual_exponent = sam_binary_exponent(as_type<uint>(sum.x)) + exponent;
    if (actual_exponent >= -126) return as_type<uint>(ldexp(sum.x + sum.y, exponent));
    const float high = ldexp(sum.x, exponent + 149), low = ldexp(sum.y, exponent + 149);
    int rounded = int(rint(high));
    const float remainder = (high - float(rounded)) + low;
    if (remainder > 0.5f || (remainder == 0.5f && (rounded & 1))) rounded++;
    if (remainder < -0.5f || (remainder == -0.5f && (rounded & 1))) rounded--;
    const uint sign = as_type<uint>(sum.x) & 0x80000000u;
    return sign | uint(abs(rounded));
}

template<bool DROP_TAIL, bool KN = false>
inline void sam_dense_small(const device float *x, const device float *weight, const device float *bias,
    device float *out, constant SamDenseShape &shape, uint2 group, uint subgroup, uint lane) {
    const uint column = group.x * 4 + subgroup, row = group.y, depth = shape.depth;
    if (column >= shape.columns || row >= shape.rows) return;
    const device float *a = x + row * depth, *b = weight + (KN ? column : column * depth);
    int largest = -2048, smallest = 2048;
    uint exceptional = 0, subnormal = 0;
    float2 sum = float2(0);
    for (uint k = lane; k < depth; k += 32) {
        const uint left = as_type<uint>(a[k]), right = as_type<uint>(b[k * (KN ? shape.columns : 1)]);
        const uint flag = sam_exception(left, right);
        exceptional |= flag;
        if (flag || (left & 0x7fffffffu) == 0 || (right & 0x7fffffffu) == 0) continue;
        const int exponent = sam_binary_exponent(left) + sam_binary_exponent(right);
        largest = max(largest, exponent);
        smallest = min(smallest, exponent);
        subnormal |= uint((left & 0x7f800000u) == 0 || (right & 0x7f800000u) == 0);
        if (!DROP_TAIL || k + 1 < depth) {
            const float av = as_type<float>(left), bv = as_type<float>(right);
            const float product = av * bv;
            sam_pair_add(sum, product);
            sam_pair_add(sum, fma(av, bv, -product));
        }
    }
    const uint addend = as_type<uint>(bias[column]);
    if (lane == 0) {
        exceptional |= sam_exception(addend, 0x3f800000u);
        if ((addend & 0x7fffffffu) != 0 && (addend & 0x7fffffffu) < 0x7f800000u) {
            const int exponent = sam_binary_exponent(addend);
            largest = max(largest, exponent);
            smallest = min(smallest, exponent);
            subnormal |= uint((addend & 0x7f800000u) == 0);
        }
    }
    exceptional = simd_or(exceptional);
    largest = simd_max(largest);
    smallest = simd_min(smallest);
    subnormal = simd_or(subnormal);
    if (exceptional) {
        if (lane == 0) out[row * shape.columns + column] = as_type<float>((exceptional & 4u) || (exceptional & 3u) == 3u ? 0x7fc00000u : exceptional & 2u ? 0xff800000u : 0x7f800000u);
        return;
    }
    if (largest == -2048) {
        if (lane == 0) out[row * shape.columns + column] = 0;
        return;
    }
    const bool scaled = subnormal || largest > 64 || smallest < -64;
    const int scale = scaled ? largest - 32 : 0;
    if (scaled) {
        sum = float2(0);
        for (uint k = lane; k < depth - uint(DROP_TAIL); k += 32) {
            const uint left = as_type<uint>(a[k]), right = as_type<uint>(b[k * (KN ? shape.columns : 1)]);
            if ((left & 0x7fffffffu) == 0 || (right & 0x7fffffffu) == 0) continue;
            const float av = sam_binary_mantissa(left), bv = sam_binary_mantissa(right);
            const int shift = sam_binary_exponent(left) + sam_binary_exponent(right) - scale;
            const float product = av * bv;
            const float residual = fma(av, bv, -product);
            sam_pair_add(sum, ldexp(product, shift));
            sam_pair_add(sum, ldexp(residual, shift));
        }
    }
    for (ushort offset = 16; offset != 0; offset >>= 1) {
        const float high = simd_shuffle_down(sum.x, offset), low = simd_shuffle_down(sum.y, offset);
        if (lane + offset < 32) {
            sam_pair_add(sum, high);
            sam_pair_add(sum, low);
        }
    }
    if (lane == 0) {
        if ((addend & 0x7fffffffu) != 0) {
            const float value = scaled ? ldexp(sam_binary_mantissa(addend), sam_binary_exponent(addend) - scale) : bias[column];
            sam_pair_add(sum, value);
        }
        out[row * shape.columns + column] = as_type<float>(sam_scaled_bits(sum, scale));
    }
}

#define SAM_DENSE_SMALL(NAME, DROP) \
[[max_total_threads_per_threadgroup(128)]] kernel void NAME(const device float *x, const device float *weight, \
    const device float *bias, device float *out, constant SamDenseShape &shape, uint2 group [[threadgroup_position_in_grid]], \
    uint subgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    sam_dense_small<DROP>(x, weight, bias, out, shape, group, subgroup, lane); \
}
SAM_DENSE_SMALL(tf_sam3_dense_small, false)
SAM_DENSE_SMALL(tf_sam3_dense_small_drop_tail, true)

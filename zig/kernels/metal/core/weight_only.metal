// The build prefixes the shared weight-only header under these Metal scalar and pointer definitions.
#include <metal_stdlib>
using namespace metal;
#define TF_INLINE inline
#define TF_DATA device
#define TF_INDEX ulong
#define TF_FROM_BITS as_type<float>
#define TF_TO_BITS as_type<uint>
#define TF_LDEXP ldexp
#define TF_FMA fma
#define TF_HALF(bits) float(as_type<half>(ushort(bits)))
// TF_WEIGHT_ONLY_HEADER

#define TF_WO_METAL(NAME, CONTROL) \
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void NAME( \
    device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]], \
    device const uchar* s [[buffer(2)]], device const uchar* b [[buffer(3)]], \
    device ushort* out [[buffer(4)]], constant TfWeightOnly& p [[buffer(5)]], \
    constant int& repetition [[buffer(6)]], uint lid [[thread_index_in_threadgroup]], \
    uint2 group [[threadgroup_position_in_grid]]) { \
    const int lane = lid & 31, col = int(group.x) * 4 + int(lid >> 5), row = int(group.y); \
    if (col >= p.n || row >= p.rows) return; \
    float total = 0.0f; \
    for (int first = 0; first < p.k; first += p.group) { \
        float partial = tf_wo_partial(x, w, s, b, p, row, col, first, lane, CONTROL); \
        for (int offset = 16; offset; offset >>= 1) { \
            partial += simd_shuffle_xor(partial, offset); \
            if (CONTROL == 4) partial = tf_wo_bf(tf_wo_round(partial)); \
        } \
        total += partial; if (CONTROL == 4) total = tf_wo_bf(tf_wo_round(total)); \
    } \
    if (CONTROL == 1 && isnan(total)) total = 0.0f; \
    if (CONTROL == 2) total += float(repetition); \
    if (lane == 0) out[ulong(row) * p.n + col] = tf_wo_round(total); \
}
TF_WO_METAL(tf_weight_only_projection, 0)
#ifdef TF_KERNEL_CONTRACT
TF_WO_METAL(tf_weight_only_projection_bad_nan, 1)
TF_WO_METAL(tf_weight_only_projection_bad_repeat, 2)
TF_WO_METAL(tf_weight_only_projection_bad_tail, 3)
TF_WO_METAL(tf_weight_only_projection_bad_sums, 4)
#endif

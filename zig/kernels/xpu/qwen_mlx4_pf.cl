// Prefill GEMM for MLX affine 4-bit (group 64) weights on the matrix engine, after the design of ggml_quant.cl / ggml_pfgemm.cl: the weight matrix is decoded once per call
// into fp16 (w = q * scale + bias in fp32, rounded to fp16) row-major W [rows][in], the activations become fp16 k-major Xt [in][Rp], and the GEMM (ggml_pfgemm.cl, 2D block
// loads, C^T = W X^T) is a chain of intel_sub_group_f16_f16_matrix_mad_k16 over the k tiles in order (fp32 accumulation): a row's value depends on its own activations and
// the weights only, so any chunking of a prompt gives the same bits.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline uint half_pair(float a, float b) {
    return (uint)as_ushort(convert_half_rte(clamp(a, -65504.0f, 65504.0f))) | ((uint)as_ushort(convert_half_rte(clamp(b, -65504.0f, 65504.0f))) << 16);
}

// One work-item a (output row, group of 64 inputs); local size 16 along rows. Row layout: words [rows][in / 8], scales / biases [rows][in / 64]. KL = 0.
// Block layout (qwen_mlx4b.repack): 16-row blocks, group g, [lane][8 words], scales / biases [(rb * groups + g) * 16 + lane]. KL = 1.
#define DEFINE_PFDEC(NAME, BLK) \
__attribute__((reqd_work_group_size(16, 8, 1))) __attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME(__global const uint *w, __global const ushort *sc, __global const ushort *bi, __global uint *Wv, uint in, uint rows) { \
    const uint row = get_global_id(0), gi = get_global_id(1); \
    const uint groups = in / 64, lane = row & 15, g = row >> 4; \
    ulong wi, si; \
    if (BLK) { \
        si = ((ulong)g * groups + gi) * 16 + lane; \
        wi = si * 8; \
    } else { \
        si = (ulong)row * groups + gi; \
        wi = (ulong)row * (in / 8) + gi * 8; \
    } \
    const float s = bf(sc[si]), b = bf(bi[si]); \
    __local uint stg_all[8 * 16 * 33]; \
    __local uint *stg = stg_all + get_local_id(1) * (16 * 33); \
    __local uint *st = stg + lane * 33; \
    const uint8 u = vload8(0, w + wi); \
    _Pragma("unroll") for (int t = 0; t < 4; t++) { \
        const uint kt = gi * 4 + t; \
        const uint x0 = t < 2 ? (t == 0 ? u.s0 : u.s2) : (t == 2 ? u.s4 : u.s6); \
        const uint x1 = t < 2 ? (t == 0 ? u.s1 : u.s3) : (t == 2 ? u.s5 : u.s7); \
        uint8 o8; \
        _Pragma("unroll") for (int d = 0; d < 8; d++) { \
            const uint x = d < 4 ? x0 : x1; \
            const int p = (d & 3) * 2; \
            o8[d] = half_pair(fma((float)((x >> (4 * p)) & 15u), s, b), fma((float)((x >> (4 * p + 4)) & 15u), s, b)); \
        } \
        _Pragma("unroll") for (int e = 0; e < 8; e++) st[t * 8 + e] = o8[e]; \
    } \
    /* the 32 words of the row's 64 inputs go out transposed through local memory: each message writes one 64 B line of one row */ \
    sub_group_barrier(CLK_LOCAL_MEM_FENCE); \
    _Pragma("unroll") for (int r = 0; r < 16; r++) { \
        __global uint *o = Wv + (ulong)(g * 16 + r) * (in / 2) + (ulong)gi * 32; \
        o[lane] = stg[r * 33 + lane]; \
        o[16 + lane] = stg[r * 33 + 16 + lane]; \
    } \
}
DEFINE_PFDEC(pfdec_mlx_row, 0)
DEFINE_PFDEC(pfdec_mlx_blk, 1)

// Activation prep: X [R][K] bf16 -> Xt [K][Rp] fp16 (k-major, the GEMM's B operand; tokens R .. Rp zero). One work-item a (token, 8 k). Grid (Rp / 64, K / 8), local 64.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void pf_prep(__global const ushort *x, __global ushort *xt, uint R, uint K, uint Rp) {
    const uint row = get_global_id(0), kb = get_global_id(1);
    ushort8 v = (ushort8)(0);
    if (row < R) v = vload8(0, x + (ulong)row * K + kb * 8);
    for (uint j = 0; j < 8; j++) xt[(ulong)(kb * 8 + j) * Rp + row] = as_ushort(convert_half_rte(clamp(bf(v[j]), -65504.0f, 65504.0f)));
}

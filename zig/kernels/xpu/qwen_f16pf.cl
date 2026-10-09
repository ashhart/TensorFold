// Activation prep for the 2D-block-load prefill GEMM (ggml_pfgemm.cl) on plain fp16 weights: X [R][K] bf16 -> Xt [K][Rp] fp16 (k-major; tokens R .. Rp zero). One work-item a (token, 8 k):
// a 16 B read, 8 two-byte writes contiguous across the tokens of a sub-group. Grid (Rp / 64, K / 8), local 64. (The same kernel as pf_prep of ggml_quant.cl, which needs no GGUF module.)
inline float bf(ushort v) { return as_float((uint)v << 16); }

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void pf_prep(__global const ushort *x, __global ushort *xt, uint R, uint K, uint Rp) {
    const uint row = get_global_id(0), kb = get_global_id(1);
    ushort8 v = (ushort8)(0);
    if (row < R) v = vload8(0, x + (ulong)row * K + kb * 8);
    for (uint j = 0; j < 8; j++) xt[(ulong)(kb * 8 + j) * Rp + row] = as_ushort(convert_half_rte(clamp(bf(v[j]), -65504.0f, 65504.0f)));
}

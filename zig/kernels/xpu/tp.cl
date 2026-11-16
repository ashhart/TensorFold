// Level Zero P2P collectives for Intel Xe2: rank-ordered FP32 sums (#pragma OPENCL FP_CONTRACT OFF) for exact bits.
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); }

// 1-shot P2P AllReduce (FP16): out[i] = fp16_rte(p0[i] + p1[i])
KERNEL(256) void tp2_allreduce_f16(
    global const half *p0,
    global const half *p1,
    global half *out,
    int n4)
{
    const int idx = get_global_id(0);
    if (idx >= n4) return;
    const float4 s = convert_float4(vload4(idx, p0)) + convert_float4(vload4(idx, p1));
    vstore4(convert_half4_rte(s), idx, out);
}

// 1-shot P2P AllReduce (FP16): out[i] = fp16_rte(((p0[i] + p1[i]) + p2[i]) + p3[i])
KERNEL(256) void tp4_allreduce_f16(
    global const half *p0,
    global const half *p1,
    global const half *p2,
    global const half *p3,
    global half *out,
    int n4)
{
    const int idx = get_global_id(0);
    if (idx >= n4) return;
    float4 s = convert_float4(vload4(idx, p0)) + convert_float4(vload4(idx, p1));
    s = s + convert_float4(vload4(idx, p2));
    s = s + convert_float4(vload4(idx, p3));
    vstore4(convert_half4_rte(s), idx, out);
}

// 2-shot P2P ReduceScatter (FP16): sums slice [off4 .. off4 + slice_n4) into rs_out[0 .. slice_n4)
KERNEL(256) void tp2_rs_f16(
    global const half *p0,
    global const half *p1,
    global half *rs_out,
    int off4,
    int slice_n4)
{
    const int idx = get_global_id(0);
    if (idx >= slice_n4) return;
    const int src = off4 + idx;
    const float4 s = convert_float4(vload4(src, p0)) + convert_float4(vload4(src, p1));
    vstore4(convert_half4_rte(s), idx, rs_out);
}

KERNEL(256) void tp4_rs_f16(
    global const half *p0,
    global const half *p1,
    global const half *p2,
    global const half *p3,
    global half *rs_out,
    int off4,
    int slice_n4)
{
    const int idx = get_global_id(0);
    if (idx >= slice_n4) return;
    const int src = off4 + idx;
    float4 s = convert_float4(vload4(src, p0)) + convert_float4(vload4(src, p1));
    s = s + convert_float4(vload4(src, p2));
    s = s + convert_float4(vload4(src, p3));
    vstore4(convert_half4_rte(s), idx, rs_out);
}

// 1-shot P2P AllReduce (BF16) for BF16/FlashNext activations
KERNEL(256) void tp2_allreduce_bf16(
    global const ushort *p0,
    global const ushort *p1,
    global ushort *out,
    int n4)
{
    const int idx = get_global_id(0);
    if (idx >= n4) return;
    const ushort4 a0 = vload4(idx, p0), a1 = vload4(idx, p1);
    ushort4 r;
    r.s0 = f2bf(bf2f(a0.s0) + bf2f(a1.s0));
    r.s1 = f2bf(bf2f(a0.s1) + bf2f(a1.s1));
    r.s2 = f2bf(bf2f(a0.s2) + bf2f(a1.s2));
    r.s3 = f2bf(bf2f(a0.s3) + bf2f(a1.s3));
    vstore4(r, idx, out);
}

KERNEL(256) void tp4_allreduce_bf16(
    global const ushort *p0,
    global const ushort *p1,
    global const ushort *p2,
    global const ushort *p3,
    global ushort *out,
    int n4)
{
    const int idx = get_global_id(0);
    if (idx >= n4) return;
    const ushort4 a0 = vload4(idx, p0), a1 = vload4(idx, p1), a2 = vload4(idx, p2), a3 = vload4(idx, p3);
    ushort4 r;
    r.s0 = f2bf(((bf2f(a0.s0) + bf2f(a1.s0)) + bf2f(a2.s0)) + bf2f(a3.s0));
    r.s1 = f2bf(((bf2f(a0.s1) + bf2f(a1.s1)) + bf2f(a2.s1)) + bf2f(a3.s1));
    r.s2 = f2bf(((bf2f(a0.s2) + bf2f(a1.s2)) + bf2f(a2.s2)) + bf2f(a3.s2));
    r.s3 = f2bf(((bf2f(a0.s3) + bf2f(a1.s3)) + bf2f(a2.s3)) + bf2f(a3.s3));
    vstore4(r, idx, out);
}

// 1-shot P2P AllReduce (FP32) for FlashNext routed MoE / FP32 partials
KERNEL(256) void tp2_allreduce_f32(
    global const float *p0,
    global const float *p1,
    global float *out,
    int n4)
{
    const int idx = get_global_id(0);
    if (idx >= n4) return;
    vstore4(vload4(idx, p0) + vload4(idx, p1), idx, out);
}

KERNEL(256) void tp4_allreduce_f32(
    global const float *p0,
    global const float *p1,
    global const float *p2,
    global const float *p3,
    global float *out,
    int n4)
{
    const int idx = get_global_id(0);
    if (idx >= n4) return;
    float4 s = vload4(idx, p0) + vload4(idx, p1);
    s = s + vload4(idx, p2);
    s = s + vload4(idx, p3);
    vstore4(s, idx, out);
}

KERNEL(256) void tp4_rs_f32(
    global const float *p0,
    global const float *p1,
    global const float *p2,
    global const float *p3,
    global float *rs_out,
    int off4,
    int slice_n4)
{
    const int idx = get_global_id(0);
    if (idx >= slice_n4) return;
    const int src = off4 + idx;
    float4 s = vload4(src, p0) + vload4(src, p1);
    s = s + vload4(src, p2);
    s = s + vload4(src, p3);
    vstore4(s, idx, rs_out);
}

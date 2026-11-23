//! Dense PLE keeps the maintained CUDA operation order with packed inputs and F32 convolution weight reads.
const std = @import("std");
const edit = @import("geometry_source.zig");
pub fn cuda(a: std.mem.Allocator) ![]u8 {
    const original = @embedFile("../../../kernels/cuda/flashnext_hc.cu");
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var gate = try edit.once(mem, try edit.function(original, "fn_ple_gate("), "fn_ple_gate(", "qwen4_ple_gate(");
    if (std.mem.count(u8, gate, "KEYS[row + d]") != 2) return error.HybridGeometrySourceChanged;
    gate = try std.mem.replaceOwned(u8, mem, gate, "KEYS[row + d]", "KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]");
    gate = try edit.once(mem, gate, "VALS[static_cast<long long>(r) * D + d]", "VALS[static_cast<long long>(r) * (S + 1) * D + d]");
    gate = try edit.once(mem, gate, "__fdiv_rn(dot, __fsqrt_rn(fd))", "__fdiv_rn(rbf(dot), __fsqrt_rn(fd))");
    // Qwen4 has no PDL launch: its copies keep their aliasing promises and start without the grid hooks.
    for ([_][2][]const u8{
        .{ "const __nv_bfloat16* KEYS, const __nv_bfloat16* VALS,", "const __nv_bfloat16* __restrict__ KEYS, const __nv_bfloat16* __restrict__ VALS," },
        .{ "const __nv_bfloat16* H, const float* NK,", "const __nv_bfloat16* __restrict__ H, const float* __restrict__ NK," },
        .{ "const float* NQ, __nv_bfloat16* GATED,", "const float* __restrict__ NQ, __nv_bfloat16* __restrict__ GATED," },
        .{ "float* PSS, float eps, int D, int S) { flashnext_pdl::enter();", "float* __restrict__ PSS, float eps, int D, int S) {" },
    }) |r| gate = try edit.once(mem, gate, r[0], r[1]);
    var conv = try edit.once(mem, try edit.function(original, "fn_ple_conv("), "fn_ple_conv(", "qwen4_ple_conv(");
    conv = try edit.once(mem, conv, "const __nv_bfloat16* CW,", "const float* __restrict__ CW,");
    for ([_][2][]const u8{
        .{ "const __nv_bfloat16* GATED, const float* PSS,", "const __nv_bfloat16* __restrict__ GATED, const float* __restrict__ PSS," },
        .{ "const float* NC, const __nv_bfloat16* TAIL,", "const float* __restrict__ NC, const __nv_bfloat16* __restrict__ TAIL," },
        .{ "__nv_bfloat16* NROW, float eps,", "__nv_bfloat16* __restrict__ NROW, float eps," },
        .{ "int DIL) { flashnext_pdl::enter();", "int DIL) {" },
    }) |r| conv = try edit.once(mem, conv, r[0], r[1]);
    conv = try edit.once(mem, conv, "bf(CW[static_cast<long long>(e) * TAPS + t])", "CW[static_cast<long long>(e) * TAPS + t]");
    const sigmoid = try edit.once(mem, try edit.function(original, "__device__ __forceinline__ float bsig("), "return rbf(tf_sigmoid(x));", "const float e = tf_exp(-fabsf(x)); return rbf(__fdiv_rn(x < 0 ? e : 1.0f, __fadd_rn(1.0f, e)));");
    const silu = try edit.once(mem, try edit.function(original, "__device__ __forceinline__ float bsilu("), "return rbf(tf_silu(x));", "const float e = tf_exp(-fabsf(x)); return rbf(__fdiv_rn(x < 0 ? __fmul_rn(x, e) : x, __fadd_rn(1.0f, e)));");
    const sum = "using tf_common::block_sum_serial;";
    if (std.mem.count(u8, original, sum) != 1) return error.HybridGeometrySourceChanged;
    return std.mem.concat(a, u8, &.{ "#include <cuda_bf16.h>\n#include \"common/glue_math.cuh\"\nusing namespace tf_glue;\n", sigmoid, "\n", silu, "\n", sum, "\n", gate, "\n", conv, "\n" });
}

//! MLX's 120-byte ComputeParams push constants, shared by every stage 1 Vulkan shader.
const std = @import("std");

/// qmm.comp's push constants (std430); shaders read the fields each op sets, the rest keep MLX's defaults.
pub const Params = extern struct {
    count: u32 = 0,
    operation: u32 = 0,
    lhs_size: u32 = 0,
    rhs_size: u32 = 0,
    reduce_size: u32 = 0,
    output_size: u32 = 0,
    lhs_offset: u32 = 0,
    rhs_offset: u32 = 0,
    output_offset: u32 = 0,
    aux_size: u32 = 0,
    aux_offset: u32 = 0,
    matrix_m: u32 = 0,
    matrix_n: u32 = 0,
    matrix_k: u32 = 0,
    flags: u32 = 0,
    alpha: f32 = 1,
    beta: f32 = 0,
    dims: u32 = 0,
    shape: [4]u32 = .{ 0, 0, 0, 0 },
    in_strides: [4]u32 = .{ 0, 0, 0, 0 },
    out_strides: [4]u32 = .{ 0, 0, 0, 0 },
};

/// Push bytes a full-Params pipeline (qmm, dequant, gather_take, copy_general) records.
pub const push_bytes: u32 = 120;
/// cast.comp reads only these first fields (count through output_offset).
pub const cast_push_bytes: u32 = 36;

comptime {
    const names = [_][]const u8{ "count", "operation", "lhs_size", "rhs_size", "reduce_size", "output_size", "lhs_offset", "rhs_offset", "output_offset", "aux_size", "aux_offset", "matrix_m", "matrix_n", "matrix_k", "flags", "alpha", "beta", "dims" };
    for (names, 0..) |name, i| if (@offsetOf(Params, name) != 4 * i) @compileError("qmm.comp push constant moved: " ++ name);
    if (@offsetOf(Params, "shape") != 72 or @offsetOf(Params, "in_strides") != 88 or @offsetOf(Params, "out_strides") != 104) @compileError("qmm.comp push constant arrays moved");
    if (@sizeOf(Params) != push_bytes) @compileError("qmm.comp push constants are 120 bytes");
}

test "cast reads only the prefix the cast pipeline records" {
    try std.testing.expect(cast_push_bytes <= push_bytes);
    try std.testing.expectEqual(@as(u32, 36), cast_push_bytes);
}

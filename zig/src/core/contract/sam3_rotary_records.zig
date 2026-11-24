const M = @import("limits.zig").MetalLaunch;

/// rotary.metal's kernels and their planted controls: 256-thread groups, reserved in the source.
pub const metal = blk: {
    const functions = [_][]const u8{ "sam3_rotary_table", "sam3_rotate", "sam3_rotate_split_f32", "sam3_rotate_split_f16" };
    var out: [2 * functions.len + 1]M = undefined;
    for (functions, 0..) |f, i| {
        out[2 * i] = .{ .name = "sam3 rotary " ++ f, .threads = 256, .tg_memory = 0, .reserved = 256 };
        out[2 * i + 1] = .{ .name = "sam3 rotary control " ++ f, .threads = 256, .tg_memory = 0, .reserved = 256 };
    }
    out[2 * functions.len] = .{ .name = "sam3 rotary sam3_rotate_split_f32_flag", .threads = 256, .tg_memory = 0, .reserved = 256 };
    break :blk out;
};

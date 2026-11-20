const M = @import("limits.zig").MetalLaunch;

/// SAM3's arithmetic kernels, their controls and bias_rows' restoring copy: 256 threads, reserved in the source.
pub const metal = blk: {
    const functions = [_][]const u8{ "sam3_bias_rows", "sam3_scale", "sam3_relu", "sam3_sigmoid", "sam3_mask_squash", "sam3_depthwise", "sam3_group_norm" };
    var out: [2 * functions.len + 1]M = undefined;
    for (functions, 0..) |f, i| {
        // group_norm's two 256-float threadgroup arrays
        const tg = if (i == functions.len - 1) 2048 else 0;
        out[2 * i] = .{ .name = "sam3 arithmetic " ++ f, .threads = 256, .tg_memory = tg, .reserved = 256 };
        out[2 * i + 1] = .{ .name = "sam3 arithmetic control " ++ f, .threads = 256, .tg_memory = tg, .reserved = 256 };
    }
    out[2 * functions.len] = .{ .name = "sam3 arithmetic bias restore sam3_copy", .threads = 256, .tg_memory = 0, .reserved = 256 };
    break :blk out;
};

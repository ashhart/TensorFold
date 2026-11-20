const M = @import("limits.zig").MetalLaunch;

/// SAM3's data-movement kernels and their planted controls: 256-thread groups, reserved in the source.
pub const metal = blk: {
    const functions = [_][]const u8{ "sam3_copy", "sam3_columns", "sam3_transpose", "sam3_tiles", "sam3_channel_fill", "sam3_conv_pack", "sam3_deconv_pack", "sam3_unfold" };
    var out: [2 * functions.len]M = undefined;
    for (functions, 0..) |f, i| {
        out[2 * i] = .{ .name = "sam3 layout " ++ f, .threads = 256, .tg_memory = 0, .reserved = 256 };
        out[2 * i + 1] = .{ .name = "sam3 layout control " ++ f, .threads = 256, .tg_memory = 0, .reserved = 256 };
    }
    break :blk out;
};

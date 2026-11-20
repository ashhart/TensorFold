const M = @import("limits.zig").MetalLaunch;

/// SAM3's flash attention tiles and their planted controls: tensor-op kernels, 128 threads reserved in the source.
pub const metal = blk: {
    const functions = [_][]const u8{ "sam3_flash_f32_d32_k32", "sam3_flash_f32_d64_k32", "sam3_flash_f16_d32_k64", "sam3_flash_f16_d64_k32" };
    var out: [2 * functions.len]M = undefined;
    for (functions, 0..) |f, i| {
        out[2 * i] = .{ .name = "sam3 flash " ++ f, .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false };
        out[2 * i + 1] = .{ .name = "sam3 flash control " ++ f, .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false };
    }
    break :blk out;
};

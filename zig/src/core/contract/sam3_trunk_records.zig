const M = @import("limits.zig").MetalLaunch;

/// SAM3's mixed trunk tiles and their planted controls: tensor-op kernels, 256 threads reserved in the source.
pub const metal = blk: {
    const functions = [_][]const u8{ "sam3_tile_project", "sam3_tile_project_gelu", "sam3_tile_project_residual", "sam3_tile_norm_f16" };
    const tg = [_]u32{ 0, 0, 0, 144 };
    var out: [2 * functions.len]M = undefined;
    for (functions, tg, 0..) |f, bytes, i| {
        out[2 * i] = .{ .name = "sam3 trunk tile " ++ f, .threads = 256, .tg_memory = bytes, .reserved = 256, .offline = false };
        out[2 * i + 1] = .{ .name = "sam3 trunk tile control " ++ f, .threads = 256, .tg_memory = bytes, .reserved = 256, .offline = false };
    }
    break :blk out;
};

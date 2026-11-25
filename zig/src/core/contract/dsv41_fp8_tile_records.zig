//! DeepSeek V4.1 FP8 rows on the MMA tile: up to 16 simdgroups (512 threads, reserved; M1 and M2 build 512).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
const names = [_][]const u8{ "r8_c8_f32", "r8_c8_bf16", "r16_c8_f32", "r16_c8_bf16", "r8_c16_f32", "r8_c16_bf16", "r16_c16_f32", "r16_c16_bf16" };
pub const records = blk: {
    var out: [names.len]M = undefined;
    for (names, 0..) |n, i| out[i] = .{ .name = "dsv41 fp8 tile tf_dsv41_fp8_tile_" ++ n, .threads = 512, .tg_memory = 18432, .reserved = 512, .m1 = 512, .m2 = 512 };
    break :blk out;
};

test "the DeepSeek V4.1 FP8 tile launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}

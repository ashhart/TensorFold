//! DeepSeek V4.1 FP8 dense rows: every launch is one 32-lane simdgroup, which the source reserves.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
const tiers = [_][]const u8{ "r1", "r2", "r4", "r8", "r16" };
pub const records = blk: {
    var out: [2 * tiers.len + 2]M = undefined;
    for (tiers, 0..) |t, i| {
        out[i] = .{ .name = "dsv41 fp8 rows part a tf_dsv41_fp8_a_" ++ t, .threads = 32, .tg_memory = 0, .reserved = 32 };
        out[tiers.len + i] = .{ .name = "dsv41 fp8 rows part b tf_dsv41_fp8_b_" ++ t, .threads = 32, .tg_memory = 0, .reserved = 32 };
    }
    out[2 * tiers.len] = .{ .name = "dsv41 fp8 rows finish tf_dsv41_fp8_finish_f32", .threads = 32, .tg_memory = 0, .reserved = 32 };
    out[2 * tiers.len + 1] = .{ .name = "dsv41 fp8 rows finish tf_dsv41_fp8_finish_bf16", .threads = 32, .tg_memory = 0, .reserved = 32 };
    break :blk out;
};

test "the DeepSeek V4.1 FP8 launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}

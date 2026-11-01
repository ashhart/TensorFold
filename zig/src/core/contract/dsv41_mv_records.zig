//! DeepSeek V4.1 small matvecs: every launch is one 32-lane simdgroup, which the source reserves.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

const tiers = [_][]const u8{ "r1c1", "r2c1", "r4c1", "r4c2", "r8c2", "r16c2" };
const types = [_][]const u8{ "bf16w_f32o", "bf16w_bf16o", "f32w_f32o", "f32w_bf16o" };

pub const records = blk: {
    var out: [types.len * tiers.len]M = undefined;
    for (types, 0..) |ty, i| for (tiers, 0..) |t, j| {
        out[i * tiers.len + j] = .{ .name = "dsv41 mv tf_dsv41_mv_" ++ ty ++ "_" ++ t, .threads = 32, .tg_memory = 0, .reserved = 32 };
    };
    break :blk out;
};

test "the DeepSeek V4.1 matvec launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}

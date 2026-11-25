//! DeepSeek V4.1's EXL3 routed experts: one 32-lane simdgroup a group, which each source reserves.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "dsv41 x3 up tf_dsv41_x3_up", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 x3 down tf_dsv41_x3_down", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 x3 act tf_dsv41_x3_act", .threads = 32, .tg_memory = 0, .reserved = 32 },
};

test "the DeepSeek V4.1 EXL3 expert launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}

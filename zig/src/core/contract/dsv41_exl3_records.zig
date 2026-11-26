//! DeepSeek V4.1's one-row EXL3 projection: one 32-lane simdgroup a group of 32 columns, which the source reserves.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{.{ .name = "dsv41 exl3 row tf_exl3_matmul_row", .threads = 32, .tg_memory = 0, .reserved = 32 }};

test "the DeepSeek V4.1 one-row EXL3 launch fits every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}

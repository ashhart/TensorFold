//! DeepSeek V4.1 Engram launches: every one is a 32-lane simdgroup (or 32 threads), which the source reserves.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 engram embed tf_dsv41_engram_embed", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 engram aq tf_dsv41_engram_aq", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 engram gate tf_dsv41_engram_gate", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 engram add tf_dsv41_engram_add", .threads = 32, .tg_memory = 0, .reserved = 32 },
};

test "the DeepSeek V4.1 Engram launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}

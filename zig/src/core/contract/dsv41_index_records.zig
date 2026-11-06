//! DeepSeek V4.1's indexer scores: prompt rows eight a simdgroup on MMA, decode rows eight simdgroups over one query.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "dsv41 ix scores tf_dsv41_ix_scores", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "dsv41 ix scores tf_dsv41_ix_scores_dec", .threads = 256, .tg_memory = 16640, .reserved = 256 },
};

test "the DeepSeek V4.1 indexer launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}

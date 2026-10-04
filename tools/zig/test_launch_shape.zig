const std = @import("std");
const c = @import("launch_shape");

test "grid ceiling preserves tails and never adds before division" {
    try std.testing.expectEqual(@as(u32, 0), c.tf_launch_blocks(0, 256));
    try std.testing.expectEqual(@as(u32, 1), c.tf_launch_blocks(1, 256));
    try std.testing.expectEqual(@as(u32, 1), c.tf_launch_blocks(256, 256));
    try std.testing.expectEqual(@as(u32, 2), c.tf_launch_blocks(257, 256));
    try std.testing.expectEqual(@as(u32, 65535), c.tf_launch_blocks(std.math.maxInt(u64), 256));
}

test "thread width changes the grid while invalid width refuses" {
    try std.testing.expectEqual(@as(u32, 2), c.tf_launch_blocks(129, 128));
    try std.testing.expectEqual(@as(u32, 0), c.tf_launch_blocks(1, 0));
}

test "empty matrices allocate no elements" {
    var count: u64 = 99;
    try std.testing.expect(c.tf_matrix_count(0, 0, 8, &count));
    try std.testing.expectEqual(@as(u64, 0), count);
}

test "matrix extent rejects element and byte overflow" {
    var count: u64 = 0;
    try std.testing.expect(!c.tf_matrix_count(std.math.maxInt(u64), 2, 8, &count));
    try std.testing.expect(!c.tf_matrix_count(1, std.math.maxInt(u64) / 8 + 1, 8, &count));
    try std.testing.expect(!c.tf_matrix_count(1, 1, 0, &count));
    try std.testing.expect(!c.tf_matrix_count(1, 0, 8, &count));
    try std.testing.expect(c.tf_matrix_count(16, 131072, 8, &count));
    try std.testing.expectEqual(@as(u64, 16 * 131072), count);
}

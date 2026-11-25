const std = @import("std");
const fwd = @import("forward.zig");

test "cache offsets from the tree origin are signed: a ring made below the origin counts back" {
    try std.testing.expectEqual(@as(i64, 128), fwd.elementsFrom(0x1000, 0x1100));
    try std.testing.expectEqual(@as(i64, -128), fwd.elementsFrom(0x1100, 0x1000));
    try std.testing.expectEqual(@as(i64, 0), fwd.elementsFrom(0x7f00_0000_0000, 0x7f00_0000_0000));
}

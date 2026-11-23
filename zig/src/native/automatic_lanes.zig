//! A family-specific ceiling on an automatic request; an explicit count stays exact for admission to refuse.
const std = @import("std");

pub fn bounded(lanes: u32, fixed: bool, limit: u32) u32 {
    return if (fixed) lanes else @min(lanes, limit);
}

test "automatic lane ceilings preserve explicit admission requests" {
    try std.testing.expectEqual(@as(u32, 8), bounded(64, false, 8));
    try std.testing.expectEqual(@as(u32, 32), bounded(64, false, 32));
    try std.testing.expectEqual(@as(u32, 8), bounded(8, true, 8));
    try std.testing.expectEqual(@as(u32, 9), bounded(9, true, 8));
    try std.testing.expectEqual(@as(u32, 32), bounded(32, true, 32));
    try std.testing.expectEqual(@as(u32, 33), bounded(33, true, 32));
    try std.testing.expectEqual(@as(u32, 0), bounded(0, false, 8));
}

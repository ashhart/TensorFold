//! The checkpoint's layer map: linear layers skip every fourth, attention starts at 3.
const std = @import("std");
const weights = @import("cuda_prompt_wt.zig");

test "linear layers skip every fourth index and attention starts at 3" {
    try std.testing.expect(weights.linearLayer(0));
    try std.testing.expect(!weights.linearLayer(3));
    try std.testing.expect(!weights.linearLayer(47));
    try std.testing.expectEqual(@as(usize, 0), weights.linearIndex(0));
    try std.testing.expectEqual(@as(usize, 2), weights.linearIndex(2));
    try std.testing.expectEqual(@as(usize, 35), weights.linearIndex(46));
    try std.testing.expectEqual(@as(usize, 0), weights.attnIndex(3));
    try std.testing.expectEqual(@as(usize, 11), weights.attnIndex(47));
}

//! Layer-index placement checks for the CUDA prompt weight layout.
const std = @import("std");
const weights = @import("cuda_prompt_wt.zig");
const linearLayer = weights.linearLayer;
const linearIndex = weights.linearIndex;
const attnIndex = weights.attnIndex;

test "linear layers skip every fourth index and attention starts at 3" {
    try std.testing.expect(linearLayer(0));
    try std.testing.expect(!linearLayer(3));
    try std.testing.expect(!linearLayer(47));
    try std.testing.expectEqual(@as(usize, 0), linearIndex(0));
    try std.testing.expectEqual(@as(usize, 2), linearIndex(2));
    try std.testing.expectEqual(@as(usize, 35), linearIndex(46));
    try std.testing.expectEqual(@as(usize, 0), attnIndex(3));
    try std.testing.expectEqual(@as(usize, 11), attnIndex(47));
}

//! cuda_reuse.zig's image rule: a kept state resumes only a request with the same images before its position.
const std = @import("std");
const reuse = @import("cuda_reuse.zig");

test "a state resumes only a request with the same images before its position" {
    const d = [_]u64{ 1, 2, 3 };
    const kept = [_]reuse.Seen{.{ .at = 5, .id = d[0] }};
    // a later image is the pass's
    try std.testing.expect(reuse.sameImages(&kept, &.{ .{ .at = 5, .id = d[0] }, .{ .at = 40, .id = d[2] } }, 20));
    // another image under the same ids
    try std.testing.expect(!reuse.sameImages(&kept, &.{.{ .at = 5, .id = d[1] }}, 20));
    try std.testing.expect(!reuse.sameImages(&kept, &.{.{ .at = 6, .id = d[0] }}, 20));
    try std.testing.expect(!reuse.sameImages(&kept, &.{}, 20)); // a text prompt never resumes an image's state
    // nor an image prompt a text state past its image
    try std.testing.expect(!reuse.sameImages(&.{}, &.{.{ .at = 5, .id = d[0] }}, 20));
    try std.testing.expect(reuse.sameImages(&.{}, &.{.{ .at = 5, .id = d[0] }}, 5)); // a text prefix before it
}

//! answer_max: a reply's budget after the think block, beside the reason's own limit.
const std = @import("std");
const stream = @import("stream.zig");

test "answer_max limits the reply after the think block, not the reasoning" {
    const gpa = std.testing.allocator;
    var s = try stream.Stream.init(gpa, .{ .id = "answer", .prompt = &.{1}, .max_new = 12, .think_end = 91, .answer_max = 2 });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 5), try s.commit(gpa, &.{ 5, 6, 7, 8, 91 }, &.{}));
    try std.testing.expect(!s.finished);
    try std.testing.expectEqual(@as(i64, 2), s.budgetLeft());
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 10, 11, 12 }, &.{}));
    try std.testing.expect(s.finished);
    try std.testing.expectEqual(stream.Reason.length, s.reason);
    var long = try stream.Stream.init(gpa, .{ .id = "long", .prompt = &.{1}, .max_new = 4, .think_end = 91, .answer_max = 2 });
    defer long.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 4), try long.commit(gpa, &.{ 5, 6, 7, 8, 9 }, &.{}));
    try std.testing.expectEqual(stream.Reason.length, long.reason);
}

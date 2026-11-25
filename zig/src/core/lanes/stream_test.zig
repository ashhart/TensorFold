//! The stream's commit, loop guard and answer limits.
const std = @import("std");
const stream = @import("stream.zig");
const Stream = stream.Stream;
const Reason = stream.Reason;
const Row = @import("logprob.zig").Row;
const loop_warm_in = stream.loop_warm_in;

test "commit stops at eos and length" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9} });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }, &.{}));
    try std.testing.expectEqualSlices(u32, &.{ 5, 9 }, s.emitted());
    try std.testing.expect(s.finished and s.reason == .stop);
}

test "a logprob stream lands a row with each token it lands, and refuses tokens without rows" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9}, .logprobs = 0 });
    defer s.deinit(gpa);
    try std.testing.expectError(error.LogprobRowsMismatch, s.commit(gpa, &.{5}, &.{}));
    const rows = [_]Row{ .{ .token = 5, .logprob = -0.5 }, .{ .token = 9, .logprob = -1 }, .{ .token = 7, .logprob = -2 } };
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }, &rows));
    try std.testing.expectEqual(@as(usize, 2), s.rows.items.len);
    try std.testing.expectEqual(@as(u32, 9), s.rows.items[1].token);
}

test "answer_max limits the reply after the think block, not the reasoning" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "answer", .prompt = &.{1}, .max_new = 12, .think_end = 91, .answer_max = 2 });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 5), try s.commit(gpa, &.{ 5, 6, 7, 8, 91 }, &.{}));
    try std.testing.expect(!s.finished);
    try std.testing.expectEqual(@as(i64, 2), s.budgetLeft());
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 10, 11, 12 }, &.{}));
    try std.testing.expect(s.finished);
    try std.testing.expectEqual(Reason.length, s.reason);
    var long = try Stream.init(gpa, .{ .id = "long", .prompt = &.{1}, .max_new = 4, .think_end = 91, .answer_max = 2 });
    defer long.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 4), try long.commit(gpa, &.{ 5, 6, 7, 8, 9 }, &.{}));
    try std.testing.expectEqual(Reason.length, long.reason);
}

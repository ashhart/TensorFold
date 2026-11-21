//! Stream commits enforce loop guards and answer limits across thinking boundaries.
const std = @import("std");
const Stream = @import("stream.zig").Stream;
const Reason = @import("stream.zig").Reason;
const loop_warm_in = @import("stream.zig").loop_warm_in;

test "loop guard stops at the cap even when no cycle fires" {
    const gpa = std.testing.allocator;
    var tokens: [20]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i);
    var guarded = try Stream.init(gpa, .{ .id = "guarded", .prompt = &.{1}, .max_new = 10, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer guarded.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 10), try guarded.commit(gpa, &tokens, &.{}));
    try std.testing.expect(guarded.finished and guarded.reason == .length);
}

test "loop guard fires, closes thinking, and latches once" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..257) |_| try tokens.append(gpa, 7);
    var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 400, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items, &.{}));
    try std.testing.expectEqual(@as(u32, 1), s.loop_period.?);
    try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
    while (s.popForce()) |token| _ = try s.commit(gpa, &.{token}, &.{});
    try std.testing.expect(!s.finished and s.force.items.len == 0);
    _ = try s.commit(gpa, &.{8}, &.{});
    try std.testing.expect(!s.finished and s.loop_period.? == 1);
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

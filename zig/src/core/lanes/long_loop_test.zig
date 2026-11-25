//! The loop guard on a stream: its cap, a short cycle's latch, and a long repeated passage's close at one token.
const std = @import("std");
const stream = @import("stream.zig");
const long_loop = @import("long_loop.zig");

/// The emitted length where the guard fired on a repeated 300-token passage, `step` tokens a round.
fn closesAt(step: usize) !usize {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..200) |_| try tokens.append(gpa, 1000 + r.uintLessThan(u32, 50000));
    var passage: [300]u32 = undefined;
    for (&passage) |*t| t.* = 1000 + r.uintLessThan(u32, 50000);
    while (tokens.items.len < 4000) try tokens.append(gpa, passage[(tokens.items.len - 200) % passage.len]);
    var s = try stream.Stream.init(gpa, .{ .id = "long", .prompt = &.{1}, .max_new = 8000, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer s.deinit(gpa);
    var at: usize = 0;
    while (at < tokens.items.len and s.loop_period == null) {
        const window = tokens.items[at..@min(at + step, tokens.items.len)];
        const take = if (stream.loopCut(&s, window)) |cut| window[0 .. cut + 1] else window;
        _ = try s.commit(gpa, take, &.{});
        at += take.len;
    }
    try std.testing.expectEqual(@as(?u32, passage.len), s.loop_period);
    try std.testing.expect(!s.think_open and std.mem.eql(u32, s.force.items, &.{ 90, 91, 92 }));
    return s.emitted().len;
}

test "the loop guard closes a long repeated passage's reasoning at one token, a token at a time or in windows" {
    // a 300-token passage repeats from token 200: 1,024 repeated tokens after its first copy
    const want = 200 + 300 + long_loop.min_repeat;
    try std.testing.expectEqual(want, try closesAt(1));
    try std.testing.expectEqual(want, try closesAt(6));
}

test "loop guard stops at the cap even when no cycle fires" {
    const gpa = std.testing.allocator;
    var tokens: [20]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i);
    var guarded = try stream.Stream.init(gpa, .{ .id = "guarded", .prompt = &.{1}, .max_new = 10, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer guarded.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 10), try guarded.commit(gpa, &tokens, &.{}));
    try std.testing.expect(guarded.finished and guarded.reason == .length);
}

test "loop guard fires, closes thinking, and latches once" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..stream.loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..257) |_| try tokens.append(gpa, 7);
    var s = try stream.Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 400, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items, &.{}));
    try std.testing.expectEqual(@as(u32, 1), s.loop_period.?);
    try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
    while (s.popForce()) |token| _ = try s.commit(gpa, &.{token}, &.{});
    try std.testing.expect(!s.finished and s.force.items.len == 0);
    _ = try s.commit(gpa, &.{8}, &.{});
    try std.testing.expect(!s.finished and s.loop_period.? == 1);
}

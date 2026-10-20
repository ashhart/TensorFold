//! The loop guard on a long repeated passage: the reasoning closes at one token, a token at a time or in windows.
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

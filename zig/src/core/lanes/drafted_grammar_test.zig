//! Structured output through the drafts wrapper: the grammar walks held drafts, and the verify sees what it walked.
const std = @import("std");
const fake = @import("fake.zig");
const be = @import("backend.zig");
const grammar = @import("grammar.zig");
const Sampling = @import("sampling.zig").Sampling;
const dt = @import("drafted_test.zig");

test "a grammar's read of fillers leaves them for the verify: both see the pending token" {
    var r: dt.Restored = undefined;
    const b = try r.init();
    defer r.deinit();
    const s = &r.s;
    const n: u64 = dt.long.len;
    const t = try b.read(try b.first(s, n));
    s.pending = t;
    try b.draft(&.{.{ .stream = s, .follow = &.{}, .first = .{ .value = t }, .rows = null, .start = n, .position = n + 1, .depth = 3 }});
    var walked: [3]u32 = undefined;
    try b.vtable.held.?(b.ptr, s, &walked);
    try std.testing.expectEqualSlices(u32, &.{ t, t, t }, &walked);
    // the verify of the same held window reads the same fillers, not the head's empty chain
    var drawn: [4]u32 = undefined;
    var echo: [3]u32 = undefined;
    const w = [_]be.Window{.{ .stream = s, .pending = t, .held = 3, .tokens = &.{}, .parents = null, .positions = &.{ n + 1, n + 2, n + 3, n + 4 } }};
    var o = [_]be.Verified{.{ .sampled = &drawn, .drafts = &echo }};
    try b.verify(&w, &o);
    try std.testing.expectEqual(@as(u32, 0), r.head.helds);
    try std.testing.expectEqualSlices(u32, &walked, &echo);
}

/// A test grammar: `length` odd tokens, then 96.
const Odd = struct {
    length: u32 = 12,
    at: u32 = 0,
    done: bool = false,

    fn rules(o: *Odd) grammar.Rules {
        return .{ .ptr = o, .vtable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free } };
    }

    fn cast(ptr: *anyopaque) *Odd {
        return @ptrCast(@alignCast(ptr));
    }

    fn allows(o: *const Odd, t: u32) bool {
        if (o.done) return false;
        return if (o.at == o.length) t == 96 else t % 2 == 1;
    }

    fn accept(ptr: *anyopaque, t: u32) anyerror!bool {
        const o = cast(ptr);
        if (!o.allows(t)) return false;
        if (o.at == o.length) o.done = true else o.at += 1;
        return true;
    }

    fn rollback(ptr: *anyopaque, n: usize) anyerror!void {
        const o = cast(ptr);
        for (0..n) |_| {
            if (o.done) o.done = false else o.at -= 1;
        }
    }

    fn terminated(ptr: *anyopaque) bool {
        return cast(ptr).done;
    }

    fn fill(ptr: *anyopaque, words: []u32) anyerror!void {
        const o = cast(ptr);
        @memset(words, 0);
        for (0..fake.vocab) |t| if (o.allows(@intCast(t))) {
            words[t / 32] |= @as(u32, 1) << @intCast(t % 32);
        };
    }

    fn drafts(_: *anyopaque, _: []u32) anyerror!usize {
        return 0;
    }

    fn free(_: *anyopaque) void {}
};

test "structured output through the wrapper: the grammar walks the drafter's held drafts, and drafted == plain" {
    for ([_]?Sampling{ null, dt.sampled }) |s| {
        var outs: [2]dt.Out = undefined;
        for (&outs, [_]dt.Mode{ .wrapped, .plain }) |*o, mode| {
            var g: Odd = .{};
            var k: grammar.Constraint = .init(g.rules(), (fake.vocab + 31) / 32, null);
            o.* = try dt.run(&.{.{ .prompt = &dt.p1, .sampling = s, .drafts = mode == .wrapped, .grammar = &k }}, mode, false);
        }
        defer for (outs) |o| dt.freeOut(o);
        try std.testing.expectEqualSlices(u32, outs[1].tokens[0], outs[0].tokens[0]);
        try std.testing.expect(outs[0].head.helds > 0 and outs[0].drafted > 0);
        var g: Odd = .{};
        for (outs[0].tokens[0]) |t| try std.testing.expect(try Odd.accept(&g, t));
        try std.testing.expect(g.done);
    }
}

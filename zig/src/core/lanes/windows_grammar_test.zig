//! Structured output on a planned window: a tree shaped as a chain grows by the grammar's own drafts as a chain does.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const grammar = @import("grammar.zig");
const win = @import("windows.zig");
const et = @import("engine_test.zig");

const gpa = std.testing.allocator;

/// A test grammar: the `want` tokens in order, then 96; its own drafts are what is left of `want`.
const Spell = struct {
    want: []const u32,
    at: usize = 0,
    done: bool = false,

    fn rules(s: *Spell) grammar.Rules {
        return .{ .ptr = s, .vtable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free } };
    }

    fn cast(ptr: *anyopaque) *Spell {
        return @ptrCast(@alignCast(ptr));
    }

    fn allows(s: *const Spell, t: u32) bool {
        if (s.done) return false;
        return if (s.at == s.want.len) t == 96 else t == s.want[s.at];
    }

    fn accept(ptr: *anyopaque, t: u32) anyerror!bool {
        const s = cast(ptr);
        if (!s.allows(t)) return false;
        if (s.at == s.want.len) s.done = true else s.at += 1;
        return true;
    }

    fn rollback(ptr: *anyopaque, n: usize) anyerror!void {
        const s = cast(ptr);
        for (0..n) |_| {
            if (s.done) s.done = false else s.at -= 1;
        }
    }

    fn terminated(ptr: *anyopaque) bool {
        return cast(ptr).done;
    }

    fn fill(ptr: *anyopaque, words: []u32) anyerror!void {
        const s = cast(ptr);
        @memset(words, 0);
        for (0..fake.vocab) |t| if (s.allows(@intCast(t))) {
            words[t / 32] |= @as(u32, 1) << @intCast(t % 32);
        };
    }

    fn drafts(ptr: *anyopaque, out: []u32) anyerror!usize {
        const s = cast(ptr);
        if (s.done) return 0;
        const n = @min(out.len, s.want.len - s.at);
        @memcpy(out[0..n], s.want[s.at..][0..n]);
        return n;
    }

    fn free(_: *anyopaque) void {}
};

test "a tree shaped as a chain takes the grammar's own drafts as chain rows, and its window builds" {
    var cfg = try et.model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    // a one-node tree, then a two-node chain whose second node the grammar refuses
    for ([_][]const u32{ &.{5}, &.{ 5, 4 } }, [_][]const i32{ &.{-1}, &.{ -1, 0 } }) |tokens, parents| {
        var g: Spell = .{ .want = &.{ 5, 7, 9, 11 } };
        var k: grammar.Constraint = .init(g.rules(), (fake.vocab + 31) / 32, null);
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &prompt, .max_new = 8, .eos = &.{96}, .drafts = true, .grammar = &k });
        defer s.deinit(gpa);
        s.pending = 3;
        var p: win.Plan = .{ .stream = &s, .position = prompt.len, .kind = .fill, .tokens = tokens, .parents = parents };
        try win.constrain(&engine, &p);
        try std.testing.expectEqualSlices(u32, &.{ 5, 7, 9, 11 }, p.tokens);
        try std.testing.expect(p.parents == null);
        try std.testing.expectEqual(@as(u32, 2), p.grammar_from);
        try std.testing.expectEqual(@as(usize, 5 * k.words), p.masks.len);
        const w = try win.build(&engine, p, false);
        try std.testing.expectEqualSlices(u64, &.{ 6, 7, 8, 9, 10 }, w.positions);
        try std.testing.expect(w.parents == null);
        try std.testing.expectEqual(@as(usize, 0), g.at);
    }
}

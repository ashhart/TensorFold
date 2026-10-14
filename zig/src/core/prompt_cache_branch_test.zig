//! Branch points: a state its conversation resumes again after a later turn extended it is not superseded by that turn.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_test.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;
const Store = pc.Store;

/// A chat prompt: `base`, `n` new tokens of `seed`, then a 7-token generation prompt (its history is the rest).
fn chat(a: std.mem.Allocator, base: []const u32, seed: u32, n: usize) ![]u32 {
    const out = try a.alloc(u32, base.len + n + 7);
    @memcpy(out[0..base.len], base);
    for (out[base.len..][0..n], 0..) |*t, i| t.* = seed << 20 | @as(u32, @intCast(i));
    for (out[base.len + n ..], 0..) |*t, i| t.* = @intCast(1000 + i);
    return out;
}

/// `n` tokens counting up from `from`.
fn seq(comptime n: usize, from: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = from + @as(u32, @intCast(i));
    return out;
}

/// One request as a host runs it: begin (lookup, restore, room), then the pass keeping at every mark.
fn request(f: *Fake, s: *Store, a: std.mem.Allocator, prompt: []const u32) !u32 {
    const p = try s.begin(a, prompt, @intCast(prompt.len - 7), &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(fresh(prompt), f.pass(s, prompt, p));
    return p.from;
}

test "a history every turn resumes stays kept when a 6k prompt needs room: its turns branch from it, not past it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{}, 310_000); // fake bytes 100 + at: three ~100k states and no 6k one
    defer s.deinit();
    const history = try a.alloc(u32, 100_117);
    for (history, 1..) |*t, i| t.* = @intCast(i);
    try std.testing.expectEqual(@as(u32, 0), try request(&f, &s, a, try chat(a, history, 0, 0)));
    for (1..3) |k| { // two turns on that history: each resumes it and keeps its own longer state
        try std.testing.expectEqual(@as(u32, 100_117), try request(&f, &s, a, try chat(a, history, @intCast(k), 2000)));
    }
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    try std.testing.expectEqual(@as(u32, 0), try request(&f, &s, a, try chat(a, &.{}, 9, 6000))); // another prompt
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted); // the first turn's state went
    try std.testing.expectEqual(@as(u32, 100_117), try request(&f, &s, a, try chat(a, history, 3, 2000)));
    try std.testing.expect(s.held <= s.budget);
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "a state resumed once and then extended is still superseded: a conversation that moved on frees it first" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 3 }, 400); // fake bytes 100 + at
    defer s.deinit();
    const other = seq(20, 50); // history 13: 113
    try std.testing.expectEqual(@as(u32, 0), try request(&f, &s, a, &other)); // the least recently used state
    const base = seq(20, 1); // 113 more
    try std.testing.expectEqual(@as(u32, 0), try request(&f, &s, a, &base));
    const grown = try chat(a, base[0..13], 2, 6); // resumes 13, keeps 19: 345 held
    try std.testing.expectEqual(@as(u32, 13), try request(&f, &s, a, grown));
    const third = seq(20, 70);
    try std.testing.expectEqual(@as(u32, 0), try request(&f, &s, a, &third)); // 113 more: the superseded 13 goes
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expect(s.find(&base, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(u32, 19), s.find(grown, &.{}, &.{}).?.at);
    try std.testing.expectEqual(@as(u32, 13), s.find(&other, &.{}, &.{}).?.at);
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "a turn's rewind state is not superseded by the history state its own prompt kept: an edit still resumes it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 3 }, 400); // fake bytes 100 + at
    defer s.deinit();
    const other = seq(20, 50); // history 13: 113, the least recently used state
    try std.testing.expectEqual(@as(u32, 0), try request(&f, &s, a, &other));
    const first = seq(10, 1); // history 8, its latest user message at 4: 104 and 108 more
    var p = try s.beginRewind(a, &first, 8, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 4, 8 }, p.marks);
    try std.testing.expectEqual(fresh(&first), f.pass(&s, &first, p));
    const third = seq(20, 70); // 113 more: the oldest goes, not the rewind state its own history extends
    try std.testing.expectEqual(@as(u32, 0), try request(&f, &s, a, &third));
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expect(s.find(&other, &.{}, &.{}) == null);
    const edited = [_]u32{ 1, 2, 3, 4, 50, 6, 7, 8, 9, 10 };
    p = try s.beginRewind(a, &edited, 8, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4), p.from);
    try std.testing.expectEqual(fresh(&edited), f.pass(&s, &edited, p));
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

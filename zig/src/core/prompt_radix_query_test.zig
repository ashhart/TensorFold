//! The store under a query's reuse identity (C6, #2444): its chunk schedule, its key's length, its prefilled rows.
const std = @import("std");
const radix = @import("prompt_radix.zig");
const testing = std.testing;

/// Pages whose holders are counted, and states that are the position they were kept at; a page is 4 tokens.
const Fake = struct {
    refs: [64]u32 = @splat(0),
    states: [16]u32 = @splat(0),
    nstates: usize = 0,

    fn of(p: *anyopaque) *Fake {
        return @ptrCast(@alignCast(p));
    }
    fn retain(p: *anyopaque, id: u32) void {
        of(p).refs[id] += 1;
    }
    fn release(p: *anyopaque, id: u32) void {
        of(p).refs[id] -= 1;
    }
    fn holders(p: *anyopaque, id: u32) u32 {
        return of(p).refs[id];
    }
    fn available(p: *anyopaque) usize {
        var n: usize = 0;
        for (of(p).refs) |r| n += @intFromBool(r == 0);
        return n;
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(p: *anyopaque, _: ?*anyopaque, at: u32) anyerror!radix.Saved {
        const f = of(p);
        f.states[f.nstates] = at;
        f.nstates += 1;
        return @ptrCast(&f.states[f.nstates - 1]);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: radix.Saved) anyerror!void {}
    fn dropState(_: *anyopaque, _: radix.Saved) void {}
    fn pages(f: *Fake) radix.Pages {
        return .{ .ptr = f, .bytes = 1, .vtable = &.{ .retain = retain, .release = release, .holders = holders, .available = available } };
    }
    fn family(f: *Fake) radix.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = dropState } };
    }
    /// A stream's own pages: one holder each.
    fn take(f: *Fake, ids: []const u32) void {
        for (ids) |id| f.refs[id] = 1;
    }
    fn end(f: *Fake, ids: []const u32) void {
        for (ids) |id| f.refs[id] -= 1;
    }
};

const planned: radix.Rules = .{ .page = 4, .planned = true, .grid = 8, .lookahead = 0, .min_gap = 1, .min_prompt = 0 };

fn seq(comptime n: usize) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = @intCast(i + 1);
    return out;
}

fn with(prompt: []const u32, starts: []const u32) radix.Query {
    var q: radix.Query = .of(prompt);
    q.starts = starts;
    return q;
}

test "a request's own start off the grid is kept and resumed; without starts the grid alone decides" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), planned);
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 8 });
    const prompt = seq(20);
    const mine = [_]u32{ 0, 1, 2 };
    f.take(&mine);
    defer f.end(&mine);
    var path: [3]u32 = undefined;
    try testing.expect(!s.keep(.of(&prompt), 12, null, &mine, &path).held); // 12 is off the grid of 8
    try testing.expect(s.keep(with(&prompt, &.{12}), 12, null, &mine, &path).held); // the request's own start
    try testing.expectEqual(@as(u32, 12), s.find(with(&prompt, &.{12})).?.end);
    try testing.expect(s.find(.of(&prompt)) == null); // a request on the grid never resumes inside its chunk at 12
}

test "a grid point inside a request's own chunk is never its resume point, nor a mark it keeps" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), planned);
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 8 });
    const prompt = seq(24);
    const mine = [_]u32{ 0, 1, 2, 3 };
    f.take(&mine);
    defer f.end(&mine);
    var path: [4]u32 = undefined;
    try testing.expect(s.keep(.of(&prompt), 16, null, &mine, &path).held); // on the grid
    const own = with(&prompt, &.{ 12, 20 }); // chunks [12, 20) and [20, ...): 16 is inside one
    try testing.expect(s.find(own) == null);
    try testing.expect(!s.keep(own, 16, null, &mine, &path).held);
    const marks = try s.marks(testing.allocator, own, 0, 23, &.{}, &.{});
    defer testing.allocator.free(marks);
    try testing.expectEqualSlices(u32, &.{20}, marks); // the history's 23 floors to the start at 20
}

test "a key of another length than its prompt reuses nothing and keeps nothing" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), .{ .page = 4, .lookahead = 0, .min_gap = 1, .min_prompt = 0 });
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 8 });
    const prompt = seq(12);
    const mine = [_]u32{ 0, 1 };
    f.take(&mine);
    defer f.end(&mine);
    var path: [2]u32 = undefined;
    try testing.expect(s.keep(.of(&prompt), 8, null, &mine, &path).held);
    var q: radix.Query = .of(&prompt);
    q.prompt_len = 11; // the key names more rows than the prompt has
    try testing.expect(!s.keep(q, 4, null, mine[0..1], path[0..1]).held);
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try s.begin(testing.allocator, q, 12, &.{}, null, &adopt);
    defer plan.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 0), plan.from);
    try testing.expectEqual(@as(usize, 0), plan.marks.len);
    try testing.expectEqual(@as(usize, 0), adopt.items.len);
}

test "a keep past the rows its pass prefills is refused: the store holds prefilled rows only" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), .{ .page = 4, .lookahead = 0, .min_gap = 1, .min_prompt = 0 });
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 8 });
    const prompt = seq(12);
    const mine = [_]u32{ 0, 1 };
    f.take(&mine);
    defer f.end(&mine);
    var path: [2]u32 = undefined;
    var q: radix.Query = .of(&prompt);
    q.prefilled = 6; // rows past 6 were decoded
    try testing.expect(!s.keep(q, 8, null, &mine, &path).held);
    try testing.expect(s.keep(q, 4, null, mine[0..1], path[0..1]).held);
}

test "a page-only store resumes only at a page edge on the request's chunk schedule" {
    var f: Fake = .{};
    var s = try radix.Store.initGroups(testing.allocator, f.family(), f.pages(), planned, .{ .legacy = false }, null);
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 0 });
    const prompt = seq(25);
    const mine = [_]u32{ 0, 1, 2, 3, 4, 5 };
    f.take(&mine);
    defer f.end(&mine);
    var path: [6]u32 = undefined;
    try testing.expect(s.keep(.of(&prompt), 24, null, &mine, &path).held);
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try s.begin(testing.allocator, with(&prompt, &.{ 12, 20 }), 25, &.{}, null, &adopt);
    defer plan.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 20), plan.from); // 24 is inside the chunk from 20; 20 is a start
    try testing.expectEqual(@as(usize, 5), adopt.items.len);
    f.end(adopt.items);
}

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

fn image(at: u32, tokens: u32, hash: u64) radix.Identity {
    return .{ .at = at, .tokens = tokens, .gh = 2, .gw = 2, .hash = hash };
}

fn withImages(prompt: []const u32, ims: []const radix.Identity) radix.Query {
    var q: radix.Query = .of(prompt);
    q.images = ims;
    return q;
}

/// Ids a resume of `q` adopts, given back at once.
fn adopted(s: *radix.Store, f: *Fake, q: radix.Query, out: []u32) ![]u32 {
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try s.begin(testing.allocator, q, q.prompt_len, &.{}, null, &adopt);
    plan.deinit(testing.allocator);
    @memcpy(out[0..adopt.items.len], adopt.items);
    f.end(adopt.items);
    return out[0..adopt.items.len];
}

const text: radix.Rules = .{ .page = 4, .lookahead = 0, .min_gap = 1, .min_prompt = 0 };

test "two images under one key at one position are two branches; the text before them stays shared" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), text);
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 8 });
    const prompt = seq(13); // rows 4..8 are an image's: equal key words, other pictures
    const a = [_]radix.Identity{image(4, 4, 0xa)};
    const b = [_]radix.Identity{image(4, 4, 0xb)};
    const mine_a = [_]u32{ 0, 1, 2 };
    const mine_b = [_]u32{ 3, 4, 5 };
    f.take(&mine_a);
    f.take(&mine_b);
    var path: [3]u32 = undefined;
    try testing.expect(s.keep(withImages(&prompt, &a), 12, null, &mine_a, &path).held);
    try testing.expect(s.keep(withImages(&prompt, &b), 12, null, &mine_b, &path).held);
    try testing.expectEqualSlices(u32, &.{ 0, 4, 5 }, &path); // page 0 is text: shared; the image's pages are b's own
    f.end(&mine_a);
    f.end(&mine_b);
    var got: [8]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, try adopted(&s, &f, withImages(&prompt, &a), &got));
    try testing.expectEqualSlices(u32, &.{ 0, 4, 5 }, try adopted(&s, &f, withImages(&prompt, &b), &got));
    const c = [_]radix.Identity{image(4, 4, 0xc)};
    try testing.expect(s.find(withImages(&prompt, &c)) == null); // a third picture: nothing past the text resumes
}

test "with a lookahead word, an image starting at a page edge makes the page before it another page" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), .{ .page = 4, .lookahead = 1, .min_gap = 1, .min_prompt = 0 });
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 8 });
    const prompt = seq(10);
    const a = [_]radix.Identity{image(4, 4, 0xa)};
    const b = [_]radix.Identity{image(4, 4, 0xb)};
    const mine_a = [_]u32{ 0, 1 };
    const mine_b = [_]u32{ 2, 3 };
    f.take(&mine_a);
    defer f.end(&mine_a);
    f.take(&mine_b);
    defer f.end(&mine_b);
    var path: [2]u32 = undefined;
    try testing.expect(s.keep(withImages(&prompt, &a), 8, null, &mine_a, &path).held);
    try testing.expect(s.keep(withImages(&prompt, &b), 8, null, &mine_b, &path).held);
    try testing.expectEqualSlices(u32, &mine_b, &path); // page 0 reads row 4, the image's first: not shared
}

test "a node split across an image keeps it on both sides, so another picture there matches neither" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), text);
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 8 });
    const one = seq(13);
    var two = seq(13);
    two[10] = 99; // diverges inside the third page: the node splits after its second page
    const a = [_]radix.Identity{image(6, 4, 0xa)}; // rows 6..10 cross the split at 8
    const mine1 = [_]u32{ 0, 1, 2 };
    const mine2 = [_]u32{ 3, 4, 5 };
    f.take(&mine1);
    defer f.end(&mine1);
    f.take(&mine2);
    defer f.end(&mine2);
    var path: [3]u32 = undefined;
    try testing.expect(s.keep(withImages(&one, &a), 12, null, &mine1, &path).held);
    try testing.expect(s.keep(withImages(&two, &a), 12, null, &mine2, &path).held);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 5 }, &path);
    const b = [_]radix.Identity{image(6, 4, 0xb)};
    var held: [3]u32 = undefined;
    var last: ?*radix.Node = null;
    const k: radix.Key = .{ .tokens = one[0..12], .images = &b };
    try testing.expectEqual(@as(usize, 1), s.tree.heldPrefix(k, 3, &held, &last)); // page 1 holds rows 6, 7 of a
    var got: [8]u32 = undefined; // both sides kept the image: each original branch still resumes whole
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, try adopted(&s, &f, withImages(&one, &a), &got));
    try testing.expectEqualSlices(u32, &.{ 0, 1, 5 }, try adopted(&s, &f, withImages(&two, &a), &got));
}

test "with a lookahead word, a split at an image's first row keeps it in the head's last window and the tail" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), .{ .page = 4, .lookahead = 1, .min_gap = 1, .min_prompt = 0 });
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 8 });
    const one = seq(14);
    var two = seq(14);
    two[10] = 99; // the node splits after its second page, at row 8
    const a = [_]radix.Identity{image(8, 4, 0xa)}; // rows 8..12 start at the cut
    const mine1 = [_]u32{ 0, 1, 2 };
    const mine2 = [_]u32{ 3, 4, 5 };
    f.take(&mine1);
    defer f.end(&mine1);
    f.take(&mine2);
    defer f.end(&mine2);
    var path: [3]u32 = undefined;
    try testing.expect(s.keep(withImages(&one, &a), 12, null, &mine1, &path).held);
    try testing.expect(s.keep(withImages(&two, &a), 12, null, &mine2, &path).held);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 5 }, &path);
    const b = [_]radix.Identity{image(8, 4, 0xb)};
    var held: [3]u32 = undefined;
    var last: ?*radix.Node = null;
    const k: radix.Key = .{ .tokens = one[0..13], .images = &b };
    try testing.expectEqual(@as(usize, 1), s.tree.heldPrefix(k, 3, &held, &last)); // page 1 reads row 8: a's
    var got: [8]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, try adopted(&s, &f, withImages(&one, &a), &got));
    try testing.expectEqualSlices(u32, &.{ 0, 1, 5 }, try adopted(&s, &f, withImages(&two, &a), &got));
}

test "a longer prompt in another namespace never counts as the same conversation moving past a reused state" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), text);
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 2 });
    const p = seq(9);
    var q = seq(9);
    q[0] = 50; // another conversation, in the same namespace
    const longer = seq(13);
    var ns1: radix.Query = .of(&p);
    ns1.namespace = 1;
    var other: radix.Query = .of(&q);
    other.namespace = 1;
    var ns2: radix.Query = .of(&longer);
    ns2.namespace = 2;
    const m1 = [_]u32{ 0, 1 };
    const m2 = [_]u32{ 2, 3 };
    const m3 = [_]u32{ 4, 5, 6 };
    f.take(&m1);
    defer f.end(&m1);
    f.take(&m2);
    defer f.end(&m2);
    f.take(&m3);
    defer f.end(&m3);
    var path: [3]u32 = undefined;
    try testing.expect(s.keep(ns1, 8, null, &m1, path[0..2]).held);
    var got: [8]u32 = undefined;
    _ = try adopted(&s, &f, ns1, &got); // resumed once: worth more than an entry nobody asked for
    try testing.expect(s.keep(other, 8, null, &m2, path[0..2]).held);
    try testing.expect(s.keep(ns2, 12, null, &m3, &path).held); // a third state: one of the two goes
    try testing.expect(s.find(ns1) != null); // the reused one stays: namespace 2's words don't extend its conversation
    try testing.expect(s.find(other) == null);
}

test "two namespaces never share a page, whatever their words" {
    var f: Fake = .{};
    var s = try radix.Store.init(testing.allocator, f.family(), f.pages(), text);
    defer s.deinit();
    s.limit(.{ .pages = 32, .snaps = 8 });
    const prompt = seq(9);
    var q1: radix.Query = .of(&prompt);
    q1.namespace = 1;
    var q2: radix.Query = .of(&prompt);
    q2.namespace = 2;
    const mine1 = [_]u32{ 0, 1 };
    const mine2 = [_]u32{ 2, 3 };
    f.take(&mine1);
    defer f.end(&mine1);
    f.take(&mine2);
    defer f.end(&mine2);
    var path: [2]u32 = undefined;
    try testing.expect(s.keep(q1, 8, null, &mine1, &path).held);
    try testing.expect(s.keep(q2, 8, null, &mine2, &path).held);
    try testing.expectEqualSlices(u32, &mine2, &path);
    try testing.expect(s.find(.of(&prompt)) == null); // the default namespace holds neither
}

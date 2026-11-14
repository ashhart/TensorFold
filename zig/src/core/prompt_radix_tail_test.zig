//! The store's writable tails (C4, #2444): a checkpoint inside a page, its rows a held copy never written again.
const std = @import("std");
const radix = @import("prompt_radix.zig");
const testing = std.testing;

/// Counted holders (streams' pages from 0, the store's held ones from 32), a page of 4 tokens; copy_rows logged.
const Fake = struct {
    refs: [64]u32 = @splat(0),
    states: [16]u32 = @splat(0),
    nstates: usize = 0,
    restored: ?u32 = null,
    /// Each copy_rows: group, src, dst, rows.
    rows: [16][4]u32 = undefined,
    nrows: usize = 0,
    fences: u32 = 0,
    fail_hold: bool = false,
    fail_rows: bool = false,
    queued: ?u32 = null, // the page the last copy_rows wrote, until a fence
    held_at_fence: bool = true,

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
    fn hold(p: *anyopaque, out: []u32) bool {
        const f = of(p);
        if (f.fail_hold) return false;
        var n: usize = 0;
        for (32..64) |id| if (n < out.len and f.refs[id] == 0) {
            out[n] = @intCast(id);
            n += 1;
        };
        if (n < out.len) return false;
        for (out) |id| f.refs[id] = 1;
        return true;
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
    fn restore(p: *anyopaque, _: ?*anyopaque, saved: radix.Saved) anyerror!void {
        of(p).restored = @as(*u32, @ptrCast(@alignCast(saved))).*;
    }
    fn dropState(_: *anyopaque, _: radix.Saved) void {}
    fn partBytes(_: *anyopaque, _: u32, _: u32) u64 {
        return 0;
    }
    fn copy(_: *anyopaque, _: ?*anyopaque, _: u32, _: u32, _: radix.Parts.Part) anyerror!void {}
    fn partRestore(_: *anyopaque, _: ?*anyopaque, _: u32, _: radix.Parts.Part) anyerror!void {}
    fn copyRows(p: *anyopaque, _: ?*anyopaque, group: u32, src: u32, dst: u32, rows: u32) anyerror!void {
        const f = of(p);
        f.rows[f.nrows] = .{ group, src, dst, rows };
        f.nrows += 1;
        f.queued = dst;
        if (f.fail_rows) return error.CopyFailed; // after its work is queued
    }
    fn fence(p: *anyopaque, _: ?*anyopaque) void {
        const f = of(p);
        f.fences += 1;
        if (f.queued) |id| f.held_at_fence = f.held_at_fence and f.refs[id] > 0;
        f.queued = null;
    }

    fn pages(f: *Fake) radix.Pages {
        return .{ .ptr = f, .bytes = 1, .vtable = &.{ .retain = retain, .release = release, .holders = holders, .available = available, .hold = hold } };
    }
    fn family(f: *Fake) radix.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = dropState } };
    }
    fn hooks(f: *Fake) radix.PartHooks {
        return .{ .ptr = f, .vtable = &.{ .bytes = partBytes, .copy = copy, .restore = partRestore, .copy_rows = copyRows, .fence = fence } };
    }
    /// A stream's own pages: one holder more each.
    fn take(f: *Fake, ids: []const u32) void {
        for (ids) |id| f.refs[id] += 1;
    }
    fn end(f: *Fake, ids: []const u32) void {
        for (ids) |id| f.refs[id] -= 1;
    }
    /// Times `id` was written by a copy.
    fn writes(f: *const Fake, id: u32) usize {
        var n: usize = 0;
        for (f.rows[0..f.nrows]) |c| n += @intFromBool(c[2] == id);
        return n;
    }
    fn refsTotal(f: *const Fake) usize {
        var n: usize = 0;
        for (f.refs) |r| n += r;
        return n;
    }
};

const unplanned: radix.Rules = .{ .page = 4, .lookahead = 0, .min_gap = 1, .min_prompt = 0 };
const planned: radix.Rules = .{ .page = 4, .planned = true, .grid = 8, .lookahead = 0, .min_gap = 1, .min_prompt = 0 };
const tails: radix.Groups = .{ .tails = true };

fn open(f: *Fake, rules: radix.Rules, groups: radix.Groups) !radix.Store {
    var s = try radix.Store.initGroups(testing.allocator, f.family(), f.pages(), rules, groups, f.hooks());
    s.limit(.{ .pages = 16, .snaps = 4 });
    return s;
}

fn seq(comptime n: usize) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = @intCast(i + 1);
    return out;
}

/// `begin`, its adopted pages and copies the stream's until it ends them.
const Resume = struct {
    plan: radix.Plan,
    adopt: std.ArrayList(u32) = .empty,

    fn of(s: *radix.Store, q: radix.Query) !Resume {
        var r: Resume = .{ .plan = undefined };
        r.plan = try s.begin(testing.allocator, q, q.prompt_len, &.{}, null, &r.adopt);
        return r;
    }
    fn end(r: *Resume, s: *radix.Store, f: *Fake) void {
        f.end(r.adopt.items);
        s.finish(null, r.plan.copies);
        r.plan.deinit(testing.allocator);
        r.adopt.deinit(testing.allocator);
    }
};

test "the same off-page position marks a tail with tails, and its page edge without" {
    var f: Fake = .{};
    var with = try open(&f, unplanned, tails);
    defer with.deinit();
    var without = try radix.Store.initGroups(testing.allocator, f.family(), f.pages(), unplanned, .{}, null);
    defer without.deinit();
    without.limit(.{ .pages = 16, .snaps = 4 });
    const prompt = seq(10);
    const a = try with.marks(testing.allocator, .of(&prompt), 0, 6, &.{}, &.{});
    defer testing.allocator.free(a);
    const b = try without.marks(testing.allocator, .of(&prompt), 0, 6, &.{}, &.{});
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u32, &.{6}, a);
    try testing.expectEqualSlices(u32, &.{4}, b);
}

test "a tail keeps its rows as a held copy, and a prompt that diverges past them resumes there through a copy" {
    var f: Fake = .{};
    var s = try open(&f, unplanned, tails);
    defer s.deinit();
    const prompt = seq(10);
    const mine = [_]u32{ 0, 1 }; // a whole page and the partial page
    f.take(&mine);
    var path: [1]u32 = undefined;
    const kept = s.keep(.of(&prompt), 6, null, &mine, &path);
    try testing.expect(kept.held);
    try testing.expectEqual(@as(usize, 1), kept.shared); // the whole page only: the stream still writes its partial one
    try testing.expectEqual(@as(usize, 2), s.tree.heldPages()); // the path page and the tail's copy
    try testing.expectEqual(@as(usize, 1), f.nrows);
    const tail_id = f.rows[0][2];
    try testing.expectEqualSlices(u32, &.{ 0, 1, tail_id, 2 }, &f.rows[0]); // prefix group 0, rows [0, 2) of page 1
    try testing.expectEqual(@as(u32, 1), f.refs[1]); // the stream's partial page stays the stream's alone
    f.end(&mine);
    const edge = s.tree.root.kids.items[0];
    try testing.expect(edge.entry == null); // a tail on an edge with no state of its own
    try testing.expectEqual(@as(usize, 1), edge.tails.items.len);

    var diverged = seq(10);
    diverged[6] = 99; // inside the tail's page, past its rows
    var r = try Resume.of(&s, .of(&diverged));
    try testing.expectEqual(@as(u32, 6), r.plan.from);
    try testing.expectEqual(@as(?u32, 6), f.restored);
    try testing.expectEqualSlices(u32, &.{0}, r.adopt.items);
    try testing.expectEqual(@as(usize, 1), r.plan.copies.len);
    try testing.expectEqual(radix.Copy{ .group = 0, .index = 1, .src = tail_id, .rows = 2 }, r.plan.copies[0]);
    try testing.expectEqual(@as(u32, 2), f.refs[tail_id]); // held for the stream until finish
    r.end(&s, &f);
    try testing.expectEqual(@as(u32, 1), f.refs[tail_id]);

    var other = seq(10);
    other[5] = 99; // inside the tail's rows: only the page edge matches, which holds no state
    var miss = try Resume.of(&s, .of(&other));
    try testing.expectEqual(@as(u32, 0), miss.plan.from);
    miss.end(&s, &f);
    try testing.expectEqual(@as(usize, 1), f.writes(tail_id)); // the keep that made it, and nothing since
}

test "a tail inside the first page hangs on the root: kept, found, resumed and evicted" {
    var f: Fake = .{};
    var s = try open(&f, unplanned, tails);
    defer s.deinit();
    const prompt = seq(10);
    const mine = [_]u32{0};
    f.take(&mine);
    var path: [0]u32 = .{};
    try testing.expect(s.keep(.of(&prompt), 3, null, &mine, &path).held);
    f.end(&mine);
    try testing.expectEqual(@as(usize, 1), s.tree.root.tails.items.len);
    const tail_id = s.tree.root.tails.items[0].tail.?.ids[0];
    var r = try Resume.of(&s, .of(&prompt));
    try testing.expectEqual(@as(u32, 3), r.plan.from);
    try testing.expectEqual(@as(usize, 0), r.adopt.items.len);
    try testing.expectEqual(radix.Copy{ .group = 0, .index = 0, .src = tail_id, .rows = 3 }, r.plan.copies[0]);
    r.end(&s, &f);
    s.limit(.{ .pages = 16, .snaps = 0 });
    try testing.expectEqual(@as(usize, 0), s.tree.snaps);
    try testing.expectEqual(@as(u32, 0), f.refs[tail_id]);
    try testing.expect(s.keep(.of(&prompt), 3, null, &mine, &path).held == false); // no state slot left
    s.limit(.{ .pages = 16, .snaps = 4 });
    f.take(&mine);
    try testing.expect(s.keep(.of(&prompt), 3, null, &mine, &path).held);
    f.end(&mine);
    try testing.expect(s.reclaim(64 - 0)); // the root's idle tail is reclaimed like a leaf's: every page back
    try testing.expectEqual(@as(usize, 0), s.tree.heldPages());
}

test "the deepest usable tail on one edge wins: three tails, a request's starts and a diverging prompt" {
    var f: Fake = .{};
    var s = try open(&f, planned, tails);
    defer s.deinit();
    const prompt = seq(12);
    for ([_]u32{ 5, 6, 7 }) |at| {
        const mine = [_]u32{ 0, 1 };
        f.take(&mine);
        var path: [1]u32 = undefined;
        try testing.expect(s.keep(.{ .key = &prompt, .prompt_len = 12, .starts = &.{at} }, at, null, &mine, &path).held);
        f.end(&mine);
    }
    try testing.expectEqual(@as(usize, 3), s.tree.root.kids.items[0].tails.items.len);
    const both = radix.Query{ .key = &prompt, .prompt_len = 12, .starts = &.{ 5, 7 } };
    try testing.expectEqual(@as(u32, 7), s.find(both).?.at());
    var diverged = seq(12);
    diverged[6] = 99; // the tail at 7 holds row 6
    const q = radix.Query{ .key = &diverged, .prompt_len = 12, .starts = &.{ 5, 7 } };
    try testing.expectEqual(@as(u32, 5), s.find(q).?.at()); // 6 is inside the chunk from 5, so not a resume point
    const at6 = radix.Query{ .key = &diverged, .prompt_len = 12, .starts = &.{6} };
    try testing.expectEqual(@as(u32, 6), s.find(at6).?.at());
}

test "a tail at a grid row inside a request's chunk is refused, and so is one off the grid without a start" {
    var f: Fake = .{};
    var s = try open(&f, planned, tails);
    defer s.deinit();
    const prompt = seq(12);
    const mine = [_]u32{ 0, 1, 2 };
    f.take(&mine);
    defer f.end(&mine);
    var path: [2]u32 = undefined;
    try testing.expect(!s.keep(.{ .key = &prompt, .prompt_len = 12, .starts = &.{5} }, 8, null, mine[0..2], path[0..2]).held);
    try testing.expect(!s.keep(.of(&prompt), 6, null, mine[0..2], path[0..1]).held); // the grid's start is 0
    try testing.expectEqual(@as(usize, 0), s.tree.snaps);
}

test "a page-only store keeps whole pages without entries, and tails as entries only with state slots" {
    var f: Fake = .{};
    var s = try open(&f, unplanned, .{ .legacy = false, .tails = true });
    defer s.deinit();
    s.limit(.{ .pages = 16, .snaps = 0 });
    const prompt = seq(10);
    const mine = [_]u32{ 0, 1 };
    var path: [1]u32 = undefined;
    f.take(&mine);
    const pages_only = s.keep(.of(&prompt), 6, null, &mine, &path);
    try testing.expect(!pages_only.held and pages_only.shared == 1); // its whole page, no tail
    try testing.expectEqual(@as(usize, 1), s.tree.heldPages());
    s.limit(.{ .pages = 16, .snaps = 2 });
    try testing.expect(s.keep(.of(&prompt), 6, null, &mine, &path).held);
    f.end(&mine);
    try testing.expectEqual(@as(usize, 1), s.tree.snaps);
    var r = try Resume.of(&s, .of(&prompt));
    try testing.expectEqual(@as(u32, 6), r.plan.from);
    try testing.expectEqual(@as(usize, 1), r.plan.copies.len);
    r.end(&s, &f);
    var other = seq(10);
    other[5] = 99;
    var edge = try Resume.of(&s, .of(&other)); // the page edge: a page-only hit with no entry
    try testing.expectEqual(@as(u32, 4), edge.plan.from);
    try testing.expectEqual(@as(usize, 0), edge.plan.copies.len);
    edge.end(&s, &f);
}

test "a later prompt that diverges inside a tail's rows does not supersede it" {
    var f: Fake = .{};
    var s = try open(&f, unplanned, tails);
    defer s.deinit();
    const a = seq(10);
    const mine = [_]u32{ 0, 1, 2 };
    f.take(&mine);
    defer f.end(&mine);
    var path: [2]u32 = undefined;
    try testing.expect(s.keep(.of(&a), 6, null, mine[0..2], path[0..1]).held);
    var r = try Resume.of(&s, .of(&a)); // resumed: worth more than a state nobody asked for
    r.end(&s, &f);
    var b = seq(12);
    b[4] = 99; // past the tail's edge, inside its rows
    try testing.expect(s.keep(.of(&b), 8, null, mine[0..2], path[0..2]).held);
    s.limit(.{ .pages = 16, .snaps = 1 });
    try testing.expectEqual(@as(u32, 6), s.find(.of(&a)).?.at()); // the tail stays; b's state went
}

test "a later tail supersedes one only under its Ns and its lookahead words" {
    var f: Fake = .{};
    var s = try radix.Store.initGroups(testing.allocator, f.family(), f.pages(), .{ .page = 4, .lookahead = 1, .min_gap = 1, .min_prompt = 0 }, tails, f.hooks());
    defer s.deinit();
    s.limit(.{ .pages = 16, .snaps = 4 });
    const mine = [_]u32{ 0, 1 };
    f.take(&mine);
    defer f.end(&mine);
    var path: [1]u32 = undefined;
    const a = seq(10);
    try testing.expect(s.keep(.{ .key = &a, .prompt_len = 10, .namespace = 1 }, 2, null, mine[1..2], path[0..0]).held);
    try testing.expect(s.keep(.of(&a), 6, null, &mine, &path).held);
    for ([_]radix.Query{ .{ .key = &a, .prompt_len = 10, .namespace = 1 }, .of(&a) }) |q| {
        var r = try Resume.of(&s, q); // each resumed once
        r.end(&s, &f);
    }
    const b = seq(12);
    try testing.expect(s.keep(.{ .key = &b, .prompt_len = 12, .namespace = 2 }, 3, null, mine[1..2], path[0..0]).held); // another Ns
    var c = seq(12);
    c[6] = 99; // the tail at 6 read row 6 as its lookahead
    try testing.expect(s.keep(.of(&c), 7, null, &mine, &path).held);
    s.limit(.{ .pages = 16, .snaps = 2 }); // the two never resumed go; neither of them superseded a resumed tail
    try testing.expectEqual(@as(u32, 2), s.find(.{ .key = &a, .prompt_len = 10, .namespace = 1 }).?.at());
    try testing.expectEqual(@as(u32, 6), s.find(.of(&a)).?.at());
}

test "tails need copy_rows and a pool that holds pages for the store" {
    var f: Fake = .{};
    try testing.expectError(error.TailsNeedHooks, radix.Store.initGroups(testing.allocator, f.family(), f.pages(), unplanned, tails, null));
    var no_hold = f.pages();
    no_hold.vtable = &.{ .retain = Fake.retain, .release = Fake.release, .holders = Fake.holders, .available = Fake.available };
    try testing.expectError(error.TailsNeedHold, radix.Store.initGroups(testing.allocator, f.family(), no_hold, unplanned, tails, f.hooks()));
}

test "a window group with other rows a page copies its own partial rows, at a tail and at a page edge" {
    var f: Fake = .{};
    const window = [_]radix.Parts.PartSpec{.{ .kind = .window, .make = .retained, .rows = 3 }};
    var s = try open(&f, unplanned, .{ .parts = &window, .tails = true });
    defer s.deinit();
    const prompt = seq(12);
    const mine = [_]u32{ 0, 1, 2 };
    const win = [_]u32{ 10, 11, 12 }; // rows [0, 9): the last partial at 7 and at 8
    f.take(&mine);
    f.take(&win);
    var path: [2]u32 = undefined;
    try testing.expect(s.keepParts(.of(&prompt), 7, null, mine[0..2], &.{.{ .base = 0, .ids = &win }}, path[0..1]).held);
    try testing.expect(s.keepParts(.of(&prompt), 8, null, mine[0..2], &.{.{ .base = 0, .ids = &win }}, path[0..2]).held);
    f.end(&mine);
    f.end(&win);
    // at 7: the tail's 3 rows of prefix page 1 (group 1, after the one part), the window's 1 row of its page 2
    var r = try Resume.of(&s, .of(prompt[0..8]));
    defer r.end(&s, &f);
    try testing.expectEqual(@as(u32, 7), r.plan.from);
    try testing.expectEqual(@as(usize, 2), r.plan.copies.len);
    try testing.expectEqual(@as(u32, 0), r.plan.copies[0].group);
    try testing.expectEqual(@as(u32, 2), r.plan.copies[0].index);
    try testing.expectEqual(@as(u32, 1), r.plan.copies[0].rows);
    try testing.expectEqual(@as(u32, 1), r.plan.copies[1].group);
    try testing.expectEqual(@as(u32, 1), r.plan.copies[1].index);
    try testing.expectEqual(@as(u32, 3), r.plan.copies[1].rows);
    var e = try Resume.of(&s, .of(&prompt)); // at 8: the edge, the window's 2 rows
    defer e.end(&s, &f);
    try testing.expectEqual(@as(u32, 8), e.plan.from);
    try testing.expectEqual(@as(usize, 1), e.plan.copies.len);
    try testing.expectEqual(@as(u32, 2), e.plan.copies[0].rows);
}

test "a tail that fails to hold, to copy or to allocate takes nothing and leaves no state" {
    var f: Fake = .{};
    var fa = std.testing.FailingAllocator.init(testing.allocator, .{});
    var s = try radix.Store.initGroups(fa.allocator(), f.family(), f.pages(), unplanned, tails, f.hooks());
    defer s.deinit();
    s.limit(.{ .pages = 16, .snaps = 4 });
    const prompt = seq(10);
    const mine = [_]u32{ 0, 1 };
    f.take(&mine);
    defer f.end(&mine);
    var path: [1]u32 = undefined;
    for ([_]bool{ true, false }) |fail_hold| {
        f.fail_hold = fail_hold;
        f.fail_rows = !fail_hold;
        try testing.expect(!s.keep(.of(&prompt), 6, null, &mine, &path).held);
        try testing.expectEqual(@as(usize, 0), s.tree.snaps);
        try testing.expectEqual(mine.len + s.tree.heldPages(), f.refsTotal()); // the path stays; no copy is left held
        try testing.expect(f.held_at_fence); // the failed copy completed before its page went back
    }
    f.fail_rows = false;
    var k: usize = 0;
    while (k < 32) : (k += 1) {
        fa.fail_index = fa.alloc_index + k;
        if (s.keep(.of(&prompt), 6, null, &mine, &path).held) break;
        try testing.expectEqual(@as(usize, 0), s.tree.snaps);
        try testing.expectEqual(mine.len + s.tree.heldPages(), f.refsTotal());
    }
    try testing.expect(k >= 3 and k < 32); // the tail's key, its images, its ids and the entry each failed once
}

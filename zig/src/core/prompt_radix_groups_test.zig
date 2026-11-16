//! The store over two prefix groups (C3, #2444): ids page-major on the path, limits counted in ids.
const std = @import("std");
const radix = @import("prompt_radix.zig");
const testing = std.testing;

/// Pages whose holders are counted, and states that are the position they were kept at; a page is 4 tokens.
const Fake = struct {
    refs: [32]u32 = @splat(0),
    states: [8]u32 = @splat(0),
    nstates: usize = 0,
    restored: ?u32 = null,

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
    fn restore(p: *anyopaque, _: ?*anyopaque, saved: radix.Saved) anyerror!void {
        const at: *u32 = @ptrCast(@alignCast(saved));
        of(p).restored = at.*;
    }
    fn dropState(_: *anyopaque, _: radix.Saved) void {}

    fn pages(f: *Fake) radix.Pages {
        return .{ .ptr = f, .bytes = 1, .vtable = &.{ .retain = retain, .release = release, .holders = holders, .available = available } };
    }
    fn family(f: *Fake) radix.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = dropState } };
    }
    /// A stream's own ids: taken pages, one holder each.
    fn take(f: *Fake, ids: []const u32) void {
        for (ids) |id| f.refs[id] = 1;
    }
};

const two: radix.Groups = .{ .prefix_rows = &.{ 4, 4 } };

fn open(f: *Fake, limit_ids: usize) !radix.Store {
    var s = try radix.Store.initGroups(testing.allocator, f.family(), f.pages(), .{ .page = 4, .lookahead = 0, .min_gap = 1, .min_prompt = 0 }, two, null);
    s.limit(.{ .pages = limit_ids, .snaps = 4 });
    return s;
}

test "two prefix groups: a keep holds each logical page's two ids, and a resume adopts them page-major" {
    var f: Fake = .{};
    var s = try open(&f, 16);
    defer s.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const mine = [_]u32{ 0, 1, 2, 3 }; // page 0: ids 0 (group 0) and 1 (group 1); page 1: 2 and 3
    f.take(&mine);
    var path: [4]u32 = undefined;
    const kept = s.keep(.of(&prompt), 8, null, &mine, &path);
    try testing.expect(kept.held);
    try testing.expectEqual(@as(usize, 4), kept.shared); // ids, not logical pages
    try testing.expectEqualSlices(u32, &mine, &path);
    try testing.expectEqual(@as(usize, 4), s.tree.heldPages());
    for (mine) |id| try testing.expectEqual(@as(u32, 2), f.refs[id]); // the stream and the store
    for (mine) |id| f.refs[id] -= 1; // the stream ends

    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try s.begin(testing.allocator, .of(&prompt), 10, &.{}, null, &adopt);
    defer testing.allocator.free(plan.marks);
    try testing.expectEqual(@as(u32, 8), plan.from);
    try testing.expectEqual(@as(?u32, 8), f.restored);
    try testing.expectEqualSlices(u32, &mine, adopt.items);
    for (adopt.items) |id| f.refs[id] -= 1;
}

test "two prefix groups: a longer prompt shares the first page's ids and adds its own, split page-major" {
    var f: Fake = .{};
    var s = try open(&f, 16);
    defer s.deinit();
    const a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 0 };
    const b = [_]u32{ 1, 2, 3, 4, 9, 9, 9, 9, 0 };
    const mine_a = [_]u32{ 0, 1, 2, 3 };
    const mine_b = [_]u32{ 4, 5, 6, 7 };
    f.take(&mine_a);
    f.take(&mine_b);
    var path: [4]u32 = undefined;
    try testing.expect(s.keep(.of(&a), 8, null, &mine_a, &path).held);
    const kept = s.keep(.of(&b), 8, null, &mine_b, &path);
    try testing.expect(kept.held);
    // the shared first page is a's ids; b's second is its own
    try testing.expectEqualSlices(u32, &.{ 0, 1, 6, 7 }, &path);
    try testing.expectEqual(@as(usize, 6), s.tree.heldPages());
}

test "two prefix groups: the page limit counts ids, so a keep needing more ids than it holds evicts or refuses" {
    var f: Fake = .{};
    var s = try open(&f, 4); // two logical pages of two groups
    defer s.deinit();
    const long = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0 };
    const mine = [_]u32{ 0, 1, 2, 3, 4, 5 };
    f.take(&mine);
    var path: [6]u32 = undefined;
    try testing.expect(!s.keep(.of(&long), 12, null, &mine, &path).held); // 6 ids pass the limit of 4
    try testing.expectEqual(@as(usize, 0), s.tree.heldPages());
    try testing.expect(s.keep(.of(&long), 8, null, mine[0..4], path[0..4]).held); // 4 ids fit
}

test "prefix groups with other rows a page than the store's are refused when it opens" {
    var f: Fake = .{};
    try testing.expectError(error.UnequalPrefixRows, radix.Store.initGroups(testing.allocator, f.family(), f.pages(), .{ .page = 4, .min_prompt = 0 }, .{ .prefix_rows = &.{ 4, 8 } }, null));
}

const page_only: radix.Groups = .{ .legacy = false };

fn openPages(f: *Fake, limit_ids: usize) !radix.Store {
    var s = try radix.Store.initGroups(testing.allocator, f.family(), f.pages(), .{ .page = 4, .lookahead = 0, .min_gap = 1, .min_prompt = 0 }, page_only, null);
    s.limit(.{ .pages = limit_ids, .snaps = 0 }); // no checkpoint: no snapshot slots needed
    return s;
}

test "a store with no checkpoint keeps pages with no snapshot slot and resumes at any matched page edge" {
    var f: Fake = .{};
    var s = try openPages(&f, 16);
    defer s.deinit();
    try testing.expect(s.on());
    const kept_prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    const mine = [_]u32{ 0, 1, 2 };
    f.take(&mine);
    var path: [3]u32 = undefined;
    try testing.expect(s.keep(.of(&kept_prompt), 12, null, &mine, &path).held);
    try testing.expectEqual(@as(usize, 0), s.tree.snaps);
    for (mine) |id| f.refs[id] -= 1;
    // a prompt that diverges inside the kept node's third page resumes after its second, adopting two pages
    const other = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 99, 99, 99, 99, 99 };
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try s.begin(testing.allocator, .of(&other), 14, &.{}, null, &adopt);
    defer testing.allocator.free(plan.marks);
    try testing.expectEqual(@as(u32, 8), plan.from);
    try testing.expectEqualSlices(u32, &.{ 0, 1 }, adopt.items);
    try testing.expectEqual(@as(?u32, null), f.restored); // nothing to restore
    for (adopt.items) |id| f.refs[id] -= 1;
    adopt.clearRetainingCapacity();
    // a prompt that ends inside it resumes at the last whole page that leaves a token to prefill
    const short = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    const plan2 = try s.begin(testing.allocator, .of(&short), 7, &.{}, null, &adopt);
    defer testing.allocator.free(plan2.marks);
    try testing.expectEqual(@as(u32, 4), plan2.from);
    try testing.expectEqualSlices(u32, &.{0}, adopt.items);
    for (adopt.items) |id| f.refs[id] -= 1;
}

test "a store with no checkpoint frees an idle leaf to keep another prompt's pages" {
    var f: Fake = .{};
    var s = try openPages(&f, 3);
    defer s.deinit();
    const a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0 };
    const mine = [_]u32{ 0, 1, 2 };
    f.take(&mine);
    var path: [3]u32 = undefined;
    try testing.expect(s.keep(.of(&a), 12, null, &mine, &path).held);
    for (mine) |id| f.refs[id] -= 1;
    const b = [_]u32{ 7, 7, 7, 7, 7, 7, 7, 7, 0 };
    const theirs = [_]u32{ 5, 6 };
    f.take(&theirs);
    var path2: [2]u32 = undefined;
    try testing.expect(s.keep(.of(&b), 8, null, &theirs, &path2).held); // room: the idle leaf of `a` went
    try testing.expect(s.tree.heldPages() <= 3);
    for (theirs) |id| f.refs[id] -= 1;
}

test "a store with no checkpoint making room for a pass's marks never frees the path that pass resumes" {
    var f: Fake = .{};
    var s = try openPages(&f, 5);
    defer s.deinit();
    const a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0 };
    const mine = [_]u32{ 0, 1, 2 };
    f.take(&mine);
    var path: [3]u32 = undefined;
    try testing.expect(s.keep(.of(&a), 12, null, &mine, &path).held);
    for (mine) |id| f.refs[id] -= 1; // the tree's alone: idle, and the oldest leaf
    const b = [_]u32{ 7, 7, 7, 7, 7, 7, 7, 7, 0 };
    const theirs = [_]u32{ 5, 6 };
    f.take(&theirs);
    var path2: [2]u32 = undefined;
    try testing.expect(s.keep(.of(&b), 8, null, &theirs, &path2).held);
    for (theirs) |id| f.refs[id] -= 1; // a newer idle leaf: 5 pages held of 5
    const longer = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    // its mark at 20 wants 2 more pages
    const plan = try s.begin(testing.allocator, .of(&longer), 21, &.{}, null, &adopt);
    defer testing.allocator.free(plan.marks);
    try testing.expectEqual(@as(u32, 12), plan.from);
    try testing.expectEqualSlices(u32, &mine, adopt.items);
    for (adopt.items) |id| f.refs[id] -= 1;
    try testing.expectEqual(@as(u32, 0), f.refs[5]); // the newer leaf went instead
}

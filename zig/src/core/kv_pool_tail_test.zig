//! The store's writable tails (C4, #2444) over a real pool: the private copy from the stream's promise, reclaim first.
const std = @import("std");
const kv = @import("kv_pool");
const radix = @import("engine_api").prompt_radix;
const Pool = kv.Pool;
const Lease = kv.Lease;
const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;

const page_bytes = 8;

/// A family whose device memory is one word a page: copy_rows copies the word and logs the page it wrote.
const Family = struct {
    pool: *Pool,
    mem: [32]u32 = @splat(0),
    wrote: [16]u32 = undefined,
    nwrote: usize = 0,

    fn of(p: *anyopaque) *Family {
        return @ptrCast(@alignCast(p));
    }
    fn bytes(_: *anyopaque, _: u32, _: u32) u64 {
        return 0;
    }
    fn copy(_: *anyopaque, _: ?*anyopaque, _: u32, _: u32, _: radix.Parts.Part) anyerror!void {}
    fn restore(_: *anyopaque, _: ?*anyopaque, _: u32, _: radix.Parts.Part) anyerror!void {}
    fn copyRows(p: *anyopaque, _: ?*anyopaque, _: u32, src: u32, dst: u32, _: u32) anyerror!void {
        const f = of(p);
        f.mem[dst] = f.mem[src];
        f.wrote[f.nwrote] = dst;
        f.nwrote += 1;
    }
    fn fence(_: *anyopaque, _: ?*anyopaque) void {}
    fn hooks(f: *Family) radix.PartHooks {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .copy = copy, .restore = restore, .copy_rows = copyRows, .fence = fence } };
    }
    fn snapBytes(_: *anyopaque, _: u32) u64 {
        return 0;
    }
    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!radix.Saved {
        return error.NoLegacy;
    }
    fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: radix.Saved) anyerror!void {}
    fn drop(_: *anyopaque, _: radix.Saved) void {}
    fn snapshots(f: *Family) radix.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = snapBytes, .save = save, .restore = snapRestore, .drop = drop } };
    }
    fn written(f: *const Family, id: u32) usize {
        var n: usize = 0;
        for (f.wrote[0..f.nwrote]) |w| n += @intFromBool(w == id);
        return n;
    }
};

const World = struct {
    pool: Pool,
    f: Family = undefined,
    store: radix.Store = undefined,

    fn init(w: *World, pages: usize) !void {
        w.pool = try Pool.init(testing.allocator, pages, page_bytes);
        w.f = .{ .pool = &w.pool };
        const rules: radix.Rules = .{ .page = 4, .lookahead = 0, .min_gap = 1, .min_prompt = 0 };
        w.store = try radix.Store.initGroups(testing.allocator, w.f.snapshots(), radix.pagesOf(Pool, &w.pool, page_bytes), rules, .{ .legacy = false, .tails = true }, w.f.hooks());
        w.store.limit(.{ .pages = pages, .snaps = 4 });
        try w.pool.joinReclaim(.{ .ctx = &w.store, .call = reclaim });
    }
    fn deinit(w: *World) void {
        w.pool.leaveReclaim(&w.store);
        w.store.deinit();
        expect(w.pool.idle() and w.pool.available() == w.pool.total()) catch @panic("pages left after close");
        w.pool.deinit();
    }
    fn reclaim(ctx: *anyopaque, want: usize) bool {
        const s: *radix.Store = @ptrCast(@alignCast(ctx));
        return s.reclaim(want);
    }
    /// A stream keeps a tail 2 rows into its second page; the store's copy is returned.
    fn keepTail(w: *World, words: []const u32) !u32 {
        var l: Lease = .{};
        defer w.pool.finish(&l);
        try w.pool.promise(&l, 2);
        try w.pool.cover(&l, 2, null);
        w.f.mem[l.ids.items[1]] = 42; // the partial page's rows
        var path: [1]u32 = undefined;
        try expect(w.store.keep(.of(words), 6, null, l.ids.items, &path).held);
        const edge = w.store.tree.root.kids.items[0];
        return edge.tails.items[0].tail.?.ids[0];
    }
};

const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

test "a resume at a tail takes its private page from its promise, and the store's copy is never written" {
    var w: World = .{ .pool = undefined };
    try w.init(16);
    defer w.deinit();
    const tail_id = try w.keepTail(&prompt);
    try expectEqual(@as(u32, 42), w.f.mem[tail_id]);

    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try w.store.begin(testing.allocator, .of(&prompt), 10, &.{}, null, &adopt);
    defer plan.deinit(testing.allocator);
    try expectEqual(@as(u32, 6), plan.from);
    var l: Lease = .{};
    try w.pool.promise(&l, 3 - adopt.items.len); // the request's three pages less the one it adopted
    try w.pool.adopt(&l, adopt.items);
    const promised = w.pool.census().promised;
    const c = plan.copies[0];
    const own = try w.pool.take(&l); // the private copy: a page of its promise
    try expectEqual(promised - 1, w.pool.census().promised);
    try Family.copyRows(&w.f, null, c.group, c.src, own, c.rows);
    try expectEqual(@as(usize, c.index), l.ids.items.len - 1); // mapped at the tail's logical page
    w.store.finish(null, plan.copies);
    try expectEqual(@as(u32, 42), w.f.mem[own]);
    try expectEqual(@as(u32, 1), w.pool.holders(tail_id)); // the store's again alone
    try expectEqual(@as(usize, 1), w.f.written(tail_id)); // the keep that made it, and nothing since
    w.pool.finish(&l);
}

test "a resume that fails after its take trims the page back to its promise and still ends its copies" {
    var w: World = .{ .pool = undefined };
    try w.init(16);
    defer w.deinit();
    const tail_id = try w.keepTail(&prompt);
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try w.store.begin(testing.allocator, .of(&prompt), 10, &.{}, null, &adopt);
    defer plan.deinit(testing.allocator);
    var l: Lease = .{};
    try w.pool.promise(&l, 2);
    try w.pool.adopt(&l, adopt.items);
    const before = w.pool.census();
    _ = try w.pool.take(&l);
    w.pool.trim(&l, adopt.items.len); // the copy failed: the page goes back to the promise
    try expectEqual(before, w.pool.census());
    w.store.finish(null, plan.copies);
    try expectEqual(@as(u32, 1), w.pool.holders(tail_id));
    w.pool.finish(&l);
}

test "a promise that needs pages reclaims an idle tail; the path a stream maps stays mapped" {
    var w: World = .{ .pool = undefined };
    try w.init(4);
    defer w.deinit();
    _ = try w.keepTail(&prompt); // the path page and the tail's copy: 2 of 4
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    var other = prompt;
    other[5] = 99; // the page edge only: the path is mapped, the tail is idle
    const plan = try w.store.begin(testing.allocator, .of(&other), 10, &.{}, null, &adopt);
    defer plan.deinit(testing.allocator);
    try expectEqual(@as(u32, 4), plan.from);
    var l: Lease = .{};
    defer w.pool.finish(&l);
    try w.pool.adopt(&l, adopt.items);
    try w.pool.promise(&l, 3); // 2 free: the tail goes first
    try expectEqual(@as(usize, 0), w.store.tree.snaps);
    try expectEqual(@as(usize, 0), w.store.tree.heldPages()); // its path held no other state: pruned
    try expectEqual(@as(u32, 1), w.pool.holders(adopt.items[0])); // still the stream's
}

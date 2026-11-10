//! The paged store's checkpoint parts (C3, #2444) over a real pool: made all or nothing, held once, reclaimed, pinned.
const std = @import("std");
const kv = @import("kv_pool");
const radix = @import("engine_api").prompt_radix;
const Pool = kv.Pool;
const Lease = kv.Lease;
const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectEqualSlices = testing.expectEqualSlices;

const page = 4; // tokens a page of the prefix group
const page_bytes = 8;

/// A family whose device memory is one word a page: a copy writes the live state's word, a restore reads it back.
const Family = struct {
    pool: *Pool,
    mem: [64]u32 = @splat(0),
    live: [2]u32 = .{ 0, 0 }, // each group's live state
    restored: [2]?u32 = .{ null, null },
    fences: u32 = 0,
    copies: u32 = 0,
    rows_copied: u32 = 0,
    fail_copy: bool = false,
    fail_rows: bool = false,
    queued: []const u32 = &.{}, // the pages the last copy wrote (queued work, until a fence)
    held_at_fence: bool = true, // every fence saw the queued pages still held: nothing went back before it
    hold_on_restore: usize = 0, // pages the restore takes from the pool (its reclaim runs)

    fn of(p: *anyopaque) *Family {
        return @ptrCast(@alignCast(p));
    }
    fn bytes(_: *anyopaque, _: u32, _: u32) u64 {
        return 20; // three pages of 8 bytes
    }
    fn copy(p: *anyopaque, _: ?*anyopaque, group: u32, _: u32, dst: radix.Parts.Part) anyerror!void {
        const f = of(p);
        for (dst.ids) |id| f.mem[id] = f.live[group];
        f.queued = dst.ids;
        if (f.fail_copy) return error.CopyFailed; // after its work is queued
        f.copies += 1;
    }
    fn restore(p: *anyopaque, _: ?*anyopaque, group: u32, src: radix.Parts.Part) anyerror!void {
        const f = of(p);
        if (f.hold_on_restore > 0) {
            var ids: [8]u32 = undefined;
            if (!f.pool.hold(ids[0..f.hold_on_restore])) return error.NoPages;
            for (ids[0..f.hold_on_restore]) |id| f.pool.release(id);
        }
        f.restored[group] = f.mem[src.ids[0]];
    }
    fn copyRows(p: *anyopaque, _: ?*anyopaque, _: u32, src: u32, dst: u32, rows: u32) anyerror!void {
        const f = of(p);
        if (f.fail_rows) return error.CopyFailed;
        f.mem[dst] = f.mem[src];
        f.rows_copied += rows;
    }
    fn fence(p: *anyopaque, _: ?*anyopaque) void {
        const f = of(p);
        f.fences += 1;
        for (f.queued) |id| f.held_at_fence = f.held_at_fence and f.pool.holders(id) > 0;
        f.queued = &.{};
    }
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
};

const copied_state = [_]radix.Parts.PartSpec{.{ .kind = .state, .make = .copied }};
const window3 = [_]radix.Parts.PartSpec{.{ .kind = .window, .make = .retained, .rows = 3 }};
const both = [_]radix.Parts.PartSpec{ .{ .kind = .window, .make = .retained, .rows = 3 }, .{ .kind = .state, .make = .copied } };

const rules: radix.Rules = .{ .page = page, .lookahead = 0, .min_gap = 1, .min_prompt = 0 };

const World = struct {
    pool: Pool,
    f: Family = undefined,
    store: radix.Store = undefined,
    reclaims: u32 = 0, // the pool asked the store

    fn init(w: *World, specs: []const radix.Parts.PartSpec, pages: usize) !void {
        w.pool = try Pool.init(testing.allocator, pages, page_bytes);
        w.f = .{ .pool = &w.pool };
        w.store = try radix.Store.initGroups(testing.allocator, w.f.snapshots(), radix.pagesOf(Pool, &w.pool, page_bytes), rules, .{ .parts = specs, .legacy = false }, w.f.hooks());
        w.store.limit(.{ .pages = pages, .snaps = 8 });
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
        @as(*World, @fieldParentPtr("store", s)).reclaims += 1;
        return s.reclaim(want);
    }
    /// A stream's lease of `n` pages.
    fn lease(w: *World, n: usize) !Lease {
        var l: Lease = .{};
        try w.pool.promise(&l, n);
        try w.pool.cover(&l, n, null);
        return l;
    }
};

const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

test "a copied part: held from the pool at keep, restored and fenced at a resume, back at eviction" {
    var w: World = .{ .pool = undefined };
    try w.init(&copied_state, 16);
    defer w.deinit();
    var l = try w.lease(2);
    w.f.live[0] = 77;
    var path: [2]u32 = undefined;
    const kept = w.store.keepParts(.of(&prompt), 8, null, l.ids.items, &.{.{ .base = 0, .ids = &.{} }}, &path);
    try expect(kept.held);
    try expectEqual(@as(usize, 5), w.store.tree.heldPages()); // two path pages, three part pages
    w.pool.finish(&l);
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try w.store.begin(testing.allocator, .of(&prompt), 14, &.{}, null, &adopt);
    defer testing.allocator.free(plan.marks);
    try expectEqual(@as(u32, 8), plan.from);
    try expectEqual(@as(?u32, 77), w.f.restored[0]);
    try expect(w.f.fences >= 2); // the keep's copies, then the restore's
    for (adopt.items) |id| w.pool.release(id);
    const before = w.pool.available();
    w.store.limit(.{ .pages = 16, .snaps = 0 }); // the entry goes, its parts with it
    try expectEqual(before + 5, w.pool.available()); // the part pages and the path the entry alone kept
}

test "a retained window with other rows a page: whole pages shared, the partial one copied, mapped at their logical index" {
    var w: World = .{ .pool = undefined };
    try w.init(&window3, 32);
    defer w.deinit();
    var l = try w.lease(2); // the prefix group
    var win = try w.lease(3); // the window: rows 0..8 on 3-row pages, the last one partial (2 rows)
    for (win.ids.items, 0..) |id, i| w.f.mem[id] = @intCast(100 + i);
    var path: [2]u32 = undefined;
    const r = [_]radix.Retained{.{ .base = 1, .ids = win.ids.items[1..3] }}; // its window omits page 0
    try expect(w.store.keepParts(.of(&prompt), 8, null, l.ids.items, &r, &path).held);
    try expectEqual(@as(u32, 2), w.f.rows_copied);
    try expectEqual(@as(u32, 2), w.pool.holders(win.ids.items[1])); // the stream and the store
    try expectEqual(@as(u32, 1), w.pool.holders(win.ids.items[2])); // the stream's partial page: never the store's
    const last_page = win.ids.items[2];
    w.f.mem[last_page] = 999; // the stream writes on: the checkpoint does not change
    w.pool.finish(&l);
    w.pool.finish(&win);
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try w.store.begin(testing.allocator, .of(&prompt), 14, &.{}, null, &adopt);
    defer plan.deinit(testing.allocator);
    try expectEqual(@as(usize, 1), plan.parts.len);
    try expectEqual(@as(u32, 1), plan.parts[0].base);
    try expectEqual(@as(usize, 1), plan.parts[0].ids.len); // page 1: whole
    try expectEqual(@as(usize, 1), plan.copies.len);
    try expectEqual(@as(u32, 2), plan.copies[0].index); // logical page 2, rows 0..2
    try expectEqual(@as(u32, 2), plan.copies[0].rows);
    try expectEqual(@as(u32, 102), w.f.mem[plan.copies[0].src]); // the rows as they were at the keep
    for (adopt.items) |id| w.pool.release(id);
    for (plan.parts[0].ids) |id| w.pool.release(id);
    const fences = w.f.fences;
    w.store.finish(null, plan.copies);
    try expectEqual(fences + 1, w.f.fences);
}

test "a part copy that fails publishes nothing: the path stays, the part pages go back after a fence" {
    var w: World = .{ .pool = undefined };
    try w.init(&both, 32);
    defer w.deinit();
    var l = try w.lease(2);
    defer w.pool.finish(&l);
    var win = try w.lease(3);
    defer w.pool.finish(&win);
    const r = [_]radix.Retained{ .{ .base = 0, .ids = win.ids.items }, .{ .base = 0, .ids = &.{} } };
    var path: [2]u32 = undefined;
    for ([_]bool{ true, false }) |rows_fail| {
        w.f.fail_rows = rows_fail;
        w.f.fail_copy = !rows_fail;
        const fences = w.f.fences;
        const kept = w.store.keepParts(.of(&prompt), 8, null, l.ids.items, &r, &path);
        try expect(!kept.held);
        try expectEqual(@as(usize, 2), kept.shared); // the path is the store's
        try expectEqual(@as(usize, 2), w.store.tree.heldPages()); // and nothing else
        try expectEqual(fences + 1, w.f.fences);
        try expect(w.f.held_at_fence); // the fence came before the pages went back
        try expectEqual(@as(usize, 0), w.store.tree.snaps);
        for (win.ids.items) |id| try expectEqual(@as(u32, 1), w.pool.holders(id));
    }
    try expectEqual(@as(u64, 2), w.store.tree.counts.failed);
}

test "two entries whose windows overlap hold each shared window page once" {
    var w: World = .{ .pool = undefined };
    try w.init(&window3, 32);
    defer w.deinit();
    var l = try w.lease(3);
    defer w.pool.finish(&l);
    var win = try w.lease(4); // rows 0..12 on 3-row pages
    defer w.pool.finish(&win);
    var path: [3]u32 = undefined;
    try expect(w.store.keepParts(.of(&prompt), 4, null, l.ids.items[0..1], &.{.{ .base = 0, .ids = win.ids.items[0..2] }}, path[0..1]).held);
    try expect(w.store.keepParts(.of(&prompt), 12, null, l.ids.items[0..3], &.{.{ .base = 0, .ids = win.ids.items[0..4] }}, path[0..3]).held);
    try expectEqual(@as(u32, 2), w.pool.holders(win.ids.items[0])); // named by both entries, held once by the store
}

test "reclaim frees a copied part while its path and retained part are a stream's; the stream keeps its pages" {
    var w: World = .{ .pool = undefined };
    try w.init(&both, 32);
    defer w.deinit();
    var l = try w.lease(2);
    defer w.pool.finish(&l);
    var win = try w.lease(3);
    defer w.pool.finish(&win);
    var path: [2]u32 = undefined;
    const r = [_]radix.Retained{ .{ .base = 0, .ids = win.ids.items[0..3] }, .{ .base = 0, .ids = &.{} } };
    try expect(w.store.keepParts(.of(&prompt), 8, null, l.ids.items, &r, &path).held);
    const before = w.pool.available();
    try expect(w.store.reclaim(before + 3)); // the copied part's 3 pages (and the partial copy) are store-only
    try expect(w.pool.available() >= before + 3);
    try expectEqual(@as(usize, 0), w.store.tree.snaps);
    for (l.ids.items) |id| try expect(w.pool.holders(id) >= 1); // the stream's pages stay
    for (win.ids.items) |id| try expectEqual(@as(u32, 1), w.pool.holders(id));
}

test "a restore that takes pool pages runs reclaim, which never evicts the entry being restored" {
    var w: World = .{ .pool = undefined };
    try w.init(&copied_state, 8);
    defer w.deinit();
    var l = try w.lease(2);
    w.f.live[0] = 5;
    var path: [2]u32 = undefined;
    try expect(w.store.keepParts(.of(&prompt), 8, null, l.ids.items, &.{.{ .base = 0, .ids = &.{} }}, &path).held);
    w.pool.finish(&l);
    w.f.hold_on_restore = w.pool.available() + 1; // one more than free: the pool asks the store
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try w.store.begin(testing.allocator, .of(&prompt), 14, &.{}, null, &adopt);
    defer testing.allocator.free(plan.marks);
    try expectEqual(@as(u32, 0), plan.from); // the restore could not get its pages: a prefill from the start
    try expectEqual(@as(?u32, null), w.f.restored[0]);
    try expectEqual(@as(usize, 0), adopt.items.len);
}

test "parts need hooks, a pool that holds pages, and rows for a retained group" {
    var pool = try Pool.init(testing.allocator, 4, page_bytes);
    defer pool.deinit();
    var f: Family = .{ .pool = &pool };
    const pages = radix.pagesOf(Pool, &pool, page_bytes);
    try testing.expectError(error.PartsNeedHooks, radix.Store.initGroups(testing.allocator, f.snapshots(), pages, rules, .{ .parts = &copied_state, .legacy = false }, null));
    var no_hold = pages;
    var vt = pages.vtable.*;
    vt.hold = null;
    no_hold.vtable = &vt;
    try testing.expectError(error.PartsNeedHold, radix.Store.initGroups(testing.allocator, f.snapshots(), no_hold, rules, .{ .parts = &copied_state, .legacy = false }, f.hooks()));
    const zero = [_]radix.Parts.PartSpec{.{ .kind = .window, .make = .retained }};
    try testing.expectError(error.RetainedRows, radix.Store.initGroups(testing.allocator, f.snapshots(), pages, rules, .{ .parts = &zero, .legacy = false }, f.hooks()));
}

test "an allocation failing at each step of a keep with parts: before the path, nothing changes; after it, the path stays alone" {
    var k: usize = 0;
    while (true) : (k += 1) {
        var pool = try Pool.init(testing.allocator, 32, page_bytes);
        defer pool.deinit();
        var f: Family = .{ .pool = &pool };
        var fa = std.testing.FailingAllocator.init(testing.allocator, .{});
        var store = try radix.Store.initGroups(fa.allocator(), f.snapshots(), radix.pagesOf(Pool, &pool, page_bytes), rules, .{ .parts = &both, .legacy = false }, f.hooks());
        store.limit(.{ .pages = 32, .snaps = 8 });
        var l: Lease = .{};
        try pool.promise(&l, 2);
        try pool.cover(&l, 2, null);
        var win: Lease = .{};
        try pool.promise(&win, 3);
        try pool.cover(&win, 3, null);
        const r = [_]radix.Retained{ .{ .base = 0, .ids = win.ids.items }, .{ .base = 0, .ids = &.{} } };
        var path: [2]u32 = undefined;
        fa.fail_index = fa.alloc_index + k;
        const kept = store.keepParts(.of(&prompt), 8, null, l.ids.items, &r, &path);
        const failed = fa.has_induced_failure;
        if (failed) {
            try expect(!kept.held);
            try expectEqual(@as(usize, 0), store.tree.snaps);
            const path_pages: usize = if (kept.shared == 2) 2 else 0; // after the insert the path is the store's
            try expectEqual(path_pages, store.tree.heldPages());
            for (win.ids.items) |id| try expectEqual(@as(u32, 1), pool.holders(id)); // no part outlived the failure
        } else try expect(kept.held);
        pool.finish(&l);
        pool.finish(&win);
        store.deinit();
        try expect(pool.idle() and pool.available() == pool.total());
        if (!failed) break;
    }
    try expect(k > 3); // the insert, the parts, the copies and the prompt each failed once
}

test "a retained group without its pages through the mark is refused, not counted" {
    var w: World = .{ .pool = undefined };
    try w.init(&window3, 16);
    defer w.deinit();
    var l = try w.lease(2);
    defer w.pool.finish(&l);
    var path: [2]u32 = undefined;
    try expect(!w.store.keepParts(.of(&prompt), 8, null, l.ids.items, &.{.{ .base = 3, .ids = &.{} }}, &path).held);
    try expectEqual(@as(usize, 0), w.store.tree.heldPages());
}

test "a keep whose parts need more than the free pages reclaims, but never the path it is building" {
    var w: World = .{ .pool = undefined };
    try w.init(&copied_state, 6);
    defer w.deinit();
    var l = try w.lease(2);
    var path: [2]u32 = undefined;
    w.f.fail_copy = true; // a failed checkpoint leaves its path alone in the tree
    try expect(!w.store.keepParts(.of(&prompt), 8, null, l.ids.items, &.{.{ .base = 0, .ids = &.{} }}, &path).held);
    w.pool.finish(&l); // the path is now the store's alone: idle, reclaimable
    var other = try w.lease(2); // 6 pages: 2 the path's, 2 this stream's, 2 free; the part needs 3
    defer w.pool.finish(&other);
    w.f.fail_copy = false;
    const asked = w.reclaims;
    const kept = w.store.keepParts(.of(&prompt), 8, null, other.ids.items, &.{.{ .base = 0, .ids = &.{} }}, &path);
    try expect(w.reclaims > asked); // the part's hold ran the reclaim chain
    try expect(!kept.held); // the only reclaimable pages were the path being built: refused, not pruned
    try expectEqual(@as(usize, 2), w.store.tree.heldPages()); // the path is still the tree's
    try expectEqual(@as(usize, 2), kept.shared);
}

test "a plan's retained ids are its own: an eviction between begin and adoption leaves them valid" {
    var w: World = .{ .pool = undefined };
    try w.init(&both, 32);
    defer w.deinit();
    var l = try w.lease(2);
    var win = try w.lease(3);
    var path: [2]u32 = undefined;
    const r = [_]radix.Retained{ .{ .base = 0, .ids = win.ids.items }, .{ .base = 0, .ids = &.{} } };
    try expect(w.store.keepParts(.of(&prompt), 8, null, l.ids.items, &r, &path).held);
    const whole = [_]u32{ win.ids.items[0], win.ids.items[1] };
    w.pool.finish(&l);
    w.pool.finish(&win);
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try w.store.begin(testing.allocator, .of(&prompt), 14, &.{}, null, &adopt);
    defer plan.deinit(testing.allocator);
    w.store.limit(.{ .pages = 32, .snaps = 0 }); // the entry goes before the stream maps anything
    try expectEqualSlices(u32, &whole, plan.parts[0].ids);
    for (plan.parts[0].ids) |id| try expectEqual(@as(u32, 1), w.pool.holders(id)); // the stream's holds alone
    for (plan.parts[0].ids) |id| w.pool.release(id);
    w.store.finish(null, plan.copies);
    for (adopt.items) |id| w.pool.release(id);
}

test "a begin whose hand-out allocation fails gives back the pages it adopted" {
    var w: World = .{ .pool = undefined };
    try w.init(&window3, 32);
    defer w.deinit();
    var l = try w.lease(2);
    var win = try w.lease(3);
    var path: [2]u32 = undefined;
    try expect(w.store.keepParts(.of(&prompt), 8, null, l.ids.items, &.{.{ .base = 0, .ids = win.ids.items }}, &path).held);
    w.pool.finish(&l);
    w.pool.finish(&win);
    var k: usize = 0;
    while (true) : (k += 1) {
        var fa = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = k });
        var adopt: std.ArrayList(u32) = .empty;
        defer adopt.deinit(testing.allocator);
        const census = w.pool.census();
        const plan = w.store.begin(fa.allocator(), .of(&prompt), 14, &.{}, null, &adopt) catch {
            try expectEqual(@as(usize, 0), adopt.items.len);
            try expectEqual(census, w.pool.census()); // every hold this begin took went back
            continue;
        };
        for (adopt.items) |id| w.pool.release(id);
        for (plan.parts[0].ids) |id| w.pool.release(id);
        w.store.finish(null, plan.copies);
        plan.deinit(fa.allocator());
        break;
    }
    try expect(k >= 3); // the marks, the parts, their ids and the copies each failed once
}

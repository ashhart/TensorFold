//! The pool under the paged prompt store: random admits, keeps, resumes and ends leave every page free or the store's.
const std = @import("std");
const kv = @import("kv_pool");
const radix = @import("engine_api").prompt_radix;
const Pool = kv.Pool;
const Lease = kv.Lease;
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const page = 4;
const layout: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = 25, .page_tokens = page, .snap_bytes = 100 };

/// A family whose state is the row it was kept at; saves and restores fail on demand.
const Family = struct {
    gpa: Allocator,
    live: usize = 0,
    fail_save: bool = false,
    fail_restore: bool = false,

    const State = struct { at: u32 };

    fn of(ptr: *anyopaque) *Family {
        return @ptrCast(@alignCast(ptr));
    }
    fn snapshots(f: *Family) radix.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 100;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!radix.Saved {
        const f = of(ptr);
        if (f.fail_save) return error.CopyFailed;
        const st = try f.gpa.create(State);
        st.* = .{ .at = at };
        f.live += 1;
        return st;
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, _: radix.Saved) anyerror!void {
        if (of(ptr).fail_restore) return error.CopyFailed;
    }
    fn drop(ptr: *anyopaque, saved: radix.Saved) void {
        const f = of(ptr);
        f.gpa.destroy(@as(*State, @ptrCast(@alignCast(saved))));
        f.live -= 1;
    }
};

/// A stream's device page table that fails its next write when told to.
const Table = struct {
    fail: bool = false,
    fn table(t: *Table) kv.Table {
        return .{ .ctx = t, .write = write };
    }
    fn write(ctx: *anyopaque, _: []const u32, _: usize) anyerror!void {
        const t: *Table = @ptrCast(@alignCast(ctx));
        if (t.fail) return error.UploadFailed;
    }
};

/// A pool, a paged store over it, and the live streams' leases, as a backend holds them.
const World = struct {
    gpa: Allocator,
    pool: Pool,
    store: radix.Store,
    family: *Family,
    streams: std.ArrayList(Lease) = .empty,
    table: Table = .{},
    refused: usize = 0,
    failed: usize = 0,

    fn reclaim(ctx: *anyopaque, want: usize) bool {
        const s: *radix.Store = @ptrCast(@alignCast(ctx));
        return s.reclaim(want);
    }

    /// A request as the backend admits it: resume, promise the rest of its need, cover it, keep at its marks.
    fn admit(w: *World, a: Allocator, prompt: []const u32, history: u32, max_new: u32) !void {
        var adopt: std.ArrayList(u32) = .empty;
        defer adopt.deinit(w.gpa);
        const plan = try w.store.begin(a, prompt, history, &.{}, null, &adopt);
        var l: Lease = .{};
        errdefer w.pool.finish(&l);
        try w.pool.adopt(&l, adopt.items);
        const need = layout.need(prompt.len, max_new, 1);
        try w.pool.promise(&l, need - adopt.items.len);
        try w.pool.cover(&l, need, w.table.table());
        for (plan.marks) |m| {
            var path: [64]u32 = undefined;
            const n = m / page;
            const kept = w.store.keep(prompt, m, null, l.ids.items[0..n], path[0..n]);
            for (path[0..kept.shared], 0..) |id, i| w.pool.share(&l, i, id);
        }
        try w.streams.append(w.gpa, l);
    }

    fn end(w: *World, i: usize) void {
        var l = w.streams.swapRemove(i);
        w.pool.finish(&l);
    }

    /// Every page's holders are the leases mapping it and the store's node holding it; the free list is the rest.
    fn check(w: *const World) !void {
        const n = w.pool.total();
        var refs: [64]u32 = @splat(0);
        var leased: [64]u32 = @splat(0);
        var promised: usize = 0;
        for (w.streams.items) |l| {
            promised += l.promised;
            for (l.ids.items) |id| {
                refs[id] += 1;
                leased[id] += 1;
            }
        }
        held(w.store.tree.root, &refs);
        try std.testing.expectEqualSlices(u32, refs[0..n], w.pool.refs);
        try std.testing.expectEqualSlices(u32, leased[0..n], w.pool.leased);
        try expectEqual(promised, w.pool.promised);
        var seen: [64]bool = @splat(false);
        for (w.pool.free[0..w.pool.nfree]) |id| {
            try expect(!seen[id] and refs[id] == 0);
            seen[id] = true;
        }
        var zero: usize = 0;
        for (refs[0..n]) |r| zero += @intFromBool(r == 0);
        try expectEqual(zero, w.pool.nfree);
    }

    fn held(node: *const radix.Node, refs: *[64]u32) void {
        for (node.pages) |id| refs[id] += 1;
        for (node.kids.items) |k| held(k, refs);
    }
};

/// One random run: conversations grow turn by turn over a shared block; streams end, fail or are refused at random.
fn run(w: *World, seed: u64, pages: usize) !void {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const family = try gpa.create(Family);
    family.* = .{ .gpa = gpa };
    w.* = .{ .gpa = gpa, .pool = try Pool.init(gpa, pages, layout.pageBytes()), .store = undefined, .family = family };
    w.store = try radix.Store.init(gpa, family.snapshots(), radix.pagesOf(Pool, &w.pool, w.pool.page_bytes), .{ .page = page, .min_gap = 8, .min_prompt = 0, .tail = true });
    w.store.limit(.{ .pages = pages / 2, .snaps = 4 });
    w.pool.reclaim = .{ .ctx = &w.store, .call = World.reclaim };
    var convs: [4][64]u32 = undefined;
    var lens: [4]u32 = @splat(0);
    for (&convs, 0..) |*c, k| for (c, 0..) |*t, i| {
        t.* = if (i < 8) @intCast(i + 1) else @intCast(1000 * (k + 1) + i); // an 8-token block every conversation shares
    };
    for (0..80) |_| {
        _ = arena.reset(.retain_capacity);
        switch (rnd.uintLessThan(u8, 10)) {
            0...4 => { // the next turn of a conversation (an edit now and then: its earlier turn no longer matches)
                const k = rnd.uintLessThan(usize, 4);
                const history = lens[k];
                lens[k] = @min(56, history + 3 + rnd.uintLessThan(u32, 9));
                if (rnd.uintLessThan(u8, 8) == 0) convs[k][rnd.uintLessThan(usize, 48) + 8] +%= 1;
                w.table.fail = rnd.uintLessThan(u8, 10) == 0;
                w.family.fail_save = rnd.uintLessThan(u8, 10) == 0;
                w.family.fail_restore = rnd.uintLessThan(u8, 12) == 0;
                w.admit(arena.allocator(), convs[k][0..lens[k]], history, rnd.uintLessThan(u32, 9)) catch |e| switch (e) {
                    error.LaneMemory => w.refused += 1, // the host requeues it until a stream ends
                    error.UploadFailed => w.failed += 1,
                    else => return e,
                };
            },
            5...7 => if (w.streams.items.len > 0) w.end(rnd.uintLessThan(usize, w.streams.items.len)), // finish or cancel
            8 => w.store.limit(.{ .pages = rnd.uintLessThan(usize, pages), .snaps = rnd.uintLessThan(usize, 6) }),
            else => _ = w.store.reclaim(rnd.uintLessThan(usize, pages)),
        }
        try w.check();
    }
}

test "random admits, keeps, resumes, finishes, cancels and failures: every page free or the store's, then all free" {
    var refused: usize = 0;
    var failed: usize = 0;
    var adopted: usize = 0;
    for (0..300) |seed| {
        var w: World = undefined;
        try run(&w, seed, 24 + seed % 17);
        refused += w.refused;
        failed += w.failed;
        adopted += w.store.tree.counts.hits;
        while (w.streams.items.len > 0) w.end(0);
        try w.check();
        try expect(w.pool.idle()); // the idle server's line: promised 0, free + store == total
        w.store.limit(.{}); // the store cleared
        try w.check();
        for (w.pool.refs) |r| try expectEqual(@as(u32, 0), r);
        try expectEqual(w.pool.total(), w.pool.nfree);
        try expectEqual(@as(usize, 0), w.family.live);
        w.streams.deinit(w.gpa);
        w.store.deinit();
        w.pool.deinit();
        w.gpa.destroy(w.family);
    }
    try expect(refused > 0 and failed > 0 and adopted > 0); // the runs met full pools, failed uploads and resumes
}

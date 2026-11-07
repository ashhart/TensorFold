//! TF_GLM_KV_SPLIT on a driver of addresses alone: the two regions and tables, each owner's ids, a follower's checks.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const state = @import("cuda_state.zig");
const pages_mod = @import("cuda_pages.zig");
const Engine = @import("cuda_engine.zig").Engine;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const gpa = std.testing.allocator;

/// The driver: each allocation a fresh address, every table write and NaN fill kept in order.
const F = struct {
    var next: u64 = 1 << 30;
    var live: usize = 0;
    var puts: usize = 0;
    var at: [16]u64 = @splat(0); // a table's address, and its words below
    var words: [16][32]u32 = undefined;
    var fills: std.ArrayList([2]u64) = .empty; // (address, words) of each NaN fill
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(u64, n + 1, 1 << 20);
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        live -= 1;
        return 0;
    }
    fn clear(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn fill(p: u64, v: c_uint, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (v == 0xFFFF_FFFF) fills.append(gpa, .{ p, n }) catch return 2;
        return 0;
    }
    fn put(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        const ids: [*]const u32 = @ptrCast(@alignCast(src.?));
        puts += 1;
        for (&at, &words) |*a, *w| if (a.* == 0 or (dst >= a.* and dst < a.* + 128)) {
            if (a.* == 0) a.* = dst;
            @memcpy(w[(dst - a.*) / 4 ..][0 .. n / 4], ids[0 .. n / 4]);
            return 0;
        };
        return 1;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    /// The first `n` words written to the table at `t`.
    fn table(t: u64, n: usize) []const u32 {
        for (at, &words) |a, *w| if (a == t) return w[0..n];
        return &.{};
    }
};

/// An engine with what the pages read: TP=2's shapes on a KDA, a DSA and a KDA layer, 4,096 rows (8 pages a window).
pub const Fake = struct {
    d: cuda.Driver,
    layers: [3]weights.Layer,
    e: Engine,

    pub fn init(f: *Fake, split: bool) !void {
        F.live = 0;
        F.puts = 0;
        F.at = @splat(0);
        F.fills.clearRetainingCapacity();
        f.d = undefined;
        f.d.api.cuMemAlloc_v2 = F.alloc;
        f.d.api.cuMemFree_v2 = F.free;
        f.d.api.cuMemsetD8_v2 = F.clear;
        f.d.api.cuMemsetD32Async = F.fill;
        f.d.api.cuMemcpyHtoD_v2 = F.put;
        f.d.api.cuStreamSynchronize = F.wait;
        for (&f.layers, [_]bool{ false, true, false }) |*l, mla| l.mixer = if (mla) .{ .mla = undefined } else .{ .kda = undefined };
        f.e = undefined;
        f.e.gpa = gpa;
        f.e.d = &f.d;
        f.e.stream = .{ .d = &f.d, .handle = null };
        f.e.capacity = 4096;
        f.e.seqs = .empty;
        f.e.pages = null;
        f.e.images = null;
        f.e.settings = .{ .kv_split = split };
        f.e.w = undefined;
        f.e.w.config = .{};
        f.e.w.share = try shard.Share.of(&f.e.w.config, 0, 2);
        f.e.w.layers = &f.layers;
        f.e.share = f.e.w.share; // rank 0 of two: an owner class each
    }

    pub fn deinit(f: *Fake, p: ?*pages_mod.Pages) void {
        if (p) |q| q.deinit();
        for (f.e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            f.e.gpa.destroy(s);
        };
        f.e.seqs.deinit(f.e.gpa);
        F.fills.clearAndFree(gpa);
    }

    /// The engine's own sequence, then a pool of `pages` beside it for `streams` sequences.
    pub fn pool(f: *Fake, streams: usize, pages: usize, leader: bool) !*pages_mod.Pages {
        try pages_mod.ownSequence(&f.e);
        return pages_mod.Pages.init(gpa, &f.e, try pages_mod.layout(gpa, &f.e, 0), streams, pages, leader);
    }

    pub fn st(f: *Fake, slot: usize) *state.State {
        return &f.e.seqs.items[slot].?.st;
    }
};

test "GLM KV split: a page's pooled keys and latent rows lie in two regions, each cache at its part, two tables a stream" {
    var f: Fake = undefined;
    try f.init(true);
    const off = blk: {
        var g: Fake = undefined;
        try g.init(false);
        defer g.deinit(null);
        break :blk try pages_mod.layout(gpa, &g.e, 0);
    };
    const l = try pages_mod.layout(gpa, &f.e, 0);
    try expectEqual(off.token_bytes, l.token_bytes); // the same page, its parts apart
    try expectEqual(off.fixed_bytes + 256, l.fixed_bytes); // the latent table, 8 entries as the arena rounds them
    const p = try f.pool(3, 16, true);
    defer f.deinit(p);
    const pooled = state.pkPageBytes(&f.e.w);
    const latent = state.latentPageBytes(&f.e.w);
    try expectEqual(state.pageBytes(&f.e.w), pooled + latent);
    try expectEqual(@as(u64, 2 * 128 * 256), pooled); // a DSA layer and the head: 128 pooled keys of 256 bytes each
    const base = p.arena.?.ptr;
    for (0..3) |slot| {
        const st = f.st(slot); // the engine's own (placed after the pool) and the two made with it
        try expectEqual(st.table + 256, st.ltable); // after the table, so a kept state's tail still ends at it
        for (st.caches, 0..) |c, j| {
            try expectEqual(base + 16 * pooled + j * 512 * 1024, c.lc);
            try expectEqual(base + j * 128 * 256, c.pk);
            try expect(c.pt == st.ltable and c.ps == latent and c.pkt == st.table and c.pks == pooled);
        }
        const h = st.mtp_caches; // the head's latent rows are a cache of the latent region like the others
        try expectEqual(base + 16 * pooled + st.caches.len * 512 * 1024, h.lc);
        try expectEqual(base + st.caches.len * 128 * 256, h.pk);
        try expect(h.pt == st.ltable and h.ps == latent and h.pkt == st.table and h.pks == pooled);
    }
}

test "GLM KV split: off, one table and the interleaved page, as before the split" {
    var f: Fake = undefined;
    try f.init(false);
    const p = try f.pool(2, 8, true);
    defer f.deinit(p);
    const st = f.st(1);
    try expectEqual(@as(u64, 0), st.ltable);
    const c = st.caches[0];
    try expect(c.pt == st.table and c.pkt == 0 and c.pks == 0 and c.ps == state.pageBytes(&f.e.w));
    try expectEqual(c.lc + 512 * 1024, c.pk);
    _ = try p.open(1, 600, 100);
    try expectEqual(@as(usize, 1), F.puts); // one table written
}

test "GLM KV split: a stream's ids reach both tables, the latent one holding the same ids while every rank holds every row" {
    var f: Fake = undefined;
    try f.init(true);
    const p = try f.pool(3, 16, true);
    defer f.deinit(p);
    const ids = try p.open(1, 1500, 100);
    try expectEqual(@as(usize, 2), F.puts);
    try std.testing.expectEqualSlices(u32, ids, F.table(f.st(1).table, ids.len));
    try std.testing.expectEqualSlices(u32, ids, F.table(f.st(1).ltable, ids.len));
}

test "GLM KV split: TF_KV_POISON fills a stream's pages in both regions, each at its own stride" {
    var f: Fake = undefined;
    try f.init(true);
    const p = try f.pool(3, 16, true);
    defer f.deinit(p);
    p.poison = true;
    const ids = try p.open(1, 600, 100);
    const pooled = state.pkPageBytes(&f.e.w);
    const latent = state.latentPageBytes(&f.e.w);
    const base = p.arena.?.ptr;
    try expectEqual(2 * ids.len, F.fills.items.len);
    for (ids, 0..) |id, i| {
        try expectEqual([2]u64{ base + id * pooled, pooled / 4 }, F.fills.items[2 * i]);
        try expectEqual([2]u64{ base + 16 * pooled + id * latent, latent / 4 }, F.fills.items[2 * i + 1]);
    }
}

test "GLM KV split: rank 0 books each stream from the roomiest owner class, every id of it, and every class idle after" {
    var f: Fake = undefined;
    try f.init(true);
    const p = try f.pool(4, 16, true); // 8 pages a class, a window each
    defer f.deinit(p);
    try expectEqual(@as(u32, 2), p.pool.classes);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, try p.open(1, 600, 100)); // all tie: class 0
    try expectEqual(@as(u32, 0), p.leases[1].owner);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 5 }, try p.open(2, 1000, 100)); // class 1 has more
    try expectEqual(@as(u32, 1), p.leases[2].owner);
    try expectEqual(@as(u32, 0), p.pool.roomiest()); // 6 against 5
    const third = try p.open(3, 600, 100);
    for (third) |id| try expectEqual(@as(u32, 0), id % 2);
    p.close(3); // a stream ended before its first decode (a cancel): its class's pages back at once
    try expectEqual(@as(usize, 6), p.pool.classFree(0));
    p.close(1);
    p.close(2);
    try expect(p.pool.idle());
    for (0..2) |c| try expect(p.pool.classFree(@intCast(c)) == 8 and p.pool.classPromised(@intCast(c)) == 0);
    try expectEqual(@as(u32, 0), p.leases[2].owner); // finish resets the lease
    try std.testing.expectEqualSlices(u32, &.{ 5, 3 }, try p.openOn(1, 600, 100, 1)); // a pinned owner, last ids back first
    try std.testing.expectError(error.BadStep, p.openOn(2, 600, 100, 2));
    try expect(!p.used[2] and p.pool.classPromised(0) == 0);
}

test "GLM KV split: a class runs out alone: admission waits for its pages though the other class has them" {
    var f: Fake = undefined;
    try f.init(true);
    const p = try f.pool(4, 16, true);
    defer f.deinit(p);
    _ = try p.openOn(1, 3000, 100, 0); // 7 of class 0's 8
    _ = try p.openOn(2, 3000, 100, 1); // 7 of class 1's
    try expectEqual(@as(usize, 2), p.pool.available());
    try expect(p.fits(100, 100) and !p.fits(600, 100)); // one page fits a class; two do not, though two are free
    try std.testing.expectError(error.LaneMemory, p.open(3, 600, 100));
    try expect(!p.used[3] and p.pool.availableIn(0) == 1 and p.pool.availableIn(1) == 1);
}

test "GLM KV split: a follower maps rank 0's ids of the stream's owner, refusing another class or owner before mapping" {
    var f: Fake = undefined;
    try f.init(true);
    const p = try f.pool(3, 16, false);
    defer f.deinit(p);
    try std.testing.expectError(error.ResidueMismatch, p.mapOwned(1, &.{ 1, 4, 5 }, 0, 1));
    try std.testing.expectError(error.BadStep, p.mapOwned(1, &.{ 2, 4 }, 0, 2)); // owner past the classes
    try expect(F.puts == 0 and !p.used[1] and f.st(1).capacity == 0); // nothing mapped or uploaded
    try p.mapOwned(1, &.{ 5, 1, 7 }, 0, 1);
    try expectEqual(@as(u32, 1), p.leases[1].owner);
    try std.testing.expectEqualSlices(u32, &.{ 5, 1, 7 }, F.table(f.st(1).ltable, 3));
    p.close(1);
    try expectEqual(@as(u32, 0), p.leases[1].owner);
    try p.map(1, &.{ 2, 0 }, 0); // `map` is owner 0's
}

test "GLM KV split: a follower whose switch differs from rank 0's pool step is refused, and makes no pool" {
    for ([_]bool{ true, false }) |mine| {
        var f: Fake = undefined;
        try f.init(mine);
        defer f.deinit(null);
        try pages_mod.ownSequence(&f.e);
        const live = F.live;
        const page = (try pages_mod.layout(gpa, &f.e, 0)).pageBytes();
        try std.testing.expectError(error.RankMismatch, pages_mod.Pages.followFlagged(gpa, &f.e, 3, 16, page, !mine));
        try expectEqual(live, F.live);
        const p = try pages_mod.Pages.followFlagged(gpa, &f.e, 3, 16, page, mine);
        p.deinit();
    }
}

test "GLM KV split: a class shorter than a stream's window is refused before anything is made; off builds as before" {
    {
        var g: Fake = undefined; // off: 14 pages build as before
        try g.init(false);
        g.deinit(try g.pool(3, 14, true));
    }
    var f: Fake = undefined;
    try f.init(true);
    defer f.deinit(null);
    try pages_mod.ownSequence(&f.e);
    const l = try pages_mod.layout(gpa, &f.e, 0);
    const live = F.live;
    try std.testing.expectError(error.ClassTooSmall, pages_mod.Pages.init(gpa, &f.e, l, 3, 14, true)); // 7 < 8
    try std.testing.expectError(error.ClassTooSmall, pages_mod.Pages.followFlagged(gpa, &f.e, 3, 14, l.pageBytes(), true));
    try std.testing.expectError(error.BadClasses, pages_mod.Pages.init(gpa, &f.e, l, 3, 17, true));
    try expectEqual(live, F.live);
}

test "GLM KV split: a request longer than a class fits admission and is refused by name, never as LaneMemory" {
    var f: Fake = undefined;
    try f.init(true);
    const p = try f.pool(3, 16, true);
    defer f.deinit(p);
    f.e.capacity = 8192; // as a table longer than a class would be, were it not refused at init
    try expect(p.fits(5000, 100)); // 11 pages: past a class of 8, so the lane host does not wait for it
    try std.testing.expectError(error.ClassTooSmall, p.open(1, 5000, 100));
    try expect(!p.used[1] and p.pool.idle());
}

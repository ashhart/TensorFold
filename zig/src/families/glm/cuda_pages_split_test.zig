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
const Fake = struct {
    d: cuda.Driver,
    layers: [3]weights.Layer,
    e: Engine,

    fn init(f: *Fake, split: bool) !void {
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
    }

    fn deinit(f: *Fake, p: ?*pages_mod.Pages) void {
        if (p) |q| q.deinit();
        for (f.e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            f.e.gpa.destroy(s);
        };
        f.e.seqs.deinit(f.e.gpa);
        F.fills.clearAndFree(gpa);
    }

    /// The engine's own sequence, then a pool of `pages` beside it for `streams` sequences.
    fn pool(f: *Fake, streams: usize, pages: usize, leader: bool) !*pages_mod.Pages {
        try pages_mod.ownSequence(&f.e);
        return pages_mod.Pages.init(gpa, &f.e, try pages_mod.layout(gpa, &f.e, 0), streams, pages, leader);
    }

    fn st(f: *Fake, slot: usize) *state.State {
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

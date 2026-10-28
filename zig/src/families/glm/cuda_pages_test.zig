//! GLM's KV pages on a driver of addresses alone: tables in row order, leases given back, a follower's ids checked.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const pages_mod = @import("cuda_pages.zig");
const Plan = @import("cuda_snapshot_layout.zig").Plan;
const page_rows = @import("cuda_state.zig").page_rows;
const sparse = @import("cuda_sparse.zig");
const prefix_mod = @import("cuda_prefix.zig");
const Kept = @import("cuda_engine.zig").Kept;
const Engine = @import("cuda_engine.zig").Engine;
const Lanes = @import("cuda_lanes.zig").Lanes;
const lanes = @import("lanes");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

/// The driver: each allocation a fresh address, the page tables written to it kept, the NaN fills counted.
const F = struct {
    var next: u64 = 1 << 30;
    var live: usize = 0;
    var fills: usize = 0;
    var at: [8]u64 = @splat(0); // a table's address, and its words below
    var words: [8][32]u32 = undefined;
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
    fn fill(_: u64, v: c_uint, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        fills += @intFromBool(v == 0xFFFF_FFFF);
        return 0;
    }
    fn put(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        const ids: [*]const u32 = @ptrCast(@alignCast(src.?));
        for (&at, &words) |*a, *w| if (a.* == 0 or (dst >= a.* and dst < a.* + 128)) {
            if (a.* == 0) a.* = dst;
            const from = (dst - a.*) / 4;
            @memcpy(w[from..][0 .. n / 4], ids[0 .. n / 4]);
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

/// An engine with what the pages read: a driver, TP=2's shapes on a KDA, a DSA and a KDA layer, 4,096 rows.
const Fake = struct {
    d: cuda.Driver,
    layers: [3]weights.Layer,
    e: Engine,

    fn init(f: *Fake, gpa: std.mem.Allocator) !void {
        F.live = 0;
        F.fills = 0;
        F.at = @splat(0);
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
        f.e.settings = .{};
        f.e.w = undefined;
        f.e.w.config = .{};
        f.e.w.share = try shard.Share.of(&f.e.w.config, 0, 2);
        f.e.w.layers = &f.layers;
    }

    /// The pages first (if any), then every slot's sequence, as the engine frees them.
    fn deinit(f: *Fake, p: ?*pages_mod.Pages) void {
        if (p) |q| q.deinit();
        for (f.e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            f.e.gpa.destroy(s);
        };
        f.e.seqs.deinit(f.e.gpa);
    }

    /// The engine's own sequence as `Engine.init` makes it when TF_KV_PAGES pages, then the pool beside it.
    fn pool(f: *Fake, l: anytype, streams: usize, pages: usize, leader: bool) !*pages_mod.Pages {
        try pages_mod.ownSequence(&f.e);
        return pages_mod.Pages.init(f.e.gpa, &f.e, l, streams, pages, leader);
    }
};

test "a paged stream takes its request's pages, its table holds them in row order, and its end gives them back" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    try expectEqual(@as(u64, 2 * 1088), l.token_bytes); // a DSA layer and the MTP layer: 1,024 + 64 bytes each
    const p = try f.pool(l, 3, 8, true);
    defer f.deinit(p);
    try expect(p.paged(0) and p.paged(1) and p.paged(2) and !p.paged(3));
    try expect(p.fits(600, 100));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, try p.open(1, 600, 100)); // 708 rows: 2 pages
    const st1 = &f.e.seqs.items[1].?.st;
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, F.table(st1.table, 2));
    try expectEqual(@as(usize, 1024), st1.capacity);
    try expect(!p.fits(3000, 100)); // 7 pages of the 6 left
    try std.testing.expectError(error.LaneMemory, p.open(2, 3000, 100));
    try expectEqual(@as(usize, 0), f.e.seqs.items[2].?.st.capacity);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4, 5, 6 }, try p.open(2, 2000, 100));
    p.close(1);
    try expectEqual(@as(usize, 0), st1.capacity);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, try p.open(1, 600, 100)); // the last page given back, first
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, F.table(st1.table, 2));
    p.close(1);
    p.close(2);
    try expect(p.pool.idle() and p.pool.census().free == 8);
    try std.testing.expectError(error.PromptTooLong, p.open(1, 4096, 0)); // past the window's table
}

test "the engine's own sequence holds no rows before its pool, then sits on the pool's pages and takes them like the others" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    try pages_mod.ownSequence(&f.e);
    const own = f.e.seqs.items[0].?;
    try expect(f.e.cur == own and f.e.slot == 0 and own.st.table != 0 and own.st.capacity == 0); // no window made
    const live = F.live;
    const p = try pages_mod.Pages.init(std.testing.allocator, &f.e, l, 2, 8, true);
    defer f.deinit(p);
    try expect(f.e.cur == own and f.e.seqs.items[0].? == own); // the same sequence: it is placed, not made again
    try expectEqual(live + 2, F.live); // the pages' arena and slot 1's sequence
    try expectEqual(p.arena.?.ptr, own.st.caches[0].lc); // its DSA layer's part of page 0 starts the arena
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, try p.open(0, 600, 100));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, F.table(own.st.table, 2));
    try expectEqual(@as(usize, 1024), own.st.capacity);
    p.close(0);
    try expect(own.st.capacity == 0 and p.pool.idle() and p.pool.census().free == 8);
}

test "a pool is refused while the engine's own sequence holds a window or was never made, and nothing is made for it" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    defer f.deinit(null);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    try std.testing.expectError(error.OwnNotPaged, pages_mod.Pages.init(std.testing.allocator, &f.e, l, 2, 8, true));
    try f.e.bind(0); // a window, as without TF_KV_PAGES
    try std.testing.expectError(error.OwnNotPaged, pages_mod.Pages.init(std.testing.allocator, &f.e, l, 2, 8, true));
    try expectEqual(@as(usize, 1), F.live); // the window alone
    try expectEqual(@as(usize, 1), f.e.seqs.items.len);
}

test "a follower maps rank 0's ids, and refuses ids past its pool or a slot that is not paged" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const p = try f.pool(l, 3, 8, false);
    defer f.deinit(p);
    try p.map(2, &.{ 5, 3, 7 }, 0);
    try std.testing.expectEqualSlices(u32, &.{ 5, 3, 7 }, F.table(f.e.seqs.items[2].?.st.table, 3));
    try expectEqual(@as(usize, 1536), f.e.seqs.items[2].?.st.capacity);
    try std.testing.expectError(error.BadStep, p.map(1, &.{8}, 0));
    try std.testing.expectError(error.BadStep, p.map(3, &.{1}, 0));
    p.close(2);
    try expectEqual(@as(usize, 0), f.e.seqs.items[2].?.st.capacity);
    try expect(p.pool.idle()); // a follower's pool lends nothing
}

test "TF_GLM_KV_SHARE at TP: two followers map rank 0's resumed table, NaN past the cache's pages, more kept refused" {
    var ids: [4]u32 = undefined;
    {
        var f: Fake = undefined; // rank 0: the prompt cache keeps the first two pages, and a resume holds them again
        try f.init(std.testing.allocator);
        const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
        const p = try f.pool(l, 3, 8, true);
        defer f.deinit(p);
        _ = try p.open(1, 1500, 100);
        for ([_]u32{ 0, 1, 0, 1 }) |id| p.pool.retain(id);
        p.close(1);
        @memcpy(&ids, try p.openWith(2, 1500, 100, &.{ 0, 1 }));
        p.close(2);
        for ([_]u32{ 0, 1 }) |id| p.pool.release(id);
        try expect(p.pool.idle());
    }
    for (1..3) |_| { // ranks 1 and 2, one after the other on the fake driver
        var f: Fake = undefined;
        try f.init(std.testing.allocator);
        const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
        const p = try f.pool(l, 3, 8, false);
        defer f.deinit(p);
        p.poison = true;
        try p.map(2, &ids, 2);
        try std.testing.expectEqualSlices(u32, &ids, F.table(f.e.seqs.items[2].?.st.table, 4));
        try expectEqual(@as(usize, 2), F.fills); // the stream's two fresh pages alone
        p.close(2);
        try std.testing.expectError(error.BadStep, p.map(2, ids[0..2], 3));
    }
}

test "a follower refuses rank 0's pool when its KV page has other bytes, and makes no pool" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    try pages_mod.ownSequence(&f.e);
    const more = l.pageBytes() + 1088 * 512; // rank 0 with one more paged layer
    try std.testing.expectError(error.RankMismatch, pages_mod.Pages.follow(std.testing.allocator, &f.e, 2, 8, more));
    try expectEqual(@as(usize, 1), F.live); // the engine's own sequence alone: no arena, no paged sequence
    const p = try pages_mod.Pages.follow(std.testing.allocator, &f.e, 2, 8, l.pageBytes());
    defer f.deinit(p);
    try expect(!p.leader and p.pool.total() == 8);
}

test "TF_KV_PAGES: a stream past the paged sequences waits for one, and its slot is never made a whole window" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const p = try f.pool(l, 2, 8, true);
    defer f.deinit(p);
    f.e.pages = p;
    var ln: Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer ln.lanes.deinit(f.e.gpa);
    var streams: [3]lanes.Stream = undefined; // as many at once as lanes: the engine's own sequence and one more
    try expectEqual(@as(usize, 2), (try ln.seat(&streams[0], 600, 100, &.{})).len);
    try expectEqual(@as(u32, 0), ln.lanes.get(&streams[0]).?.slot); // the engine's own takes pages too
    try expectEqual(@as(usize, 2), (try ln.seat(&streams[1], 600, 100, &.{})).len);
    const live = F.live;
    try std.testing.expectError(error.LaneMemory, ln.seat(&streams[2], 600, 100, &.{})); // slot 2: past the paged ones
    try expectEqual(live, F.live); // nothing allocated for it
    try expectEqual(@as(usize, 2), f.e.seqs.items.len);
    try expectEqual(@as(usize, 2), ln.lanes.count());
    _ = ln.lanes.remove(&streams[0]); // a stream ends: the one that waited takes its paged sequence
    p.close(0);
    _ = try ln.seat(&streams[2], 600, 100, &.{});
    try expectEqual(@as(u32, 0), ln.lanes.get(&streams[2]).?.slot);
    try expectEqual(@as(u32, 0), f.e.slot);
}

test "TF_KV_PAGES: the lane host admits a request by the pool alone, the engine's own sequence free or not" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const p = try f.pool(l, 2, 8, true);
    defer f.deinit(p);
    f.e.pages = p;
    var ln: Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer ln.lanes.deinit(f.e.gpa);
    try expect(ln.fits(3000, 100)); // 7 of the 8 pages
    var streams: [2]lanes.Stream = undefined;
    _ = try ln.seat(&streams[0], 600, 100, &.{}); // slot 0: 2 pages
    _ = try ln.seat(&streams[1], 2000, 100, &.{}); // slot 1: 5 pages
    _ = ln.lanes.remove(&streams[0]);
    p.close(0);
    try expect(!ln.fits(3000, 100)); // the engine's own is free, but only 3 pages are
    try expect(ln.fits(1000, 100));
}

test "TF_KV_POISON: every page a stream takes is filled with NaN before its rows are written" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const p = try f.pool(l, 1, 8, true);
    defer f.deinit(p);
    p.poison = true;
    _ = try p.open(0, 1200, 100);
    try expectEqual(@as(usize, 3), F.fills);
}

test "a stream resuming a kept prompt takes the prompt cache's pages first: its table starts with them, NaN spares them" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const p = try f.pool(l, 3, 8, true);
    defer f.deinit(p);
    p.poison = true;
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, try p.open(1, 1500, 100));
    p.pool.retain(0); // the prompt cache keeps the first two pages
    p.pool.retain(1);
    p.close(1);
    p.pool.retain(0); // its resume holds them once more for the stream
    p.pool.retain(1);
    F.fills = 0;
    const ids = try p.openWith(2, 1500, 100, &.{ 0, 1 });
    try expectEqual(@as(usize, 4), ids.len);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, ids[0..2]);
    try std.testing.expectEqualSlices(u32, ids, F.table(f.e.seqs.items[2].?.st.table, 4));
    try expectEqual(@as(usize, 2), F.fills); // the two fresh pages alone
    try expectEqual(@as(u32, 2), p.pool.holders(0)); // the cache and the stream
    p.close(2);
    try expectEqual(@as(u32, 1), p.pool.holders(0)); // the cache's alone again
    p.pool.release(0);
    p.pool.release(1);
    try expect(p.pool.idle() and p.pool.census().free == 8);
}

test "a resume that cannot have its fresh pages gives the prompt cache's back, and nothing is mapped" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const p = try f.pool(l, 3, 8, true);
    defer f.deinit(p);
    _ = try p.open(1, 1500, 100); // pages 0 to 3
    p.pool.retain(0);
    p.close(1);
    _ = try p.open(0, 3000, 100); // 7 of the 7 free pages
    p.pool.retain(0); // the resume's hold
    try std.testing.expectError(error.LaneMemory, p.openWith(2, 1500, 100, &.{0}));
    try expectEqual(@as(u32, 1), p.pool.holders(0)); // the cache's alone
    try expectEqual(@as(usize, 0), f.e.seqs.items[2].?.st.capacity);
    p.close(0);
    p.pool.release(0);
    try expect(p.pool.idle() and p.pool.census().free == 8);
}

test "a kept state's copy leaves the prompt cache's pages out: past them, two pools a paged cache and the head's rows" {
    var f: Fake = undefined;
    try f.init(std.testing.allocator);
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const p = try f.pool(l, 2, 8, true);
    defer f.deinit(p);
    const ids = try p.open(1, 3000, 100); // 7 pages
    const st = &f.e.seqs.items[1].?.st;
    const c = &f.e.w.config;
    const at = 2048;
    const held = at / page_rows;
    const whole = try Plan.init(st, c, at, at, 0);
    const kept = try Plan.init(st, c, at, at, held);
    const latent = c.kv_lora * 2; // a row's
    const pooled = c.i_dim * 2; // a pool's
    // the DSA layer's and the MTP head's caches: their 2,048 latent rows and 512 pools stay on the held pages
    try expectEqual(whole.len - kept.len, 2 * (at * latent + at / sparse.pool * pooled));
    var parts: usize = 0;
    for (kept.parts[0..kept.count]) |part| if (part.span != 0) {
        parts += 1;
        try expectEqual(2 * pooled, part.len); // the pooled keys' two rows past the last whole pool
        try expectEqual(part.live + @as(u64, ids[held]) * part.page, kept.piece(part, 0).live); // the next page's first
    };
    try expectEqual(@as(usize, 2), parts);
    // the head 8 rows past the mark: those rows are copied from the next page too
    const ahead = try Plan.init(st, c, at, at + 8, held);
    var rows: usize = 0;
    for (ahead.parts[0..ahead.count]) |part| if (part.span == page_rows * latent) {
        rows += part.len / latent;
        try expectEqual(part.live + @as(u64, ids[held]) * part.page, ahead.piece(part, 0).live);
    };
    try expectEqual(@as(usize, 8), rows);
    try std.testing.expectError(error.SnapshotShape, Plan.init(st, c, at, at, held + 1)); // a page past the mark
}

/// The prompt cache's states without a device: each a Kept, live until the store drops it.
const States = struct {
    gpa: std.mem.Allocator,
    next: u64 = 1,
    live: usize = 0,

    fn of(p: *anyopaque) *States {
        return @ptrCast(@alignCast(p));
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(p: *anyopaque, _: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const x = of(p);
        const k = try x.gpa.create(Kept);
        k.* = .{ .id = x.next, .at = at };
        x.next += 1;
        x.live += 1;
        return k;
    }
    fn drop(p: *anyopaque, saved: *anyopaque) void {
        const x = of(p);
        x.gpa.destroy(@as(*Kept, @ptrCast(@alignCast(saved))));
        x.live -= 1;
    }
    const vtable: prefix_mod.radix.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = prefix_mod.Prefix.restore, .drop = drop };
};

test "TF_GLM_KV_SHARE: a pass's pages below its mark go to the prompt cache, and a later pass resumes on those pages" {
    const gpa = std.testing.allocator;
    var f: Fake = undefined;
    try f.init(gpa);
    const l = try pages_mod.layout(gpa, &f.e, 0);
    const p = try f.pool(l, 3, 8, true);
    defer f.deinit(p);
    var states: States = .{ .gpa = gpa };
    const x = try prefix_mod.Prefix.initWith(gpa, p, 2048, 4, .{ .ptr = &states, .vtable = &States.vtable });
    defer x.deinit();
    var owner: prefix_mod.Owner = .{};
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(gpa);
    var a: [3000]u32 = undefined;
    for (&a, 1..) |*t, i| t.* = @intCast(i);
    // a cold pass: a mark on the grid below its end, where the tree takes the pages below it and the state
    const cold = try x.begin(&a, 0, &.{}, &owner, &adopt);
    defer gpa.free(cold.marks);
    try std.testing.expectEqualSlices(u32, &.{2048}, cold.marks);
    try expect(cold.from == 0 and owner.pending == null and adopt.items.len == 0);
    var kept: [4]u32 = undefined;
    @memcpy(&kept, (try p.openWith(1, a.len, 100, adopt.items))[0..4]);
    x.keep(1, &a, 2048);
    try expectEqual(@as(usize, 1), states.live);
    p.close(1); // its pages past the mark free again, those below it the tree's
    for (kept) |id| try expectEqual(@as(u32, 1), p.pool.holders(id));
    // the same 2,049 tokens, another tail: the pass resumes at 2,048 on the tree's pages, its state pending
    var b = a;
    for (b[2049..]) |*t| t.* += 10_000;
    const warm = try x.begin(&b, 0, &.{}, &owner, &adopt);
    defer gpa.free(warm.marks);
    try expectEqual(@as(u32, 2048), warm.from);
    try expectEqual(@as(u32, 2048), owner.pending.?.at);
    try std.testing.expectEqualSlices(u32, &kept, adopt.items);
    try std.testing.expectEqualSlices(u32, &kept, (try p.openWith(2, b.len, 100, adopt.items))[0..4]);
    for (kept) |id| try expectEqual(@as(u32, 2), p.pool.holders(id)); // the tree and the stream
    adopt.clearRetainingCapacity();
    p.close(2);
    // another token at the mark: the head's row 2,047 read it, so the last page is another page and nothing resumes
    var c = a;
    for (c[2048..]) |*t| t.* += 10_000;
    owner = .{};
    const miss = try x.begin(&c, 0, &.{}, &owner, &adopt);
    defer gpa.free(miss.marks);
    try expect(miss.from == 0 and owner.pending == null and adopt.items.len == 0);
    // a stream that needs the tree's pages: its promise evicts the state, and the pages are the stream's
    _ = try p.openWith(0, c.len, 100, &.{}); // 7 pages: the 4 free and 3 of the tree's
    try expectEqual(@as(usize, 0), states.live);
    p.close(0);
    const gone = try x.begin(&a, 0, &.{}, &owner, &adopt);
    defer gpa.free(gone.marks);
    try expect(gone.from == 0 and owner.pending == null and adopt.items.len == 0);
    try expect(p.pool.idle() and p.pool.census().free == 8);
}

fn made(gpa: std.mem.Allocator) !void {
    var f: Fake = undefined;
    try f.init(gpa);
    defer f.e.seqs.deinit(gpa);
    const l = try pages_mod.layout(gpa, &f.e, 0);
    try pages_mod.ownSequence(&f.e);
    const own = f.e.seqs.items[0].?;
    defer { // the engine frees its own sequence, its pool made or not
        own.st.deinit();
        gpa.destroy(own);
    }
    const p = try pages_mod.Pages.init(gpa, &f.e, l, 3, 8, true);
    p.deinit();
    for (f.e.seqs.items[1..]) |q| if (q) |s| {
        s.st.deinit();
        gpa.destroy(s);
    };
}

test "every host allocation failure while making the pages leaves no device buffer behind" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(gpa: std.mem.Allocator) !void {
            made(gpa) catch |err| {
                try expectEqual(@as(usize, 0), F.live);
                return err;
            };
            try expectEqual(@as(usize, 0), F.live);
        }
    }.run, .{});
}

//! GLM's disk image on a driver whose device memory is host memory: sizes against the snapshot plan, a round trip.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const image = @import("cuda_disk_image.zig");
const State = @import("cuda_state.zig").State;
const Plan = @import("cuda_snapshot_layout.zig").Plan;
const pages = @import("cuda_snapshot_pages.zig");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;

/// Device addresses are host addresses: allocations from the page allocator, copies and fills as memcpy and memset.
const H = struct {
    var held: [16]struct { ptr: u64 = 0, len: usize = 0 } = @splat(.{});
    fn bytes(p: u64, n: usize) []u8 {
        return @as([*]u8, @ptrFromInt(p))[0..n];
    }
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        const m = std.heap.page_allocator.alloc(u8, n) catch return 2;
        for (&held) |*h| if (h.ptr == 0) {
            h.* = .{ .ptr = @intFromPtr(m.ptr), .len = n };
            out.* = h.ptr;
            return 0;
        };
        std.heap.page_allocator.free(m);
        return 2;
    }
    fn free(p: u64) callconv(.c) cuda.abi.Result {
        for (&held) |*h| if (h.ptr == p) {
            std.heap.page_allocator.free(bytes(h.ptr, h.len));
            h.* = .{};
            return 0;
        };
        return 1;
    }
    fn set(p: u64, v: u8, n: usize) callconv(.c) cuda.abi.Result {
        @memset(bytes(p, n), v);
        return 0;
    }
    fn dtod(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memcpy(bytes(dst, n), bytes(src, n));
        return 0;
    }
    fn dtoh(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], bytes(src, n));
        return 0;
    }
    fn htod(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        @memcpy(bytes(dst, n), @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = set;
        d.api.cuMemcpyDtoDAsync_v2 = dtod;
        d.api.cuMemcpyDtoH_v2 = dtoh;
        d.api.cuMemcpyHtoD_v2 = htod;
        d.api.cuStreamSynchronize = wait;
        return d;
    }
};

const capacity = 6144; // past the index rings' 4,096 rows, so a state at the end holds a full ring

/// TP=2 rank 0's shapes on KDA, DSA, KDA, DSA layers.
const Model = struct {
    layers: [4]weights.Layer,
    w: weights.Weights,

    fn init(m: *Model) !void {
        for (&m.layers, [_]bool{ false, true, false, true }) |*l, mla| l.mixer = if (mla) .{ .mla = undefined } else .{ .kda = undefined };
        m.w = undefined;
        m.w.config = .{};
        m.w.share = try shard.Share.of(&m.w.config, 0, 2);
        m.w.layers = &m.layers;
    }
};

test "GLM disk image: the lease bytes the image counts are the snapshot plan's, at each position and head length" {
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    for ([_]bool{ false, true }) |kv8| {
        var st = try State.initFormat(std.testing.allocator, &d, &m.w, capacity, kv8);
        defer st.deinit();
        for ([_]bool{ false, true }) |head| {
            const sh = try image.Shape.of(&st, &m.w.config, capacity, head);
            for ([_]u32{ 2048, 4096, 6144 }) |at| {
                for ([_]u64{ at - 1, at, sh.headRows(at) }) |mtp| {
                    const plan = try Plan.init(&st, &m.w.config, at, if (head) mtp else null);
                    try expectEqual(@as(u64, plan.len), image.leaseBytes(sh, at, if (head) mtp else 0));
                }
            }
        }
    }
}

test "GLM disk image: the store's rows are by position from 0 in whole blocks; rings, pools in progress and the head are its own" {
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    var st = try State.initFormat(std.testing.allocator, &d, &m.w, capacity, true);
    defer st.deinit();
    const sh = try image.Shape.of(&st, &m.w.config, capacity, true);
    const a = std.testing.allocator;
    var list: std.ArrayList(image.Part) = .empty;
    defer list.deinit(a);
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(a);
    const at: u32 = 6144;
    try image.parts(sh, at, &list, a);
    try image.segments(sh, at, at, &segs, a);
    var total: u64 = 0;
    var rows: usize = 0;
    for (list.items) |p| {
        total += p.bytes;
        if (p.row == 0) continue;
        rows += 1;
        try expect(256 % p.per == 0);
        try expectEqual(@as(u64, at / p.per) * p.row, p.bytes); // a pooled part counts the whole pools alone
    }
    try expectEqual(@as(usize, 3 * 2), rows); // each DSA layer's latent rows, scales and whole pools
    const last = segs.items[segs.items.len - 1];
    try expectEqual(total, last.img + last.len);
    try expectEqual(@as(u64, 2 * 256), list.items[7].bytes); // layer 0's pools past `at`: the spare rows
    try expectError(error.SnapshotShape, image.segments(sh, at, sh.headRows(at) + 1, &segs, a));
}

test "GLM disk image: a kept state's image read out of its lease restores every planned byte into another sequence" {
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    const a = std.testing.allocator;
    const c = &m.w.config;
    var st = try State.initFormat(a, &d, &m.w, capacity, false);
    defer st.deinit();
    for (H.bytes(st.arena.buf.ptr, st.arena.buf.len), 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
    var pool = try pages.Pool.init(a, &d, 64 << 20);
    defer pool.deinit();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const at: u32 = 2048;
    const mtp: u64 = at - 1; // a head one row behind: its last image row is not in the lease
    const sh = try image.Shape.of(&st, c, capacity, true);
    const plan = try Plan.init(&st, c, at, mtp);
    var kept = try pool.take(plan.len);
    defer pool.give(&kept);
    try pool.move(stream, &kept, &plan, true);
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(a);
    try image.segments(sh, at, mtp, &segs, a);
    const last = segs.items[segs.items.len - 1];
    const img = try a.alloc(u8, @intCast(last.img + last.len));
    defer a.free(img);
    var words = (image.Words{ .at = at, .mtp_len = mtp, .base = 0, .mtp_base = 0, .cur = 0b10 }).encode();
    const out: image.Copy = .{ .d = &d, .pool = &pool, .lease = &kept, .segs = segs.items, .words = &words };
    const cut = [_]u64{ 0, 1000, plan.len / 2, img.len }; // the store asks for ranges, not the whole image
    for (cut[0..3], cut[1..]) |lo, hi| try out.run(lo, img[@intCast(lo)..@intCast(hi)], true);
    for (segs.items) |s| if (s.from == .zeros) for (img[@intCast(s.img)..][0..@intCast(s.len)]) |b| try expectEqual(@as(u8, 0), b);
    // a blank state as a disk read takes it (room for the head's whole window), filled from the image
    var blank = try pool.take(@intCast(image.leaseBytes(sh, at, sh.headRows(at))));
    defer pool.give(&blank);
    var back: [6]u64 = undefined;
    const in: image.Copy = .{ .d = &d, .pool = &pool, .lease = &blank, .segs = segs.items, .words = &back };
    try in.run(0, img, false);
    const w = try image.Words.decode(back, at);
    try expectEqual(mtp, w.mtp_len);
    try expectEqual(@as(u64, 0b10), w.cur);
    try expectError(error.DiskWords, image.Words.decode(back, at + 2048));
    var fresh = try State.initFormat(a, &d, &m.w, capacity, false);
    defer fresh.deinit();
    const into = try Plan.init(&fresh, c, at, w.mtp_len);
    try expectEqual(plan.len, into.len);
    try pool.move(stream, &blank, &into, false);
    for (plan.parts[0..plan.count], into.parts[0..into.count]) |x, y| {
        try std.testing.expectEqualSlices(u8, H.bytes(x.live, x.len), H.bytes(y.live, y.len));
    }
}

test "GLM disk image: the store's image is rank 0's then each other rank's prefix, and routes every byte once" {
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    const a = std.testing.allocator;
    var st = try State.initFormat(a, &d, &m.w, capacity, true);
    defer st.deinit();
    const sh = try image.Shape.of(&st, &m.w.config, capacity, true);
    const at: u32 = 4096;
    const prefixes = [_]u64{ sh.prefix, sh.prefix - 4096, sh.prefix + 512 }; // TP=3: heads one by one
    var list: std.ArrayList(image.Part) = .empty;
    defer list.deinit(a);
    try image.storeParts(sh, &prefixes, at, &list, a);
    var total: u64 = 0;
    for (list.items) |p| total += p.bytes;
    try expectEqual(image.imageBytes(sh, at) + prefixes[1] + prefixes[2], total);
    var pieces: std.ArrayList(image.Piece) = .empty;
    defer pieces.deinit(a);
    var seen: [3]u64 = .{ 0, 0, 0 };
    var shared: u64 = 0;
    const cuts = [_]u64{ 0, 40, image.Words.bytes + sh.prefix - 8, image.imageBytes(sh, at) + 100, total }; // ranges across the seams
    for (cuts[0 .. cuts.len - 1], cuts[1..]) |lo, hi| {
        pieces.clearRetainingCapacity();
        try image.route(sh, &prefixes, at, lo, hi - lo, &pieces, a);
        var n: u64 = 0;
        for (pieces.items) |p| {
            n += p.len;
            seen[p.rank] += p.len;
            if (p.shared) shared += p.len;
            if (p.rank > 0) try expect(p.off >= image.Words.bytes and p.off + p.len <= image.Words.bytes + prefixes[p.rank] and !p.shared);
            if (p.shared) try expect(p.rank == 0 and p.off >= image.Words.bytes + sh.prefix);
        }
        try expectEqual(hi - lo, n);
    }
    try expectEqual(image.imageBytes(sh, at), seen[0]);
    try expectEqual(prefixes[1], seen[1]);
    try expectEqual(prefixes[2], seen[2]);
    try expectEqual(image.imageBytes(sh, at) - image.Words.bytes - sh.prefix, shared);
    try expectError(error.SnapshotShape, image.route(sh, &prefixes, at, total - 4, 8, &pieces, a));
}

/// The fields of an engine the kept-state functions read, over a host-memory driver.
const Fake = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    c: @import("config.zig").Config,
    cur: struct { st: State, in_head: usize = 0 },
    capacity: usize = capacity,
    mtp: bool = true,
    snaps: std.AutoHashMapUnmanaged(u64, @import("cuda_engine_types.zig").Snap) = .empty,
    snap_pages: ?pages.Pool = null,
};

test "GLM disk image: a disk read's blank state is refused until sealed, and its seal checks the words" {
    const snaps = @import("cuda_snapshots.zig");
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    const a = std.testing.allocator;
    var e: Fake = .{ .gpa = a, .d = &d, .stream = .{ .d = &d, .handle = null }, .c = m.w.config, .cur = .{ .st = try State.initFormat(a, &d, &m.w, capacity, false) } };
    defer e.cur.st.deinit();
    e.snap_pages = try pages.Pool.init(a, &d, 256 << 20);
    defer e.snap_pages.?.deinit();
    try e.snaps.ensureTotalCapacity(a, 8);
    defer e.snaps.deinit(a);
    const at: u32 = 2048;
    try snaps.blankAs(&e, 7, at);
    defer {
        var s = e.snaps.get(7).?;
        snaps.free(&e, &s);
    }
    try expectError(error.StateKept, snaps.blankAs(&e, 7, at));
    try expectError(error.UnsealedState, snaps.restore(&e, 7));
    try expectError(error.DiskWords, snaps.seal(&e, 7, at + 9, .{ 0, 0 }, 0)); // a head past its verify window
    try expectError(error.DiskWords, snaps.seal(&e, 7, at, .{ 0, 0 }, @as(u64, 1) << @intCast(e.cur.st.cur.len))); // a copy bit no layer has
    try expectError(error.DiskWords, snaps.seal(&e, 7, at - 1, .{ at + 1, 0 }, 0)); // a main ring based past the state
    try expectError(error.DiskWords, snaps.seal(&e, 7, at - 1, .{ 0, at }, 0)); // a head ring based past its rows
    try snaps.seal(&e, 7, at - 1, .{ 3, 5 }, 0b01);
    const s = e.snaps.get(7).?;
    const sh = try image.Shape.of(&e.cur.st, &e.c, capacity, true);
    try expectEqual(image.leaseBytes(sh, at, at - 1), @as(u64, s.len));
    try expectEqual(@as(usize, at - 1), s.mtp_len);
    try expectEqual([2]usize{ 3, 5 }, s.bases);
    try std.testing.expectEqualSlices(u1, &.{ 1, 0 }, s.cur);
    try expectError(error.StateKept, snaps.seal(&e, 7, at, .{ 0, 0 }, 0));
    const far: u32 = capacity - 1024; // a state past the rings' rows
    const back: u32 = far - @as(u32, @intCast(e.cur.st.ring_rows)); // the least base whose ring still holds its rows
    try snaps.blankAs(&e, 8, far);
    defer {
        var held = e.snaps.get(8).?;
        snaps.free(&e, &held);
    }
    try expectError(error.DiskWords, snaps.seal(&e, 8, far, .{ back - 1, back }, 0)); // a main ring too far back
    try expectError(error.DiskWords, snaps.seal(&e, 8, far, .{ back, back - 1 }, 0)); // a head ring too far back
    try snaps.seal(&e, 8, far, .{ back, back }, 0);
    try snaps.blankAs(&e, 9, far);
    defer {
        var held = e.snaps.get(9).?;
        snaps.free(&e, &held);
    }
    try snaps.seal(&e, 9, 0, .{ back, 0 }, 0); // no head rows yet (as without drafts): the main ring holds the state's rows
}

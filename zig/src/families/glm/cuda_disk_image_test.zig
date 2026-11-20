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
pub const H = struct {
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
    fn contextDevice(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn deviceAttribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0; // a discrete card: cuda.memory.download copies straight to the host
        return 0;
    }
    pub fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = set;
        d.api.cuMemcpyDtoDAsync_v2 = dtod;
        d.api.cuMemcpyDtoH_v2 = dtoh;
        d.api.cuMemcpyHtoD_v2 = htod;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxGetDevice = contextDevice;
        d.api.cuDeviceGetAttribute = deviceAttribute;
        return d;
    }
};

pub const capacity = 6144; // past the index rings' 4,096 rows, so a state at the end holds a full ring

/// TP=2 rank 0's shapes on KDA, DSA, KDA, DSA layers.
pub const Model = struct {
    layers: [4]weights.Layer,
    w: weights.Weights,

    pub fn init(m: *Model) !void {
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
                    const plan = try Plan.init(&st, &m.w.config, at, if (head) mtp else null, 0);
                    try expectEqual(@as(u64, plan.len), image.leaseBytes(sh, at, if (head) mtp else 0));
                }
            }
        }
    }
}

test "GLM disk image: the longest state is a position the store's parts take, so the staging floor prices it" {
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    const a = std.testing.allocator;
    // capacities off and on a pool's edge: rows.attach prices `longest` and falls back to its default on an error
    for ([_]usize{ capacity, capacity + 1, capacity + 3 }) |cap| {
        var st = try State.initFormat(a, &d, &m.w, cap, false);
        defer st.deinit();
        const sh = try image.Shape.of(&st, &m.w.config, cap, true);
        const at = image.longest(sh);
        try expect(at <= cap and cap - at < 4 and at % 4 == 0);
        var parts: std.ArrayList(image.Part) = .empty;
        defer parts.deinit(a);
        try image.storeParts(sh, &.{ sh.prefix, sh.prefix }, at, &parts, a);
        try expect(parts.items.len > 0);
        try expectError(error.SnapshotShape, image.storeParts(sh, &.{ sh.prefix, sh.prefix }, at + 4, &parts, a));
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
    try image.segments(sh, at, at, &segs, a, true);
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
    try expectError(error.SnapshotShape, image.segments(sh, at, sh.headRows(at) + 1, &segs, a, true));
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
    var pool = try pages.Pool.init(a, &d, 64 << 20, false);
    defer pool.deinit();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const at: u32 = 2048;
    const mtp: u64 = at - 1; // a head one row behind: its last image row is not in the lease
    const sh = try image.Shape.of(&st, c, capacity, true);
    const plan = try Plan.init(&st, c, at, mtp, 0);
    var kept = try pool.take(plan.len);
    defer pool.give(&kept);
    try pool.move(stream, &kept, &plan, true);
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(a);
    try image.segments(sh, at, mtp, &segs, a, true);
    const last = segs.items[segs.items.len - 1];
    const img = try a.alloc(u8, @intCast(last.img + last.len));
    defer a.free(img);
    var words = (image.Words{ .at = at, .mtp_len = mtp, .base = 0, .mtp_base = 0, .cur = 0b10, .world = 2 }).encode();
    const out: image.Copy = .{ .d = &d, .pool = &pool, .lease = &kept, .segs = segs.items, .words = &words };
    const cut = [_]u64{ 0, 1000, plan.len / 2, img.len }; // the store asks for ranges, not the whole image
    for (cut[0..3], cut[1..]) |lo, hi| try out.run(lo, img[@intCast(lo)..@intCast(hi)], true);
    for (segs.items) |s| if (s.from == .zeros) for (img[@intCast(s.img)..][0..@intCast(s.len)]) |b| try expectEqual(@as(u8, 0), b);
    // a blank state as a disk read takes it (room for the head's whole window), filled from the image
    var blank = try pool.take(@intCast(image.leaseBytes(sh, at, sh.headRows(at))));
    defer pool.give(&blank);
    var back: [image.Words.count]u64 = undefined;
    const in: image.Copy = .{ .d = &d, .pool = &pool, .lease = &blank, .segs = segs.items, .words = &back };
    try in.run(0, img, false);
    const w = try image.Words.decode(back, at, 2, .unsplit);
    try expectEqual(mtp, w.mtp_len);
    try expectEqual(@as(u64, 0b10), w.cur);
    try expectError(error.DiskWords, image.Words.decode(back, at + 2048, 2, .unsplit));
    var fresh = try State.initFormat(a, &d, &m.w, capacity, false);
    defer fresh.deinit();
    const into = try Plan.init(&fresh, c, at, w.mtp_len, 0);
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
        try image.route(sh, &prefixes, at, lo, hi - lo, &pieces, a, null);
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
    try expectError(error.SnapshotShape, image.route(sh, &prefixes, at, total - 4, 8, &pieces, a, null));
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
    share_pages: bool = false,
    share: struct { rank: usize = 0, world: usize = 1 } = .{},
    room: ?@import("cuda_owner_attn.zig").Room = null,
};

test "GLM disk image: a disk read's blank state is refused until sealed, and its seal checks the words" {
    const snaps = @import("cuda_snapshots.zig");
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    const a = std.testing.allocator;
    var e: Fake = .{ .gpa = a, .d = &d, .stream = .{ .d = &d, .handle = null }, .c = m.w.config, .cur = .{ .st = try State.initFormat(a, &d, &m.w, capacity, false) } };
    defer e.cur.st.deinit();
    e.snap_pages = try pages.Pool.init(a, &d, 256 << 20, false);
    defer e.snap_pages.?.deinit();
    try e.snaps.ensureTotalCapacity(a, 8);
    defer e.snaps.deinit(a);
    const at: u32 = 2048;
    try snaps.blankAs(&e, 7, at, null);
    defer {
        var s = e.snaps.get(7).?;
        snaps.free(&e, &s);
    }
    try expectError(error.StateKept, snaps.blankAs(&e, 7, at, null));
    try expectError(error.UnsealedState, snaps.restore(&e, 7));
    try expectError(error.DiskWords, snaps.seal(&e, 7, at + @import("cuda_state.zig").window_rows + 1, .{ 0, 0 }, 0)); // a head past its verify window
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
    try snaps.blankAs(&e, 8, far, null);
    defer {
        var held = e.snaps.get(8).?;
        snaps.free(&e, &held);
    }
    try expectError(error.DiskWords, snaps.seal(&e, 8, far, .{ back - 1, back }, 0)); // a main ring too far back
    try expectError(error.DiskWords, snaps.seal(&e, 8, far, .{ back, back - 1 }, 0)); // a head ring too far back
    try snaps.seal(&e, 8, far, .{ back, back }, 0);
    try snaps.blankAs(&e, 9, far, null);
    defer {
        var held = e.snaps.get(9).?;
        snaps.free(&e, &held);
    }
    try snaps.seal(&e, 9, 0, .{ back, 0 }, 0); // no head rows yet (as without drafts): the main ring holds the state's rows
}

test "GLM disk image: the words record the world, the layout and the owner, and a read refuses another by name" {
    const at: u32 = 4096;
    const w: image.Words = .{ .at = at, .mtp_len = 7, .base = 1, .mtp_base = 2, .cur = 0b1, .world = 3 };
    const x = w.encode();
    try expectEqual(@as(u64, 72), image.Words.bytes);
    try expectEqual(@as(usize, image.Words.bytes), @sizeOf(@TypeOf(x)));
    // an unsplit state: every rank holds every row, so no rank owns it
    try std.testing.expectEqualSlices(u64, &.{ 3, 0, image.Words.no_owner }, x[6..]);
    try expectEqual(w, try image.Words.decode(x, at, 3, .unsplit));
    try expectError(error.DiskWorld, image.Words.decode(x, at, 2, .unsplit)); // written by three ranks, read by two
    try expectError(error.DiskWorld, image.Words.decode(x, at, 4, .unsplit));
    const split: image.Words.Layout = @fromBackingInt(@intCast(1));
    try expectError(error.DiskLayout, image.Words.decode(x, at, 3, split)); // this server splits, the image does not
    var y = x;
    y[7] = 1; // an image of a split layout read unsplit
    try expectError(error.DiskLayout, image.Words.decode(y, at, 3, .unsplit));
    const owned = try image.Words.decode(y[0..8].* ++ [1]u64{2}, at, 3, split);
    try expectEqual(@as(u64, 2), owned.owner);
    try expectError(error.DiskLayout, image.Words.decode(y, at, 3, split)); // a split image names an owner rank
    try expectError(error.DiskLayout, image.Words.decode(y[0..8].* ++ [1]u64{3}, at, 3, split));
    y = x;
    y[8] = 0; // an unsplit image that names an owner
    try expectError(error.DiskLayout, image.Words.decode(y, at, 3, .unsplit));
    y = x;
    y[0] = 2; // the format comes first: a format this server does not read is DiskWords, whatever its layout
    try expectError(error.DiskWords, image.Words.decode(y, at, 2, .unsplit));
}

/// A state's store image at `at` for two ranks, rank 1's prefix a copy of rank 0's: what the load's probe gathers.
fn storeImage(a: std.mem.Allocator, d: *const cuda.Driver, pool: *pages.Pool, st: *const State, c: *const @import("config.zig").Config, sh: image.Shape, at: u32, mtp: u64) ![]u8 {
    const plan = try Plan.init(st, c, at, mtp, 0);
    var kept = try pool.take(plan.len);
    defer pool.give(&kept);
    try pool.move(.{ .d = d, .handle = null }, &kept, &plan, true);
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(a);
    try image.segments(sh, at, mtp, &segs, a, true);
    const own = image.imageBytes(sh, at);
    const out = try a.alloc(u8, @intCast(own + sh.prefix));
    errdefer a.free(out);
    @memset(out[0..image.Words.bytes], 0); // the words are not the probe's question
    const copy: image.Copy = .{ .d = d, .pool = pool, .lease = &kept, .segs = segs.items, .words = undefined };
    try copy.run(image.Words.bytes, out[image.Words.bytes..@intCast(own)], true);
    @memcpy(out[@intCast(own)..], out[image.Words.bytes..][0..@intCast(sh.prefix)]);
    return out;
}

test "GLM disk image: the probe's hash leaves out what a prompt pass does not write, whatever the slot ran before" {
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    const a = std.testing.allocator;
    const c = &m.w.config;
    var st = try State.initFormat(a, &d, &m.w, capacity, false);
    defer st.deinit();
    for (H.bytes(st.arena.buf.ptr, st.arena.buf.len), 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
    var pool = try pages.Pool.init(a, &d, 64 << 20, false);
    defer pool.deinit();
    const at: u32 = 2048;
    const mtp: u64 = at - 1; // the head a row behind: its last pool unfinished
    const sh = try image.Shape.of(&st, c, capacity, true);
    const prefixes = [_]u64{ sh.prefix, sh.prefix };
    var loose: std.ArrayList(image.Span) = .empty;
    defer loose.deinit(a);
    try image.unwritten(sh, &prefixes, &.{ sh.window, sh.window }, at, mtp, &loose, a);
    try expectError(error.SnapshotShape, image.unwritten(sh, &prefixes, &.{ sh.window + 1, sh.window }, at, mtp, &loose, a));
    const before = try storeImage(a, &d, &pool, &st, c, sh, at, mtp);
    defer a.free(before);
    // what load's timed windows leave in the slot: a window's rows, keys past `at`'s whole pools, the first-draft row
    const row = c.i_dim * 2;
    const pool_rows = @import("cuda_sparse.zig").pool;
    @memset(H.bytes(st.proj, st.caches[0].lc - st.proj), 0xa5);
    for (st.caches) |x| @memset(H.bytes(x.pk + at / pool_rows * row, 2 * row), 0xa5);
    @memset(H.bytes(st.mtp_caches.pk + mtp / pool_rows * row, 2 * row), 0xa5);
    @memset(H.bytes(st.prompt_hidden, c.hidden * 2), 0xa5);
    const after = try storeImage(a, &d, &pool, &st, c, sh, at, mtp);
    defer a.free(after);
    try expect(!std.mem.eql(u8, before, after)); // the image carries them: hashed whole, two loads named two stores
    image.clear(loose.items, 0, before);
    image.clear(loose.items, 0, after);
    try std.testing.expectEqualSlices(u8, before, after);
    // what the prompt writes still names the store: a KDA state, a whole pool's keys
    for ([_]u64{ st.conv, st.caches[1].pk }) |p| {
        H.bytes(p, 1)[0] ^= 1;
        defer H.bytes(p, 1)[0] ^= 1;
        const changed = try storeImage(a, &d, &pool, &st, c, sh, at, mtp);
        defer a.free(changed);
        image.clear(loose.items, 0, changed);
        try expect(!std.mem.eql(u8, before, changed));
    }
}

test "GLM disk image: a split copy off its home lacks the latent runs, its lease the rest in order, asked for none" {
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    const a = std.testing.allocator;
    var st = try State.initFormat(a, &d, &m.w, capacity, false);
    defer st.deinit();
    const sh = try image.Shape.of(&st, &m.w.config, capacity, true);
    const at: u32 = 4096;
    var runs: std.ArrayList([2]u64) = .empty;
    defer runs.deinit(a);
    try image.latentRuns(sh, at, &runs, a);
    try expectEqual(@as(usize, sh.layers + 1), runs.items.len); // each DSA layer's, then the head's
    for (runs.items[0..sh.layers]) |r| try expectEqual(@as(u64, at) * sh.latent, r[1]);
    try expectEqual(@as(u64, sh.headRows(at)) * sh.latent, runs.items[sh.layers][1]);
    for ([_]u64{ at, at - 1, 0 }) |mtp| {
        var home: std.ArrayList(image.Seg) = .empty;
        defer home.deinit(a);
        var off: std.ArrayList(image.Seg) = .empty;
        defer off.deinit(a);
        try image.segments(sh, at, mtp, &home, a, true);
        try image.segments(sh, at, mtp, &off, a, false);
        var absent: usize = 0;
        var next: u64 = 0; // lease bytes follow each other, the absent runs left out
        var img: u64 = 0; // and the image's bytes each once, in order
        for (off.items) |s| {
            try expectEqual(img, s.img);
            img += s.len;
            switch (s.from) {
                .absent => {
                    try expectEqual(runs.items[absent], [2]u64{ s.img, s.len });
                    absent += 1;
                },
                .lease => |l| {
                    try expectEqual(next, l);
                    next += s.len;
                },
                else => {},
            }
        }
        try expectEqual(runs.items.len, absent);
        var lat: u64 = 0;
        for (runs.items[0..sh.layers]) |r| lat += r[1];
        try expectEqual(image.leaseBytes(sh, at, mtp) - lat - mtp * sh.latent, next); // the head's latent rows too
        const last = home.items[home.items.len - 1];
        try expectEqual(last.img + last.len, img); // the same image, home or not
    }
    // asked for an absent run, a copy refuses by name and reads nothing
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(a);
    try image.segments(sh, at, at, &segs, a, false);
    var pool = try pages.Pool.init(a, &d, 64 << 20, false);
    defer pool.deinit();
    var lease = try pool.take(@intCast(image.leaseBytes(sh, at, at)));
    defer pool.give(&lease);
    var words: [image.Words.count]u64 = @splat(0);
    const cp: image.Copy = .{ .d = &d, .pool = &pool, .lease = &lease, .segs = segs.items, .words = &words };
    var buf: [64]u8 = undefined;
    try expectError(error.KvSplitAbsent, cp.run(runs.items[0][0] + 8, &buf, true));
    try cp.run(runs.items[0][0] - 64, &buf, true); // the bytes before it are this copy's
}

test "GLM disk image: a split state's latent runs route to its home at the home's offsets, every other byte as before" {
    var m: Model = undefined;
    try m.init();
    var d = H.driver();
    const a = std.testing.allocator;
    var st = try State.initFormat(a, &d, &m.w, capacity, true);
    defer st.deinit();
    const sh = try image.Shape.of(&st, &m.w.config, capacity, true);
    const at: u32 = 4096;
    const prefixes = [_]u64{ sh.prefix, sh.prefix + 512 };
    var runs: std.ArrayList([2]u64) = .empty;
    defer runs.deinit(a);
    try image.latentRuns(sh, at, &runs, a);
    var lat: u64 = 0;
    for (runs.items) |r| lat += r[1];
    const total = image.imageBytes(sh, at) + prefixes[1];
    var pieces: std.ArrayList(image.Piece) = .empty;
    defer pieces.deinit(a);
    for ([_]u32{ 0, 1 }) |home| {
        var seen: [2]u64 = .{ 0, 0 };
        var shared: u64 = 0;
        var homed: u64 = 0;
        const cuts = [_]u64{ 0, 40, runs.items[0][0] + 100, runs.items[1][0] + 7, image.imageBytes(sh, at) + 100, total };
        for (cuts[0 .. cuts.len - 1], cuts[1..]) |lo, hi| {
            pieces.clearRetainingCapacity();
            try image.route(sh, &prefixes, at, lo, hi - lo, &pieces, a, home);
            var n: u64 = 0;
            for (pieces.items) |p| {
                n += p.len;
                seen[p.rank] += p.len;
                if (p.shared) shared += p.len;
                if (p.rank != home or p.shared or p.off < image.Words.bytes + prefixes[home]) continue; // its own prefix
                const img = p.off + prefixes[0] - prefixes[home]; // where it sits in rank 0's image
                for (runs.items) |r| if (img >= r[0] and img < r[0] + r[1]) {
                    try expect(img + p.len <= r[0] + r[1]);
                    homed += p.len;
                };
            }
            try expectEqual(hi - lo, n);
        }
        try expectEqual(lat, homed); // every latent byte to the home, none shared
        try expectEqual(image.imageBytes(sh, at) - image.Words.bytes - sh.prefix - lat, shared);
        try expectEqual(total, seen[0] + seen[1]);
    }
    try expectError(error.SnapshotShape, image.route(sh, &prefixes, at, 0, 8, &pieces, a, 2)); // a home past the ranks
}

test "GLM disk image: the bytes past a probe's loose spans are read alone, so a copy off the home reads no latent run" {
    const a = std.testing.allocator;
    var out: std.ArrayList(image.Span) = .empty;
    defer out.deinit(a);
    const spans = [_]image.Span{ .{ .off = 40, .len = 10 }, .{ .off = 0, .len = 5 }, .{ .off = 45, .len = 15 }, .{ .off = 70, .len = 0 }, .{ .off = 90, .len = 30 } };
    try image.gaps(&spans, 100, &out, a);
    try std.testing.expectEqualSlices(image.Span, &.{ .{ .off = 5, .len = 35 }, .{ .off = 60, .len = 30 } }, out.items);
    out.clearRetainingCapacity();
    try image.gaps(&.{}, 8, &out, a);
    try std.testing.expectEqualSlices(image.Span, &.{.{ .off = 0, .len = 8 }}, out.items);
    out.clearRetainingCapacity();
    try image.gaps(&.{.{ .off = 0, .len = 8 }}, 8, &out, a);
    try expectEqual(@as(usize, 0), out.items.len);
}

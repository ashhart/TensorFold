//! The prefill line ring: during an expert-major prefill every decoder layer's pool lines.

const std = @import("std");
const cuda = @import("cuda");
const pool_ = @import("../offload/cuda_exl3_pool.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const attribution_ = @import("cuda_exl3_prefill_attribution.zig");

const layers_n = wt.layers_n;
const routed_n = wt.experts_n;
const table_n = routed_n + 1;
const batch_ring = 1024;
const release_ring = 4096;
/// Copies a batch holds before its table upload and event.
const batch_copies = 16;
/// Pinned snapshots of a layer's pointer tables, one a batch in flight (a batch uploads its snapshot, never the live
const snap_ring = 64;

const SlotState = enum(u8) { free, pending };

pub const Stats = struct {
    copies: u64 = 0,
    dma_requested_bytes: u64 = 0,
    hit_picks: u64 = 0,
    miss_picks: u64 = 0,
    hit_expert_requests: u64 = 0,
    miss_expert_requests: u64 = 0,
    wasted: u64 = 0,
    ring_full: u64 = 0,
    release_waits: u64 = 0,
    compute_waits: u64 = 0,
    reseeded: u64 = 0,

    pub fn observeDemand(stats: *Stats, present: bool, picks: u32) void {
        if (picks == 0) return;
        if (present) {
            stats.hit_picks += picks;
            stats.hit_expert_requests += 1;
        } else {
            stats.miss_picks += picks;
            stats.miss_expert_requests += 1;
        }
    }
};

pub const Ring = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    pools: [layers_n]*pool_.Pool,
    addr: []u64,
    /// Each slot's projection offsets and room (its pool's line layout: under expert parallel a rank's layers hold
    off: [][3]u32,
    room_of: [][3]u32,
    state: []SlotState,
    owner: []u32, // (layer << 16 | id) + 1, 0: none
    epoch_of: []u64,
    next: usize = 0,
    pending_n: usize = 0,
    copy: cuda.Stream,
    cev: [batch_ring]cuda.Event,
    batches: u64 = 0,
    rev: [release_ring]cuda.Event,
    epoch: u64 = 0,
    copy_ordered_epoch: u64 = 0,
    tlay: cuda.exl3_pool.Tables,
    /// Pinned images of each layer's tables (pointers and widths) as the device holds them during the prefill.
    shadow: cuda.HostBuffer,
    snaps: cuda.HostBuffer,
    /// Slot of each layer's expert in the current chunk, or -1; the batch that copied it.
    slot: [layers_n][routed_n]i32 = undefined,
    batch_of: [layers_n][routed_n]u64 = undefined,
    /// Prefill member counts a layer has seen (the guess order for the next chunks).
    counts: [layers_n][routed_n]u32 = undefined,
    /// The open batch: its layer and copies.
    open_layer: ?usize = null,
    open_n: usize = 0,
    /// The routed ids this rank holds (expert parallel, cuda_exl3_tp.zig): a prefetch never guesses another rank's.
    own_lo: usize = 0,
    own_hi: usize = routed_n,
    stats: Stats = .{},
    telemetry_id: u64 = 0,
    attribution: ?attribution_.State = null,

    /// A ring over `pools` (one per decoder layer, every line the same layout); error.Invalid otherwise.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, pools: []const *pool_.Pool, telemetry: bool) !*Ring {
        if (pools.len != layers_n) return error.Invalid;
        // the tables' entries: 513 (routed and shared), or 512 on an expert-parallel rank without the shared expert
        const te = pools[0].lines.spec.experts;
        if (te != table_n and te != routed_n) return error.Invalid;
        var n: usize = 0;
        for (pools) |p| {
            if (p.lines.spec.experts != te) return error.Invalid;
            n += p.opt.lines;
        }
        const r = try gpa.create(Ring);
        errdefer gpa.destroy(r);
        r.* = .{ .gpa = gpa, .d = d, .pools = undefined, .off = undefined, .room_of = undefined, .addr = undefined, .state = undefined, .owner = undefined, .epoch_of = undefined, .copy = undefined, .cev = undefined, .rev = undefined, .tlay = cuda.exl3_pool.Tables.init(te), .shadow = undefined, .snaps = undefined };
        r.telemetry_id = try @import("../model/exl3_telemetry.zig").identity();
        @memcpy(&r.pools, pools);
        r.addr = try gpa.alloc(u64, n);
        errdefer gpa.free(r.addr);
        r.off = try gpa.alloc([3]u32, n);
        errdefer gpa.free(r.off);
        r.room_of = try gpa.alloc([3]u32, n);
        errdefer gpa.free(r.room_of);
        r.state = try gpa.alloc(SlotState, n);
        errdefer gpa.free(r.state);
        r.owner = try gpa.alloc(u32, n);
        errdefer gpa.free(r.owner);
        r.epoch_of = try gpa.alloc(u64, n);
        errdefer gpa.free(r.epoch_of);
        r.attribution = try attribution_.State.init(gpa, n, telemetry);
        errdefer if (r.attribution) |*state| state.deinit(gpa);
        var k: usize = 0;
        for (pools) |p| for (0..p.opt.lines) |i| {
            const lay = p.lines.layout;
            r.addr[k] = p.lines.at(i, 0);
            for (0..3) |w| {
                const stop = if (w < 2) lay.off[w + 1] else lay.line_bytes;
                r.off[k][w] = @intCast(lay.off[w]);
                r.room_of[k][w] = @intCast(stop - lay.off[w]);
            }
            k += 1;
        };
        r.copy = try cuda.Stream.init(d, true);
        errdefer r.copy.deinit();
        var made_c: usize = 0;
        errdefer for (r.cev[0..made_c]) |*e| e.deinit();
        for (&r.cev) |*e| {
            e.* = try cuda.Event.init(d, false);
            made_c += 1;
        }
        var made_r: usize = 0;
        errdefer for (r.rev[0..made_r]) |*e| e.deinit();
        for (&r.rev) |*e| {
            e.* = try cuda.Event.init(d, false);
            made_r += 1;
        }
        r.shadow = try cuda.HostBuffer.alloc(d, layers_n * r.tlay.bytes);
        errdefer r.shadow.free();
        r.snaps = try cuda.HostBuffer.alloc(d, snap_ring * r.tlay.k2[0]);
        for (&r.counts) |*c| @memset(c, 0);
        return r;
    }

    pub fn deinit(r: *Ring) void {
        r.copy.synchronize() catch {};
        r.snaps.free();
        r.shadow.free();
        for (&r.rev) |*e| e.deinit();
        for (&r.cev) |*e| e.deinit();
        r.copy.deinit();
        if (r.attribution) |*state| state.deinit(r.gpa);
        r.gpa.free(r.epoch_of);
        r.gpa.free(r.room_of);
        r.gpa.free(r.off);
        r.gpa.free(r.owner);
        r.gpa.free(r.state);
        r.gpa.free(r.addr);
        r.gpa.destroy(r);
    }

    pub fn slots(r: *const Ring) usize {
        return r.addr.len;
    }

    fn shadowOf(r: *Ring, layer: usize) []u8 {
        return r.shadow.bytes[layer * r.tlay.bytes ..][0..r.tlay.bytes];
    }

    fn ptrEntry(r: *Ring, layer: usize, which: usize, id: usize) *align(1) u64 {
        return @ptrCast(r.shadowOf(layer)[r.tlay.ptr[which] + id * 8 ..][0..8]);
    }

    fn k2Entry(r: *Ring, layer: usize, which: usize, id: usize) *align(1) i32 {
        return @ptrCast(r.shadowOf(layer)[r.tlay.k2[which] + id * 4 ..][0..4]);
    }

    /// Before a prefill: nothing of the pools' may run (`s` drained), every slot free, each layer's table image read
    pub fn begin(r: *Ring, s: cuda.Stream) !void {
        try s.synchronize();
        for (r.pools) |p| try p.lines.drain();
        @memset(r.state, .free);
        @memset(r.owner, 0);
        @memset(r.epoch_of, 0);
        r.next = 0;
        r.pending_n = 0;
        r.copy_ordered_epoch = r.epoch;
        r.open_layer = null;
        r.open_n = 0;
        for (&r.slot) |*row| @memset(row, -1);
        for (r.pools, 0..) |p, l| {
            const img = r.shadowOf(l);
            try p.lines.tables.download(0, img);
            for (0..routed_n) |id| for (0..3) |w| {
                r.ptrEntry(l, w, id).* = 0;
                r.k2Entry(l, w, id).* = p.k2[id][w];
            };
            try upload(r.copy, p.lines.tables.ptr, img);
        }
        // the images above are read before the host rewrites them
        try r.copy.synchronize();
    }

    /// A new deep chunk: every layer's slots of the last one are stale (nothing reads them any more).
    pub fn newChunk(r: *Ring) !void {
        try r.closeBatch();
        for (r.state, r.owner, 0..) |*st, *ow, index| {
            if (st.* == .pending) {
                r.stats.wasted += 1;
                if (r.attribution) |*state| state.unused(index);
            }
            st.* = .free;
            ow.* = 0;
        }
        r.pending_n = 0;
        for (&r.slot, 0..) |*row, l| {
            @memset(row, -1);
            for (0..routed_n) |id| for (0..3) |w| {
                r.ptrEntry(l, w, id).* = 0;
            };
        }
    }

    /// After a prefill: every pool empty (policy state, routed table entries), the device idle; then, with `reseed`
    pub fn end(r: *Ring, s: cuda.Stream, reseed: bool) !void {
        try r.closeBatch();
        try r.copy.synchronize();
        try s.synchronize();
        for (r.pools, 0..) |p, l| {
            const img = r.shadowOf(l);
            for (0..routed_n) |id| for (0..3) |w| {
                r.ptrEntry(l, w, id).* = 0;
                r.k2Entry(l, w, id).* = 0;
            };
            try p.lines.tables.upload(0, img);
            p.v = pool_.offload.Vram.init(p.slots_tab, p.line_tab, p.stamp_tab, p.layer_tab);
            p.prev_n = 0;
            @memset(p.lines.line_pending, false);
            @memset(p.lines.line_epoch, 0);
        }
        if (!reseed) return;
        for (r.pools, 0..) |p, l| {
            var order: [routed_n]u16 = undefined;
            var n: usize = 0;
            for (0..routed_n) |id| {
                if (r.counts[l][id] == 0) continue;
                order[n] = @intCast(id);
                n += 1;
            }
            const Ctx = struct {
                c: *const [routed_n]u32,
                fn hotter(c: @This(), a: u16, b: u16) bool {
                    if (c.c[a] != c.c[b]) return c.c[a] > c.c[b];
                    return a < b;
                }
            };
            std.mem.sort(u16, order[0..n], Ctx{ .c = &r.counts[l] }, Ctx.hotter);
            const take = @min(n, p.opt.lines, p.opt.max_rows * p.opt.slots);
            if (take == 0) continue;
            const rows = (take + p.opt.slots - 1) / p.opt.slots;
            var fake: [routed_n + 16]i32 = undefined;
            const picks = fake[0 .. rows * p.opt.slots];
            @memset(picks, -1);
            for (order[0..take], 0..) |id, i| picks[i] = id;
            const rd = try p.plan(picks, rows);
            // the pool's copy stream does the copies; the ring's stream takes the gates and the release
            for (0..rd.n_passes) |j| {
                try p.stage(rd, j, r.copy);
                try p.release(rd, j, r.copy);
            }
            r.stats.reseeded += rd.misses_();
        }
    }

    fn closeBatch(r: *Ring) !void {
        const l = r.open_layer orelse return;
        const p = r.pools[l];
        // the pointer tables only (widths were written at begin), from this batch's own snapshot
        const n = r.tlay.k2[0];
        const b = r.batches;
        if (b >= snap_ring) try r.cev[(b - snap_ring) % batch_ring].synchronize();
        const snap = r.snaps.bytes[(b % snap_ring) * n ..][0..n];
        @memcpy(snap, r.shadowOf(l)[0..n]);
        try upload(r.copy, p.lines.tables.ptr, snap);
        try r.cev[r.batches % batch_ring].record(r.copy);
        r.batches += 1;
        r.open_layer = null;
        r.open_n = 0;
    }

    /// The copy stream waits until the pass that last read slot `k` finished.
    fn afterReaders(r: *Ring, k: usize) !void {
        const e = r.epoch_of[k];
        if (e == 0 or e <= r.copy_ordered_epoch) return;
        const ev = r.rev[e % release_ring];
        if (r.epoch - e >= release_ring) {
            // its event was recorded over by a newer pass: wait for that one on the host (no device cycle)
            try ev.synchronize();
            r.copy_ordered_epoch = e;
            return;
        }
        if (try ev.done()) {
            r.copy_ordered_epoch = e;
            return;
        }
        try r.copy.wait(ev);
        r.copy_ordered_epoch = e;
        r.stats.release_waits += 1;
    }

    pub fn isLoaded(r: *const Ring, layer: usize, id: usize) bool {
        return r.slot[layer][id] >= 0;
    }

    /// Free slots ahead of the cursor before a pending one (the ring's room).
    pub fn room(r: *const Ring) usize {
        return r.slots() - r.pending_n;
    }

    /// Copies expert `id` of `layer` into the next free slot; false when the ring holds no free slot.
    pub fn load(r: *Ring, layer: usize, id: usize, source: attribution_.Source) !bool {
        if (r.slot[layer][id] >= 0) return true;
        if (r.pending_n == r.slots()) {
            r.stats.ring_full += 1;
            return false;
        }
        const p = r.pools[layer];
        const src = p.src.routed[id];
        // the oldest slots are consumed first; skip any still pending, or too small for this expert
        var tried: usize = 0;
        while (r.state[r.next] == .pending or !fits(r.room_of[r.next], src)) {
            r.next = (r.next + 1) % r.slots();
            tried += 1;
            if (tried == r.slots()) {
                r.stats.ring_full += 1;
                return false;
            }
        }
        const k = r.next;
        r.next = (r.next + 1) % r.slots();
        // the open batch's snapshot first (it must not see the evicted expert gone).
        if (r.open_layer != null and (r.open_layer.? != layer or r.open_n == batch_copies)) try r.closeBatch();
        try r.afterReaders(k);
        if (r.owner[k] != 0) {
            const o = r.owner[k] - 1;
            const ol = o >> 16;
            const oid = o & 0xffff;
            r.slot[ol][oid] = -1;
            for (0..3) |w| r.ptrEntry(ol, w, oid).* = 0;
        }
        const dst = r.addr[k];
        const offsets: [3]usize = .{ r.off[k][0], r.off[k][1], r.off[k][2] };
        const capacity = try std.math.add(usize, offsets[2], r.room_of[k][2]);
        _ = try cuda.exl3_pool.dma.directLines(r.copy, dst, capacity, offsets, src, p.src.pinned_bank);
        for (0..3) |w| r.ptrEntry(layer, w, id).* = dst + r.off[k][w];
        r.slot[layer][id] = @intCast(k);
        r.batch_of[layer][id] = r.batches;
        r.owner[k] = @intCast(((layer << 16) | id) + 1);
        r.state[k] = .pending;
        r.pending_n += 1;
        r.open_layer = layer;
        r.open_n += 1;
        r.stats.copies += 1;
        r.stats.dma_requested_bytes += src[0].len + src[1].len + src[2].len;
        if (r.attribution) |*state| state.copied(k, source, src[0].len + src[1].len + src[2].len);
        return true;
    }

    /// The compute stream waits for every copy (and table write) of `ids` of `layer`.
    pub fn gate(r: *Ring, s: cuda.Stream, layer: usize, ids: []const u16) !void {
        var last: ?u64 = null;
        for (ids) |id| {
            if (id >= routed_n) continue;
            const b = r.batch_of[layer][id];
            last = if (last) |x| @max(x, b) else b;
        }
        const b = last orelse return;
        if (b == r.batches) try r.closeBatch(); // its batch is still open
        const ev = r.cev[b % batch_ring];
        if (r.batches - b > batch_ring) {
            try r.copy.synchronize();
            return;
        }
        try s.wait(ev);
        r.stats.compute_waits += 1;
    }

    /// After a pass that read `ids` of `layer` was enqueued on `s`: its slots become free once it finishes.
    pub fn release(r: *Ring, s: cuda.Stream, layer: usize, ids: []const u16) !void {
        r.epoch += 1;
        try r.rev[r.epoch % release_ring].record(s);
        for (ids) |id| {
            if (id >= routed_n) continue;
            const k: usize = @intCast(r.slot[layer][id]);
            r.epoch_of[k] = r.epoch;
            if (r.state[k] == .pending) {
                r.state[k] = .free;
                r.pending_n -= 1;
            }
        }
    }

    /// Experts loaded for `layer` that its picks did not use: their slots are free again (never read).
    pub fn drop(r: *Ring, layer: usize, used: []const u32) void {
        for (0..routed_n) |id| {
            if (used[id] != 0) continue;
            const k = r.slot[layer][id];
            if (k < 0) continue;
            const ku: usize = @intCast(k);
            if (r.state[ku] == .pending) {
                r.state[ku] = .free;
                r.pending_n -= 1;
                r.stats.wasted += 1;
                if (r.attribution) |*state| state.unused(ku);
            }
        }
    }

    /// Tops `layer` up to `want` loaded experts with the ones not loaded, hottest (prefill counts).
    pub fn prefetch(r: *Ring, layer: usize, want: usize, keep_free: usize, source: attribution_.Source) !usize {
        var order: [routed_n]u16 = undefined;
        var n: usize = 0;
        var have: usize = 0;
        for (0..routed_n) |id| {
            if (r.slot[layer][id] >= 0) {
                have += 1;
                continue;
            }
            if (id < r.own_lo or id >= r.own_hi) continue;
            order[n] = @intCast(id);
            n += 1;
        }
        if (have >= want) return 0;
        const Ctx = struct {
            a: *const [routed_n]u32,
            b: *const [routed_n]u32,
            fn hotter(c: @This(), x: u16, y: u16) bool {
                if (c.a[x] != c.a[y]) return c.a[x] > c.a[y];
                if (c.b[x] != c.b[y]) return c.b[x] > c.b[y];
                return x < y;
            }
        };
        const p = r.pools[layer];
        std.mem.sort(u16, order[0..n], Ctx{ .a = &r.counts[layer], .b = &p.counts[p.opt.layer] }, Ctx.hotter);
        var queued: usize = 0;
        for (order[0..n]) |id| {
            if (have + queued == want or r.room() <= keep_free) break;
            if (!try r.load(layer, id, source)) break;
            queued += 1;
        }
        try r.closeBatch();
        return queued;
    }

    pub fn flush(r: *Ring) !void {
        try r.closeBatch();
    }
};

fn fits(room: [3]u32, src: [3][]const u8) bool {
    for (room, src) |n, s| if (s.len > n) return false;
    return true;
}

fn upload(s: cuda.Stream, dst: u64, bytes: []const u8) !void {
    try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
}

test "exl3: a ring slot takes an expert only if each projection fits its line's room" {
    const big: [300]u8 = @splat(0);
    const room: [3]u32 = .{ 256, 512, 256 };
    try std.testing.expect(fits(room, .{ big[0..256], big[0..300], big[0..10] }));
    try std.testing.expect(!fits(room, .{ big[0..300], big[0..10], big[0..10] }));
    try std.testing.expect(fits(room, .{ &.{}, &.{}, &.{} }));
}

const ReleaseFixture = @import("cuda_exl3_prefill_ring_fixture.zig").Fixture;

test "exl3: prefill attribution contracts" {
    _ = @import("cuda_exl3_prefill_attribution_test.zig");
}

test "exl3: prefill ring queues one release fence for a pass's pending slot copies" {
    var driver: cuda.Driver = undefined;
    const ring = try std.testing.allocator.create(Ring);
    defer std.testing.allocator.destroy(ring);
    var epochs: [32]u64 = @splat(7);
    ReleaseFixture.init(&driver, ring, &epochs, 7);
    for (epochs, 0..) |_, index| {
        try ring.afterReaders(index);
        try upload(ring.copy, 42, &.{1});
    }
    try std.testing.expectEqualSlices(ReleaseFixture.Call, &.{ .query, .wait }, ReleaseFixture.calls[0..2]);
    try std.testing.expectEqual(@as(usize, 34), ReleaseFixture.count);
    for (ReleaseFixture.calls[2..ReleaseFixture.count]) |call| try std.testing.expectEqual(ReleaseFixture.Call.copy, call);
    try std.testing.expectEqual(@as(u64, 7), ring.copy_ordered_epoch);
    try std.testing.expectEqual(@as(u64, 1), ring.stats.release_waits);
}

test "exl3: prefill ring queued fences cover older epochs but never a newer reader" {
    var driver: cuda.Driver = undefined;
    const ring = try std.testing.allocator.create(Ring);
    defer std.testing.allocator.destroy(ring);
    var epochs = [_]u64{ 4, 2, 4, 5, 0, 1, 6 };
    ReleaseFixture.init(&driver, ring, &epochs, 6);
    for (0..6) |index| try ring.afterReaders(index);
    try std.testing.expectEqualSlices(ReleaseFixture.Call, &.{ .query, .wait, .query, .wait }, ReleaseFixture.calls[0..ReleaseFixture.count]);
    try std.testing.expectEqual(ring.rev[5].handle, ReleaseFixture.last_event);
    ReleaseFixture.query_result = cuda.abi.success;
    try ring.afterReaders(6);
    try ring.afterReaders(3);
    try std.testing.expectEqual(@as(usize, 5), ReleaseFixture.count);
    try std.testing.expectEqual(@as(u64, 6), ring.copy_ordered_epoch);
    try std.testing.expectEqual(@as(u64, 2), ring.stats.release_waits);
}

test "exl3: prefill ring retains the host fallback at the release event wrap boundary" {
    var driver: cuda.Driver = undefined;
    const ring = try std.testing.allocator.create(Ring);
    defer std.testing.allocator.destroy(ring);
    var epochs = [_]u64{ 7, 7 };
    for ([_]u64{ release_ring - 1, release_ring, release_ring + 1 }) |distance| {
        ReleaseFixture.init(&driver, ring, &epochs, 7 + distance);
        try ring.afterReaders(0);
        try ring.afterReaders(1);
        if (distance < release_ring) {
            try std.testing.expectEqualSlices(ReleaseFixture.Call, &.{ .query, .wait }, ReleaseFixture.calls[0..ReleaseFixture.count]);
            try std.testing.expectEqual(@as(u64, 1), ring.stats.release_waits);
        } else {
            try std.testing.expectEqualSlices(ReleaseFixture.Call, &.{.sync}, ReleaseFixture.calls[0..ReleaseFixture.count]);
            try std.testing.expectEqual(@as(u64, 0), ring.stats.release_waits);
        }
        try std.testing.expectEqual(ring.rev[7].handle, ReleaseFixture.last_event);
        try std.testing.expectEqual(@as(u64, 7), ring.copy_ordered_epoch);
    }
}

test "exl3: prefill ring failed release dependencies cannot authorize subsequent copies" {
    var driver: cuda.Driver = undefined;
    const ring = try std.testing.allocator.create(Ring);
    defer std.testing.allocator.destroy(ring);
    var epochs = [_]u64{7};
    ReleaseFixture.init(&driver, ring, &epochs, 7);
    ReleaseFixture.wait_result = cuda.abi.error_not_ready;
    try std.testing.expectError(error.NotReady, ring.afterReaders(0));
    try std.testing.expectEqual(@as(u64, 0), ring.copy_ordered_epoch);
    try std.testing.expectEqual(@as(u64, 0), ring.stats.release_waits);
    ReleaseFixture.wait_result = cuda.abi.success;
    try ring.afterReaders(0);
    try std.testing.expectEqualSlices(ReleaseFixture.Call, &.{ .query, .wait, .query, .wait }, ReleaseFixture.calls[0..ReleaseFixture.count]);
    try std.testing.expectEqual(@as(u64, 7), ring.copy_ordered_epoch);

    ReleaseFixture.init(&driver, ring, &epochs, 7 + release_ring);
    ReleaseFixture.sync_result = cuda.abi.error_not_ready;
    try std.testing.expectError(error.NotReady, ring.afterReaders(0));
    try std.testing.expectEqual(@as(u64, 0), ring.copy_ordered_epoch);
    ReleaseFixture.sync_result = cuda.abi.success;
    try ring.afterReaders(0);
    try std.testing.expectEqualSlices(ReleaseFixture.Call, &.{ .sync, .sync }, ReleaseFixture.calls[0..ReleaseFixture.count]);
    try std.testing.expectEqual(@as(u64, 7), ring.copy_ordered_epoch);
}

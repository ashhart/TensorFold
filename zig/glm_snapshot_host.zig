//! GLM kept-state sizing and transfers through the actual rank owner with a recording CUDA driver.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;

const Sizing = struct {
    var next: u64 = 1 << 30;
    fn alloc(out: *u64, bytes: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(usize, bytes, 256);
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn zero(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

test "GLM snapshots: a short prefix fits 4 GiB even with a 524288-token configured window" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Sizing.alloc;
    d.api.cuMemFree_v2 = Sizing.free;
    d.api.cuMemsetD8_v2 = Sizing.zero;
    var w: glm.weights.Weights = undefined;
    w.config = .{};
    w.share = try glm.shard.Share.of(&w.config, 0, 2);
    var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
    w.layers = layers[0..w.config.layers];
    for (w.layers, 0..) |*l, i| {
        l.* = undefined;
        l.mixer = switch (w.config.kind(@intCast(i))) {
            .mla => .{ .mla = undefined },
            .kda => .{ .kda = undefined },
        };
    }
    var e: glm.engine.Engine = undefined;
    var seq: @TypeOf(e.cur.*) = undefined;
    seq.st = try glm.forward.State.init(t.allocator, &d, &w, glm.native.capacity(524288));
    defer seq.st.deinit();
    e.cur = &seq;
    e.c = w.config;
    e.mtp = true;
    e.capacity = seq.st.capacity;
    for ([_]u32{ 2048, 34816, 266240 }) |at| {
        std.debug.print("GLM snapshot sizing rank=0 world=2 at={d} bytes={d} arena={d}\n", .{ at, glm.native.snap_bytes(&e, at), seq.st.arena.buf.len });
    }
    const short = glm.native.snap_bytes(&e, 2048);
    const longer = glm.native.snap_bytes(&e, 34816);
    try t.expect(short < 4 * @as(u64, 1 << 30));
    try t.expect(short < longer);
    try t.expect(longer < seq.st.arena.buf.len);
    try t.expectEqual(@as(u64, 0), short % (2 << 20));
}

const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const pc = @import("engine_api").prompt_cache;
const adapter = @import("copies");

test "GLM snapshots: grid and committed off-grid read sets survive copies on both unequal rank shares" {
    var d = rec.init();
    const before = cuda.usage(false).device;
    for (0..3) |rank| {
        var f: Fixture = undefined;
        try f.init(&d, rank, 3, true);
        defer f.deinit();
        try f.e.prepareSnapshots(3 * f.e.stateBytes(8192) + 17);
        const allocs = rec.allocs;
        for ([_]usize{ 2048, 3073, 4096, 8192 }) |at| for ([_]usize{ at - 1, at, at + 7 }) |head| {
            try f.mark(at, head, @intCast(head % 251));
            const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
            defer t.allocator.free(saved);
            const bases = [2]usize{ f.seq.st.base, f.seq.st.mtp_base };
            const id = try f.e.keepState();
            @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xE9);
            try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
            f.seq.in_head = 3;
            try f.e.restoreState(id);
            try f.expectPrefix(saved, at, head);
            try t.expectEqual(at, f.seq.st.pos);
            try t.expectEqual(head, f.seq.st.mtp_len);
            try t.expectEqual(@as(u1, @truncate(at)), f.seq.st.cur[0]);
            try t.expectEqual(bases[0], f.seq.st.base);
            try t.expectEqual(bases[1], f.seq.st.mtp_base);
            try t.expectEqual(@as(usize, 0), f.seq.in_head);
            const tail = f.seq.st.caches[0].lc + at * f.e.c.kv_lora * 2;
            try t.expectEqual(@as(u8, 0xE9), rec.bytes(tail, 1)[0]);
            const charge = f.e.snaps.get(id).?.lease.ids.items.len * glm.snapshots.pages.page;
            try t.expect(charge <= f.e.stateBytes(at));
            try t.expectEqual(f.e.snap_pages.?.arena.?.len, charge + f.e.snap_pages.?.spare());
            f.e.dropState(id);
            // In production the following gate or the next Leader step drains these notes.
            f.e.dropped.clearRetainingCapacity();
            try t.expectEqual(allocs, rec.allocs);
        };
    }
    try t.expectEqual(before, cuda.usage(false).device);
}

test "GLM snapshots: actual store charges held plus spare pages, evicts and protects a borrowed state" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, false);
    defer f.deinit();
    const budget = 2 * f.e.stateBytes(2048) + 31;
    _ = try glm.native.snap_ready(&f.e, budget);
    const family: pc.Snapshots = .{ .ptr = &f.e, .vtable = &adapter.Snaps(glm.native, null).vtable };
    var store = pc.Store.init(t.allocator, family, adapter.cacheRules(glm.native), budget);
    defer store.deinit();
    const allocs = rec.allocs;
    var prompt: [2049]u32 = @splat(1);
    for (0..9) |i| {
        prompt[0] = @intCast(i);
        try f.mark(2048, 2048, @intCast(i));
        try t.expect(store.keep(&prompt, 2048, null, &.{}, &.{}));
        try t.expect(store.held + glm.native.snap_spare(&f.e) <= budget);
        try t.expectEqual(allocs, rec.allocs);
        try t.expect(store.entries.items.len <= 2);
    }
    try t.expect(store.counts.evicted > 0);
    const found = try store.lookup(t.allocator, &prompt, 2048, &.{}, &.{}, &.{});
    defer t.allocator.free(found.marks);
    const borrowed = found.entry orelse return error.ExpectedHit;
    defer store.release(borrowed);
    const kept: *glm.engine.Kept = @ptrCast(@alignCast(borrowed.saved));
    const id = kept.id;
    for (10..15) |i| {
        prompt[0] = @intCast(i);
        try f.mark(2048, 2048, @intCast(i));
        try t.expect(store.keep(&prompt, 2048, null, &.{}, &.{}));
        try t.expect(f.e.holds(id));
        try t.expect(store.held + glm.native.snap_spare(&f.e) <= budget);
    }
}

test "GLM snapshots: copy faults drain before pages return and a restore can retry without allocation" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    try f.e.prepareSnapshots(2 * f.e.stateBytes(2048));
    try f.mark(2048, 2047, 77);
    const free = f.e.snap_pages.?.spare();
    const syncs = rec.syncs;
    rec.fail_copy = rec.copies + 2;
    try t.expectError(error.NotReady, f.e.keepState());
    try t.expectEqual(free, f.e.snap_pages.?.spare());
    try t.expectEqual(@as(u32, 0), f.e.snaps.count());
    try t.expect(rec.syncs > syncs);
    rec.fail_copy = null;
    const id = try f.e.keepState();
    const allocs = rec.allocs;
    const held = f.e.snap_pages.?.spare();
    rec.fail_copy = rec.copies + 3;
    try t.expectError(error.NotReady, f.e.restoreState(id));
    try t.expectEqual(held, f.e.snap_pages.?.spare());
    try t.expect(f.e.holds(id));
    rec.fail_copy = null;
    try f.e.restoreState(id);
    try t.expectEqual(allocs, rec.allocs);
    f.seq.st.pending = 1;
    try t.expectError(error.NotAtMark, f.e.keepState());
}

test "GLM snapshots: fragmented pages join a larger state and cannot exceed the admitted budget" {
    var d = rec.init();
    const p = glm.snapshots.pages;
    var pool = try p.Pool.init(t.allocator, &d, 5 * p.page + p.page - 1);
    defer pool.deinit();
    var a = try pool.take(1);
    var b = try pool.take(2 * p.page);
    defer pool.give(&b);
    var c = try pool.take(2 * p.page);
    pool.give(&a);
    pool.give(&c);
    var joined = try pool.take(3 * p.page);
    defer pool.give(&joined);
    try t.expectEqual(@as(u64, 0), pool.spare());
    try t.expectError(error.NoSnapshotRoom, pool.take(1));
    try t.expectEqual(@as(usize, 1), rec.allocs);
}

const Control = struct {
    share: struct { rank: usize },
    dropped: std.ArrayList(u32) = .empty,
    announce: u64 = 0,
    words: [8]f64 = @splat(0),
    calls: usize = 0,
    held: u64 = 1,
    prepares: usize = 0,
    fail_prepare: bool = false,
    peer_failed: bool = false,
    position: u32 = 0,
    kept_at: u32 = 0,
    fail_shared: bool = false,
    pub fn shared(c: *@This(), word: f64) !f64 {
        if (c.fail_shared) return error.PeerFailed;
        const i = c.calls;
        c.calls += 1;
        if (c.share.rank == 0) c.words[i] = word;
        return if (c.peer_failed) 1 else c.words[i];
    }
    pub fn dropState(c: *@This(), id: u64) void {
        if (id == c.held) c.held = 0;
    }
    pub fn keepAs(c: *@This(), id: u64) !void {
        if (c.held != 0) return error.PeerStillHoldsEvictedPages;
        c.held = id;
        c.kept_at = c.position;
    }
    pub fn prepareSnapshots(c: *@This(), _: u64) !void {
        c.prepares += 1;
        if (c.fail_prepare) return error.OutOfMemory;
    }
    fn mark(ctx: *anyopaque, at: u32) void {
        const c: *Control = @ptrCast(@alignCast(ctx));
        c.held = 2;
        c.kept_at = at;
        c.announce = 2;
    }
};

test "GLM snapshots: cancellation publishes the completed prefix and evictions before ranks rebind" {
    var leader: Control = .{ .share = .{ .rank = 0 }, .position = 2048 };
    defer leader.dropped.deinit(t.allocator);
    try leader.dropped.append(t.allocator, 1);
    const marks: glm.engine.Marks = .{ .at = &.{}, .ctx = &leader, .call = Control.mark, .interrupted = Control.mark };
    try glm.snapshots.interrupted(&leader, marks, 2048);
    try t.expectEqualSlices(f64, &.{ 3, 4 }, leader.words[0..leader.calls]);
    var peer: Control = .{ .share = .{ .rank = 1 }, .position = 2048, .words = leader.words };
    try glm.snapshots.interrupted(&peer, null, 2048);
    leader.position = 0;
    peer.position = 0;
    try t.expectEqual(@as(u64, 2), peer.held);
    try t.expectEqual(@as(u32, 2048), leader.kept_at);
    try t.expectEqual(leader.kept_at, peer.kept_at);
    try t.expectEqual(@as(u64, 0), leader.announce);
    try t.expectEqual(@as(usize, 0), leader.dropped.items.len);
    leader.calls = 0;
    peer.calls = 0;
    try glm.snapshots.interrupted(&leader, null, 0);
    peer.words = leader.words;
    try glm.snapshots.interrupted(&peer, null, 0);
    try t.expectEqual(@as(usize, 1), leader.calls);
    try t.expectEqual(@as(f64, 0), leader.words[0]);
    try t.expectEqual(@as(u64, 2), peer.held);
    peer.fail_shared = true;
    try t.expectError(error.PeerFailed, glm.snapshots.interrupted(&peer, null, 0));
}

test "GLM snapshots: in-chunk eviction reaches the follower before its replacement takes pages" {
    var leader: Control = .{ .share = .{ .rank = 0 }, .announce = 2 };
    defer leader.dropped.deinit(t.allocator);
    try leader.dropped.append(t.allocator, 1);
    try glm.snapshots.gate(&leader);
    try t.expectEqualSlices(f64, &.{ 3, 4 }, leader.words[0..leader.calls]);
    try t.expectEqual(@as(usize, 0), leader.dropped.items.len);
    try t.expectEqual(@as(u64, 0), leader.announce);
    var peer: Control = .{ .share = .{ .rank = 1 }, .words = leader.words };
    try glm.snapshots.gate(&peer);
    try t.expectEqual(@as(u64, 2), peer.held);
    try t.expectEqual(leader.calls, peer.calls);
    leader.calls = 0;
    try glm.snapshots.gate(&leader);
    try t.expectEqual(@as(usize, 1), leader.calls);
    try t.expectEqual(@as(f64, 0), leader.words[0]);
}

test "GLM snapshots: allocation failure still reaches startup agreement on either rank" {
    var leader: Control = .{ .share = .{ .rank = 0 }, .fail_prepare = true };
    try t.expectError(error.OutOfMemory, glm.snapshots.agree(&leader, 1 << 30));
    try t.expectEqual(@as(usize, 1), leader.prepares);
    try t.expectEqual(@as(usize, 1), leader.calls);
    try t.expectEqual(@as(f64, 1), leader.words[0]);
    var peer: Control = .{ .share = .{ .rank = 1 }, .peer_failed = true };
    try t.expectError(error.PeerSnapshotCache, glm.snapshots.agree(&peer, 1 << 30));
    try t.expectEqual(@as(usize, 1), peer.prepares);
    try t.expectEqual(@as(usize, 1), peer.calls);
}

test "GLM snapshots: host allocation failures return pages without publishing a state" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 1, true);
    defer f.deinit();
    try f.e.prepareSnapshots(2 * f.e.stateBytes(2048));
    try f.mark(2048, 2048, 88);
    const free = f.e.snap_pages.?.spare();
    // Lease ids, then parity metadata: both happen before the first copy.
    for (0..2) |index| {
        var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = index });
        f.e.gpa = failing.allocator();
        f.e.snap_pages.?.pages.gpa = failing.allocator();
        try t.expectError(error.OutOfMemory, f.e.keepState());
        try t.expectEqual(free, f.e.snap_pages.?.spare());
        try t.expectEqual(@as(u32, 0), f.e.snaps.count());
    }
    f.e.gpa = t.allocator;
    f.e.snap_pages.?.pages.gpa = t.allocator;
}

test "GLM snapshots: interleaved stream restores match their solo read sets with and without the MTP head" {
    var d = rec.init();
    for ([_]bool{ false, true }) |mtp| {
        var a: Fixture = undefined;
        var b: Fixture = undefined;
        try a.init(&d, 0, 2, mtp);
        defer a.deinit();
        try b.init(&d, 0, 2, mtp);
        defer b.deinit();
        try a.e.prepareSnapshots(3 * a.e.stateBytes(4096));
        try a.mark(2048, 2047, 17);
        const first = try t.allocator.dupe(u8, rec.bytes(a.seq.st.arena.buf.ptr, a.seq.st.arena.buf.len));
        defer t.allocator.free(first);
        const id_a = try a.e.keepState();
        try b.mark(4096, 4103, 29);
        const second = try t.allocator.dupe(u8, rec.bytes(b.seq.st.arena.buf.ptr, b.seq.st.arena.buf.len));
        defer t.allocator.free(second);
        a.e.cur = &b.seq;
        const id_b = try a.e.keepState();
        const allocs = rec.allocs;
        for (0..3) |_| {
            // Restore into a different live arena as a fresh lane would, then alternate both logical streams.
            try b.mark(99, 100, 0xAA);
            try a.e.restoreState(id_a);
            try b.expectPrefix(first, 2048, 2047);
            a.e.cur = &a.seq;
            try a.mark(99, 100, 0xBB);
            try a.e.restoreState(id_b);
            try a.expectPrefix(second, 4096, 4103);
            a.e.cur = &b.seq;
        }
        try t.expectEqual(allocs, rec.allocs);
        a.e.cur = &a.seq;
    }
}

test "GLM snapshots: a failed store publication cancels an unannounced keep without creating a peer state" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, true);
    defer f.deinit();
    try f.e.prepareSnapshots(f.e.stateBytes(2048));
    try f.mark(2048, 2048, 11);
    const free = f.e.snap_pages.?.spare();
    const id = try f.e.keepState();
    f.e.dropState(id); // Store.keep can fail its entry append after snap_save succeeded.
    try t.expectEqual(@as(u64, 0), f.e.announce);
    try t.expectEqual(@as(usize, 0), f.e.dropped.items.len);
    try t.expectEqual(free, f.e.snap_pages.?.spare());
    const published = try f.e.keepState();
    f.e.announce = 0; // the preceding successful gate published this id on the follower
    f.e.dropState(published);
    try t.expectEqualSlices(u32, &.{@intCast(published)}, f.e.dropped.items);
}

test "GLM snapshots: pool construction failures refund device accounting and free partial metadata" {
    var d = rec.init();
    const before = cuda.usage(false).device;
    for (0..3) |index| {
        var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = index });
        try t.expectError(error.OutOfMemory, glm.snapshots.pages.Pool.init(failing.allocator(), &d, 2 << 20));
    }
    rec.fail_alloc = true;
    try t.expectError(error.NotReady, glm.snapshots.pages.Pool.init(t.allocator, &d, 2 << 20));
    rec.fail_alloc = false;
    try t.expectEqual(before, cuda.usage(false).device);
    var empty = try glm.snapshots.pages.Pool.init(t.allocator, &d, (2 << 20) - 1);
    defer empty.deinit();
    try t.expectEqual(@as(u64, 0), empty.spare());
    try t.expectError(error.NoSnapshotRoom, empty.take(1));
    try t.expectError(error.Overflow, glm.snapshots.pages.rounded(std.math.maxInt(usize)));
}

test "GLM snapshots: FP8 prefix rows and separate scales restore for every rank and MTP mode" {
    var d = rec.init();
    for ([_]bool{ false, true }) |mtp| for (0..3) |rank| {
        var f: Fixture = undefined;
        try f.initFormat(&d, rank, 3, mtp, true);
        defer f.deinit();
        try f.e.prepareSnapshots(2 * f.e.stateBytes(4096));
        try f.mark(3073, 3080, 71);
        const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
        defer t.allocator.free(saved);
        const allocs = rec.allocs;
        const id = try f.e.keepState();
        @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xE9);
        try f.e.restoreState(id);
        try f.expectPrefix(saved, 3073, 3080);
        for (f.seq.st.caches) |c| {
            try t.expectEqual(@as(u8, 0xE9), rec.bytes(c.lc + 3073 * f.e.c.kv_lora, 1)[0]);
            try t.expectEqual(@as(u8, 0xE9), rec.bytes(c.ls + 3073 * @sizeOf(f32), 1)[0]);
        }
        try t.expectEqual(allocs, rec.allocs);
    };
}

//! Real Qwen page admission and radix eviction on a fake CUDA queue, without model weights.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const kv = @import("core").kv_pool;
const radix = qwen.prefix.radix;
const t = std.testing;
const D = cuda.paging_test_device.Device;
const gpa = t.allocator;
const capacity = 131072;
const layout: kv.Layout = .{
    .fixed_bytes = 0,
    .token_bytes = 33280,
    .page_tokens = 512,
    .snap_bytes = 0,
    .run_bytes = 12 * 33280 * 512,
};
const replies = [_]usize{ 1408, 1920, 3456, 2944, 1408, 3456, 1408, 2432, 3968, 1408, 1408, 1920, 1920, 1408, 1408 };

fn engine(d: *cuda.Driver) qwen.engine.Engine {
    var e: qwen.engine.Engine = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = D.stream };
    e.kv8 = false;
    e.n_taps = 0;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = capacity, .linear = @splat(false) };
    e.g.linear[0] = true;
    return e;
}

const World = struct {
    d: cuda.Driver,
    e: qwen.engine.Engine,
    pages: *qwen.pages.Pages,
    store: radix.Store,
    reclaims: usize = 0,
    snapshots: usize = 0,
    drops: usize = 0,
    active_refs: [644]u32 = undefined,
    active_leases: [644]u32 = undefined,
    kept: [15]u32 = undefined,

    fn save(ctx: *anyopaque, _: ?*anyopaque, at: u32) anyerror!radix.Saved {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        const s = try gpa.create(u32);
        s.* = at;
        w.snapshots += 1;
        return @ptrCast(s);
    }

    fn restore(_: *anyopaque, _: ?*anyopaque, _: radix.Saved) anyerror!void {}

    fn drop(ctx: *anyopaque, saved: radix.Saved) void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        const s: *u32 = @ptrCast(@alignCast(saved));
        std.debug.assert(s.* == 7680);
        gpa.destroy(s);
        w.snapshots -= 1;
        w.drops += 1;
    }

    fn bytes(_: *anyopaque, _: u32) u64 {
        return 4;
    }

    fn reclaim(ctx: *anyopaque, want: usize) bool {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        w.reclaims += 1;
        return w.store.reclaim(want);
    }

    fn init(w: *World) !void {
        w.reclaims = 0;
        w.snapshots = 0;
        w.drops = 0;
        w.d = D.driver();
        w.e = engine(&w.d);
        w.pages = try qwen.pages.Pages.init(gpa, &w.e, layout, capacity, 0, 16, 644);
        errdefer w.pages.deinit();
        w.store = try radix.Store.init(gpa, .{ .ptr = w, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, radix.pagesOf(kv.Pool, &w.pages.pool, layout.pageBytes()), .{ .page = 512 });
        errdefer w.store.deinit();
        w.store.limit(.{ .pages = 644, .snaps = 64 });
        for (replies) |reply| _ = try w.pages.open(8192, reply);
        try t.expectEqual(@as(usize, 486), w.pages.pool.census().leased);
        try t.expectEqual(@as(usize, 158), w.pages.pool.available());
        var transient: kv.Lease = .{};
        defer w.pages.pool.finish(&transient);
        try w.pages.pool.promise(&transient, 158);
        try w.pages.pool.cover(&transient, 158, null);
        for (&w.kept, 0..) |*id, i| id.* = transient.ids.items[9 + 10 * i];
        var prompt: [8192]u32 = undefined;
        for (&prompt, 0..) |*id, i| id.* = @intCast(i);
        var path: [15]u32 = undefined;
        const held = w.store.keep(&prompt, 7680, null, &w.kept, &path);
        try t.expect(held.held and held.shared == 15);
        w.pages.pool.finish(&transient);
        try w.pages.pool.joinReclaim(.{ .ctx = w, .call = reclaim });
        try t.expectEqual(@as(usize, 143), w.pages.pool.available());
        try t.expectEqual(@as(usize, 15), w.pages.pool.reclaimable());
        try t.expect(w.pages.pool.fits(layout.need(8192, 2432, qwen.pages.window_rows)));
        try t.expectEqual(@as(?u32, null), w.pages.pool.findRun(12));
        @memcpy(&w.active_refs, w.pages.pool.refs);
        @memcpy(&w.active_leases, w.pages.pool.leased);
    }

    fn unchanged(w: *World) !void {
        for (w.active_refs, w.active_leases, 0..) |r, l, id| if (l > 0) {
            try t.expectEqual(r, w.pages.pool.refs[id]);
            try t.expectEqual(l, w.pages.pool.leased[id]);
        };
    }

    fn opened(w: *World) !void {
        _ = try w.pages.open(8192, 2432);
        try t.expectEqual(@as(usize, 519), w.pages.pool.census().leased);
        for (w.pages.lanes) |lane| try t.expect(lane.used);
        try w.unchanged();
    }

    fn deinit(w: *World) void {
        w.pages.pool.leaveReclaim(w);
        w.store.deinit();
        std.debug.assert(w.snapshots == 0);
        for (w.pages.lanes) |lane| if (lane.used) {
            _ = w.pages.close(lane.cache);
        };
        w.pages.deinit();
    }
};

test "Qwen cached run: production admission reclaims a run without ending any of fifteen lanes" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    var native: qwen.native.Lanes = .{ .gpa = gpa, .io = t.io, .e = &w.e, .capacity = capacity, .target = undefined, .pages = w.pages, .paged_only = true };
    try t.expect(qwen.native.fits(&native, 8192, 2432));
    try t.expect(w.reclaims > 0 and w.drops == 1);
    try w.unchanged();
    try w.opened();
}

test "Qwen cached run: direct page open reclaims before claiming its run" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    try w.opened();
    try t.expect(w.reclaims > 0 and w.drops == 1);
}

test "Qwen cached run: actual radix eviction is sufficient without closing an active lane" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    try t.expect(w.store.reclaim(w.pages.pool.available() + 1));
    try t.expect(w.pages.pool.findRun(12) != null);
    try t.expect(w.pages.fits(8192, 2432));
    try w.unchanged();
    try w.opened();
}

test "Qwen cached run: the eligibility scan is pure and refuses empty or oversized runs" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    const before = w.pages.pool.census();
    try t.expect(w.pages.pool.canReclaimRun(12));
    try t.expect(!w.pages.pool.canReclaimRun(0));
    try t.expect(!w.pages.pool.canReclaimRun(645));
    try t.expectEqual(before, w.pages.pool.census());
    try t.expectEqual(@as(usize, 0), w.reclaims);
    try t.expectEqual(@as(usize, 0), w.drops);
    try w.unchanged();
}

test "Qwen cached run: active and doubly held separators prevent eviction" {
    for ([_]bool{ false, true }) |adopted| {
        var w: World = undefined;
        try w.init();
        defer w.deinit();
        for (w.kept) |id| w.pages.pool.retain(id);
        var held: kv.Lease = .{};
        if (adopted) try w.pages.pool.adopt(&held, &w.kept);
        defer if (adopted) w.pages.pool.finish(&held) else {
            for (w.kept) |id| w.pages.pool.release(id);
        };
        const before = w.pages.pool.census();
        try t.expect(!w.pages.pool.canReclaimRun(12));
        try t.expect(!w.pages.fits(8192, 2432));
        try t.expectEqual(before, w.pages.pool.census());
        try t.expectEqual(@as(usize, 0), w.reclaims);
        try t.expectEqual(@as(usize, 0), w.drops);
        try w.unchanged();
    }
}

test "Qwen cached run: an existing free run leaves the cache intact" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    try t.expect(w.pages.close(w.pages.lanes[0].cache));
    try t.expect(w.pages.pool.findRun(12) != null);
    try t.expect(w.pages.fits(8192, 2432));
    try t.expectEqual(@as(usize, 0), w.reclaims);
    try t.expectEqual(@as(usize, 0), w.drops);
    try t.expectEqual(@as(usize, 1), w.snapshots);
}

test "Qwen cached run: invalid lengths and insufficient total room do not purge cache" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    try t.expect(!w.pages.fits(capacity + 1, 0));
    try t.expect(!w.pages.fits(8192, capacity));
    try t.expectError(error.InvalidQwenPrompt, w.pages.open(capacity + 1, 0));
    try t.expectError(error.LaneMemory, w.pages.open(capacity, 0));
    try t.expectEqual(@as(usize, 0), w.reclaims);
    try t.expectEqual(@as(usize, 0), w.drops);
    try w.unchanged();
}

test "Qwen cached run: absent and non-progressing reclaimers refuse without looping" {
    const Stuck = struct {
        fn call(ctx: *anyopaque, _: usize) bool {
            const w: *World = @ptrCast(@alignCast(ctx));
            w.reclaims += 1;
            return true;
        }
    };
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    w.pages.pool.leaveReclaim(&w);
    try t.expect(!w.pages.fits(8192, 2432));
    try t.expectEqual(@as(usize, 0), w.reclaims);
    try w.pages.pool.joinReclaim(.{ .ctx = &w, .call = Stuck.call });
    try t.expect(!w.pages.fits(8192, 2432));
    try t.expectEqual(@as(usize, 1), w.reclaims);
    try t.expectEqual(@as(usize, 0), w.drops);
    try w.unchanged();
}

test "Qwen cached run: a full set of real lanes leaves a kept prefix intact" {
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    try w.opened();
    var prompt: [8192]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(i);
    var path: [15]u32 = undefined;
    const held = w.store.keep(&prompt, 7680, null, w.pages.lanes[0].lease.ids.items[0..15], &path);
    try t.expect(held.held);
    const calls = w.reclaims;
    const drops = w.drops;
    try t.expect(!w.pages.fits(8192, 2432));
    try t.expectEqual(calls, w.reclaims);
    try t.expectEqual(drops, w.drops);
    try t.expectEqual(@as(usize, 1), w.snapshots);
}

const PrefixWorld = struct {
    d: cuda.Driver,
    e: qwen.engine.Engine,
    pages: *qwen.pages.Pages,
    prefix: *qwen.prefix.Prefix,
    ids: [8192]u32,

    fn countCase(w: *PrefixWorld, shared: bool) !void {
        w.d = D.driver();
        w.e = engine(&w.d);
        for (&w.ids, 0..) |*id, i| id.* = @intCast(i);
        w.pages = try qwen.pages.Pages.init(gpa, &w.e, layout, capacity, 0, 16, 644);
        errdefer w.pages.deinit();
        w.prefix = try qwen.prefix.Prefix.initWithGrid(gpa, &w.e, w.pages, 64, 512);
        errdefer w.prefix.deinit();
        for (0..14) |_| _ = try w.pages.open(8192, 6000);
        _ = try w.pages.open(8192, if (shared) 17280 else 15744);
        const c = w.pages.lanes[0].cache;
        c.len = 7680;
        c.written = 7680;
        if (shared) {
            try w.prefix.keep(c, &w.ids, 7680);
            try t.expectEqual(@as(usize, 622), w.pages.pool.census().leased);
            try t.expectEqual(@as(usize, 22), w.pages.pool.available());
            try t.expectEqual(@as(usize, 0), w.pages.pool.reclaimable());
        } else {
            var transient: kv.Lease = .{};
            defer w.pages.pool.finish(&transient);
            try w.pages.pool.promise(&transient, 25);
            try w.pages.pool.cover(&transient, 25, null);
            var owner: qwen.prefix.Owner = .{ .cache = c };
            var path: [15]u32 = undefined;
            try t.expect(w.prefix.store.keep(&w.ids, 7680, &owner, transient.ids.items[0..15], &path).held);
            w.pages.pool.finish(&transient);
            try t.expectEqual(@as(usize, 619), w.pages.pool.census().leased);
            try t.expectEqual(@as(usize, 10), w.pages.pool.available());
            try t.expectEqual(@as(usize, 15), w.pages.pool.reclaimable());
        }
        try t.expectEqual(@as(usize, 1), D.hosts);
    }

    fn init(w: *PrefixWorld) !void {
        w.d = D.driver();
        w.e = engine(&w.d);
        for (&w.ids, 0..) |*id, i| id.* = @intCast(i);
        w.pages = try qwen.pages.Pages.init(gpa, &w.e, layout, capacity, 0, 16, 644);
        errdefer w.pages.deinit();
        w.prefix = try qwen.prefix.Prefix.initWithGrid(gpa, &w.e, w.pages, 64, 512);
        errdefer w.prefix.deinit();
        for (replies) |reply| _ = try w.pages.open(8192, reply);
        var transient: kv.Lease = .{};
        defer w.pages.pool.finish(&transient);
        try w.pages.pool.promise(&transient, 158);
        try w.pages.pool.cover(&transient, 158, null);
        var kept: [15]u32 = undefined;
        for (&kept, 0..) |*id, i| id.* = transient.ids.items[9 + 10 * i];
        const c = w.pages.lanes[0].cache;
        c.len = 7680;
        c.written = 7680;
        var owner: qwen.prefix.Owner = .{ .cache = c };
        var path: [15]u32 = undefined;
        try t.expect(w.prefix.store.keep(&w.ids, 7680, &owner, &kept, &path).held);
        w.pages.pool.finish(&transient);
        try t.expectEqual(@as(usize, 1), D.hosts);
        try t.expectEqual(@as(?u32, null), w.pages.pool.findRun(12));
    }

    fn deinit(w: *PrefixWorld) void {
        for (w.pages.lanes) |lane| if (lane.used) {
            _ = w.pages.close(lane.cache);
        };
        w.prefix.deinit();
        w.pages.deinit();
        std.debug.assert(D.hosts == 0 and D.freed_pending == 0);
    }
};

test "Qwen cached run: direct prefix open evicts before borrowing a pinned snapshot and cancels cleanly" {
    var w: PrefixWorld = undefined;
    try w.init();
    defer w.deinit();
    const resumed = try w.prefix.open(&w.ids, 2432, 0, &.{});
    defer gpa.free(resumed.plan.marks);
    try t.expectEqual(@as(u32, 0), resumed.plan.from);
    try t.expectEqual(@as(usize, 0), D.hosts);
    try t.expectEqual(@as(usize, 519), w.pages.pool.census().leased);
    try t.expect(w.pages.close(resumed.cache));
    try t.expectEqual(@as(usize, 486), w.pages.pool.census().leased);
    try t.expectEqual(@as(usize, 0), w.pages.pool.promised);
    try t.expectEqual(@as(usize, 0), D.freed_pending);
    const again = try w.prefix.open(&w.ids, 2432, 0, &.{});
    defer gpa.free(again.plan.marks);
    try t.expect(w.pages.close(again.cache));
}

test "Qwen cached run: prefix length and count refusals preserve the pinned state" {
    var w: PrefixWorld = undefined;
    try w.init();
    defer w.deinit();
    try t.expectError(error.InvalidQwenPrompt, w.prefix.open(&w.ids, capacity, 0, &.{}));
    try t.expectError(error.LaneMemory, w.prefix.open(&w.ids, capacity - w.ids.len, 0, &.{}));
    try t.expectEqual(@as(usize, 1), D.hosts);
    try t.expectEqual(@as(usize, 1), w.prefix.store.tree.snaps);
    try t.expectEqual(@as(usize, 0), D.freed_pending);
}

test "Qwen cached run: pending adopted snapshot cannot be purged for a run" {
    var w: PrefixWorld = undefined;
    try w.init();
    defer w.deinit();
    var owner: qwen.prefix.Owner = .{};
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(gpa);
    const plan = try w.prefix.begin(&w.ids, 0, &.{}, &owner, &adopt);
    defer gpa.free(plan.marks);
    defer for (adopt.items) |id| w.pages.pool.release(id);
    try t.expect(owner.pending != null and plan.from == 7680);
    try t.expect(!w.pages.pool.canReclaimRun(12));
    try t.expectError(error.LaneMemory, w.prefix.open(&w.ids, 2432, 0, &.{}));
    try t.expectEqual(@as(usize, 1), D.hosts);
    try t.expectEqual(@as(usize, 1), w.prefix.store.tree.snaps);
}

test "Qwen cached run: a prefix upload failure releases its reclaimed run after completion proof" {
    var w: PrefixWorld = undefined;
    try w.init();
    defer w.deinit();
    D.fail_wait_in = 1;
    try t.expectError(error.NotReady, w.prefix.open(&w.ids, 2432, 0, &.{}));
    try t.expectEqual(@as(usize, 0), D.hosts);
    try t.expectEqual(@as(usize, 486), w.pages.pool.census().leased);
    try t.expectEqual(@as(usize, 0), w.pages.pool.promised);
    try t.expectEqual(@as(usize, 0), D.freed_pending);
    const again = try w.prefix.open(&w.ids, 2432, 0, &.{});
    defer gpa.free(again.plan.marks);
    try t.expect(w.pages.close(again.cache));
}

test "Qwen cached run: a cached path already charged to an active lane fits without its full cold count" {
    var w: PrefixWorld = undefined;
    try w.countCase(true);
    defer w.deinit();
    try t.expect(w.pages.pool.findRun(12) != null);
    try t.expect(!w.pages.pool.fits(layout.need(8192, 2432, qwen.pages.window_rows)));
    var native: qwen.native.Lanes = .{ .gpa = gpa, .io = t.io, .e = &w.e, .capacity = capacity, .target = undefined, .pages = w.pages, .paged_only = true };
    try t.expect(!qwen.native.fits(&native, 8192, 2432));
    try t.expectError(error.LaneMemory, w.pages.prepareShared(8192, 2432, 16));
    const resumed = try w.prefix.open(&w.ids, 2432, 0, &.{});
    defer gpa.free(resumed.plan.marks);
    try t.expectEqual(@as(u32, 7680), resumed.plan.from);
    try t.expectEqual(@as(usize, 640), w.pages.pool.census().leased);
    try t.expectEqual(@as(usize, 1), D.hosts);
    try t.expectEqual(@as(u64, 0), w.prefix.store.tree.counts.evicted);
    try t.expect(w.pages.close(resumed.cache));
    try t.expectEqual(@as(usize, 622), w.pages.pool.census().leased);
}

test "Qwen cached run: cache-only matching pages cannot count both as credit and reclaimable room" {
    var w: PrefixWorld = undefined;
    try w.countCase(false);
    defer w.deinit();
    const before = w.pages.pool.census();
    try t.expect(w.pages.pool.canReclaimRun(12));
    try t.expect(w.pages.pool.fits(layout.need(8192, 2432, qwen.pages.window_rows) - 15));
    try t.expectError(error.LaneMemory, w.prefix.open(&w.ids, 2432, 0, &.{}));
    try t.expectEqual(before, w.pages.pool.census());
    try t.expectEqual(@as(usize, 1), D.hosts);
    try t.expectEqual(@as(u64, 0), w.prefix.store.tree.counts.evicted);
}

test "Qwen cached run: a nonmatching prefix retains cold count admission" {
    var w: PrefixWorld = undefined;
    try w.countCase(true);
    defer w.deinit();
    var other = w.ids;
    other[0] += 1;
    try t.expect(w.prefix.store.find(&other) == null);
    try t.expectError(error.LaneMemory, w.prefix.open(&other, 2432, 0, &.{}));
    try t.expectEqual(@as(usize, 22), w.pages.pool.available());
    try t.expectEqual(@as(usize, 1), D.hosts);
    try t.expectEqual(@as(u64, 0), w.prefix.store.tree.counts.evicted);
}

test "Qwen cached run: maximum credit traverses parent pages and leaves a full-page prompt one token" {
    var w: PrefixWorld = undefined;
    try w.countCase(true);
    defer w.deinit();
    const c = w.pages.lanes[0].cache;
    c.len = 5120;
    c.written = 5120;
    try w.prefix.keep(c, &w.ids, 5120);
    const found = w.prefix.store.find(&w.ids).?;
    try t.expect(found.pages.len == 5 and found.parent.?.pages.len == 10);
    const deepest = try w.prefix.open(&w.ids, 2432, 0, &.{});
    defer gpa.free(deepest.plan.marks);
    try t.expectEqual(@as(u32, 7680), deepest.plan.from);
    try t.expect(w.pages.close(deepest.cache));
    const earlier = try w.prefix.open(w.ids[0..7680], 2432, 0, &.{});
    defer gpa.free(earlier.plan.marks);
    try t.expectEqual(@as(u32, 5120), earlier.plan.from);
    try t.expectEqual(@as(usize, 644), w.pages.pool.census().leased);
    try t.expectEqual(@as(usize, 2), D.hosts);
    try t.expect(w.pages.close(earlier.cache));
    try t.expectEqual(@as(usize, 622), w.pages.pool.census().leased);
}

test "Qwen cached run: an earlier idle store cannot hide a later reclaimer" {
    const Idle = struct {
        calls: usize = 0,
        fn call(ctx: *anyopaque, _: usize) bool {
            const idle: *@This() = @ptrCast(@alignCast(ctx));
            idle.calls += 1;
            return true;
        }
    };
    var w: World = undefined;
    try w.init();
    defer w.deinit();
    w.pages.pool.leaveReclaim(&w);
    var idle: Idle = .{};
    try w.pages.pool.joinReclaim(.{ .ctx = &idle, .call = Idle.call });
    defer w.pages.pool.leaveReclaim(&idle);
    try w.pages.pool.joinReclaim(.{ .ctx = &w, .call = World.reclaim });
    try t.expect(w.pages.fits(8192, 2432));
    try t.expectEqual(@as(usize, 1), idle.calls);
    try t.expectEqual(@as(usize, 1), w.reclaims);
    try t.expectEqual(@as(usize, 1), w.drops);
    try w.opened();
    try w.unchanged();
}

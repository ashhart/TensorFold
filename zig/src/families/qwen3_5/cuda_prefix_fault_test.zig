//! TF_QWEN_KV_SHARE's pages through failed allocations, lanes, promises, uploads and copies, on the fake device.
const std = @import("std");
const cuda = @import("cuda");
const pages = @import("cuda_pages.zig");
const Prefix = @import("cuda_prefix.zig").Prefix;
const E = @import("cuda_engine.zig").Engine;
const D = cuda.paging_test_device.Device;
const t = std.testing;

const page = 256;
const capacity = 3 * 4096;

fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = D.stream };
    e.kv8 = false;
    e.n_taps = 0;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = capacity, .linear = @splat(false) };
    e.g.linear[0] = true;
    return e;
}

/// No page leased or promised and no lane open; each page free or held once, by the store.
fn settled(p: *pages.Pages) !void {
    const c = p.pool.census();
    try t.expectEqual(@as(usize, 0), c.leased);
    try t.expectEqual(@as(usize, 0), c.promised);
    try t.expectEqual(c.total, c.free + c.kept);
    for (p.pool.refs) |r| try t.expect(r <= 1);
    for (p.lanes) |x| try t.expect(!x.used);
}

fn prompt(a: std.mem.Allocator, n: usize) ![]u32 {
    const out = try a.alloc(u32, n);
    for (out, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
    return out;
}

/// The pool keeps cold state at 4096 and a longer prompt's page-edge state at 8960.
const World = struct {
    d: cuda.Driver,
    e: E,
    p: *pages.Pages,
    x: *Prefix,
    ids: []u32,
    next: []u32,
    initial: []u8,
    state: []u8,
    base_devices: usize,

    const SaveFault = enum { none, device_alloc, copy, wait };

    fn seed(w: *World, ptr: u64, ids: []const u32) void {
        for (D.bytes(ptr, w.e.g.stateBytes()), 0..) |*b, i| {
            const token = ids[i % ids.len] ^ @as(u32, @truncate(i * 31));
            b.* = @truncate(token);
        }
    }

    fn init(w: *World, gpa: std.mem.Allocator) !void {
        try w.start(gpa, false, .none);
    }

    fn gpu(w: *World, gpa: std.mem.Allocator, fault: SaveFault) !void {
        try w.start(gpa, true, fault);
    }

    fn start(w: *World, gpa: std.mem.Allocator, use_gpu: bool, fault: SaveFault) !void {
        const env = @import("sys").env;
        const name = "TF_QWEN_KV_SNAPSHOT_GPU";
        _ = env.unsetenv(name);
        if (use_gpu) _ = env.setenv(name, "1", 1);
        defer _ = env.unsetenv(name);
        w.d = D.driver();
        w.e = engine(&w.d);
        const l = pages.layout(&w.e, capacity, page, 0, 0);
        w.p = try pages.Pages.init(gpa, &w.e, l, capacity, 0, 2, 3 * l.need(capacity, 0, pages.window_rows));
        w.base_devices = D.devices;
        errdefer w.p.deinit();
        w.x = try Prefix.init(gpa, &w.e, w.p, 4);
        errdefer w.x.deinit();
        w.ids = try prompt(t.allocator, 9000);
        errdefer t.allocator.free(w.ids);
        w.next = try prompt(t.allocator, 9100);
        errdefer t.allocator.free(w.next);
        const r = try w.x.open(w.ids, 16, 9000, &.{}, null);
        defer w.x.gpa.free(r.plan.marks);
        w.initial = try t.allocator.dupe(u8, D.bytes(r.cache.state.ptr, w.e.g.stateBytes()));
        errdefer t.allocator.free(w.initial);
        w.seed(r.cache.state.ptr, w.ids[0..4096]);
        w.state = try t.allocator.dupe(u8, D.bytes(r.cache.state.ptr, w.e.g.stateBytes()));
        errdefer t.allocator.free(w.state);
        switch (fault) {
            .none => {},
            .device_alloc => D.fail_alloc = true,
            .copy => D.fail_dtod = true,
            .wait => D.fail_wait_in = 2,
        }
        try w.x.keep(r.cache, w.ids, null, 4096);
        if (fault == .copy or fault == .wait) {
            try t.expectEqual(@as(usize, 0), w.x.store.tree.snaps);
            try t.expectEqual(@as(usize, 1), w.x.store.tree.counts.failed);
            try t.expectEqual(@as(usize, 0), D.freed_pending);
            try w.x.keep(r.cache, w.ids, null, 4096);
        }
        try t.expect(w.p.close(r.cache));
        try settled(w.p);
        try t.expectEqual(@as(usize, 4096 / page), w.p.pool.census().kept);
    }

    /// The store's pages back with its states: every page free and every snapshot buffer gone.
    fn deinit(w: *World) !void {
        defer t.allocator.free(w.initial);
        defer t.allocator.free(w.state);
        defer t.allocator.free(w.ids);
        defer t.allocator.free(w.next);
        defer w.p.deinit();
        w.x.deinit();
        try t.expectEqual(w.p.pool.total(), w.p.pool.census().free);
        for (w.p.pool.refs) |r| try t.expectEqual(@as(u32, 0), r);
        try t.expectEqual(@as(usize, 0), D.hosts);
        try t.expectEqual(w.base_devices, D.devices);
    }

    /// A resume of `next` succeeds from 4096, keeps a page-edge mark at 8960, and closes its lane again.
    fn resumes(w: *World) !void {
        const r = try w.x.open(w.next, 16, 9100, &.{}, null);
        defer w.x.gpa.free(r.plan.marks);
        try t.expectEqual(@as(u32, 4096), r.plan.from);
        try t.expectEqualSlices(u32, &.{8960}, r.plan.marks);
        try t.expectEqualSlices(u8, w.state, D.bytes(r.cache.state.ptr, w.e.g.stateBytes()));
        try t.expect(w.p.close(r.cache));
        try settled(w.p);
    }
};

test "Qwen share faults: each allocation of a resume failing in turn leaves the pages settled" {
    var failing = std.testing.FailingAllocator.init(t.allocator, .{});
    var w: World = undefined;
    try w.init(failing.allocator());
    var failures: usize = 0;
    var i: usize = 0;
    while (true) : (i += 1) {
        failing.fail_index = failing.alloc_index + i;
        const r = w.x.open(w.next, 16, 9100, &.{}, null) catch |err| {
            try t.expectEqual(error.OutOfMemory, err);
            try settled(w.p);
            failures += 1;
            continue;
        };
        failing.fail_index = std.math.maxInt(usize);
        w.x.gpa.free(r.plan.marks);
        try t.expect(w.p.close(r.cache));
        try settled(w.p);
        break;
    }
    try t.expect(failures > 2); // the store's plan, the marks on the grid, the lane's adoption and cover
    try w.resumes();
    try w.deinit();
}

test "Qwen share faults: a resume's table upload or state copy failing leaves the pages settled" {
    var w: World = undefined;
    try w.init(t.allocator);
    var failures: usize = 0;
    var n: usize = 1;
    while (true) : (n += 1) {
        D.fail_wait_in = n;
        const r = w.x.open(w.next, 16, 9100, &.{}, null) catch |err| {
            try t.expectEqual(error.NotReady, err);
            try t.expectEqual(@as(usize, 0), D.fail_wait_in);
            try settled(w.p);
            failures += 1;
            continue;
        };
        D.fail_wait_in = 0;
        w.x.gpa.free(r.plan.marks);
        try t.expect(w.p.close(r.cache));
        break;
    }
    try t.expect(failures >= 2); // the adopted pages' table, the state put back
    try w.resumes();
    try w.deinit();
}

test "Qwen share faults: a resume without a lane or a promise leaves the pages settled" {
    var w: World = undefined;
    try w.init(t.allocator);
    // every lane open: the resume's adopted pages go back
    const a = try w.p.open(100, 16);
    const b = try w.p.open(100, 16);
    try t.expectError(error.LaneMemory, w.x.open(w.next, 16, 9100, &.{}, null));
    try t.expect(w.p.close(b));
    try t.expect(w.p.close(a));
    try settled(w.p);
    // every free page promised elsewhere: the store gives back no page the resume holds, and the promise fails
    var hog: @import("core").kv_pool.Lease = .{};
    try w.p.pool.promise(&hog, w.p.pool.available());
    try t.expectError(error.LaneMemory, w.x.open(w.next, 16, 9100, &.{}, null));
    w.p.pool.finish(&hog);
    try settled(w.p);
    try w.resumes();
    try w.deinit();
}

test "Qwen share faults: a keep whose swapped table fails to upload leaves the lane holding its own pages" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    var w: World = undefined;
    try w.init(t.allocator);
    // a second pass of the same prompt on its own pages: its keep at 4096 swaps in the store's
    const c = try w.p.open(w.ids.len, 16);
    const lane = for (w.p.lanes) |*x| {
        if (x.cache == c) break x;
    } else unreachable;
    const mine = try t.allocator.dupe(u32, lane.lease.ids.items[0 .. 4096 / page]);
    defer t.allocator.free(mine);
    D.fail_wait_in = 2; // keep's own wait, then the swap's table upload
    try t.expectError(error.QwenLaneTable, w.x.keep(c, w.ids, null, 4096));
    try t.expectEqualSlices(u32, mine, lane.lease.ids.items[0..mine.len]);
    for (mine) |id| try t.expect(w.p.pool.leased[id] == 1);
    // a keep that uploads: the lane maps the store's pages and its own copies go back
    try w.x.keep(c, w.ids, null, 4096);
    for (mine, lane.lease.ids.items[0..mine.len]) |own, now| try t.expect(own != now and w.p.pool.refs[own] == 0);
    try t.expect(w.p.close(c));
    try settled(w.p);
    try w.deinit();
}

test "Qwen share faults: an unsalted snapshot is not restored under a salt" {
    var w: World = undefined;
    try w.init(t.allocator);
    const r = try w.x.open(w.next, 16, 9100, &.{}, 1);
    const gpa = w.x.gpa;
    defer gpa.free(r.plan.marks);
    try t.expectEqual(@as(u32, 0), r.plan.from);
    try t.expect(w.p.close(r.cache));
    try settled(w.p);
    try w.resumes(); // the unsalted request still resumes its own state
    try w.deinit();
}

test "Qwen share faults: device snapshots restore D2D and eviction frees their bounded buffer" {
    var w: World = undefined;
    try w.gpu(t.allocator, .none);
    try t.expectEqual(@as(usize, 0), D.hosts);
    try t.expectEqual(@as(usize, 1), D.dtods);
    const saved = D.last_dtod;
    try w.resumes();
    try t.expectEqual(@as(usize, 2), D.dtods);
    try t.expectEqual(saved.src, D.last_dtod.dst);
    try t.expectEqual(saved.dst, D.last_dtod.src);
    try t.expectEqual(saved.len, D.last_dtod.len);

    const live = D.devices;
    w.x.store.limit(.{ .pages = w.p.pool.total(), .snaps = 0 });
    try t.expectEqual(@as(usize, 0), w.x.store.tree.snaps);
    try t.expectEqual(live - 1, D.devices);
    const fresh = try w.x.open(w.next, 16, 9100, &.{}, null);
    defer w.x.gpa.free(fresh.plan.marks);
    try t.expectEqual(@as(u32, 0), fresh.plan.from);
    try t.expectEqual(@as(usize, 2), D.dtods);
    try t.expectEqualSlices(u8, w.initial, D.bytes(fresh.cache.state.ptr, w.e.g.stateBytes()));
    w.seed(fresh.cache.state.ptr, w.next[0..4096]);
    try t.expectEqualSlices(u8, w.state, D.bytes(fresh.cache.state.ptr, w.e.g.stateBytes()));
    try t.expect(w.p.close(fresh.cache));
    try settled(w.p);
    try w.deinit();
}

test "Qwen share faults: snapshot device allocation failure falls back to pinned host" {
    var w: World = undefined;
    try w.gpu(t.allocator, .device_alloc);
    try t.expectEqual(@as(usize, 1), D.hosts);
    try t.expectEqual(@as(usize, 0), D.dtods);
    try t.expectEqual(@as(usize, 1), w.x.store.tree.snaps);
    try w.resumes();
    try t.expectEqual(@as(usize, 0), D.dtods);
    try settled(w.p);
    try w.deinit();
}

test "Qwen share faults: failed device snapshot copies drain before retry without stale leases" {
    inline for (.{ World.SaveFault.copy, World.SaveFault.wait }) |fault| {
        var w: World = undefined;
        try w.gpu(t.allocator, fault);
        try t.expectEqual(@as(usize, 0), D.hosts);
        try t.expectEqual(@as(usize, 1), w.x.store.tree.snaps);
        try t.expectEqual(@as(usize, 1), w.x.store.tree.counts.failed);
        try t.expectEqual(@as(usize, 0), D.freed_pending);
        try t.expectEqual(@as(usize, 2), D.dtods);
        try w.resumes();
        try t.expectEqual(@as(usize, 3), D.dtods);
        try w.deinit();
    }
}

test "Qwen share faults: a failed D2D restore keeps its snapshot and settles every page lease" {
    var w: World = undefined;
    try w.gpu(t.allocator, .none);
    D.fail_dtod = true;
    try t.expectError(error.NotReady, w.x.open(w.next, 16, 9100, &.{}, null));
    try t.expectEqual(@as(usize, 1), w.x.store.tree.snaps);
    try t.expectEqual(@as(usize, 0), D.freed_pending);
    try settled(w.p);
    try w.resumes();
    try t.expectEqual(@as(usize, 3), D.dtods);
    try w.deinit();
}

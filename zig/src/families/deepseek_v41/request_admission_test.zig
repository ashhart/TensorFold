//! Host transport adapter around the actual target admission routine, shared store and reservation owner.

const std = @import("std");
const t = std.testing;
const kv = @import("kv");
const sessions = kv.sessions;
const admission = @import("request_admission.zig");
const fams = [_]sessions.Family{.{ .name = "kv", .ratio = 1, .row_bytes = 1 }};

const Bound = struct {
    released: bool = false,
    fn bytes(_: *anyopaque) u64 {
        return 1;
    }
    fn blobs(_: *anyopaque, _: std.mem.Allocator) ![]sessions.disk.Blob {
        return error.Unused;
    }
    fn release(p: *anyopaque) void {
        @as(*Bound, @ptrCast(@alignCast(p))).released = true;
    }
    fn value(b: *Bound) sessions.store.Bounded {
        return .{ .ptr = b, .vtable = &.{ .nbytes = bytes, .blobs = blobs, .release = release } };
    }
};

const Host = struct {
    gpa: std.mem.Allocator,
    store: *sessions.Store,
    kx: struct { slot: *sessions.Slot },
    sent: usize = 0,
    spills: usize = 0,
    damage: bool = false,
    fail: ?anyerror = null,

    pub fn find(h: *Host, prompt: []const u32) !?sessions.store.Hit {
        const ids = try h.gpa.alloc(i32, prompt.len);
        defer h.gpa.free(ids);
        for (prompt, ids) |id, *out| out.* = @intCast(id);
        return h.store.find(41, ids);
    }
    pub fn poolPlan(h: *Host, n: u64, max_new: u64, hit: ?sessions.store.Hit, held: []const u32, extra: u32, spills: *std.ArrayList(u32)) !kv.sess.Plan {
        return kv.sess.admit(h.store, hit, n, max_new, h.kx.slot.capacity, held, extra, spills);
    }
    pub fn resumePlanned(h: *Host, _: []const u32, need: u32, hit: ?sessions.store.Hit, spills: []const u32) !admission.Resumed {
        h.sent += 1;
        if (h.fail) |err| return err;
        for (spills) |id| _ = try h.store.evict(id);
        try h.store.settleAll();
        h.spills += spills.len;
        h.kx.slot.reserve(need);
        if (hit) |entry| {
            if (h.damage) {
                try h.store.drop(entry.id);
                return .{ .damaged = true };
            }
            _ = try h.store.restoreRam(entry.id, h.kx.slot);
            return .{ .at = entry.pos };
        }
        return .{};
    }
};

fn save(store: *sessions.Store, slot: *sessions.Slot, ids: []const i32, bound: *Bound) !u32 {
    try slot.ensure(ids.len);
    const id = (try store.save(slot, bound.value(), ids, 41, 0)).?;
    _ = try slot.releaseAll();
    return id;
}

fn stockGrowth(world: u32, rank: u32) !void {
    const a = t.allocator;
    var pool = try sessions.Pool.init(a, .{ .families = &fams, .page = 512 }, 2100224, world, rank);
    defer pool.deinit();
    var host = try sessions.HostPool.init(a, &pool);
    defer host.deinit();
    var store = try sessions.Store.init(a, t.io, &pool, host.store(), null, .{ .ram_bytes = 1 << 30 });
    defer store.deinit();
    const idle = try pool.newSlot(2100224);
    const active = try pool.newSlot(655360);
    const next = try pool.newSlot(655360);
    const ids = try a.alloc(i32, 1990000);
    defer a.free(ids);
    @memset(ids, 7);
    var bound: Bound = .{};
    _ = try save(&store, idle, ids, &bound);
    const prompt = try a.alloc(u32, 117029);
    defer a.free(prompt);
    @memset(prompt, 9);
    var h: Host = .{ .gpa = a, .store = &store, .kx = .{ .slot = active } };
    try t.expect(pool.available() > 0 and @as(u64, pool.available()) * pool.page < prompt.len);
    _ = try admission.resumeRequest(&h, prompt, 256);
    try t.expect(h.spills > 0 and bound.released);
    try t.expect(active.quota.? >= (prompt.len + 256 + kv.sess.slack + 511) / 512);
    // A concurrent request cannot consume the active request's unmapped reservation.
    var h2: Host = .{ .gpa = a, .store = &store, .kx = .{ .slot = next } };
    _ = try admission.resumeRequest(&h2, prompt, 256);
    try active.ensure(65536);
    try next.ensure(65536);
    try active.ensure(prompt.len + 256);
    try next.ensure(prompt.len + 256);
    _ = try active.releaseAll();
    _ = try next.releaseAll();
    try t.expectEqual(pool.npages, pool.available());
}

test "pool growth: stock 117029-token admission reclaims idle ownership before the first piece on each rank" {
    try stockGrowth(1, 0);
    try stockGrowth(2, 0);
    try stockGrowth(2, 1);
}

test "pool growth: active reservations refuse without mutation, partial prefill rollback permits the next request" {
    const a = t.allocator;
    var pool = try sessions.Pool.init(a, .{ .families = &fams, .page = 16 }, 512, 1, 0);
    defer pool.deinit();
    var host = try sessions.HostPool.init(a, &pool);
    defer host.deinit();
    var store = try sessions.Store.init(a, t.io, &pool, host.store(), null, .{});
    defer store.deinit();
    const active = try pool.newSlot(512);
    const next = try pool.newSlot(512);
    var h: Host = .{ .gpa = a, .store = &store, .kx = .{ .slot = active } };
    const prompt: [300]u32 = @splat(4);
    _ = try admission.resumeRequest(&h, &prompt, 64);
    try active.ensure(128);
    const free = pool.freePages();
    const available = pool.available();
    var h2: Host = .{ .gpa = a, .store = &store, .kx = .{ .slot = next } };
    try t.expectError(error.LaneMemory, admission.resumeRequest(&h2, &prompt, 64));
    try t.expectEqual(@as(usize, 0), h2.sent);
    try t.expectEqual(free, pool.freePages());
    try t.expectEqual(available, pool.available());
    try t.expectEqual(@as(u32, 0), next.len);
    try t.expect(next.quota == null);
    // A refused growth leaves the partial prompt untouched, then release returns pages and its remaining quota.
    try t.expectError(error.PoolExhausted, next.ensure(512));
    try t.expectEqual(@as(u32, 8), active.len);
    _ = try active.releaseAll();
    _ = try admission.resumeRequest(&h2, &prompt, 64);
    try next.ensure(prompt.len + 64);
    _ = try next.releaseAll();
    try t.expectEqual(pool.npages, pool.available());
}

test "pool growth: a damaged RAM hit re-prices its lost shared credit, device and transport errors propagate" {
    const a = t.allocator;
    var pool = try sessions.Pool.init(a, .{ .families = &fams, .page = 16 }, 512, 1, 0);
    defer pool.deinit();
    var host = try sessions.HostPool.init(a, &pool);
    defer host.deinit();
    var store = try sessions.Store.init(a, t.io, &pool, host.store(), null, .{});
    defer store.deinit();
    const idle = try pool.newSlot(512);
    const active = try pool.newSlot(512);
    const ids: [128]i32 = @splat(7);
    var bound: Bound = .{};
    _ = try save(&store, idle, &ids, &bound);
    const prompt: [300]u32 = @splat(7);
    var h: Host = .{ .gpa = a, .store = &store, .kx = .{ .slot = active }, .damage = true };
    const r = try admission.resumeRequest(&h, &prompt, 64);
    try t.expect(r.damaged and r.at == 0 and h.sent == 2);
    try active.ensure(prompt.len + 64);
    _ = try active.releaseAll();
    for ([_]anyerror{ error.CudaFailed, error.PeerFailed }) |err| {
        h.fail = err;
        try t.expectError(err, admission.resumeRequest(&h, &prompt, 64));
        try t.expectEqual(@as(u32, 0), active.len);
    }
}

test {
    _ = @import("pool_growth_test.zig");
}

test "pool growth: a planned restore miss preserves the original reply reservation" {
    var pool = try sessions.Pool.init(t.allocator, .{ .families = &fams, .page = 16 }, 512, 1, 0);
    defer pool.deinit();
    var host = try sessions.HostPool.init(t.allocator, &pool);
    defer host.deinit();
    var store = try sessions.Store.init(t.allocator, t.io, &pool, host.store(), null, .{});
    defer store.deinit();
    const slot = try pool.newSlot(512);
    slot.reserve(24);
    var h: Host = .{ .gpa = t.allocator, .store = &store, .kx = .{ .slot = slot } };
    const prompt: [128]u32 = @splat(7);
    try admission.afterMiss(&h, &prompt, 0);
    try t.expectEqual(@as(?u32, 24), slot.quota);
    try slot.ensure(320);
    _ = try slot.releaseAll();
}

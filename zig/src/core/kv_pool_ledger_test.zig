//! A pool counted in a kv_ledger.Ledger (#1943): it admits exactly what it admitted alone, and shares the bytes.
const std = @import("std");
const api = @import("engine_api");
const kv = @import("kv_pool");
const Ledger = api.kv_ledger.Ledger;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

/// The lane host's view of a pool (lane_host_ledger.count), built by hand for a pool with no host.
fn counted(p: *kv.Pool, l: *Ledger) void {
    const view = struct {
        fn read(ctx: *const anyopaque) api.kv_ledger.Pages {
            const q: *const kv.Pool = @ptrCast(@alignCast(ctx));
            return .{
                .total = q.total(),
                .available = q.available(),
                .reclaimable = q.reclaimable(),
                .unit = q.unit(),
            };
        }
        fn detach(ctx: *anyopaque) void {
            const q: *kv.Pool = @ptrCast(@alignCast(ctx));
            q.ledger = null;
        }
    };
    l.* = .ofPages(.{ .ctx = p, .read = view.read, .detach = view.detach });
    p.ledger = .{ .ctx = l, .want = Ledger.wantOf };
}

/// A store that keeps pages and gives up its oldest when a promise asks; it records every reclaim's `want`.
const Store = struct {
    pool: *kv.Pool,
    held: std.ArrayList(u32) = .empty,
    asked: std.ArrayList(usize) = .empty,
    fn call(ctx: *anyopaque, want: usize) bool {
        const s: *Store = @ptrCast(@alignCast(ctx));
        s.asked.append(std.testing.allocator, want) catch {};
        while (s.pool.available() < want and s.held.items.len > 0) s.pool.release(s.held.orderedRemove(0));
        return s.pool.available() >= want;
    }
    fn deinit(s: *Store) void {
        s.held.deinit(std.testing.allocator);
        s.asked.deinit(std.testing.allocator);
    }
};

/// One side of the walk: a pool, its store and its streams' leases.
const Side = struct {
    pool: kv.Pool,
    store: Store = undefined,
    leases: [4]kv.Lease = .{ .{}, .{}, .{}, .{} },

    fn init(s: *Side, pages: usize, page_bytes: u64) !void {
        s.* = .{ .pool = try kv.Pool.init(std.testing.allocator, pages, page_bytes) };
        s.store = .{ .pool = &s.pool };
        s.pool.reclaim = .{ .ctx = &s.store, .call = Store.call };
    }

    fn deinit(s: *Side) void {
        for (&s.leases) |*l| s.pool.finish(l);
        for (s.store.held.items) |id| s.pool.release(id);
        s.store.deinit();
        s.pool.deinit();
    }

    /// One step of the walk: `op` picks what, `n` how many pages, `i` which lease; the outcome as a number.
    fn step(s: *Side, op: u8, n: usize, i: usize) !usize {
        const l = &s.leases[i];
        switch (op) {
            0 => return @intFromBool(if (s.pool.promise(l, n)) true else |_| false),
            1 => { // take what it promised, then the store keeps a copy of its first page
                const want = l.ids.items.len + l.promised;
                try s.pool.cover(l, want, null);
                if (l.ids.items.len > 0) {
                    s.pool.retain(l.ids.items[0]);
                    try s.store.held.append(std.testing.allocator, l.ids.items[0]);
                }
                return want;
            },
            2 => {
                s.pool.finish(l);
                return 0;
            },
            3 => {
                var ids: [3]u32 = undefined;
                const k = @min(n, ids.len);
                if (!s.pool.hold(ids[0..k])) return 0;
                try s.store.held.appendSlice(std.testing.allocator, ids[0..k]);
                return 1;
            },
            else => return @intFromBool(s.pool.fits(n)),
        }
    }
};

fn walk(page_bytes: u64) !void {
    var alone: Side = undefined;
    try alone.init(24, page_bytes);
    defer alone.deinit();
    var both: Side = undefined;
    try both.init(24, page_bytes);
    defer both.deinit();
    var ledger: Ledger = undefined;
    counted(&both.pool, &ledger);
    var rng = std.Random.DefaultPrng.init(1943);
    const r = rng.random();
    for (0..4000) |_| {
        const op = r.uintLessThan(u8, 5);
        const n = r.uintLessThan(usize, 30);
        const i = r.uintLessThan(usize, 4);
        try expectEqual(try alone.step(op, n, i), try both.step(op, n, i));
        try expectEqual(alone.pool.census(), both.pool.census());
        for (0..30) |k| try expectEqual(alone.pool.fits(k), both.pool.fits(k));
    }
    try std.testing.expectEqualSlices(usize, alone.store.asked.items, both.store.asked.items); // the same reclaims
    try expect(alone.store.asked.items.len > 10 and ledger.idle());
    ledger.detach();
    try expect(both.pool.ledger == null);
}

test "kv ledger over a pool: a walk of promises, takes, holds and reclaims admits exactly what the pool alone did" {
    try walk(4096);
    try walk(0); // pages of no bytes count one byte each, so the ledger still counts them
}

test "kv ledger over a pool: the pool's pages are a sub-count of its bytes, which leases beside them share" {
    var p = try kv.Pool.init(std.testing.allocator, 8, 100);
    defer p.deinit();
    var l: Ledger = undefined;
    counted(&p, &l);
    try expectEqual(@as(u64, 800), l.total);
    var a: kv.Lease = .{};
    try p.promise(&a, 3);
    try expectEqual(@as(u64, 500), l.free()); // promised pages count as soon as they are promised
    try p.cover(&a, 2, null);
    try expectEqual(@as(u64, 500), l.free()); // taking them changes nothing in the ledger
    var bytes: api.kv_ledger.Lease = .{};
    try l.promise(&bytes, 250); // an unpaged lease beside the pool: three pages' worth are left, not five
    try expect(p.fits(2) and !p.fits(3));
    try std.testing.expectError(error.LaneMemory, p.promise(&a, 3));
    try expect(!l.fits(251));
    l.finish(&bytes);
    try p.promise(&a, 5);
    try expectEqual(@as(u64, 0), l.free());
    p.finish(&a);
    try expect(p.idle() and l.idle() and l.free() == l.total);
    l.detach();
}

//! A lease's retired pages (C5, #2444 P4): no page is reused before its readers are done, and the counts stay exact.
const std = @import("std");
const kv = @import("kv_pool.zig");
const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

/// The change notice's view: `available()`, and each count against a scan of the pages themselves.
const Seen = struct {
    available: usize = 0,
    bad: usize = 0,
    fn call(ctx: *anyopaque, p: *const kv.Pool) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.available = p.available();
        if (!exact(p)) s.bad += 1;
    }
};

/// Whether the pool's counts match a scan: free, pending, leased, kept and the store's alone, page by page.
fn exact(p: *const kv.Pool) bool {
    var free: usize = 0;
    var leased: usize = 0;
    var kept: usize = 0;
    var alone: usize = 0;
    for (p.refs, p.leased, 0..) |r, l, i| {
        free += @intFromBool(r == 0 and !kv.retirement.held(p, i));
        leased += @intFromBool(l > 0);
        kept += @intFromBool(r > l);
        alone += @intFromBool(r == 1 and l == 0);
    }
    const r = &p.retired;
    for (r.pending.items) |id| if (p.refs[id] != 0 or r.tokens[id] <= r.settled) return false;
    if (p.nfree < p.promised) return false;
    return free == p.nfree and
        leased == p.nleased and kept == p.nkept and alone == p.nreclaimable;
}

/// A store that holds pages beside the leases, and lets one go when the pool asks.
const Store = struct {
    held: [4]u32 = undefined,
    n: usize = 0,
    fn keep(s: *Store, p: *kv.Pool, id: u32) void {
        p.retain(id);
        s.held[s.n] = id;
        s.n += 1;
    }
    fn reclaim(ctx: *anyopaque, _: usize) bool {
        const s: *Store = @ptrCast(@alignCast(ctx));
        if (s.n == 0) return false;
        s.n -= 1;
        pool.?.release(s.held[s.n]);
        return true;
    }
    var pool: ?*kv.Pool = null;
};

test "a retired page waits for its readers: counts and notices unchanged until settle, then free and promised again" {
    var p = try kv.Pool.init(testing.allocator, 1, 64);
    defer p.deinit();
    var seen: Seen = .{};
    p.notify = .{ .ctx = &seen, .call = Seen.call };
    var l: kv.Lease = .{};
    try p.promise(&l, 1);
    const id = try p.take(&l);
    try kv.retirement.retire(&p, &l, 0, 7);
    try expectEqual(kv.retirement.none, l.ids.items[0]);
    try expectEqual(@as(usize, 0), seen.available); // not free: a forward up to 7 may still read it
    try expectEqual(@as(usize, 0), l.promised); // its promise comes back only when it is free
    try expectEqual(@as(usize, 1), p.census().pending);
    try expect(!p.idle());
    try kv.retirement.settle(&p, 6);
    try expectEqual(@as(usize, 1), p.census().pending);
    try kv.retirement.settle(&p, 7);
    try expectEqual(@as(usize, 0), p.census().pending);
    try expectEqual(@as(usize, 1), l.promised);
    try expectEqual(id, try p.take(&l)); // the same page again, after its readers
    try expectEqual(@as(usize, 0), seen.bad);
    p.finish(&l);
    try expect(p.idle());
}

test "a retired page the store keeps waits for its token whenever the store lets it go" {
    var p = try kv.Pool.init(testing.allocator, 2, 64);
    defer p.deinit();
    var store: Store = .{};
    Store.pool = &p;
    defer Store.pool = null;
    try p.joinReclaim(.{ .ctx = &store, .call = Store.reclaim });
    var l: kv.Lease = .{};
    try p.promise(&l, 2);
    const a = try p.take(&l);
    _ = try p.take(&l);
    store.keep(&p, a);
    try kv.retirement.retire(&p, &l, 0, 3);
    try expectEqual(@as(u32, 1), p.holders(a)); // the store's alone
    _ = Store.reclaim(&store, 1); // evicted before settle: pending, not free
    try expectEqual(@as(usize, 1), p.census().pending);
    try expectEqual(@as(usize, 0), p.available());
    try kv.retirement.settle(&p, 3); // the pool is full: the credit is granted from the page now free
    try expectEqual(@as(usize, 1), l.promised);
    _ = try p.take(&l);

    // the store holds a retired page through settle: the credit's promise reclaims it, free at once
    store.keep(&p, l.ids.items[1]);
    try kv.retirement.retire(&p, &l, 1, 4);
    try kv.retirement.settle(&p, 4);
    try expectEqual(@as(usize, 1), l.promised);
    try expectEqual(@as(usize, 0), p.census().pending);
    p.finish(&l);
    try expect(p.idle());
    p.leaveReclaim(&store);
}

test "a store's page retired and evicted after settle is free at once" {
    var p = try kv.Pool.init(testing.allocator, 4, 64);
    defer p.deinit();
    var l: kv.Lease = .{};
    try p.promise(&l, 1);
    const a = try p.take(&l);
    p.retain(a); // the store
    try kv.retirement.retire(&p, &l, 0, 2);
    try kv.retirement.settle(&p, 2); // room: the credit needs no reclaim
    p.release(a);
    try expectEqual(@as(usize, 0), p.census().pending);
    try expectEqual(@as(usize, 4), p.nfree);
    p.finish(&l);
    try expect(p.idle());
}

test "no take, run or aligned run gets a pending page" {
    var p = try kv.Pool.init(testing.allocator, 4, 64);
    defer p.deinit();
    var l: kv.Lease = .{};
    try p.promise(&l, 4);
    try p.cover(&l, 4, null);
    try kv.retirement.retire(&p, &l, 1, 9); // page 1 pending: no run of 4, and none of 2 at 0
    var m: kv.Lease = .{};
    p.trim(&l, 1); // pages 2 and 3 free; the retired entry's promise is its credit's
    p.unpromise(&l);
    try p.promise(&m, 2);
    try expectEqual(@as(?u32, 2), p.findRun(2));
    try expectEqual(@as(?u32, 2), p.findAligned(2, 2));
    try expectEqual(@as(?u32, null), p.findRun(3)); // [1, 4) holds the pending page
    try expectEqual(@as(?u32, null), p.findAligned(3, 1));
    try p.coverRun(&m, 2, 2, null);
    for (m.ids.items) |id| try expect(id != 1);
    try kv.retirement.settle(&p, 9);
    try expectEqual(@as(usize, 1), p.nfree);
    p.finish(&m);
    p.finish(&l);
    try expect(p.idle());
}

test "a store page with readers pending is not reclaimable capacity; it is once they settle" {
    var p = try kv.Pool.init(testing.allocator, 1, 64);
    defer p.deinit();
    var l: kv.Lease = .{};
    try p.promise(&l, 1);
    const a = try p.take(&l);
    p.retain(a); // the store
    try kv.retirement.retire(&p, &l, 0, 7);
    p.finish(&l);
    try expectEqual(@as(usize, 0), p.reclaimable());
    try expect(!p.fits(1));
    try kv.retirement.settle(&p, 7);
    try expectEqual(@as(usize, 1), p.reclaimable());
    try expect(p.fits(1));
    p.release(a);
    try expect(p.idle());
}

test "a settle that cannot grant a credit fails and grants it on a later settle, once" {
    var p = try kv.Pool.init(testing.allocator, 2, 64);
    defer p.deinit();
    var seen: Seen = .{};
    p.notify = .{ .ctx = &seen, .call = Seen.call };
    var l: kv.Lease = .{};
    var other: kv.Lease = .{};
    try p.promise(&l, 2);
    const a = try p.take(&l);
    _ = try p.take(&l);
    p.retain(a); // the store keeps it, and no reclaim chain can free it
    try kv.retirement.retire(&p, &l, 0, 3);
    try kv.retirement.retire(&p, &l, 1, 3); // freed: one credit can be granted, the other cannot
    try expectError(error.LaneMemory, kv.retirement.settle(&p, 3));
    try expectEqual(@as(usize, 1), l.promised); // one granted
    try expectEqual(@as(usize, 1), p.retired.credits.items.len);
    try expectError(error.LaneMemory, kv.retirement.settle(&p, 3)); // nothing new: no second grant
    try expectEqual(@as(usize, 1), l.promised);
    p.release(a); // the store lets it go: free (its readers are done)
    try kv.retirement.settle(&p, 3);
    try expectEqual(@as(usize, 2), l.promised);
    try expectError(error.LaneMemory, p.promise(&other, 1));
    try expectEqual(@as(usize, 0), seen.bad);
    p.finish(&l);
    try expect(p.idle());
}

test "a share into an empty entry takes the store's page with nothing to give back" {
    var p = try kv.Pool.init(testing.allocator, 2, 64);
    defer p.deinit();
    var held: [1]u32 = undefined;
    try expect(p.hold(&held));
    var l: kv.Lease = .{};
    try kv.retirement.skip(&p, &l, 1);
    p.share(&l, 0, held[0]);
    try expectEqual(held[0], l.ids.items[0]);
    try expectEqual(@as(u32, 2), p.holders(held[0]));
    try expect(exact(&p));
    p.finish(&l);
    p.release(held[0]);
    try expect(p.idle());
}

test "a retire whose allocations fail changes nothing, at each of them, and its retry succeeds" {
    for (0..3) |j| { // a fresh pool's first retire: the pending list's room, the tokens, the credit
        var fail = std.testing.FailingAllocator.init(testing.allocator, .{});
        var p = try kv.Pool.init(fail.allocator(), 2, 64);
        defer p.deinit();
        var l: kv.Lease = .{};
        defer p.finish(&l);
        try p.promise(&l, 1);
        const a = try p.take(&l);
        fail.fail_index = fail.alloc_index + j;
        try testing.expectError(error.OutOfMemory, kv.retirement.retire(&p, &l, 0, 1));
        try expectEqual(a, l.ids.items[0]);
        try expectEqual(@as(u32, 1), p.holders(a));
        try expect(exact(&p));
        fail.fail_index = std.math.maxInt(usize);
        try kv.retirement.retire(&p, &l, 0, 1);
        try expectEqual(kv.retirement.none, l.ids.items[0]);
        try kv.retirement.settle(&p, 1);
    }
}

test "skipped entries name no page: a take after them sits at its logical index, and finish passes them by" {
    var p = try kv.Pool.init(testing.allocator, 4, 64);
    defer p.deinit();
    var l: kv.Lease = .{};
    try kv.retirement.skip(&p, &l, 3);
    try p.promise(&l, 1);
    const id = try p.take(&l);
    try expectEqual(@as(usize, 4), l.ids.items.len);
    try expectEqual(id, l.ids.items[3]);
    for (l.ids.items[0..3]) |e| try expectEqual(kv.retirement.none, e);
    p.trim(&l, 2); // the page back to the promise, the skipped entries dropped with no promise
    try expectEqual(@as(usize, 1), l.promised);
    p.finish(&l);
    try expect(p.idle());
}

test "a lease that ends with credits owed drops them: the pool is idle once its pages settle" {
    var p = try kv.Pool.init(testing.allocator, 2, 64);
    defer p.deinit();
    var l: kv.Lease = .{};
    try p.promise(&l, 2);
    try p.cover(&l, 2, null);
    try kv.retirement.retire(&p, &l, 0, 5);
    p.finish(&l);
    try expect(!p.idle()); // page 0 still pending
    try kv.retirement.settle(&p, 5);
    try expect(p.idle());
}

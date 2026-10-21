//! The page pool's own tests: the layout's arithmetic, the load plan, leases, and cleanup after every injected failure.
const std = @import("std");
const kv = @import("kv_pool.zig");
const Pool = kv.Pool;
const Lease = kv.Lease;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const mib: u64 = 1 << 20;

/// Nemotron 3.5 Lightning on CUDA: 6 attention layers and the MTP head's, 2 KV heads of 128, Mamba states in a slot.
const nemotron: kv.Layout = .{ .fixed_bytes = 64 * mib, .token_bytes = 7 << 10, .page_tokens = 512, .snap_bytes = 48 * mib };

test "a 2,048-token prompt with 4,096 reply tokens and a 16-row window reserves 13 pages of 512 rows, 45.5 MiB" {
    try expectEqual(@as(u64, 3584 * 1024), nemotron.pageBytes());
    try expectEqual(@as(usize, 13), nemotron.need(2048, 4096, 16));
    try expectEqual(@as(u64, 45 * mib + mib / 2), 13 * nemotron.pageBytes());
    try expectEqual(@as(usize, 12), nemotron.need(2000, 4096, 16));
    try expectEqual(@as(usize, 1), nemotron.need(1, 0, 16));
    try expectEqual(@as(usize, 2), nemotron.need(497, 0, 16)); // one row past a page edge takes the next page
    try expectEqual(@as(usize, 1), nemotron.need(496, 0, 16));
}

test "the plan: the room pays the engine's own sequence, the other streams' fixed parts, slots and copied KV, then pages" {
    const g: u64 = 1 << 30;
    const p = try kv.plan(nemotron, .{ .room = 40 * g, .engine = g, .lanes = 4, .cache = 2 * g });
    try expectEqual(@as(u32, 4), p.lanes);
    try expectEqual(@as(usize, 21), p.snaps); // half the cache in 48 MiB slots
    try expectEqual(2 * g - 21 * 48 * mib, p.copied);
    try expectEqual(@as(usize, @intCast((40 * g - g - 3 * 64 * mib - 2 * g) / nemotron.pageBytes())), p.pages);
    const named = try kv.plan(nemotron, .{ .room = 40 * g, .engine = g, .lanes = 1, .cache = 2 * g, .snaps = 4 });
    try expectEqual(@as(usize, 4), named.snaps);
    try expectEqual(2 * g - 4 * 48 * mib, named.copied);
    const none = try kv.plan(nemotron, .{ .room = 40 * g, .engine = g, .lanes = 2 });
    try expectEqual(@as(usize, 0), none.snaps);
    try expectEqual(@as(u64, 0), none.copied);
    try expectEqual(@as(usize, @intCast((39 * g - 64 * mib) / nemotron.pageBytes())), none.pages);
    // streams beside the engine's: each its fixed part and a page, or fewer are served (a named count refuses)
    const small = 2 * g + 2 * (64 * mib + nemotron.pageBytes()) + mib;
    try expectEqual(@as(u32, 3), (try kv.plan(nemotron, .{ .room = small, .engine = 2 * g, .lanes = 8 })).lanes);
    try std.testing.expectError(error.TooMany, kv.plan(nemotron, .{ .room = small, .engine = 2 * g, .lanes = 8, .fixed = true }));
    try std.testing.expectError(error.NoStream, kv.plan(nemotron, .{ .room = g, .engine = 2 * g, .lanes = 1 }));
    // the cache gets what the streams leave: a page each comes first
    const tight = try kv.plan(nemotron, .{ .room = small - mib + 200 * mib, .engine = 2 * g, .lanes = 3, .cache = 16 * g });
    try expectEqual(@as(usize, 2), tight.pages);
    try expectEqual(@as(usize, 2), tight.snaps);
    try expectEqual(@as(u64, 104 * mib), tight.copied);
    // no more pages than the streams beside the engine's hold at once, each at its window
    const bounded = try kv.plan(nemotron, .{ .room = 40 * g, .engine = g, .lanes = 4, .window = 16384 + 16 });
    try expectEqual(@as(usize, 3 * 33), bounded.pages);
    try expectEqual(kv.split(2 * g, 48 * mib, null).snaps, p.snaps);
    try expectEqual(@as(usize, kv.most_snaps), kv.split(16 * g, 48 * mib, 1000).snaps);
}

test "a family without snapshot slots (snap_bytes 0) keeps its whole cache as copied rows beside the pages" {
    const g: u64 = 1 << 30;
    // GLM-5.3-Flash on CUDA at TP=2: KDA states and rings fixed a stream, 13,056 bytes a token, kept states copied
    const copies: kv.Layout = .{ .fixed_bytes = 180 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    const s = kv.split(16 * g, 0, null);
    try expectEqual(@as(usize, 0), s.snaps);
    try expectEqual(16 * g, s.copied);
    try expectEqual(16 * g, kv.split(16 * g, 0, 4).copied); // --keep names no slots here
    const p = try kv.plan(copies, .{ .room = 60 * g, .engine = 0, .lanes = 4, .cache = 16 * g });
    try expectEqual(@as(usize, 0), p.snaps);
    try expectEqual(16 * g, p.copied);
    try expectEqual(@as(usize, @intCast((44 * g - 3 * 180 * mib) / copies.pageBytes())), p.pages);
}

test "a cache holds when the room past the engine and every stream's fixed part and page is at least its bytes" {
    const g: u64 = 1 << 30;
    const streams = 3 * (64 * mib + nemotron.pageBytes());
    try expect(kv.holds(nemotron, .{ .room = g + streams + 16 * g, .engine = g, .lanes = 4, .cache = 16 * g }));
    try expect(!kv.holds(nemotron, .{ .room = g + streams + 16 * g - 1, .engine = g, .lanes = 4, .cache = 16 * g }));
    const p = try kv.plan(nemotron, .{ .room = g + streams + 16 * g, .engine = g, .lanes = 4, .cache = 16 * g });
    try expectEqual(16 * g, p.snaps * nemotron.snap_bytes + p.copied); // what holds is what plan gives the cache whole
    try expect(!kv.holds(nemotron, .{ .room = g, .engine = 2 * g, .lanes = 4, .cache = 0 }));
}

test "a promise holds pages from every other lease, take turns it into pages, finish gives all of it back" {
    var p = try Pool.init(std.testing.allocator, 8, 100);
    defer p.deinit();
    var a: Lease = .{};
    var b: Lease = .{};
    try p.promise(&a, 5);
    try expectEqual(@as(usize, 3), p.available());
    try std.testing.expectError(error.LaneMemory, p.promise(&b, 4)); // the second waits
    try expectEqual(@as(usize, 0), b.promised);
    try p.cover(&a, 3, null);
    try expectEqual(@as(usize, 2), a.promised);
    try expectEqual(@as(usize, 3), p.available());
    try std.testing.expectError(error.PastReservation, p.cover(&a, 6, null)); // never past the reservation
    try expectEqual(@as(usize, 3), a.ids.items.len);
    try expectEqual(@as(usize, 2), a.promised);
    p.finish(&a);
    try p.promise(&b, 8); // then it runs
    try p.cover(&b, 8, null);
    try expectEqual(@as(usize, 0), p.available());
    p.finish(&b);
    try expect(p.idle());
    try expectEqual(@as(usize, 8), p.nfree);
}

test "a page released with no holder is refused, and so is a store's release of a page only a stream holds" {
    var p = try Pool.init(std.testing.allocator, 4, 100);
    defer p.deinit();
    try std.testing.expectError(error.NotHeld, p.drop(2));
    var a: Lease = .{};
    try p.promise(&a, 1);
    const id = (try p.take(&a));
    try std.testing.expectError(error.NotHeld, p.drop(id)); // the store never held it
    p.retain(id); // the store keeps it now
    try p.drop(id);
    try std.testing.expectError(error.NotHeld, p.drop(id));
    p.finish(&a);
    try std.testing.expectError(error.NotHeld, p.drop(id));
    try expect(p.idle());
}

/// The device page tables of a test's streams: the `fail_at`-th write of any fails, as a failed upload does.
const Tables = struct {
    writes: usize = 0,
    fail_at: ?usize = null,
    entries: [4][64]u32 = @splat(@splat(std.math.maxInt(u32))),
    at: [4]Slot = undefined,

    const Slot = struct { t: *Tables, k: usize };

    fn table(t: *Tables, k: usize) kv.Table {
        t.at[k] = .{ .t = t, .k = k };
        return .{ .ctx = &t.at[k], .write = write };
    }

    fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
        const slot: *Slot = @ptrCast(@alignCast(ctx));
        const t = slot.t;
        defer t.writes += 1;
        if (t.fail_at == t.writes) return error.UploadFailed;
        @memcpy(t.entries[slot.k][from..ids.len], ids[from..]);
    }
};

/// Three streams growing in turn, a fourth adopting a page; the first error stops it, the pool left as it was.
fn growth(p: *Pool, t: *Tables, leases: *[4]Lease) !void {
    for (leases[0..3], 0..) |*l, i| try p.promise(l, 4 + i);
    for (1..5) |n| for (leases[0..3], 0..) |*l, k| {
        const before = .{ l.ids.items.len, l.promised, p.nfree, p.promised };
        p.cover(l, n, t.table(k)) catch |e| {
            try expectEqual(before, .{ l.ids.items.len, l.promised, p.nfree, p.promised }); // rolled back whole
            return e;
        };
    };
    p.retain(leases[0].ids.items[0]); // a store's resume retains a page of the first for the fourth
    try p.adopt(&leases[3], leases[0].ids.items[0..1]);
}

/// Every lease's covered pages are its device table's entries; then every lease ends and the pool is whole again.
fn settle(p: *Pool, t: *const Tables, leases: *[4]Lease) !void {
    for (leases, 0..) |*l, k| {
        const covered = if (k < 3) l.ids.items.len else 0;
        try std.testing.expectEqualSlices(u32, l.ids.items[0..covered], t.entries[k][0..covered]);
        p.finish(l);
    }
    try expect(p.idle());
    try expectEqual(p.total(), p.nfree);
}

test "every table upload failure in turn rolls back what it took: nothing leaked, nothing freed twice" {
    var k: usize = 0;
    while (true) : (k += 1) {
        var p = try Pool.init(std.testing.allocator, 16, 100);
        defer p.deinit();
        var t: Tables = .{ .fail_at = k };
        var leases: [4]Lease = .{ .{}, .{}, .{}, .{} };
        const failed = if (growth(&p, &t, &leases)) |_| false else |e| blk: {
            try expectEqual(error.UploadFailed, e);
            break :blk true;
        };
        try settle(&p, &t, &leases);
        if (!failed) break;
    }
    try expectEqual(@as(usize, 12), k); // one failure for each write the run makes
}

test "every allocation failure in turn: the pool and its leases free what they hold" {
    var i: usize = 0;
    while (true) : (i += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = i });
        const gpa = failing.allocator();
        var t: Tables = .{};
        var p = Pool.init(gpa, 16, 100) catch continue;
        defer p.deinit();
        var leases: [4]Lease = .{ .{}, .{}, .{}, .{} };
        growth(&p, &t, &leases) catch |e| try expectEqual(error.OutOfMemory, e);
        try settle(&p, &t, &leases);
        if (!failing.has_induced_failure) break;
    }
    try expect(i > 3);
}

test "a promise that does not fit asks the store to reclaim first, and fails only when it still does not fit" {
    var p = try Pool.init(std.testing.allocator, 4, 100);
    defer p.deinit();
    const Kept = struct {
        pool: *Pool,
        held: std.ArrayList(u32) = .empty,
        fn call(ctx: *anyopaque, want: usize) bool {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            while (s.pool.available() < want) s.pool.release(s.held.pop() orelse return false);
            return true;
        }
    };
    var store: Kept = .{ .pool = &p };
    defer store.held.deinit(std.testing.allocator);
    var a: Lease = .{};
    try p.promise(&a, 3);
    try p.cover(&a, 3, null);
    for (a.ids.items) |id| {
        p.retain(id);
        try store.held.append(std.testing.allocator, id);
    }
    p.finish(&a); // the store alone holds three pages
    try expectEqual(@as(usize, 1), p.available());
    try expectEqual(@as(usize, 3), p.reclaimable());
    try expect(p.fits(4) and !p.fits(5));
    p.reclaim = .{ .ctx = &store, .call = Kept.call };
    var b: Lease = .{};
    try std.testing.expectError(error.LaneMemory, p.promise(&b, 5));
    try expectEqual(@as(usize, 0), b.promised);
    try p.promise(&b, 4);
    p.finish(&b);
    try expect(p.idle());
}

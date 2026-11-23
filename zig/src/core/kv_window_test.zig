//! A stream's window pages over a real pool: the checkpoints' geometry, the live bound, band tables, keeps, resumes.
const std = @import("std");
const kv = @import("kv_pool.zig");
const win = @import("kv_window.zig");
const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;

test "the checkpoints' sliding layers fill their pages: Gemma 4 31B and Kolibri-1 exactly, Gemma 4 26B-A4B with a fifth spare" {
    const g31 = try win.geometry(1024, 1, 256, 50, 2 * 16 * 256 * 2, 256 * 10 * 2 * 4 * 512 * 2);
    try expectEqual(@as(u32, 5), g31.per_band);
    try expectEqual(@as(u32, 10), g31.bands);
    try expectEqual(@as(u64, 0), g31.padding);
    const kolibri = try win.geometry(513, 64, 512, 40, 2 * 4 * 128 * 2, 512 * 10 * 2 * 4 * 128 * 2);
    try expectEqual(@as(u32, 10), kolibri.per_band);
    try expectEqual(@as(u32, 4), kolibri.bands);
    try expectEqual(@as(u64, 0), kolibri.padding);
    const g26 = try win.geometry(1024, 1, 256, 25, 2 * 8 * 256 * 2, 256 * 5 * 2 * 2 * 512 * 2);
    try expectEqual(@as(u32, 2), g26.per_band);
    try expectEqual(@as(u32, 13), g26.bands);
    try expectEqual(@as(u64, 1 << 20), g26.padding); // of 5 MiB
    try testing.expectError(error.WindowPageTooSmall, win.geometry(1024, 1, 256, 4, 1 << 20, 1 << 20));
}

test "a forward at c loads from first(c): the window's first key, rounded down to the kernels' widest tile" {
    const g = try win.geometry(1024, 64, 256, 1, 8, 2048);
    try expectEqual(@as(u32, 0), win.first(g, 0));
    try expectEqual(@as(u32, 0), win.first(g, 1023));
    try expectEqual(@as(u32, 0), win.first(g, 1024)); // key 1, rounded to 0
    try expectEqual(@as(u32, 960), win.first(g, 2000)); // key 977
    const one = try win.geometry(1024, 1, 256, 1, 8, 2048);
    try expectEqual(@as(u32, 977), win.first(one, 2000));
}

/// Each band's table as the device would hold it: logical page j of band b at rows[b][j].
const Device = struct {
    rows: [2][64]u32 = @splat(@splat(0xdead)),
    fail: bool = false,
    fn write(ctx: *anyopaque, band: u32, ids: []const u32, from: usize) anyerror!void {
        const d: *Device = @ptrCast(@alignCast(ctx));
        if (d.fail) return error.UploadFailed;
        @memcpy(d.rows[band][from..][0..ids.len], ids);
    }
    fn tables(d: *Device) win.Tables {
        return .{ .ctx = d, .write = write };
    }
};

/// Every page a forward from `k` through `end` can load is held, every band, and its table names it.
fn live(w: *const win.Window, d: *const Device, k: u32, end: u32) !void {
    const lo = win.first(w.g, k) / w.g.rows;
    for (lo..(end + w.g.rows - 1) / w.g.rows) |j| for (0..w.g.bands) |b| {
        const id = w.lease.ids.items[j * w.g.bands + b];
        try expect(id != kv.retirement.none);
        try expectEqual(id, d.rows[b][j]);
    };
}

fn exactPool(p: *const kv.Pool) !void {
    var free: usize = 0;
    for (p.refs, 0..) |r, i| free += @intFromBool(r == 0 and !kv.retirement.held(p, i));
    try expectEqual(free, p.nfree);
}

test "a prefill in chunks, then decode: held and unsettled pages stay within the promise, and each band's table follows" {
    const g = try win.geometry(512, 1, 128, 4, 8, 2 * 128 * 8); // two layers a band, two bands
    var pool = try kv.Pool.init(testing.allocator, 64, 2 * 128 * 8);
    defer pool.deinit();
    var dev: Device = .{};
    var w: win.Window = .{ .g = g };
    defer w.finish(&pool);
    const bound = win.promise(g, 256, 0);
    try pool.promise(&w.lease, bound);
    var token: u64 = 0;
    var at: u32 = 0;
    while (at < 2048) : (at += 256) { // prefill: a chunk from `at`
        token += 1;
        try w.step(&pool, at, at + 256, token, dev.tables()); // this step's retired pages still owe their promise
        var held: usize = 0;
        for (w.lease.ids.items) |id| held += @intFromBool(id != kv.retirement.none);
        try expect(held + pool.census().pending <= bound);
        try live(&w, &dev, at, at + 256);
        const lo = win.first(g, at) / g.rows;
        for (0..2) |b| for (0..lo) |j| try expectEqual(kv.retirement.none, dev.rows[b][j]);
        try kv.retirement.settle(&pool, token); // the forward's work is done
    }
    for (0..300) |_| { // decode, one row a step
        token += 1;
        try w.step(&pool, at, at + 1, token, dev.tables());
        try live(&w, &dev, at, at + 1);
        try kv.retirement.settle(&pool, token);
        at += 1;
    }
    try exactPool(&pool);
    try expectEqual(@as(usize, 0), pool.census().pending);
}

test "a keep's parts are each band's pages from first(at), and a resume places copies at their logical index" {
    const g = try win.geometry(300, 1, 128, 2, 8, 128 * 8); // one layer a band, two bands
    var pool = try kv.Pool.init(testing.allocator, 32, 128 * 8);
    defer pool.deinit();
    var dev: Device = .{};
    var w: win.Window = .{ .g = g };
    try pool.promise(&w.lease, win.promise(g, 256, 0));
    try w.step(&pool, 0, 512, 1, dev.tables());
    try kv.retirement.settle(&pool, 1);
    try w.step(&pool, 512, 700, 2, dev.tables()); // retires pages below first(512) = 213: page 0
    try kv.retirement.settle(&pool, 2);
    var out: [2]win.Band = undefined;
    try w.parts(700, testing.allocator, &out); // rows [401, 700): pages 3 to 5, the last partial
    defer for (out) |o| testing.allocator.free(o.ids);
    for (out, 0..) |o, b| {
        try expectEqual(@as(u32, 3), o.base);
        try expectEqual(@as(usize, 3), o.ids.len);
        for (o.ids, 3..) |id, j| try expectEqual(w.lease.ids.items[j * 2 + b], id);
    }
    try testing.expectError(error.WindowPagesGone, w.parts(400, testing.allocator, &out)); // page 0 is gone
    w.finish(&pool);

    var r: win.Window = .{ .g = g };
    defer r.finish(&pool);
    try pool.promise(&r.lease, win.promise(g, 256, 0));
    try r.resumeAt(&pool, 700, 3, dev.tables());
    try expectEqual(@as(usize, 12), r.lease.ids.items.len); // 3 pages skipped, 3 taken, a band each
    for (r.lease.ids.items[0..6]) |id| try expectEqual(kv.retirement.none, id);
    for (r.lease.ids.items[6..]) |id| try expect(id != kv.retirement.none);
    try expectEqual(r.lease.ids.items[6], dev.rows[0][3]);
    try expectEqual(r.lease.ids.items[7], dev.rows[1][3]);
    try r.step(&pool, 700, 701, 3, dev.tables()); // decode on from the resume: nothing below first(700) to retire
    try kv.retirement.settle(&pool, 3);
}

test "a step that cannot take a page fails with every count exact, and the window still finishes clean" {
    const g = try win.geometry(256, 1, 128, 1, 8, 128 * 8);
    var pool = try kv.Pool.init(testing.allocator, 4, 128 * 8);
    defer pool.deinit();
    var dev: Device = .{};
    var w: win.Window = .{ .g = g };
    try pool.promise(&w.lease, 2);
    try testing.expectError(error.PastReservation, w.step(&pool, 0, 512, 1, dev.tables()));
    try exactPool(&pool);
    w.finish(&pool);
    try expect(pool.idle());
}

test "a step that fails to take or to map leaves nothing taken, and its retry is the step it would have been" {
    const g = try win.geometry(256, 1, 128, 2, 8, 128 * 8); // two bands
    var pool = try kv.Pool.init(testing.allocator, 16, 128 * 8);
    defer pool.deinit();
    var dev: Device = .{};
    var w: win.Window = .{ .g = g };
    defer w.finish(&pool);
    try pool.promise(&w.lease, 3); // one band set short of what the step wants
    try testing.expectError(error.PastReservation, w.step(&pool, 0, 256, 1, dev.tables()));
    try expectEqual(@as(usize, 0), w.lease.ids.items.len); // no half set of bands
    try expectEqual(@as(usize, 3), w.lease.promised);
    try pool.promise(&w.lease, 1);
    dev.fail = true;
    try testing.expectError(error.UploadFailed, w.step(&pool, 0, 256, 2, dev.tables()));
    try expectEqual(@as(usize, 0), w.lease.ids.items.len);
    dev.fail = false;
    try w.step(&pool, 0, 256, 3, dev.tables());
    try live(&w, &dev, 0, 256);
    try exactPool(&pool);
}

test "a step that fails while it retires leaves what it retired to the caller's settle, and finish leaves the pool idle" {
    const g = try win.geometry(128, 1, 128, 1, 8, 128 * 8);
    var pool = try kv.Pool.init(testing.allocator, 16, 128 * 8);
    defer pool.deinit();
    var dev: Device = .{};
    var w: win.Window = .{ .g = g };
    try pool.promise(&w.lease, win.promise(g, 256, 0));
    try w.step(&pool, 0, 512, 1, dev.tables());
    try kv.retirement.settle(&pool, 1);
    dev.fail = true; // the retired pages' table write fails
    try testing.expectError(error.UploadFailed, w.step(&pool, 400, 401, 2, dev.tables()));
    try expectEqual(@as(usize, 2), pool.census().pending); // pages 0 and 1: forward 1 may still read them
    var out: [1]win.Band = undefined;
    try testing.expectError(error.WindowPagesGone, w.parts(300, testing.allocator, &out)); // its pages are gone
    dev.fail = false;
    try w.step(&pool, 400, 401, 3, dev.tables()); // forward 2 never ran: the retry writes their tables
    try expectEqual(kv.retirement.none, dev.rows[0][0]);
    try kv.retirement.settle(&pool, 3); // forward 3 done: every page retired so far is free
    try expectEqual(@as(usize, 0), pool.census().pending);
    w.finish(&pool);
    try expect(pool.idle());
}

test "a step settles nothing: a page retired for forward t stays out of take, runs and available() until settle(t)" {
    const g = try win.geometry(128, 1, 128, 1, 8, 128 * 8); // one band
    var pool = try kv.Pool.init(testing.allocator, 8, 128 * 8);
    defer pool.deinit();
    var dev: Device = .{};
    var w: win.Window = .{ .g = g };
    defer w.finish(&pool);
    try pool.promise(&w.lease, win.promise(g, 128, 0));
    try w.step(&pool, 0, 128, 4, dev.tables());
    try kv.retirement.settle(&pool, 4);
    var a: kv.Lease = .{};
    try pool.promise(&a, 1);
    const page = try pool.take(&a);
    try kv.retirement.retire(&pool, &a, 0, 5); // forward 5, queued, is the last to read it
    pool.finish(&a); // its lease ends; the page still waits for forward 5
    const before = pool.available();
    try w.step(&pool, 256, 384, 5, dev.tables()); // the window's step for forward 5: retires its page 0, takes two
    for (w.lease.ids.items) |id| try expect(id != page);
    try expect(kv.retirement.held(&pool, page));
    try expectEqual(@as(usize, 2), pool.census().pending); // a's page and the window's page 0
    try expectEqual(before, pool.available());
    for (1..pool.total() + 1) |n| if (pool.findRun(n)) |at| try expect(page < at or page >= at + n);
    try exactPool(&pool);
    try kv.retirement.settle(&pool, 5); // forward 5's work is done
    try expect(!kv.retirement.held(&pool, page));
    try expectEqual(@as(usize, 0), pool.census().pending);
    try expectEqual(before + 1, pool.available()); // a's page; the window's comes back as its promise
    try exactPool(&pool);
}

/// A prompt store that keeps window pages and lets one go when the pool asks.
const Kept = struct {
    held: [4]u32 = undefined,
    n: usize = 0,
    var pool: ?*kv.Pool = null;
    fn reclaim(ctx: *anyopaque, _: usize) bool {
        const s: *Kept = @ptrCast(@alignCast(ctx));
        if (s.n == 0) return false;
        s.n -= 1;
        pool.?.release(s.held[s.n]);
        return true;
    }
};

test "a full pool: a kept page the window retires is the store's alone, reclaimed for its credit at settle" {
    const g = try win.geometry(128, 1, 128, 1, 8, 128 * 8);
    const bound = win.promise(g, 128, 0);
    var pool = try kv.Pool.init(testing.allocator, bound, 128 * 8); // room for this window alone
    defer pool.deinit();
    var store: Kept = .{};
    Kept.pool = &pool;
    defer Kept.pool = null;
    try pool.joinReclaim(.{ .ctx = &store, .call = Kept.reclaim });
    defer pool.leaveReclaim(&store);
    var dev: Device = .{};
    var w: win.Window = .{ .g = g };
    try pool.promise(&w.lease, bound);
    try w.step(&pool, 0, 256, 1, dev.tables());
    try kv.retirement.settle(&pool, 1);
    var out: [1]win.Band = undefined;
    try w.parts(128, testing.allocator, &out); // a keep at 128: page 0
    pool.retain(out[0].ids[0]);
    store.held[0] = out[0].ids[0];
    store.n = 1;
    testing.allocator.free(out[0].ids);
    var token: u64 = 1;
    var at: u32 = 256;
    while (at < 1024) : (at += 128) { // each step retires a page while the pool has no free page beyond the promise
        token += 1;
        try w.step(&pool, at, at + 128, token, dev.tables());
        try kv.retirement.settle(&pool, token); // a round's end: no keep or restore holds the store's page
        try expectEqual(@as(usize, 0), pool.census().pending);
    }
    try expectEqual(@as(usize, 0), store.n); // reclaimed for a credit
    w.finish(&pool);
    try expect(pool.idle());
}

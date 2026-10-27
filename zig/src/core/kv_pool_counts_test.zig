//! Independent array census after ownership changes, rollback, and deterministic mixed lease/cache traffic.
const std = @import("std");
const kv = @import("kv_pool.zig");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

fn scan(p: *const kv.Pool) !void {
    var free: usize = 0;
    var kept: usize = 0;
    var leased: usize = 0;
    var reclaimable: usize = 0;
    for (p.refs, p.leased, 0..) |r, l, id| {
        try expect(r >= l);
        if (r == 0) free += 1;
        if (r > l) kept += 1;
        if (l != 0) leased += 1;
        if (r == 1 and l == 0) reclaimable += 1;
        var listed: usize = 0;
        for (p.free[0..p.nfree]) |f| if (f == id) {
            listed += 1;
        };
        try equal(@as(usize, if (r == 0) 1 else 0), listed);
    }
    try equal(free, p.nfree);
    try expect(p.promised <= free);
    try equal(kv.Census{ .total = p.refs.len, .free = free, .promised = p.promised, .kept = kept, .leased = leased }, p.census());
    try equal(reclaimable, p.reclaimable());
    for (0..p.total() + 2) |need| try equal(free - p.promised + reclaimable >= need, p.fits(need));
    try equal(p.promised == 0 and leased == 0 and free + kept == p.total(), p.idle());
}

test "page accounting: overlapping cache and multiple lease holders count unique pages" {
    var p = try kv.Pool.init(std.testing.allocator, 8, 4096);
    defer p.deinit();
    var a: kv.Lease = .{};
    var b: kv.Lease = .{};
    var c: kv.Lease = .{};
    var d: kv.Lease = .{};
    defer p.finish(&a);
    defer p.finish(&b);
    defer p.finish(&c);
    defer p.finish(&d);
    try scan(&p);
    try p.promise(&a, 4);
    try scan(&p);
    const id = try p.take(&a);
    try scan(&p);
    _ = try p.take(&a);
    try scan(&p);
    p.retain(id);
    try scan(&p);
    p.retain(id);
    try scan(&p);
    try p.drop(id);
    try scan(&p);
    p.finish(&a);
    try scan(&p);
    try equal(@as(usize, 1), p.reclaimable());
    p.retain(id);
    try scan(&p);
    try p.adopt(&b, &.{id});
    try scan(&p);
    p.retain(id);
    try p.adopt(&c, &.{id});
    try scan(&p);
    try p.promise(&d, 1);
    _ = try p.take(&d);
    try scan(&p);
    p.share(&d, 0, id);
    try scan(&p);
    p.share(&d, 0, id);
    try scan(&p);
    try equal(@as(usize, 1), p.census().kept);
    try equal(@as(usize, 1), p.census().leased);
    try p.drop(id);
    try scan(&p);
    try std.testing.expectError(error.NotHeld, p.drop(id));
    try scan(&p);
    p.finish(&b);
    try scan(&p);
    p.finish(&c);
    try scan(&p);
    p.finish(&d);
    try scan(&p);
    try std.testing.expectError(error.NotHeld, p.drop(id));
    try scan(&p);
    try expect(p.idle());
}

const FailingTable = struct {
    calls: usize = 0,
    fail_at: usize,

    fn write(ctx: *anyopaque, _: []const u32, _: usize) !void {
        const t: *@This() = @ptrCast(@alignCast(ctx));
        const fail = t.calls == t.fail_at;
        t.calls += 1;
        if (fail) return error.UploadFailed;
    }

    fn table(t: *@This()) kv.Table {
        return .{ .ctx = t, .write = write };
    }
};

test "page accounting: contiguous runs and table failures conserve every page class" {
    var p = try kv.Pool.init(std.testing.allocator, 32, 4096);
    defer p.deinit();
    var a: kv.Lease = .{};
    defer p.finish(&a);
    try p.promise(&a, 8);
    try scan(&p);
    const start = p.claimRun(&a, 3).?;
    try scan(&p);
    try equal(@as(usize, 3), p.census().leased);
    p.releaseRun(start, 3);
    try scan(&p);
    p.unpromise(&a);
    try scan(&p);
    try p.promise(&a, 8);
    var table: FailingTable = .{ .fail_at = 0 };
    try std.testing.expectError(error.UploadFailed, p.coverRun(&a, 4, 4, table.table()));
    try scan(&p);
    try equal(@as(usize, 8), a.promised);
    try equal(@as(usize, 0), a.ids.items.len);
    try p.coverRun(&a, 4, 4, table.table());
    try scan(&p);
    try equal(@as(u32, 0), a.ids.items[0] % 4);
    table.fail_at = table.calls;
    try std.testing.expectError(error.UploadFailed, p.cover(&a, 8, table.table()));
    try scan(&p);
    try equal(@as(usize, 4), a.ids.items.len);
    try equal(@as(usize, 4), a.promised);
    try std.testing.expectError(error.PastReservation, p.cover(&a, 9, null));
    try scan(&p);
    p.finish(&a);
    try scan(&p);
    try expect(p.idle());
}

fn allocationSequence(p: *kv.Pool) !void {
    var a: kv.Lease = .{};
    var b: kv.Lease = .{};
    defer {
        p.finish(&b);
        scan(p) catch @panic("census after adopted lease cleanup");
        p.finish(&a);
        scan(p) catch @panic("census after allocation rollback");
    }
    try p.promise(&a, 24);
    try scan(p);
    p.cover(&a, 3, null) catch |err| {
        try scan(p);
        return err;
    };
    try scan(p);
    p.retain(a.ids.items[0]);
    try scan(p);
    p.adopt(&b, a.ids.items[0..1]) catch |err| {
        try scan(p);
        return err;
    };
    try scan(p);
    p.cover(&a, 24, null) catch |err| {
        try scan(p);
        return err;
    };
    try scan(p);
}

test "page accounting: every allocation failure preserves the full-scan oracle" {
    var at: usize = 0;
    while (at < 32) : (at += 1) {
        var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = at });
        var p = kv.Pool.init(allocator.allocator(), 32, 4096) catch continue;
        defer p.deinit();
        allocationSequence(&p) catch |err| try equal(error.OutOfMemory, err);
        try scan(&p);
        try expect(p.idle());
        if (!allocator.has_induced_failure) break;
    }
    try expect(at < 32 and at > 4);
}

test "page accounting: deterministic mixed traffic agrees with an independent ownership model" {
    const pages = 32;
    var p = try kv.Pool.init(std.testing.allocator, pages, 4096);
    defer p.deinit();
    var leases: [6]kv.Lease = @splat(.{});
    var held: [pages]u32 = @splat(0);
    var random = std.Random.DefaultPrng.init(0x591e_4392);
    const rng = random.random();
    for (0..4096) |_| {
        const l = &leases[rng.uintLessThan(usize, leases.len)];
        const id = rng.uintLessThan(u32, pages);
        switch (rng.uintLessThan(u8, 7)) {
            0 => if (p.available() > 0) {
                try p.promise(l, 1);
                try scan(&p);
                _ = try p.take(l);
            },
            1 => if (p.refs[id] > 0 and held[id] < 4) {
                p.retain(id);
                held[id] += 1;
            },
            2 => if (held[id] > 0) {
                try p.drop(id);
                held[id] -= 1;
            },
            3 => if (l.ids.items.len > 0 and p.refs[id] > 0 and std.mem.indexOfScalar(u32, l.ids.items, id) == null) {
                p.share(l, rng.uintLessThan(usize, l.ids.items.len), id);
            },
            4 => if (p.refs[id] > 0 and std.mem.indexOfScalar(u32, l.ids.items, id) == null) {
                p.retain(id);
                try scan(&p);
                try p.adopt(l, &.{id});
            },
            5 => p.finish(l),
            6 => if (p.available() > 0) {
                const n = @min(p.available(), 3);
                try p.promise(l, n);
                try scan(&p);
                try p.coverRun(l, l.ids.items.len + n, 4, null);
            },
            else => unreachable,
        }
        try scan(&p);
        var mapped: [pages]u32 = @splat(0);
        for (leases) |lease| for (lease.ids.items) |page| {
            mapped[page] += 1;
        };
        for (0..pages) |page| {
            try equal(mapped[page], p.leased[page]);
            try equal(mapped[page] + held[page], p.refs[page]);
        }
    }
    for (&leases) |*lease| {
        p.finish(lease);
        try scan(&p);
    }
    for (&held, 0..) |*n, id| while (n.* > 0) : (n.* -= 1) {
        p.release(@intCast(id));
        try scan(&p);
    };
    try expect(p.idle());
    try equal(@as(usize, pages), p.nfree);
}

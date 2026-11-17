//! Admission by owner class (#2444 C1) over a pool counted in a ledger: tests 1, 2, 4 and 6, and the invariants.
const std = @import("std");
const api = @import("engine_api");
const kv = @import("kv_pool");
const lg = api.kv_ledger;
const classes = lg.classes;
const Ledger = lg.Ledger;
const Need = classes.Need;
const expect = std.testing.expect;
const equal = std.testing.expectEqual;
const t = std.testing.allocator;

/// The lane host's view of a pool (lane_host_ledger.count), its classes too, built by hand for a pool with no host.
fn counted(p: *kv.Pool, l: *Ledger) void {
    const view = struct {
        fn read(ctx: *const anyopaque) lg.Pages {
            const q: *const kv.Pool = @ptrCast(@alignCast(ctx));
            return .{ .total = q.total(), .available = q.available(), .reclaimable = q.reclaimable(), .unit = q.unit(), .classes = q.classes };
        }
        fn class(ctx: *const anyopaque, c: u32) lg.Class {
            const q: *const kv.Pool = @ptrCast(@alignCast(ctx));
            return .{ .cap = q.classCap(), .available = q.availableIn(c) };
        }
        fn detach(ctx: *anyopaque) void {
            const q: *kv.Pool = @ptrCast(@alignCast(ctx));
            q.ledger = null;
        }
    };
    l.* = .ofPages(.{ .ctx = p, .read = view.read, .detach = view.detach, .class = if (p.classes > 1) view.class else null });
    p.ledger = .{ .ctx = l, .want = Ledger.wantOf };
}

/// W=2 on 16 pages (8 a class), class 0 left with `free0` available and class 1 with `free1`, by promises.
fn twoClasses(p: *kv.Pool, l: *Ledger, hold: *[2]kv.Lease, free0: usize, free1: usize) !void {
    p.* = try kv.Pool.initClasses(t, 16, 4096, 2);
    counted(p, l);
    if (free0 < 8) try p.reserve(&hold[0], 8 - free0, 0);
    if (free1 < 8) try p.reserve(&hold[1], 8 - free1, 1);
}

test "kv classes 1: classes of 6 and 2 available, a request of 7 from one class waits, though 8 are free" {
    var p: kv.Pool = undefined;
    var l: Ledger = undefined;
    var hold: [2]kv.Lease = .{ .{}, .{} };
    try twoClasses(&p, &l, &hold, 6, 2);
    defer p.deinit();
    defer for (&hold) |*x| p.finish(x);
    try equal(@as(usize, 8), p.available());
    try expect(p.fits(7)); // the pool's total test admits it: the planted control is exactly this
    try expect(!classes.fitsNeed(&l, .{ .one = .{ .n = 7 } }));
    try expect(classes.fitsNeed(&l, .{ .one = .{ .n = 6 } }) and !classes.fitsNeed(&l, .{ .one = .{ .n = 3, .in = 1 } }));
    var x: kv.Lease = .{};
    try std.testing.expectError(error.LaneMemory, p.reserve(&x, 7, null)); // what admission said, the reservation does
    try equal(@as(u32, 0), x.owner);
    try p.reserve(&x, 6, null);
    try equal(@as(u32, 0), x.owner); // the roomiest, chosen once inside the reservation
    p.finish(&x);
}

test "kv classes 2: a spread over W=2 needing 3 and 2 waits while class 1 has 1 available" {
    var p: kv.Pool = undefined;
    var l: Ledger = undefined;
    var hold: [2]kv.Lease = .{ .{}, .{} };
    try twoClasses(&p, &l, &hold, 8, 1);
    defer p.deinit();
    defer for (&hold) |*x| p.finish(x);
    const s: Need.Spread = .{ .k0 = 0, .k1 = 5 }; // pages 0, 2, 4 from class 0; 1, 3 from class 1
    try equal(@as(usize, 3), Need.inClass(s, 2, 0));
    try equal(@as(usize, 2), Need.inClass(s, 2, 1));
    try expect(p.fits(5)); // 9 available in all: a total-only test admits it
    try expect(!classes.fitsNeed(&l, .{ .spread = s }));
    try expect(classes.fitsNeed(&l, .{ .spread = .{ .k0 = 2, .k1 = 5 } })); // 2 and 4 from class 0, 3 from class 1
    try equal(@as(?u32, null), classes.neverNeed(&l, .{ .spread = .{ .k0 = 0, .k1 = 16 } }));
    try equal(@as(?u32, 1), classes.neverNeed(&l, .{ .spread = .{ .k0 = 1, .k1 = 18 } })); // 9 odd pages: class 1's 8
}

test "kv classes 4: after takes, trims and finishes in both classes, each class's census is idle again" {
    var p = try kv.Pool.initClasses(t, 12, 4096, 2);
    defer p.deinit();
    var l: Ledger = undefined;
    counted(&p, &l);
    var a: kv.Lease = .{};
    var b: kv.Lease = .{};
    try p.reserve(&a, 5, null); // class 0 (the roomiest, lowest on ties)
    try p.reserve(&b, 4, null); // class 1, now the roomiest
    try equal(@as(u32, 1), b.owner);
    try p.cover(&a, 5, null);
    try p.cover(&b, 3, null);
    p.trim(&a, 2); // three pages back to a's promise, in class 0
    try equal(@as(usize, 3), p.classPromised(0));
    try equal(@as(usize, 1), p.classPromised(1));
    try p.cover(&a, 4, null);
    p.trim(&b, 0);
    p.finish(&a);
    p.finish(&b);
    try expect(p.idle() and l.idle());
    for (0..2) |c| {
        try equal(p.classCap(), p.classFree(@intCast(c)));
        try equal(@as(usize, 0), p.classPromised(@intCast(c)));
    }
}

/// The two invariants: class promises sum to the pool's, and no class promises more than it holds free.
fn invariants(p: *const kv.Pool) !void {
    var promised: usize = 0;
    var free: usize = 0;
    for (0..p.classes) |c| {
        try expect(p.classPromised(@intCast(c)) <= p.classFree(@intCast(c)));
        promised += p.classPromised(@intCast(c));
        free += p.classFree(@intCast(c));
    }
    try equal(p.promised, promised);
    try equal(p.nfree, free);
}

/// What a failed reservation, a cancel or a requeue must leave as it was: every count and the lease's owner.
const Snap = struct { cfree: [kv.max_classes]usize, cpromised: [kv.max_classes]usize, promised: usize, nfree: usize };

fn snap(p: *const kv.Pool) Snap {
    return .{ .cfree = p.cfree, .cpromised = p.cpromised, .promised = p.promised, .nfree = p.nfree };
}

test "kv classes: interleaved reservations in W=2 and W=3 hold both invariants after every step" {
    for ([_]u32{ 2, 3 }) |w| for ([_]u64{ 1, 2, 3 }) |seed| {
        var p = try kv.Pool.initClasses(t, 8 * w, 4096, w);
        defer p.deinit();
        var l: Ledger = undefined;
        counted(&p, &l);
        var ls: [4]kv.Lease = @splat(.{});
        defer for (&ls) |*x| p.finish(x);
        var held: std.ArrayList(u32) = .empty;
        defer held.deinit(t);
        defer for (held.items) |id| p.release(id);
        var bytes: lg.Lease = .{}; // an unpaged lease beside the pool: the bytes test inside each class's
        defer l.finish(&bytes);
        var prng = std.Random.DefaultPrng.init(0xC1_0000 + seed);
        const r = prng.random();
        var failed: usize = 0;
        for (0..600) |_| {
            const x = &ls[r.uintLessThan(usize, ls.len)];
            switch (r.uintLessThan(u8, 7)) {
                0, 1 => {
                    const n = 1 + r.uintLessThan(usize, 6);
                    const fresh = x.promised == 0 and x.ids.items.len == 0;
                    const in: ?u32 = if (!fresh or r.boolean()) null else r.uintLessThan(u32, w);
                    const said = classes.fitsNeed(&l, .{ .one = .{ .n = n, .in = in } });
                    const before = snap(&p);
                    const owner = x.owner;
                    if (p.reserve(x, n, in)) |_| {
                        if (fresh) try expect(said); // admission's answer is the reservation's, for a new lease
                    } else |_| {
                        failed += 1;
                        if (fresh) try expect(!said);
                        try equal(before, snap(&p)); // a failed reservation changes no count
                        try equal(owner, x.owner); // and leaves the owner where it was
                    }
                },
                2 => if (x.promised > 0) try p.cover(x, x.ids.items.len + 1 + r.uintLessThan(usize, x.promised), null),
                3 => p.trim(x, r.uintLessThan(usize, x.ids.items.len + 1)),
                4 => if (r.boolean()) p.finish(x) else p.unpromise(x),
                5 => {
                    var two: [2]u32 = undefined;
                    if (p.hold(&two)) try held.appendSlice(t, &two) else if (held.pop()) |id| p.release(id);
                },
                else => if (bytes.bytes == 0) l.promise(&bytes, 4096 * r.uintLessThan(u64, 4)) catch {} else l.finish(&bytes),
            }
            try invariants(&p);
            for (&ls) |*y| for (y.ids.items) |id| try equal(y.owner, id % w); // a lease's ids, all of its class
        }
        try expect(failed > 10);
    };
}

test "kv classes: a failed reservation, a cancel and a requeue leave every count and the owner as they were" {
    var p: kv.Pool = undefined;
    var l: Ledger = undefined;
    var hold: [2]kv.Lease = .{ .{}, .{} };
    try twoClasses(&p, &l, &hold, 3, 5);
    defer p.deinit();
    defer for (&hold) |*x| p.finish(x);
    const s0 = snap(&p);
    var x: kv.Lease = .{};
    try std.testing.expectError(error.LaneMemory, p.reserve(&x, 6, 1)); // class 1 lacks them (a resumed prefix's class)
    try equal(s0, snap(&p));
    try equal(@as(u32, 0), x.owner);
    try p.reserve(&x, 4, null); // the roomiest: class 1
    try equal(@as(u32, 1), x.owner);
    try std.testing.expectError(error.LaneMemory, p.reserve(&x, 2, null)); // a second promise stays in its class
    try std.testing.expectError(error.BadClass, p.reserve(&x, 1, 0)); // and never moves to another
    try equal(@as(u32, 1), x.owner);
    try p.cover(&x, 3, null);
    p.trim(&x, 1);
    p.finish(&x); // cancelled mid-stream
    try equal(s0, snap(&p));
    try equal(@as(u32, 0), x.owner);
    try p.reserve(&x, 5, null);
    p.finish(&x); // requeued: promised, nothing taken
    try equal(s0, snap(&p));
}

/// A store that keeps pages and gives up its oldest when a promise asks (kv_pool_ledger_test's).
const Store = struct {
    pool: *kv.Pool,
    held: std.ArrayList(u32) = .empty,
    fn call(ctx: *anyopaque, want: usize) bool {
        const s: *Store = @ptrCast(@alignCast(ctx));
        while (s.pool.available() < want and s.held.items.len > 0) s.pool.release(s.held.orderedRemove(0));
        return s.pool.available() >= want;
    }
};

/// One class, counted in a ledger with bytes beside it: 3000 seeded steps traced, and C1's tests asked at each.
fn walk(out: *std.ArrayList(u64)) !void {
    var p = try kv.Pool.init(t, 24, 100);
    defer p.deinit();
    var l: Ledger = undefined;
    counted(&p, &l);
    var s: Store = .{ .pool = &p };
    defer s.held.deinit(t);
    try p.joinReclaim(.{ .ctx = &s, .call = Store.call });
    var ls: [4]kv.Lease = @splat(.{});
    var bytes: lg.Lease = .{};
    var prng = std.Random.DefaultPrng.init(0xC1);
    const r = prng.random();
    for (0..3000) |_| {
        const x = &ls[r.uintLessThan(usize, ls.len)];
        const n = r.uintLessThan(usize, 12);
        var got: u64 = 0;
        switch (r.uintLessThan(u8, 7)) {
            0 => got = @intFromBool(if (p.reserve(x, n, null)) true else |_| false),
            1 => if (x.promised > 0) {
                try p.cover(x, x.ids.items.len + 1 + r.uintLessThan(usize, x.promised), null);
                p.retain(x.ids.items[0]);
                try s.held.append(t, x.ids.items[0]);
                got = x.ids.getLast();
            },
            2 => p.trim(x, r.uintLessThan(usize, x.ids.items.len + 1)),
            3 => p.finish(x),
            4 => {
                var ids: [3]u32 = undefined;
                const k = @min(n, ids.len);
                if (p.hold(ids[0..k])) try s.held.appendSlice(t, ids[0..k]);
            },
            5 => if (s.held.items.len > 0) p.release(s.held.orderedRemove(r.uintLessThan(usize, s.held.items.len))),
            else => if (bytes.bytes == 0) l.promise(&bytes, 100 * n) catch {} else l.finish(&bytes),
        }
        const c = p.census();
        try out.appendSlice(t, &.{ got, c.free, c.promised, c.kept, c.leased, p.available(), p.reclaimable(), l.free() });
        for (0..14) |k| {
            try out.appendSlice(t, &.{ @intFromBool(p.fits(k)), l.want(k), @intFromBool(l.fits(100 * k)) });
            try equal(p.fits(k), classes.fitsNeed(&l, .{ .one = .{ .n = k } })); // one class: Pool.fits exactly
            try equal(l.want(k), classes.wantIn(&l, 0, k));
            try equal(@as(?u32, null), classes.neverNeed(&l, .{ .one = .{ .n = k + 30 } })); // never past the total
        }
    }
    for (&ls) |*x| p.finish(x);
    l.finish(&bytes);
    for (s.held.items) |id| p.release(id);
    try expect(p.idle() and l.idle());
    try out.appendSlice(t, &.{p.nfree});
}

test "kv classes 6: one class counted in a ledger hands out #2415's pages, counts and answers, step for step" {
    var out: std.ArrayList(u64) = .empty;
    defer out.deinit(t);
    try walk(&out);
    try equal(@as(usize, 150001), out.items.len);
    try equal(@as(u64, 0xf005c622d55ebd68), std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(out.items))); // af17f6c8fc
}

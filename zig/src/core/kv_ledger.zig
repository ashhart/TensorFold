//! A byte budget that admitted requests promise from: leases' bytes, and a paged pool's pages as a sub-count.
const std = @import("std");

/// The bytes one holder (an admitted job) has promised past its standing stream, if it is `bound` to one.
pub const Lease = struct { bytes: u64 = 0, bound: bool = false };

/// Streams paid for at load (the engine's own sequence, `free_streams`), each at `each` bytes: a job takes one free.
pub const Standing = struct { each: u64 = 0, streams: u32 = 0 };

/// A paged pool's counts in pages, and the bytes a page counts for (kv_pool.Pool's, read through `Counted`).
pub const Pages = struct { total: usize, available: usize, reclaimable: usize, unit: u64 };

/// A pool the ledger counts: its counts, and its way back to its own pages (kv_pool stays a module of its own).
pub const Counted = struct {
    ctx: *anyopaque,
    read: *const fn (ctx: *const anyopaque) Pages,
    detach: *const fn (ctx: *anyopaque) void,
};

/// What streams' caches may take at once; the caller serializes every call (the lane host's lock, then the pool's).
pub const Ledger = struct {
    total: u64,
    promised: u64 = 0, // the leases' bytes, the pool's pages apart
    standing: Standing = .{}, // in `total`, never promised: a job bound to one promises only what it adds past it
    bound: u32 = 0, // the jobs on standing streams now
    pool: ?Counted = null, // a paged family's pool: its pages promised or held count here, its own test defers here

    /// A ledger whose whole budget is `c`'s pages: a paged family's alone admits exactly what its pool did.
    pub fn ofPages(c: Counted) Ledger {
        const n = c.read(c.ctx);
        return .{ .total = n.total * n.unit, .pool = c };
    }

    fn pages(l: *const Ledger) Pages {
        const c = l.pool orelse return .{ .total = 0, .available = 0, .reclaimable = 0, .unit = 1 };
        return c.read(c.ctx);
    }

    /// Bytes free now, nothing reclaimed: the total past the standing streams, the leases and the pool's pages.
    pub fn free(l: *const Ledger) u64 {
        const n = l.pages();
        return l.total -| l.standing.each *| l.standing.streams -| l.promised -| (n.total - n.available) * n.unit;
    }

    /// What a job costing `bytes` promises now: past a standing stream's bytes while one is free.
    pub fn price(l: *const Ledger, bytes: u64) u64 {
        return if (l.bound < l.standing.streams) bytes -| l.standing.each else bytes;
    }

    /// Whether a job costing `bytes` fits beside what is promised now, the store's reclaimable pages counted as room.
    pub fn fits(l: *const Ledger, bytes: u64) bool {
        const n = l.pages();
        return l.price(bytes) <= l.free() + n.reclaimable * n.unit;
    }

    /// Whether `bytes` more fit now with nothing reclaimed.
    pub fn holds(l: *const Ledger, bytes: u64) bool {
        return bytes <= l.free();
    }

    /// The pool's available pages at which `n` more pages fit: what its promise tests and asks the store to reclaim.
    pub fn want(l: *const Ledger, n: usize) usize {
        const c = l.pages();
        const short = (n *| c.unit) -| l.free();
        return c.available +| @as(usize, @intCast((short + c.unit - 1) / c.unit));
    }

    /// `want` for the pool's gate (kv_pool.Gate), `ctx` the ledger.
    pub fn wantOf(ctx: *const anyopaque, n: usize) usize {
        const l: *const Ledger = @ptrCast(@alignCast(ctx));
        return l.want(n);
    }

    /// The pool counts on its own pages again and leaves this ledger.
    pub fn detach(l: *Ledger) void {
        const c = l.pool orelse return;
        c.detach(c.ctx);
        l.pool = null;
    }

    /// Promise a job costing `bytes` to an empty `lease`, on a free standing stream first; LaneMemory: price unfit.
    pub fn promise(l: *Ledger, lease: *Lease, bytes: u64) error{LaneMemory}!void {
        std.debug.assert(lease.bytes == 0 and !lease.bound);
        const p = l.price(bytes);
        if (!l.holds(p)) return error.LaneMemory;
        lease.bound = l.bound < l.standing.streams;
        l.bound += @intFromBool(lease.bound);
        l.promised += p;
        lease.bytes = p;
    }

    /// `lease`'s bytes and standing stream go back and it is empty again; an empty lease returns nothing.
    pub fn finish(l: *Ledger, lease: *Lease) void {
        l.promised -= lease.bytes;
        l.bound -= @intFromBool(lease.bound);
        lease.* = .{};
    }

    /// No lease holds bytes or a standing stream; the standing streams themselves stay (the pool's census is its own).
    pub fn idle(l: *const Ledger) bool {
        return l.promised == 0 and l.bound == 0;
    }
};

test "kv ledger: a promise fits up to the total and not a byte past it" {
    var l: Ledger = .{ .total = 10 };
    var a: Lease = .{};
    var b: Lease = .{};
    try l.promise(&a, 6);
    try std.testing.expect(l.fits(4));
    try std.testing.expect(!l.fits(5));
    try std.testing.expectError(error.LaneMemory, l.promise(&b, 5));
    try std.testing.expectEqual(@as(u64, 6), l.promised);
    try std.testing.expectEqual(@as(u64, 0), b.bytes);
    try l.promise(&b, 4);
    try std.testing.expect(!l.fits(1));
    try std.testing.expect(l.fits(0));
}

test "kv ledger: finish returns a lease once and leaves the ledger idle" {
    var l: Ledger = .{ .total = 8 };
    var a: Lease = .{};
    try l.promise(&a, 8);
    try std.testing.expect(!l.idle());
    l.finish(&a);
    try std.testing.expect(l.idle());
    l.finish(&a); // an empty lease returns nothing
    try std.testing.expect(l.idle());
    try l.promise(&a, 3); // an emptied lease takes a new promise
    try std.testing.expectEqual(@as(u64, 3), l.promised);
}

test "kv ledger: a zero total admits only empty promises" {
    var l: Ledger = .{ .total = 0 };
    var a: Lease = .{};
    try std.testing.expect(l.fits(0));
    try std.testing.expectError(error.LaneMemory, l.promise(&a, 1));
    try std.testing.expect(l.idle());
}

test "kv ledger: a standing stream takes the first job, which promises only what it adds past it" {
    var l: Ledger = .{ .total = 30, .standing = .{ .each = 10, .streams = 1 } };
    var a: Lease = .{};
    var b: Lease = .{};
    var c: Lease = .{};
    try std.testing.expectEqual(@as(u64, 20), l.free()); // the standing stream is never promised to anyone
    try std.testing.expect(l.fits(30) and !l.fits(31)); // a free standing stream pays its first 10
    try l.promise(&a, 12);
    try std.testing.expect(a.bound and a.bytes == 2 and l.bound == 1);
    try std.testing.expect(l.fits(18) and !l.fits(19)); // no standing stream left: full price
    try l.promise(&b, 18);
    try std.testing.expect(!b.bound);
    try std.testing.expectError(error.LaneMemory, l.promise(&c, 1));
    l.finish(&a); // its standing stream is free again: the next job is bound
    try std.testing.expect(!l.idle());
    try l.promise(&c, 10);
    try std.testing.expect(c.bound and c.bytes == 0);
    l.finish(&b);
    try std.testing.expect(!l.idle()); // a bound job of no bytes still holds its stream
    l.finish(&c);
    try std.testing.expect(l.idle()); // the standing stream itself is not left at stop
    try std.testing.expectEqual(@as(u64, 20), l.free());
}

test "kv ledger: no standing stream prices every job in full, as before" {
    var l: Ledger = .{ .total = 10 };
    var a: Lease = .{};
    try std.testing.expectEqual(@as(u64, 7), l.price(7));
    try l.promise(&a, 7);
    try std.testing.expect(!a.bound and a.bytes == 7);
    l.finish(&a);
    try std.testing.expect(l.idle());
}

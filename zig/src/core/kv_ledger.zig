//! A byte budget that admitted requests promise from: leases' bytes, and a paged pool's pages as a sub-count.
const std = @import("std");

/// The bytes one holder (an admitted job) has promised; empty when it holds none.
pub const Lease = struct { bytes: u64 = 0 };

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

    /// Bytes free now, nothing reclaimed: the total past the leases and the pool's pages promised or held.
    pub fn free(l: *const Ledger) u64 {
        const n = l.pages();
        return l.total -| l.promised -| (n.total - n.available) * n.unit;
    }

    /// Whether `bytes` more fit beside what is promised now, the store's reclaimable pages counted as room.
    pub fn fits(l: *const Ledger, bytes: u64) bool {
        const n = l.pages();
        return bytes <= l.free() + n.reclaimable * n.unit;
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

    /// Promise `bytes` to an empty `lease`; LaneMemory, with nothing changed, when they do not fit now.
    pub fn promise(l: *Ledger, lease: *Lease, bytes: u64) error{LaneMemory}!void {
        std.debug.assert(lease.bytes == 0);
        if (!l.holds(bytes)) return error.LaneMemory;
        l.promised += bytes;
        lease.bytes = bytes;
    }

    /// `lease`'s bytes go back and it is empty again; an empty lease returns nothing.
    pub fn finish(l: *Ledger, lease: *Lease) void {
        l.promised -= lease.bytes;
        lease.bytes = 0;
    }

    /// No lease holds bytes (the pool's census reports its own pages at stop).
    pub fn idle(l: *const Ledger) bool {
        return l.promised == 0;
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

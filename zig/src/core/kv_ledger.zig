//! A byte budget that admitted requests promise from: a total, the promised bytes and one lease a holder.
const std = @import("std");

/// The bytes one holder (an admitted job) has promised; empty when it holds none.
pub const Lease = struct { bytes: u64 = 0 };

/// What unpaged streams' caches may take at once; the caller serializes every call (the lane host's lock).
pub const Ledger = struct {
    total: u64,
    promised: u64 = 0,

    /// Whether `bytes` more fit beside what is promised now.
    pub fn fits(l: *const Ledger, bytes: u64) bool {
        return bytes <= l.total -| l.promised;
    }

    /// Promise `bytes` to an empty `lease`; LaneMemory, with nothing changed, when they do not fit now.
    pub fn promise(l: *Ledger, lease: *Lease, bytes: u64) error{LaneMemory}!void {
        std.debug.assert(lease.bytes == 0);
        if (!l.fits(bytes)) return error.LaneMemory;
        l.promised += bytes;
        lease.bytes = bytes;
    }

    /// `lease`'s bytes go back and it is empty again; an empty lease returns nothing.
    pub fn finish(l: *Ledger, lease: *Lease) void {
        l.promised -= lease.bytes;
        lease.bytes = 0;
    }

    /// Nothing promised.
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

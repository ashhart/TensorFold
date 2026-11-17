//! Resource classes in admission (#2444 C1): the pages a request needs a class, tested by the ledger a class.
const kv = @import("kv_ledger.zig");
const Ledger = kv.Ledger;

/// Pages a request needs, by owner class (kv_pool_classes.zig: class c holds ids c, c + W, c + 2W ...).
pub const Need = union(enum) {
    /// All `n` from one class: `in` when a resumed prefix fixed it, else the one the reservation chooses.
    one: One,
    /// Logical pages [k0, k1), page k from class k mod W (DeepSeek's span); admission only, no lease takes it yet.
    spread: Spread,

    pub const One = struct { n: usize, in: ?u32 = null };
    pub const Spread = struct { k0: usize, k1: usize };

    pub fn total(d: Need) usize {
        return switch (d) {
            .one => |o| o.n,
            .spread => |s| s.k1 - s.k0,
        };
    }

    /// Of a spread over `w` classes, the pages class `c` gives.
    pub fn inClass(s: Spread, w: u32, c: u32) usize {
        return below(s.k1, w, c) - below(s.k0, w, c);
    }

    /// Logical pages k < x with k mod w == c.
    fn below(x: usize, w: u32, c: u32) usize {
        return if (x > c) (x - c + w - 1) / w else 0;
    }
};

/// Class `c` as the pool reads it; one class is the pool itself.
fn class(l: *const Ledger, p: kv.Pages, c: u32) kv.Class {
    const k = l.pool.?;
    return if (k.class) |f| f(k.ctx, c) else .{ .cap = p.total, .available = p.available };
}

/// Class `c`'s available pages at which `n` more fit there and in the pool's bytes (one class: exactly `want`).
pub fn wantIn(l: *const Ledger, c: u32, n: usize) usize {
    const p = l.pages();
    if (p.classes <= 1) return l.want(n);
    return @max(n, class(l, p, c).available +| (l.want(n) -| p.available)); // short of the bytes: past what it has
}

/// Whether `d` could be promised now: the pool's bytes, then each class it takes from.
pub fn fitsNeed(l: *const Ledger, d: Need) bool {
    const p = l.pages();
    if (p.classes <= 1) return p.available + p.reclaimable >= l.want(d.total()); // one class: Pool.fits exactly
    if (p.available < l.want(d.total())) return false; // no store reclaims above one class (Pool.joinReclaim)
    switch (d) {
        .one => |o| {
            if (o.in) |c| return class(l, p, c).available >= o.n;
            for (0..p.classes) |c| if (class(l, p, @intCast(c)).available >= o.n) return true; // the roomiest's test
            return false;
        },
        .spread => |s| {
            for (0..p.classes) |c| if (class(l, p, @intCast(c)).available < Need.inClass(s, p.classes, @intCast(c))) return false;
            return true;
        },
    }
}

/// The class `d` needs more of than it holds at all, so no release makes it fit (null: none; one class: never).
pub fn neverNeed(l: *const Ledger, d: Need) ?u32 {
    const p = l.pages();
    if (p.classes <= 1) return null;
    switch (d) {
        .one => |o| return if (o.n > class(l, p, o.in orelse 0).cap) o.in orelse 0 else null, // classes are equal
        .spread => |s| {
            for (0..p.classes) |c| if (Need.inClass(s, p.classes, @intCast(c)) > class(l, p, @intCast(c)).cap) return @intCast(c);
            return null;
        },
    }
}

//! A pool's owner classes, DeepSeek's residue book: ids c, c + W, c + 2W ... are class c's, each on its own free stack.
const std = @import("std");
const kv = @import("kv_pool.zig");
const Pool = kv.Pool;

test {
    _ = @import("kv_pool_class_test.zig");
    _ = @import("kv_pool_trace_test.zig");
}

/// Owner classes a pool holds at most: GLM's ranks (cuda_ranks.max_world).
pub const max = 16;

/// `plan` over `l.classes` owner classes: the pages floored to a multiple of them, and each class holds a window.
pub fn plan(l: kv.Layout, b: kv.Budget) error{ NoStream, TooMany, ClassTooSmall }!kv.Plan {
    var p = try kv.plan(l, b);
    if (l.classes <= 1) return p;
    p.pages -= p.pages % l.classes;
    if (p.pages / l.classes < l.pages(b.window)) return error.ClassTooSmall;
    return p;
}

/// Ids a class holds.
pub fn classCap(p: *const Pool) usize {
    return p.cap;
}

pub fn classFree(p: *const Pool, c: u32) usize {
    return p.cfree[c];
}

pub fn classPromised(p: *const Pool, c: u32) usize {
    return p.cpromised[c];
}

/// Class `c`'s pages free and not promised.
pub fn availableIn(p: *const Pool, c: u32) usize {
    return p.cfree[c] - p.cpromised[c];
}

/// The owner rule (#1313's): the class with the most available pages, the lowest on ties.
pub fn roomiest(p: *const Pool) u32 {
    var best: u32 = 0;
    for (1..p.classes) |c| if (availableIn(p, @intCast(c)) > availableIn(p, best)) {
        best = @intCast(c);
    };
    return best;
}

/// `fits` for one lease: above one class, the roomiest class must hold all `need` (one class: exactly `fits`).
pub fn fitsAny(p: *const Pool, need: usize) bool {
    return p.fits(need) and (p.classes == 1 or availableIn(p, roomiest(p)) >= need);
}

/// A free id back on its class's stack.
pub fn push(p: *Pool, id: u32) void {
    const c = id % p.classes;
    p.free[c * p.cap + p.cfree[c]] = id;
    p.cfree[c] += 1;
    p.nfree += 1;
}

/// The top of class `c`'s free stack.
pub fn pop(p: *Pool, c: u32) u32 {
    p.cfree[c] -= 1;
    p.nfree -= 1;
    return p.free[c * p.cap + p.cfree[c]];
}

/// A page given to `l` from outside its promise (a store's): of its owner's class, or the book is broken.
pub fn owned(p: *const Pool, l: *const kv.Lease, id: u32) void {
    if (id % p.classes != l.owner) std.debug.panic("kv pool: page {d} of class {d} given to a lease of class {d}", .{ id, id % p.classes, l.owner });
}

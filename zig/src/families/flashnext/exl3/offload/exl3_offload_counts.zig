//! Flash Next EXL3 offload: lifetime routing counts and the expert-id span a rank owns.

const std = @import("std");
const pol = @import("exl3_offload_policy.zig");

const shape = pol.shape;
const appearances = pol.appearances;

pub fn countOf(counts: []const u32, id: u16) u32 {
    if (id >= counts.len) return 0;
    return counts[id];
}

pub fn countAt(counts: *const [shape.layers][shape.experts]u32, layer: u8, id: u16) u32 {
    if (layer >= shape.layers or id >= shape.experts) return 0;
    return counts[layer][id];
}

pub const UsedOrder = struct {
    counts: []const u32,
    ids: []const u16,
};

/// Lifetime count descending, then this round's appearances, then lower id.
pub fn usedFirst(ctx: UsedOrder, a: u16, b: u16) bool {
    const ca = countOf(ctx.counts, a);
    const cb = countOf(ctx.counts, b);
    if (ca != cb) return ca > cb;
    const aa = appearances(ctx.ids, a);
    const ab = appearances(ctx.ids, b);
    if (aa != ab) return aa > ab;
    return a < b;
}

/// Add one to each routed id in `ids`.
pub fn observe(counts: []u32, ids: []const u16) error{Length}!void {
    if (counts.len != shape.experts) return error.Length;
    for (ids) |id| {
        if (id >= shape.experts) continue;
        counts[id] +|= 1;
    }
}

/// Expert ids this rank stores. Rank 0 is `0..256`, rank 1 is `256..512`.
pub const Owned = struct { lo: u16, hi: u16 };

pub fn allOwned() Owned {
    return .{ .lo = 0, .hi = @intCast(shape.experts) };
}

pub fn inOwned(owned: Owned, id: u16) bool {
    return id >= owned.lo and id < owned.hi;
}

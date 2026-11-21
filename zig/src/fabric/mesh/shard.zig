//! A tensor's share of a group, declared once: what is cut, in which units, by which rule (cluster/canon.zig's).
const std = @import("std");
const canon = @import("canon");
const Group = @import("mesh.zig").Group;

/// What is cut: a column split's output rows, a row split's input columns, a whole copy, or the head's vocabulary.
pub const Kind = enum { column, row, replicated, vocab };

/// canon: canon.partition at equal weights (largest remainder, ties to the lower rank); even: refuse uneven.
pub const Rule = enum { canon, even };

/// When an empty share refuses: only on the rank left empty, or on every rank once units are below world.
pub const Empty = enum { rank, any };

pub const Decl = struct {
    kind: Kind,
    /// Elements per indivisible unit (1 head, 64 columns, 128 EXL3 tiles, one head's width).
    unit: u64,
    rule: Rule = .canon,
    empty: Empty = .rank,
    /// Widths of the segments of a packed output (q | k | v), each cut by the same rule; empty: one segment.
    segments: []const u64 = &.{},
    /// Fewer units than ranks: each unit kept whole by world / units consecutive ranks (KV heads).
    replicate: bool = false,
};

pub const Range = struct {
    lo: u64,
    hi: u64,

    pub fn len(r: Range) u64 {
        return r.hi - r.lo;
    }
};

pub const Error = error{ BadGroup, NotOnUnits, EmptyShare, Uneven, ReplicaMismatch };

pub const max_world = 256;

/// The rank's range of `extent` elements, or the first of BadGroup, NotOnUnits, ReplicaMismatch, Uneven, EmptyShare.
pub fn range(d: Decl, extent: u64, g: Group) Error!Range {
    if (g.world == 0 or g.world > max_world or g.rank >= g.world) return error.BadGroup;
    if (d.unit == 0 or extent % d.unit != 0 or extent / d.unit > std.math.maxInt(u32)) return error.NotOnUnits;
    if (d.kind == .replicated) return .{ .lo = 0, .hi = extent };
    const units: u32 = @intCast(extent / d.unit);
    if (d.replicate and units < g.world) {
        if (units == 0 or g.world % units != 0) return error.ReplicaMismatch;
        const head = g.rank / (g.world / units);
        return .{ .lo = head * d.unit, .hi = (head + 1) * d.unit };
    }
    if (d.rule == .even and units % g.world != 0) return error.Uneven;
    if (d.empty == .any and units < g.world) return error.EmptyShare;
    const ones: [max_world]u64 = @splat(1);
    var out: [max_world]canon.Range = undefined;
    canon.partition(units, ones[0..g.world], out[0..g.world]);
    const r = out[g.rank];
    if (r.len() == 0) return error.EmptyShare;
    return .{ .lo = @as(u64, r.begin) * d.unit, .hi = @as(u64, r.end) * d.unit };
}

/// One range per segment of a packed output, each offset by the segments before it (`extent` is their sum).
pub fn ranges(d: Decl, extent: u64, g: Group, out: []Range) Error![]const Range {
    if (d.segments.len == 0) {
        out[0] = try range(d, extent, g);
        return out[0..1];
    }
    var base: u64 = 0;
    for (d.segments, out[0..d.segments.len]) |w, *r| {
        const one = try range(d, w, g);
        const end = std.math.add(u64, base, w) catch return error.NotOnUnits;
        if (end > extent) return error.NotOnUnits;
        r.* = .{ .lo = base + one.lo, .hi = base + one.hi };
        base = end;
    }
    if (base != extent) return error.NotOnUnits;
    return out[0..d.segments.len];
}

/// A range derived from `r`, num / den units for each of its units (MiniMax's query heads from its key heads).
pub fn derived(r: Range, num: u64, den: u64) Range {
    const g = num / den;
    return .{ .lo = r.lo * g, .hi = r.hi * g };
}

const testing = std.testing;
const tensorGroup = @import("mesh.zig").tensorGroup;

test "canon cuts whole units, the first ranks one more, and refuses only the empty rank" {
    const d: Decl = .{ .kind = .column, .unit = 64 };
    try testing.expectEqual(Range{ .lo = 0, .hi = 192 }, try range(d, 320, tensorGroup(0, 2)));
    try testing.expectEqual(Range{ .lo = 192, .hi = 320 }, try range(d, 320, tensorGroup(1, 2)));
    // 4 units over 5 ranks: ranks 0-3 one unit each, rank 4 none
    try testing.expectEqual(Range{ .lo = 0, .hi = 64 }, try range(d, 256, tensorGroup(0, 5)));
    try testing.expectError(error.EmptyShare, range(d, 256, tensorGroup(4, 5)));
    var any = d;
    any.empty = .any;
    try testing.expectError(error.EmptyShare, range(any, 256, tensorGroup(0, 5)));
}

test "even refuses uneven splits, and replication keeps a unit on world / units ranks" {
    const even: Decl = .{ .kind = .column, .unit = 1, .rule = .even };
    try testing.expectError(error.Uneven, range(even, 3, tensorGroup(0, 4)));
    try testing.expectEqual(Range{ .lo = 2, .hi = 4 }, try range(even, 8, tensorGroup(1, 4)));
    const kv: Decl = .{ .kind = .column, .unit = 128, .rule = .even, .replicate = true };
    try testing.expectError(error.Uneven, range(kv, 6 * 128, tensorGroup(0, 4)));
    try testing.expectEqual(Range{ .lo = 128, .hi = 256 }, try range(kv, 2 * 128, tensorGroup(2, 4)));
    try testing.expectError(error.ReplicaMismatch, range(kv, 3 * 128, tensorGroup(0, 4)));
}

test "a replicated tensor is whole on every rank" {
    const d: Decl = .{ .kind = .replicated, .unit = 1 };
    try testing.expectEqual(Range{ .lo = 0, .hi = 8 }, try range(d, 8, tensorGroup(0, 2)));
    try testing.expectEqual(Range{ .lo = 0, .hi = 8 }, try range(d, 8, tensorGroup(1, 2)));
}

test "check order: the group first, then units" {
    const d: Decl = .{ .kind = .column, .unit = 64, .rule = .even };
    try testing.expectError(error.BadGroup, range(d, 100, tensorGroup(0, 0)));
    try testing.expectError(error.BadGroup, range(d, 100, tensorGroup(2, 2)));
    try testing.expectError(error.BadGroup, range(d, 64 * 257, tensorGroup(0, 257)));
    try testing.expectError(error.NotOnUnits, range(d, 100, tensorGroup(0, 3)));
}

test "segments cut each part and keep their offsets" {
    const d: Decl = .{ .kind = .column, .unit = 1, .rule = .even, .segments = &.{ 8, 4, 4 } };
    var out: [3]Range = undefined;
    const got = try ranges(d, 16, tensorGroup(1, 2), &out);
    try testing.expectEqualSlices(Range, &.{ .{ .lo = 4, .hi = 8 }, .{ .lo = 10, .hi = 12 }, .{ .lo = 14, .hi = 16 } }, got);
    try testing.expectError(error.NotOnUnits, ranges(d, 15, tensorGroup(1, 2), &out));
    const big: Decl = .{ .kind = .column, .unit = 1 << 31, .segments = &.{ 1 << 63, 1 << 63 } };
    try testing.expectError(error.NotOnUnits, ranges(big, 0, tensorGroup(0, 1), &out));
}

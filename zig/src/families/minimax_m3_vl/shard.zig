//! The family's TP2 rank placement, TF's documented rank pattern (docs/multi-rank-receipts.md: the
//! GLM/Nemotron precedent — this family's TF_MINIMAXM3_RANK/TF_MINIMAXM3_WORLD, NCCL_COMM_ID on rank
//! 0's address) laid over the load plan: each rank runs one contiguous range of the checkpoint's
//! layers, rank 0 additionally runs the past-layer tensors (the embed, the final norm, the head) and
//! reads the logits. This stage receipts the per-rank bytes (the real pack's upload receipts are the
//! minimax-m3-tp2 command); the serving attach — each rank's launch graph over its range, NCCL
//! between the waves — rides this same split.

const std = @import("std");
const weights = @import("weights.zig");

/// This stage's placement: two GB10 nodes (the single-GB10 pack cannot fit one node). A wider world
/// needs its own receipts before `max_world` moves.
pub const max_world: u32 = 2;

/// This rank's place: TF_MINIMAXM3_RANK of TF_MINIMAXM3_WORLD — both, or neither (never one alone).
pub const Place = struct { rank: u32, world: u32 };

pub fn place() !Place {
    const r = std.c.getenv("TF_MINIMAXM3_RANK") orelse return error.NoRank;
    const w = std.c.getenv("TF_MINIMAXM3_WORLD") orelse return error.NoRank;
    const rank = std.fmt.parseInt(u32, std.mem.span(r), 10) catch return error.NoRank;
    const world = std.fmt.parseInt(u32, std.mem.span(w), 10) catch return error.NoRank;
    return checked(.{ .rank = rank, .world = world });
}

/// A place the family serves at this stage: rank inside a world of 1 or `max_world`.
pub fn checked(p: Place) !Place {
    if (p.world < 1 or p.world > max_world or p.rank >= p.world) return error.NoRank;
    return p;
}

/// The layers rank `rank` of `world` runs: the head rank's first, the worker's after — the fork's
/// M3.1 precedent (head layers, then worker layers) laid on the plan: at 60 layers, rank 0 runs
/// layers 0..29, rank 1 runs 30..59.
pub const Range = struct { first: u32, count: u32 };

pub fn range(layers: u32, world: u32, rank: u32) Range {
    const base = layers / world;
    const rem = layers % world;
    return .{ .first = rank * base + @min(rank, rem), .count = base + @intFromBool(rank < rem) };
}

fn has(r: Range, layer: u32) bool {
    return layer >= r.first and layer < r.first + r.count;
}

/// Does this load item belong to rank `rank`: its layer sits in the rank's range, or (past-layer
/// tensors: the embed, the final norm, the head) on rank 0 — the rank that embeds and reads logits.
pub fn owns(it: *const weights.Item, layers: u32, world: u32, rank: u32) bool {
    if (world == 1) return true;
    if (it.layer == weights.past) return rank == 0;
    return has(range(layers, world, rank), it.layer);
}

/// The rank's checkpoint bytes over the whole load: the receipt the TP2 upload asserts against
/// (rank 0's + rank 1's must equal `load.bytes()` exactly — no replication in this split).
pub fn bytesOf(l: *const weights.Load, world: u32, rank: u32) u64 {
    var n: u64 = 0;
    for (l.items) |*it| {
        if (owns(it, l.config.layers, world, rank)) n += it.bytes;
    }
    return n;
}

/// The rank's item count over the whole load.
pub fn itemsOf(l: *const weights.Load, world: u32, rank: u32) usize {
    var n: usize = 0;
    for (l.items) |*it| {
        if (owns(it, l.config.layers, world, rank)) n += 1;
    }
    return n;
}

// --- tests: the range math first, then the fixture plan's split by bytes ---

test "the layer split's ranges: even halves at 2, the remainder carried by the first ranks" {
    try std.testing.expectEqual(Range{ .first = 0, .count = 30 }, range(60, 2, 0));
    try std.testing.expectEqual(Range{ .first = 30, .count = 30 }, range(60, 2, 1));
    try std.testing.expectEqual(Range{ .first = 0, .count = 20 }, range(60, 3, 0));
    try std.testing.expectEqual(Range{ .first = 20, .count = 20 }, range(60, 3, 1));
    try std.testing.expectEqual(Range{ .first = 40, .count = 20 }, range(60, 3, 2));
    try std.testing.expectEqual(Range{ .first = 0, .count = 1 }, range(2, 3, 0));
    try std.testing.expectEqual(Range{ .first = 1, .count = 1 }, range(2, 3, 1));
    try std.testing.expectEqual(Range{ .first = 2, .count = 0 }, range(2, 3, 2)); // a short rank owns nothing this receipt
    try std.testing.expectError(error.NoRank, checked(.{ .rank = 2, .world = 2 }));
    try std.testing.expectError(error.NoRank, checked(.{ .rank = 0, .world = 3 }));
}

test "the fixture load splits exactly: the two ranks' bytes and items sum to the whole plan" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try weights.checkpointFixture(gpa, io, tmp.dir, false);
    const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(path);
    const c = @import("exl3.zig").synthConfig();
    var p = try @import("exl3.zig").discover(gpa, io, path, &c);
    defer p.deinit();
    var l = try weights.plan(gpa, io, path, &c, &p);
    defer l.deinit();
    // world 2 over the 2-layer fixture: rank 0 = layer 0 plus every past-layer tensor.
    const r0 = bytesOf(&l, 2, 0);
    const r1 = bytesOf(&l, 2, 1);
    try std.testing.expectEqual(l.bytes(), r0 + r1); // disjoint: no group on both ranks, none on neither
    try std.testing.expectEqual(itemsOf(&l, 2, 0) + itemsOf(&l, 2, 1), l.items.len);
    // the head is the plan's last item and it is rank 0's; every past-layer tensor is rank 0's
    try std.testing.expect(owns(&l.items[l.head], c.layers, 2, 0));
    try std.testing.expect(!owns(&l.items[l.head], c.layers, 2, 1));
    try std.testing.expect(owns(&l.items[0], c.layers, 2, 0)); // the embed goes with rank 0's first wave
    try std.testing.expect(!owns(&l.items[0], c.layers, 2, 1));
    // world 1 (TP1): everything on rank 0
    try std.testing.expectEqual(l.bytes(), bytesOf(&l, 1, 0));
    try std.testing.expectEqual(l.bytes(), bytesOf(&l, 1, 1));
}
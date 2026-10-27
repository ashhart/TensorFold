//! The routed MoE of MLX affine experts: router, pick rule, plan, gate and up, activation, down, slots summed in order.

const std = @import("std");
const ops = @import("ops/ops.zig");

const Ops = ops.Ops;
const Tensor = ops.Tensor;
const Kind = ops.Kind;
const Projection = ops.Projection;

/// A layer's experts, shared expert last: gate and up stacked as one (E + 1, 2 NI, ...) weight, down (E + 1, D, ...).
pub const Experts = struct {
    fused: Projection,
    down: Projection,
    count: usize,
    width: usize,
    dims: usize,
    limit: f32 = 0,
};

/// The router's rows in fp32 (E + 1, D), the shared expert's gate last, and the experts.
pub const Routed = struct {
    rows32: u64,
    experts: Experts,
    top_k: usize,
    /// The router's rows, E + 1: the model's expert count, whatever share of the experts this rank holds.
    rows: usize,
    /// A rank's expert ids (int32, E + 1): each id's place among its own, -1 for another rank's; zero on one rank.
    remap: u64 = 0,
    /// Plan the rows' own pairs, not a power-of-two bucket's (no graph replays the launches at other row counts).
    exact: bool = false,

    /// Routed experts, the shared one not counted.
    pub fn count(r: Routed) usize {
        return r.rows - 1;
    }
};

/// experts.py: the decode tile's most rows, the GEMM tile's rows and where it starts, items' pair cap.
const LANE_ROWS = 8;
const BLOCK_ROWS = 128;
const BLOCK_FROM = 64;
const TILE = 16;

/// MoEBuffers' size for `rows` rows: the next power of two from 16.
pub fn bucket(rows: usize) usize {
    const bits = if (rows <= 1) 0 else std.math.log2_int_ceil(usize, rows);
    return @as(usize, 1) << @intCast(@max(4, bits));
}

/// max_items(pairs, experts, min(TILE, LANE_ROWS)): an item a used expert plus one a tile of pairs past its first.
pub fn capacity(pairs: usize, experts: usize) usize {
    return @min(pairs, experts) + pairs / @min(TILE, LANE_ROWS);
}

/// tile_for: the decode tile below BLOCK_FROM rows, the GEMM tile from there; a lane round's rows keep the decode tile.
pub fn tileFor(o: Ops, rows: usize) usize {
    return if (rows < BLOCK_FROM or o.window) LANE_ROWS else BLOCK_ROWS;
}

/// `x` (rows, D) to (rows, D) in its dtype: the top-k experts plus the shared one; a tp rank returns its fp32 share.
pub fn run(o: Ops, kind: Kind, r: Routed, x: Tensor, rows: usize, prefill: bool) ops.Error!Tensor {
    _ = prefill; // the affine experts' arithmetic does not depend on it (the plan's tile comes from tile_for)
    const ex = r.experts;
    const slots = r.top_k + 1;
    const size = if (r.exact) rows else bucket(rows);
    const pairs = rows * slots;
    const cap = capacity(size * slots, r.rows);
    const logits = try o.arena.of(f32, rows * r.rows);
    const pick = try o.arena.of(i32, size * slots);
    const wts = try o.arena.of(f32, size * slots);
    const items = try o.arena.of(i32, cap * 3);
    const members = try o.arena.of(i32, size * slots);
    try o.moeRouter(x, r.rows32, logits, rows, ex.dims, r.rows);
    const count: usize = cap;
    var tile: usize = undefined;
    if (r.remap != 0) {
        // every rank picks the same pairs, runs the ones it holds (the others group under an id past its experts)
        try o.moeSelect(logits, pick, wts, null, 0, rows, r.count(), r.top_k);
        const local = try o.arena.of(i32, size * slots);
        try o.moeLocalize(pick, r.remap, local, pairs, ex.count);
        tile = tileFor(o, rows);
        try o.moeRoute(local, pairs, r.rows, tile, members, items, cap);
        try o.moeForeignItems(items, cap, ex.count);
        const both = try o.projectRouted(x, ex.fused, items, count, members, pairs, slots, @min(tile, rows));
        const act = try o.arena.take(pairs * ex.width * kind.size());
        const act_t: Tensor = .{ .ptr = act, .kind = kind };
        try o.moeAct(both, act_t, pairs, ex.width, ex.limit);
        const y = try o.projectRouted(act_t, ex.down, items, count, members, pairs, 1, @min(tile, rows));
        try o.moeZeroForeign(y, local, ex.count, pairs, ex.dims);
        const share = try o.arena.of(f32, rows * ex.dims);
        try o.moeCombine(y, wts, .{ .ptr = share, .kind = .f32 }, rows, slots, ex.dims);
        return .{ .ptr = share, .kind = .f32 };
    }
    const p = try expertParts(o, kind, r, x, rows, logits, pick, wts, items, members, cap);
    const out = try o.arena.take(rows * ex.dims * kind.size());
    try o.moeCombine(p.y, wts, .{ .ptr = out, .kind = kind }, rows, slots, ex.dims);
    return .{ .ptr = out, .kind = kind };
}

/// The routed slots a row (fp32) and their weights before the sum: a single rank only.
pub const Parts = struct { y: u64, wts: u64, slots: usize };

pub fn parts(o: Ops, kind: Kind, r: Routed, x: Tensor, rows: usize) ops.Error!Parts {
    const slots = r.top_k + 1;
    const size = if (r.exact) rows else bucket(rows);
    const cap = capacity(size * slots, r.rows);
    const logits = try o.arena.of(f32, rows * r.rows);
    const pick = try o.arena.of(i32, size * slots);
    const wts = try o.arena.of(f32, size * slots);
    const items = try o.arena.of(i32, cap * 3);
    const members = try o.arena.of(i32, size * slots);
    try o.moeRouter(x, r.rows32, logits, rows, r.experts.dims, r.rows);
    const p = try expertParts(o, kind, r, x, rows, logits, pick, wts, items, members, cap);
    return .{ .y = p.y, .wts = wts, .slots = slots };
}

/// Select, plan, gate and up, the activation and down: the rows' slots before the combine.
fn expertParts(o: Ops, kind: Kind, r: Routed, x: Tensor, rows: usize, logits: u64, pick: u64, wts: u64, items: u64, members: u64, cap: usize) ops.Error!struct { y: u64 } {
    const ex = r.experts;
    const slots = r.top_k + 1;
    const pairs = rows * slots;
    var count: usize = cap;
    var tile: usize = undefined;
    if (rows == 1) {
        try o.moeSelect(logits, pick, wts, .{ .items = items, .members = members }, cap, 1, r.count(), r.top_k);
        tile = 1;
        count = slots;
    } else {
        try o.moeSelect(logits, pick, wts, null, 0, rows, r.count(), r.top_k);
        tile = tileFor(o, rows);
        try o.moeRoute(pick, pairs, r.rows, tile, members, items, cap);
    }
    // the gate and up's activation is the launch's epilogue when the tile takes the shape
    const act_t: Tensor = if (try o.projectRoutedAct(x, ex.fused, items, count, members, pairs, slots, @min(tile, rows), ex.limit)) |t| t else blk: {
        const both = try o.projectRouted(x, ex.fused, items, count, members, pairs, slots, @min(tile, rows));
        const act = try o.arena.take(pairs * ex.width * kind.size());
        const t: Tensor = .{ .ptr = act, .kind = kind };
        try o.moeAct(both, t, pairs, ex.width, ex.limit);
        break :blk t;
    };
    const y = try o.projectRouted(act_t, ex.down, items, count, members, pairs, 1, @min(tile, rows));
    return .{ .y = y };
}

test "buffer sizes follow moe.run and max_items" {
    try std.testing.expectEqual(@as(usize, 16), bucket(1));
    try std.testing.expectEqual(@as(usize, 16), bucket(16));
    try std.testing.expectEqual(@as(usize, 32), bucket(17));
    try std.testing.expectEqual(@as(usize, 2048), bucket(2048));
    try std.testing.expectEqual(@as(usize, 129 + 16 * 9 / 8), capacity(16 * 9, 129));
}

test "the routed products take the decode tile below 64 rows and in a lane round, the GEMM tile from 64 rows" {
    const o: Ops = .{ .l = undefined, .bf16 = true, .stream = undefined, .arena = undefined };
    var round = o;
    round.window = true;
    try std.testing.expectEqual(@as(usize, LANE_ROWS), tileFor(o, 1));
    try std.testing.expectEqual(@as(usize, LANE_ROWS), tileFor(o, BLOCK_FROM - 1));
    try std.testing.expectEqual(@as(usize, BLOCK_ROWS), tileFor(o, BLOCK_FROM));
    try std.testing.expectEqual(@as(usize, BLOCK_ROWS), tileFor(o, 2048));
    try std.testing.expectEqual(@as(usize, LANE_ROWS), tileFor(round, BLOCK_FROM));
    try std.testing.expectEqual(@as(usize, LANE_ROWS), tileFor(round, 2048));
}

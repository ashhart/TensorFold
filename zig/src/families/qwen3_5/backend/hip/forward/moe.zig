//! The routed MoE of MLX affine experts: router, pick rule, plan, gate and up, activation, down, slots summed in order.

const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;

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
fn tileFor(o: Ops, rows: usize) usize {
    return if (rows < BLOCK_FROM or o.window) LANE_ROWS else BLOCK_ROWS;
}

/// `x` (rows, D) to (rows, D) in its dtype: the top-k experts plus the shared one; a tp rank returns its fp32 share.
pub fn run(o: Ops, m: *const view.Model, r: view.Routed, x: Tensor, rows: usize, prefill: bool) hip.ops.Error!Tensor {
    _ = prefill; // the affine experts' arithmetic does not depend on it (the plan's tile comes from tile_for)
    const ex = r.experts;
    const slots = r.top_k + 1;
    const size = bucket(rows);
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
        const act = try o.arena.take(pairs * ex.width * m.act.size());
        const act_t: Tensor = .{ .ptr = act, .kind = m.act };
        try o.moeAct(both, act_t, pairs, ex.width, ex.limit);
        const y = try o.projectRouted(act_t, ex.down, items, count, members, pairs, 1, @min(tile, rows));
        try o.moeZeroForeign(y, local, ex.count, pairs, ex.dims);
        const share = try o.arena.of(f32, rows * ex.dims);
        try o.moeCombine(y, wts, .{ .ptr = share, .kind = .f32 }, rows, slots, ex.dims);
        return .{ .ptr = share, .kind = .f32 };
    }
    const p = try expertParts(o, m, r, x, rows, logits, pick, wts, items, members, cap);
    const out = try o.arena.take(rows * ex.dims * m.act.size());
    try o.moeCombine(p.y, wts, .{ .ptr = out, .kind = m.act }, rows, slots, ex.dims);
    return .{ .ptr = out, .kind = m.act };
}

/// The routed slots a row (fp32) and their weights before the sum: a single rank only.
pub const Parts = struct { y: u64, wts: u64, slots: usize };

pub fn parts(o: Ops, m: *const view.Model, r: view.Routed, x: Tensor, rows: usize) hip.ops.Error!Parts {
    const slots = r.top_k + 1;
    const size = bucket(rows);
    const cap = capacity(size * slots, r.rows);
    const logits = try o.arena.of(f32, rows * r.rows);
    const pick = try o.arena.of(i32, size * slots);
    const wts = try o.arena.of(f32, size * slots);
    const items = try o.arena.of(i32, cap * 3);
    const members = try o.arena.of(i32, size * slots);
    try o.moeRouter(x, r.rows32, logits, rows, r.experts.dims, r.rows);
    const p = try expertParts(o, m, r, x, rows, logits, pick, wts, items, members, cap);
    return .{ .y = p.y, .wts = wts, .slots = slots };
}

/// Select, plan, gate and up, the activation and down: the rows' slots before the combine.
fn expertParts(o: Ops, m: *const view.Model, r: view.Routed, x: Tensor, rows: usize, logits: u64, pick: u64, wts: u64, items: u64, members: u64, cap: usize) hip.ops.Error!struct { y: u64 } {
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
        const act = try o.arena.take(pairs * ex.width * m.act.size());
        const t: Tensor = .{ .ptr = act, .kind = m.act };
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

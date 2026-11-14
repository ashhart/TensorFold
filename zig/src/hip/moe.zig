//! The routed MoE of MLX affine experts: router, pick rule, plan, gate and up, activation, down, slots summed in order.

const std = @import("std");
const ops = @import("ops/ops.zig");

const Ops = ops.Ops;
const Tensor = ops.Tensor;
const Kind = ops.Kind;
const Projection = ops.Projection;

/// A shared expert of its own width, run beside the routed stack: gate and up as one (2 NI, K) projection, then down.
pub const Shared = struct { gate_up: Projection, down: Projection };

/// A layer's experts, shared last in the stack, or beside it in `shared` at its own bits and group (`count` is then E).
pub const Experts = struct {
    fused: Projection,
    down: Projection,
    /// E + 1 when the shared expert is the stack's last expert, E when it is `shared`.
    count: usize,
    width: usize,
    dims: usize,
    limit: f32 = 0,
    shared: ?Shared = null,
};

/// The router's rows in fp32 (E + 1, D), the shared expert's gate last, and the experts.
pub const Routed = struct {
    rows32: u64,
    /// The same rows as the bf16 they are held in (the fp32 rows' values widened), which a round reads instead.
    rows16: u64 = 0,
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
    return tileOf(rows, o.window);
}

/// tile_for's rule on its own, for a fixture that plans as serving does.
pub fn tileOf(rows: usize, window: bool) usize {
    return if (rows < BLOCK_FROM or window) LANE_ROWS else BLOCK_ROWS;
}

/// The stacked shared expert's tile: it holds every row, so a prompt pass never cuts it below the GEMM tile.
pub fn sharedTileFor(o: Ops, rows: usize) usize {
    return sharedTileOf(rows, o.window, o.prefill);
}

/// sharedTileFor's rule on its own, for a fixture that plans as serving does.
pub fn sharedTileOf(rows: usize, window: bool, prefill: bool) usize {
    // decode and lane rounds keep tile_for's: their launches take m as the most rows an item holds
    return if (prefill and !window) BLOCK_ROWS else tileOf(rows, window);
}

/// The routed launches' m before the rows cap it: the most rows a plan's item holds, the shared one's where it stacks.
pub fn itemRowsFor(o: Ops, rows: usize, shared: bool) usize {
    return itemRowsOf(rows, o.window, o.prefill, shared);
}

/// itemRowsFor's rule on its own, for a fixture that plans as serving does.
pub fn itemRowsOf(rows: usize, window: bool, prefill: bool, shared: bool) usize {
    const tile = tileOf(rows, window);
    return if (shared) @max(tile, sharedTileOf(rows, window, prefill)) else tile;
}

/// The average rows an expert of `experts` takes of `rows` tokens' `top_k` routed slots, rounded up; 0 without experts.
pub fn expertRows(rows: usize, top_k: usize, experts: usize) u32 {
    if (experts == 0) return 0;
    return @intCast((rows * top_k + experts - 1) / experts);
}

/// `x` (rows, D) to (rows, D) in its dtype: the top-k experts plus the shared one; a tp rank returns its fp32 share.
pub fn run(o: Ops, kind: Kind, r: Routed, x: Tensor, rows: usize, prefill: bool) ops.Error!Tensor {
    _ = prefill; // the affine experts' arithmetic does not depend on it (the plan's tiles come from `o`)
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
    try o.moeRouterRows(x, r.rows32, r.rows16, logits, rows, ex.dims, r.rows);
    const count: usize = cap;
    var tile: usize = undefined;
    if (r.remap != 0) {
        // every rank picks the same pairs, runs the ones it holds (the others group under an id past its experts)
        try o.moeSelect(logits, pick, wts, null, 0, rows, r.count(), r.top_k);
        const local = try o.arena.of(i32, size * slots);
        try o.moeLocalize(pick, r.remap, local, pairs, ex.count);
        tile = tileFor(o, rows);
        // a rank's local ids don't name the shared expert: every expert keeps the routed tile
        try o.moeRoute(local, pairs, r.rows, tile, null, tile, members, items, cap);
        try o.moeForeignItems(items, cap, ex.count);
        const expert_rows = expertRows(rows, r.top_k, r.count());
        const both = try o.projectRouted(x, ex.fused, items, count, members, pairs, slots, @min(tile, rows), expert_rows);
        const act = try o.arena.take(pairs * ex.width * kind.size());
        const act_t: Tensor = .{ .ptr = act, .kind = kind };
        try o.moeAct(both, act_t, pairs, ex.width, ex.limit);
        const y = try o.projectRouted(act_t, ex.down, items, count, members, pairs, 1, @min(tile, rows), expert_rows);
        // the shared pair's id is another rank's here: its row reads zero, and rank 0 adds its own gate's branch
        try o.moeZeroForeign(y, local, ex.count, pairs, ex.dims);
        const share = try o.arena.of(f32, rows * ex.dims);
        if (ex.shared) |sh| {
            const s = try sharedRun(o, sh, x, rows, kind, ex.limit);
            try o.moeCombineGate(y, wts, s, .{ .ptr = share, .kind = .f32 }, rows, slots, ex.dims);
            return .{ .ptr = share, .kind = .f32 };
        }
        try o.moeCombine(y, wts, .{ .ptr = share, .kind = .f32 }, rows, slots, ex.dims);
        return .{ .ptr = share, .kind = .f32 };
    }
    const p = try expertParts(o, kind, r, x, rows, logits, pick, wts, items, members, cap);
    const out = try o.arena.take(rows * ex.dims * kind.size());
    if (ex.shared) |sh| {
        const s = try sharedRun(o, sh, x, rows, kind, ex.limit);
        try o.moeCombineGate(p.y, wts, s, .{ .ptr = out, .kind = kind }, rows, slots, ex.dims);
        return .{ .ptr = out, .kind = kind };
    }
    try o.moeCombine(p.y, wts, .{ .ptr = out, .kind = kind }, rows, slots, ex.dims);
    return .{ .ptr = out, .kind = kind };
}

/// down(silu(gate) * up) of `x`'s rows at the shared expert's own widths: (rows, dims) fp32, as the combine reads it.
fn sharedRun(o: Ops, sh: Shared, x: Tensor, rows: usize, kind: Kind, limit: f32) ops.Error!u64 {
    const width = sh.gate_up.n / 2;
    const both = try o.project(x, sh.gate_up, rows, true);
    const act: Tensor = .{ .ptr = try o.arena.take(rows * width * kind.size()), .kind = kind };
    try o.moeAct(both.ptr, act, rows, width, limit);
    return (try o.project(act, sh.down, rows, true)).ptr;
}

/// The routed slots and weights before the sum: one rank, and only where the shared expert stacks in `y`.
pub const Parts = struct { y: u64, wts: u64, slots: usize };

pub fn parts(o: Ops, kind: Kind, r: Routed, x: Tensor, rows: usize) ops.Error!Parts {
    std.debug.assert(r.experts.shared == null);
    const slots = r.top_k + 1;
    const size = if (r.exact) rows else bucket(rows);
    const cap = capacity(size * slots, r.rows);
    const logits = try o.arena.of(f32, rows * r.rows);
    const pick = try o.arena.of(i32, size * slots);
    const wts = try o.arena.of(f32, size * slots);
    const items = try o.arena.of(i32, cap * 3);
    const members = try o.arena.of(i32, size * slots);
    try o.moeRouterRows(x, r.rows32, r.rows16, logits, rows, r.experts.dims, r.rows);
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
        // the stacked shared expert is the last id; beside the stack its items are zeroed below, so it keeps the tile
        const shared: ?usize = if (ex.shared == null) r.count() else null;
        try o.moeRoute(pick, pairs, r.rows, tile, shared, sharedTileFor(o, rows), members, items, cap);
        // the launches' m: the most rows an item holds, the shared expert's whole item in a prompt pass
        tile = itemRowsFor(o, rows, shared != null);
    }
    // beside the stack the shared pair's id is the stack's first id past its experts: zero its item, never read its row
    if (ex.shared != null) try o.moeForeignItems(items, count, ex.count);
    const expert_rows = expertRows(rows, r.top_k, r.count());
    // the gate and up's activation is the launch's epilogue when the tile takes the shape
    const act_t: Tensor = if (try o.projectRoutedAct(x, ex.fused, items, count, members, pairs, slots, @min(tile, rows), ex.limit)) |t| t else blk: {
        const both = try o.projectRouted(x, ex.fused, items, count, members, pairs, slots, @min(tile, rows), expert_rows);
        const act = try o.arena.take(pairs * ex.width * kind.size());
        const t: Tensor = .{ .ptr = act, .kind = kind };
        try o.moeAct(both, t, pairs, ex.width, ex.limit);
        break :blk t;
    };
    const y = try o.projectRouted(act_t, ex.down, items, count, members, pairs, 1, @min(tile, rows), expert_rows);
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

test "the shared expert takes the GEMM tile in a prompt pass at any row count, the routed tile in decode and lane rounds" {
    const o: Ops = .{ .l = undefined, .bf16 = true, .stream = undefined, .arena = undefined };
    var prompt = o;
    prompt.prefill = true;
    var round = o;
    round.window = true;
    for ([_]usize{ 2, 33, BLOCK_FROM - 1, BLOCK_FROM, 2048 }) |rows| {
        try std.testing.expectEqual(@as(usize, BLOCK_ROWS), sharedTileFor(prompt, rows));
        // the routed experts keep tile_for's in the same prompt pass
        try std.testing.expectEqual(tileFor(o, rows), tileFor(prompt, rows));
        try std.testing.expectEqual(tileFor(o, rows), sharedTileFor(o, rows));
        try std.testing.expectEqual(@as(usize, LANE_ROWS), sharedTileFor(round, rows));
        // the launches' m holds the plan's largest item: the shared one only where it stacks in a prompt pass
        try std.testing.expectEqual(@as(usize, BLOCK_ROWS), itemRowsFor(prompt, rows, true));
        try std.testing.expectEqual(tileFor(prompt, rows), itemRowsFor(prompt, rows, false));
        try std.testing.expectEqual(tileFor(o, rows), itemRowsFor(o, rows, true));
        try std.testing.expectEqual(@as(usize, LANE_ROWS), itemRowsFor(round, rows, true));
    }
}

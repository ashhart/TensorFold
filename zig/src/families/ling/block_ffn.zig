//! Ling's MLPs on CUDA: SwiGLU over projections in their loaded formats, and the MoE block (router, groups, experts).
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("graph.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");
const ling = @import("cuda.zig");

const glue = graph.glue;
const Ops = graph.Ops;
const Config = @import("config.zig").Config;

/// Gate and up into [gate | up] rows, SwiGLU with min(silu(g), limit) (0: none), down into `out` [rows][hidden].
pub fn mlp(o: Ops, b: *const state.Buffers, m: weights.Mlp, x: u64, ldx: usize, rows: usize, limit: f32, out: u64) !void {
    const w = m.gate.outputs();
    if (rows == 1) { // one row is its own columns: no copies
        try o.mm(m.gate, x, ldx, 1, b.gu);
        try o.mm(m.up, x, ldx, 1, b.gu + w * 2);
    } else {
        try o.mmCols(m.gate, x, ldx, rows, b.tmp, b.gu, 2 * w, 0);
        try o.mmCols(m.up, x, ldx, rows, b.tmp, b.gu, 2 * w, w);
    }
    try glue.swigluWith(o.m.swiglu, o.s, b.gu, b.act, b.xs, limit, rows, w);
    try o.mm(m.down, b.act, w, rows, out);
}

/// The MoE block on b.normed into b.y: routed experts' slots summed (weights fp32), plus the shared expert.
pub fn moe(o: Ops, b: *const state.Buffers, e: weights.Moe, g: *const graph.Graph, layer: u32, rows: usize, picks: ?u64) !void {
    const c = &g.w.config;
    const D: usize = c.hidden;
    const E: usize = c.experts;
    if (e.experts != E) return error.PartialExperts; // the router picks among every expert
    const top: usize = c.topk;
    const slots = top + 1;
    const lim = ling.limits(c, layer);
    try glue.router(o.set(), o.s, b.normed, D, e.router, b.mlog, b.rpart, rows, D, E);
    try glue.select(o.set(), o.s, b.mlog, if (e.bias != 0) e.bias else g.zero_bias, b.pick, b.wts, rows, top, E, c.routed_scale, c.norm_topk, ling.groups(c));
    if (picks) |p| try o.copy(b.pick, slots * 4, p, slots * 4, rows, slots * 4);
    switch (e.routed) {
        .grouped => |l| try grouped(o, b, l, rows, slots, E),
        .lanes => |ls| for (ls, 0..) |x, j| try mlp(o, b, x, b.normed, D, rows, lim.routed, b.ey + j * rows * D * 2),
    }
    if (e.shared) |s| try mlp(o, b, s, b.normed, D, rows, e.shared_limit, b.sh);
    var a: cuda.Args = .{};
    inline for (.{ b.ey, b.pick, b.wts, if (e.shared != null) b.sh else @as(u64, 0), b.y, @as(i64, @intCast(rows)), @as(i64, @intCast(D)), @as(c_int, @intCast(slots)), @as(c_int, @intCast(top)), @as(c_int, @intFromBool(e.routed == .lanes)) }) |v| a.add(v);
    try cuda.launch.launch(o.m.combine, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((D + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
}

/// Grouped experts on every row's top-k slots (the shared slot skipped): gate-up SwiGLU, then down to bf16 slot rows.
fn grouped(o: Ops, b: *const state.Buffers, l: cuda.experts.Layer, rows: usize, slots: usize, count: usize) !void {
    const pairs = rows * slots;
    const tile: usize = if (o.prompt) cuda.experts.promptTile(l.format) else 16;
    const skip: i32 = @intCast(count);
    try o.m.router.route(o.s, b.pick, pairs, count + 1, tile, b.plan);
    const items = cuda.grouped.maxItems(pairs, count + 1, tile);
    const g = o.m.grouped();
    const up: cuda.experts.Rows = .{ .x = b.normed, .stride = l.dims, .slots = slots };
    const down: cuda.experts.Rows = .{ .x = b.eact, .stride = l.width };
    if (o.prompt) {
        try g.prompt(o.s, .up, up, l, b.plan, b.eact, items, skip);
        try g.prompt(o.s, .down_bf16, down, l, b.plan, b.ey, items, skip);
    } else {
        try g.decode(o.s, .up, up, l, b.plan, b.eact, items, skip);
        try g.decode(o.s, .down_bf16, down, l, b.plan, b.ey, items, skip);
    }
}

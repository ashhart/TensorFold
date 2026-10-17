//! Qwen3.5 MoE's routed experts on CUDA (qwen3_5_moe): the router's top-k, the grouped plan, the experts and combine.
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");

/// The routed rows' scratch: logits, picks and weights a row, the plan, the pairs' activations and sums.
pub const Scratch = struct {
    logits: u64 = 0,
    picks: u64 = 0,
    wts: u64 = 0,
    members: u64 = 0,
    items: u64 = 0,
    counts: u64 = 0,
    rank: u64 = 0,
    hist: u64 = 0,
    act: u64 = 0,
    y: u64 = 0,

    /// Carves it from `t` (the engine's scratch taker) for `most` rows.
    pub fn carve(t: anytype, g: E.Geometry, most: usize) !Scratch {
        const pairs = most * g.slots();
        const count = g.experts + 1;
        return .{
            .logits = try t.alloc(most * count * 4),
            .picks = try t.alloc(pairs * 4),
            .wts = try t.alloc(pairs * 4),
            .members = try t.alloc(pairs * 4),
            .items = try t.alloc(3 * cuda.grouped.maxItems(pairs, count, 16) * 4),
            .counts = try t.alloc(2 * 4),
            .rank = try t.alloc(pairs * 4),
            .hist = try t.alloc(((pairs + 1023) / 1024) * count * 4),
            .act = try t.alloc(pairs * g.intermediate * 2),
            .y = try t.alloc(pairs * g.hidden * 4),
        };
    }
};

/// Routed experts over `total` normed rows at x into out; `wide` rows take the prefill kernels, else decode's.
pub fn rows(e: *E.Engine, router: u64, layer: cuda.experts.Layer, x: u64, out: u64, total: usize, wide: bool) !void {
    const g = e.g;
    const s = e.routed;
    const slots = g.slots();
    const count = g.experts + 1;
    const pairs = total * slots;
    const tile: usize = if (wide) 64 else 16;
    const items = cuda.grouped.maxItems(pairs, count, tile);
    const plan: cuda.grouped.Plan = .{ .members = s.members, .items = s.items, .counts = s.counts, .rank = s.rank, .hist = s.hist };
    const grouped: cuda.experts.Grouped = .{ .affine4 = &e.experts, .affine3 = &e.experts3 };
    try e.moe.route(e.stream, x, g.hidden, total, router, s.logits, g.hidden, count);
    try e.moe.pick(e.stream, s.logits, s.picks, s.wts, total, g.experts, g.top_k);
    try e.plan.route(e.stream, s.picks, pairs, count, tile, plan);
    const rows_in: cuda.experts.Rows = .{ .x = x, .stride = g.hidden, .slots = slots };
    const act_in: cuda.experts.Rows = .{ .x = s.act, .stride = g.intermediate };
    if (wide) {
        try grouped.prompt(e.stream, .up, rows_in, layer, plan, s.act, items, -1);
        try grouped.prompt(e.stream, .down_bf16, act_in, layer, plan, s.y, items, -1);
    } else {
        try grouped.decode(e.stream, .up, rows_in, layer, plan, s.act, items, -1);
        try grouped.decode(e.stream, .down_f32, act_in, layer, plan, s.y, items, -1);
    }
    try e.moe.combine(e.stream, s.y, !wide, s.wts, out, total, slots, g.hidden);
}

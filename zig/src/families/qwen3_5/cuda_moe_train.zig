//! Packed shared/routed expert adjoints; weights stay in the engine's arena.
const cuda = @import("cuda");
const c = @import("cuda_learn_shape.zig");
const train = @import("cuda_train.zig");
const to = @import("cuda_train_ops.zig");
const hooks = @import("cuda_learn_hooks.zig");
const q4 = cuda.train_q4;

pub fn memory(g: c.Geometry) usize {
    var dense = g;
    dense.experts = 0;
    const R = train.pack_rows;
    const kept = g.layers - train.keepLow(g.layers);
    const q = g.intermediate * g.hidden / 2 + g.intermediate * g.hidden / 16;
    const removed = kept * (R * 4 * g.intermediate + 3 * q) + 3 * R * g.intermediate * 4;
    const slots = g.top_k + 1;
    const pairs = R * slots;
    const total = g.experts + 1;
    const scratch = (R * total + 3 * pairs + total + 2 * q4.tileCap(pairs, total) + pairs * (5 * g.intermediate + g.hidden)) * 4;
    return train.memory(dense) - removed + 2 * g.layers * R * slots * 4 + scratch;
}

pub fn carve(t: anytype) !void {
    const g = t.e.g;
    const rows = train.pack_rows;
    const pairs = rows * g.slots();
    const total = g.experts + 1;
    t.moe_scratch = .{
        .logits = try t.device(rows * total * 4),
        .order = try t.device(pairs * 4),
        .inv = try t.device(pairs * 4),
        .offsets = try t.device(total * 4),
        .tiles = try t.device(2 * q4.tileCap(pairs, total) * 4),
        .gu = try t.device(pairs * 2 * g.intermediate * 4),
        .dact = try t.device(pairs * g.intermediate * 4),
        .dgu = try t.device(pairs * 2 * g.intermediate * 4),
        .dxp = try t.device(pairs * g.hidden * 4),
        .a = try t.device(pairs * 4),
        .dlog = 0,
    };
}

pub fn backward(t: anytype, layer: usize, kp: hooks.LayerKeep, pocket: ?to.Pocket, rows: usize) !void {
    const e = t.e;
    const g = e.g;
    const m = e.layers[layer].mlp.moe;
    if (m.experts.format != .affine4) return error.UnsupportedLearningFormat;
    const ops: q4.Ops = .{ .k = &t.mk.?, .s = e.stream };
    const kept: q4.MoeKept = .{ .x = kp.y, .ids = kp.ids, .wts = kp.wts, .served_rounding = true };
    const tables: q4.PackedExperts = .{ .up = m.experts.up, .down = m.experts.down, .routed = g.experts, .width = g.intermediate, .dim = g.hidden, .group = 64 };
    try ops.zero(t.dy, rows * g.hidden);
    try ops.expertsBackPacked(tables, m.router, kept, t.grad, t.dy, t.moe_scratch, rows, g.top_k);
    if (pocket) |pk0| {
        var pk = pk0;
        pk.row_gate = kp.wts + g.top_k * 4;
        pk.gate_stride = g.slots();
        pk.a_shared = t.moe_scratch.a + g.top_k * 4;
        try t.ops().pocketBack(pk, kp.y, t.grad, t.dy, rows, g.hidden);
    }
    try ops.routeBack(m.router, kept, t.dy, t.moe_scratch, rows, g.top_k, g.experts, g.hidden);
}

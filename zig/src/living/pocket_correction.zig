//! Contrastive outputs for a correction: fit the new answer and subtract the old answer's loss gradient.
const std = @import("std");
const mode = @import("train_mode.zig");
const subspace = @import("subspace.zig");
const pack = @import("pocket_pack.zig");
const pocket = @import("pocket.zig");
const weight: f32 = 0.5;

/// One old answer each step, rotating through the training questions; only the new subject neurons receive its push.
pub fn add(p: anytype, t: anytype, grad: []f32, positive: usize) !void {
    if (p.against.len == 0) return;
    const ex = p.against[p.v.round % p.against.len];
    const row = pocket.subjectRow(ex) orelse return error.CorrectionSubjectMissing;
    var mask: u64 = 0;
    for (p.taken.items) |x| {
        if (x.key != .subject) return error.CorrectionSubjectMissing;
        const l = p.moe.items[x.slot];
        mask |= @as(u64, 1) << @intCast(l);
        @memset(t.deltaAt(0, 1, l), 0);
    }
    if (mask == 0) return error.CorrectionNeuronsMissing;
    const sq: mode.Seq = .{ .ids = ex.ids, .start = ex.start, .decide = ex.decide, .nudge = .{ .layers = .{ 0, mask, 0 }, .rows = .{ 0, row, 0 } } };
    var result: [1]mode.Result = undefined;
    try t.pass(&.{sq}, .target, pack.turn, &result);
    if (!std.math.isFinite(result[0].loss)) return error.NonfiniteCorrectionGradient;
    const d = p.hidden;
    for (p.taken.items, 0..) |x, j| {
        const l = p.moe.items[x.slot];
        const push = t.pushAt(0, 1, l);
        for (push) |v| if (!std.math.isFinite(v)) return error.NonfiniteCorrectionGradient;
        const activation = p.act(x.slot, x.neuron, t.normed(l, t.spans[0].row(row)));
        subtract(grad[j * d ..][0..d], @as(f32, @floatFromInt(positive)) * activation, push);
    }
}

/// Negative old-answer gradient, alongside the existing positive new-answer gradient in the same output solve.
pub fn subtract(grad: []f32, activation: f32, push: []const f32) void {
    subspace.axpy(grad, -weight * activation, push);
}

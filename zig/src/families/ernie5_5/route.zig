//! ERNIE 5.5's router (vLLM-ERNIE's ernie5_5_routing): two sigmoid scores, the top k of score + bias, weights scaled.
const std = @import("std");

/// The released router: 512 experts, each scored from two logits (gate.weight's row, then weight_1's), top 10.
pub const experts = 512;
pub const topk = 10;
/// The most picks a row may ask for (the kernel's shared slots).
pub const most_topk = 32;

/// rows of 2 x experts fp32 logits, ls apart; picks (int32) and weights (fp32) k a row.
pub const Shape = struct { rows: usize, k: usize = topk, ls: usize = 2 * experts };

/// k picks from 1 to most_topk, logit rows at least both routers wide, and rows that fit the grid.
pub fn check(sh: Shape) error{RouteShape}!void {
    if (sh.k == 0 or sh.k > most_topk or sh.ls < 2 * experts) return error.RouteShape;
    if (sh.rows > std.math.maxInt(i32)) return error.RouteShape;
}

pub fn sigmoid(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}

/// Expert e's score: sigmoid of the first router's logit plus sigmoid of the second's.
pub fn score(l: []const f32, e: usize) f64 {
    return sigmoid(l[e]) + sigmoid(l[experts + e]);
}

/// What expert e is ranked by: its score plus its correction bias, a NaN as -inf (never picked ahead of a number).
pub fn rank(l: []const f32, bias: []const f32, e: usize) f64 {
    const v = score(l, e) + bias[e];
    return if (std.math.isNan(v)) -std.math.inf(f64) else v;
}

/// A row in float64: the k best ranks (ties to the lower id) in ascending ids, weights score / max(sum, 1e-20) * scale.
pub fn route(l: []const f32, bias: []const f32, scale: []const f32, ids: []u32, w: []f64) void {
    var taken: [experts]bool = @splat(false);
    for (ids) |*id| {
        var best: usize = experts;
        for (0..experts) |e| {
            if (taken[e]) continue;
            if (best == experts or rank(l, bias, e) > rank(l, bias, best)) best = e;
        }
        taken[best] = true;
        id.* = @intCast(best);
    }
    std.mem.sort(u32, ids, {}, std.sort.asc(u32));
    weights(l, scale, ids, w);
}

/// The picks' weights: each score over the picks' summed scores (at least 1e-20), times the expert's scale.
pub fn weights(l: []const f32, scale: []const f32, ids: []const u32, w: []f64) void {
    var total: f64 = 0;
    for (ids) |id| total += score(l, id);
    for (w, ids) |*x, id| x.* = score(l, id) / @max(total, 1e-20) * scale[id];
}

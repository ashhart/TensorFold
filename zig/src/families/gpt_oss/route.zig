//! Host top-k selection from router projection-plus-bias logits. Normalization is an explicit separate operation.
const std = @import("std");
const config = @import("config.zig");
pub const Selected = struct { ids: [config.max_topk]u32, logits: [config.max_topk]f64, len: usize };

/// Reference convention: descending finite logits, lower expert ID first on ties. This does not choose weights.
pub fn select(c: *const config.Config, logits: []const f64) !Selected {
    if (logits.len != c.experts or c.topk == 0 or c.topk > config.max_topk or c.topk > logits.len) return error.BadRouterShape;
    var out: Selected = .{ .ids = @splat(0), .logits = @splat(-std.math.inf(f64)), .len = c.topk };
    for (logits, 0..) |score, id| {
        if (!std.math.isFinite(score)) return error.NonfiniteRouter;
        var at: usize = 0;
        while (at < out.len and score <= out.logits[at]) : (at += 1) {}
        if (at == out.len) continue;
        var end = out.len - 1;
        while (end > at) : (end -= 1) {
            out.logits[end] = out.logits[end - 1];
            out.ids[end] = out.ids[end - 1];
        }
        out.logits[at] = score;
        out.ids[at] = @intCast(id);
    }
    return out;
}

/// Explicit host reference for softmax over selected logits; the packet does not establish its model integration.
pub fn softmaxSelected(logits: []const f64, out: []f64) !void {
    if (logits.len == 0 or out.len != logits.len) return error.BadRouterShape;
    var maximum = -std.math.inf(f64);
    for (logits) |v| {
        if (!std.math.isFinite(v)) return error.NonfiniteRouter;
        maximum = @max(maximum, v);
    }
    var sum: f64 = 0;
    for (logits) |v| sum += @exp(v - maximum);
    for (out, logits) |*weight, v| weight.* = @exp(v - maximum) / sum;
}

//! MoE routing and combine in float64, the references every backend's MoE glue is checked against (cuda/moe_glue.zig).
const std = @import("std");

/// Sigmoid routing, as cuda/moe_glue.zig's Route.
pub const Route = struct { experts: usize, topk: usize, shared: bool, normalize: bool = false, scale: f64 = 1 };

/// A row's picks by the fp32 logit + bias (the kernels' ranking exactly; ties: lower id), weights in float64.
pub fn topk(logits: []const f32, bias: ?[]const f32, r: Route, pick: []u32, wts: []f64) void {
    var taken: [4096]bool = @splat(false);
    var sum: f64 = 0;
    for (0..r.topk) |k| {
        var best: ?usize = null;
        var bv: f32 = -std.math.inf(f32);
        for (logits[0..r.experts], 0..) |l, e| {
            if (taken[e]) continue;
            const v = l + (if (bias) |b| b[e] else 0); // one fp32 rounding, as the kernel's __fadd_rn
            if (best == null or v > bv) {
                best = e;
                bv = v;
            }
        }
        taken[best.?] = true;
        pick[k] = @intCast(best.?);
        wts[k] = 1 / (1 + @exp(-@as(f64, logits[best.?])));
        sum += wts[k];
    }
    for (wts[0..r.topk]) |*w| w.* = (if (r.normalize) w.* / sum else w.*) * r.scale;
    if (r.shared) {
        pick[r.topk] = @intCast(r.experts);
        wts[r.topk] = 1;
    }
}

/// One row: out[d] = sum over slots of w[k] * y[k][d].
pub fn combine(y: []const f64, w: []const f64, dims: usize, out: []f64) void {
    @memset(out, 0);
    for (w, 0..) |wk, k| for (out, y[k * dims ..][0..dims]) |*o, v| {
        o.* += wk * v;
    };
}

test "picks rank by logit plus bias, weigh by the logit, and ties go to the lower id" {
    var pick: [3]u32 = undefined;
    var wts: [3]f64 = undefined;
    topk(&.{ 0, 1, 1, -1 }, &.{ 0, 0, 0, 5 }, .{ .experts = 4, .topk = 2, .shared = true }, &pick, &wts);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 4 }, &pick);
    try std.testing.expectApproxEqRel(@as(f64, 1 / (1 + @exp(@as(f64, 1)))), wts[0], 1e-15);
    try std.testing.expectEqual(@as(f64, 1), wts[2]);
    topk(&.{ 2, 2 }, null, .{ .experts = 2, .topk = 2, .shared = false, .normalize = true, .scale = 2 }, pick[0..2], wts[0..2]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, pick[0..2]);
    try std.testing.expectApproxEqRel(@as(f64, 1), wts[0], 1e-15);
    var out: [2]f64 = undefined;
    combine(&.{ 1, 2, 10, 20 }, &.{ 0.5, 0.25 }, 2, &out);
    try std.testing.expectEqualSlices(f64, &.{ 3, 6 }, &out);
}

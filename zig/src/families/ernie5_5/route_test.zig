const std = @import("std");
const r = @import("route.zig");

const e = r.experts;

/// vLLM-ERNIE's _ernie5_5_routing_torch: experts by score + bias (then id), the first k, weighed without the bias.
fn vendor(l: []const f32, bias: []const f32, scale: []const f32, ids: []u32, w: []f64) void {
    var order: [e]u32 = undefined;
    for (&order, 0..) |*o, i| o.* = @intCast(i);
    const Ctx = struct { l: []const f32, bias: []const f32 };
    std.mem.sort(u32, &order, Ctx{ .l = l, .bias = bias }, struct {
        fn before(c: Ctx, a: u32, b: u32) bool {
            const x = r.rank(c.l, c.bias, a);
            const y = r.rank(c.l, c.bias, b);
            return x > y or (x == y and a < b);
        }
    }.before);
    @memcpy(ids, order[0..ids.len]);
    std.mem.sort(u32, ids, {}, std.sort.asc(u32));
    var sum: f64 = 0;
    for (ids) |id| sum += r.sigmoid(l[id]) + r.sigmoid(l[e + id]);
    for (w, ids) |*x, id| x.* = (r.sigmoid(l[id]) + r.sigmoid(l[e + id])) / @max(sum, 1e-20) * scale[id];
}

fn fill(seed: u64, l: *[2 * e]f32, bias: *[e]f32, scale: *[e]f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (l) |*x| x.* = rng.floatNorm(f32) * 2;
    for (bias, 0..) |*b, i| b.* = if (i % 3 == 0) 0.3 * rng.float(f32) else 0;
    for (scale) |*s| s.* = 0.5 + rng.float(f32) * 2;
}

test "the router is vLLM-ERNIE's: two sigmoids summed, picked by score + bias, weighed without it, scaled per expert" {
    var l: [2 * e]f32 = undefined;
    var bias: [e]f32 = undefined;
    var scale: [e]f32 = undefined;
    for (0..4) |seed| {
        fill(seed, &l, &bias, &scale);
        var ids: [r.topk]u32 = undefined;
        var w: [r.topk]f64 = undefined;
        var vi: [r.topk]u32 = undefined;
        var vw: [r.topk]f64 = undefined;
        r.route(&l, &bias, &scale, &ids, &w);
        vendor(&l, &bias, &scale, &vi, &vw);
        try std.testing.expectEqualSlices(u32, &vi, &ids);
        for (vw, w) |v, g| try std.testing.expectApproxEqRel(v, g, 1e-14);
    }
    // the bias picks an expert its score alone would not, and its weight leaves the bias out
    @memset(&l, -4);
    for (0..r.topk) |i| l[i] = 3;
    @memset(&bias, 0);
    bias[100] = 5;
    @memset(&scale, 1);
    var ids: [r.topk]u32 = undefined;
    var w: [r.topk]f64 = undefined;
    r.route(&l, &bias, &scale, &ids, &w);
    try std.testing.expectEqual(@as(u32, 100), ids[r.topk - 1]);
    try std.testing.expect(w[r.topk - 1] < 0.01);
}

test "ties go to the lower id, a NaN is never picked ahead of a number, and tiny scores divide by 1e-20" {
    var l: [2 * e]f32 = @splat(-30);
    var bias: [e]f32 = @splat(0);
    const scale: [e]f32 = @splat(2);
    for (1..r.topk) |i| l[i] = 4;
    l[20] = 1;
    l[22] = 1; // 20 and 22 tie for the last pick
    l[0] = std.math.nan(f32);
    bias[0] = 9;
    var ids: [r.topk]u32 = undefined;
    var w: [r.topk]f64 = undefined;
    r.route(&l, &bias, &scale, &ids, &w);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 20 }, &ids);
    @memset(&l, -100);
    r.route(&l, &bias, &scale, &ids, &w);
    try std.testing.expectApproxEqRel(2 * r.sigmoid(-100) / 1e-20 * 2, w[0], 1e-12);
}

test "check refuses no picks, more than the kernel holds, and logit rows narrower than both routers" {
    try r.check(.{ .rows = 300 });
    try std.testing.expectError(error.RouteShape, r.check(.{ .rows = 1, .k = 0 }));
    try std.testing.expectError(error.RouteShape, r.check(.{ .rows = 1, .k = r.most_topk + 1 }));
    try std.testing.expectError(error.RouteShape, r.check(.{ .rows = 1, .ls = 2 * e - 1 }));
}

//! CPU f64 oracle for one Flash Next MoE block: router, routed experts, shared expert, combine.

const std = @import("std");
const format = @import("../pack/exl3_format.zig");
const cpu = @import("exl3_cpu.zig");

/// One decoded EXL3 matrix W[k][n] in f64. `y[j] = sum_i x[i] * w[i * n + j]`.
pub const Dense = struct {
    k: usize,
    n: usize,
    w: []f64,

    pub fn init(gpa: std.mem.Allocator, halves: u16, trellis: []const i16, tiles_k: usize, tiles_n: usize, suh: []const f16, svh: []const f16) !Dense {
        const k = tiles_k * 16;
        const n = tiles_n * 16;
        if (k % format.hadamard_dim != 0 or n % format.hadamard_dim != 0) return error.Length;
        if (suh.len != k or svh.len != n or trellis.len != tiles_k * tiles_n * @as(usize, halves) * 8) return error.Length;
        const w = try gpa.alloc(f64, k * n);
        errdefer gpa.free(w);
        const scratch = try gpa.alloc(f64, 2 * k * n);
        defer gpa.free(scratch);
        format.dequantize(.{ .halves = halves }, .mul1, trellis, tiles_k, tiles_n, suh, svh, w, scratch);
        return .{ .k = k, .n = n, .w = w };
    }

    pub fn deinit(d: *Dense, gpa: std.mem.Allocator) void {
        gpa.free(d.w);
        d.* = undefined;
    }

    pub fn apply(d: Dense, x: []const f64, y: []f64) error{Length}!void {
        if (x.len != d.k or y.len != d.n) return error.Length;
        for (0..d.n) |j| {
            var sum: f64 = 0;
            for (0..d.k) |i| sum += x[i] * d.w[i * d.n + j];
            y[j] = sum;
        }
    }
};

/// down(silu(gate x) * up x). `gate_buf`, `up_buf`, `glu_buf` are `gate.n` long; `out` is `down.n` long.
pub fn expert(gate: Dense, up: Dense, down: Dense, x: []const f64, gate_buf: []f64, up_buf: []f64, glu_buf: []f64, out: []f64) error{Length}!void {
    if (gate.n != up.n or down.k != gate.n or down.n != gate.k) return error.Length;
    if (gate_buf.len != gate.n or up_buf.len != gate.n or glu_buf.len != gate.n) return error.Length;
    try gate.apply(x, gate_buf);
    try up.apply(x, up_buf);
    try cpu.swiglu(gate_buf, up_buf, glu_buf, std.math.inf(f64));
    try down.apply(glu_buf, out);
}

/// `logits` is `experts + 1` values, the last being the shared-gate dot.
pub fn routeFromLogits(logits: []const f64, topk: usize, picks: []u16, weights: []f64, shared_weight: *f64) error{Length}!void {
    if (logits.len < 2 or topk == 0 or topk > logits.len - 1 or picks.len != topk or weights.len != topk) return error.Length;
    const experts = logits.len - 1;
    var top = logits[0];
    for (logits[0..experts]) |v| top = @max(top, v);
    var total: f64 = 0;
    for (logits[0..experts]) |v| total += @exp(v - top);
    var taken: [1024]bool = @splat(false);
    if (experts > taken.len) return error.Length;
    var kept: f64 = 0;
    for (0..topk) |k| {
        var best: usize = 0;
        var best_v = -std.math.inf(f64);
        for (logits[0..experts], 0..) |v, i| {
            if (!taken[i] and v > best_v) {
                best_v = v;
                best = i;
            }
        }
        taken[best] = true;
        picks[k] = @intCast(best);
        weights[k] = @exp(best_v - top) / total;
        kept += weights[k];
    }
    for (weights) |*w| w.* /= kept;
    shared_weight.* = 1.0 / (1.0 + @exp(-logits[experts]));
}

/// Row-major `[rows][x.len]` f16 gate dotted with `x`, then the one-row shared gate.
pub fn logitsF16(gate: []const f16, rows: usize, shared_gate: []const f16, x: []const f64, logits: []f64) error{Length}!void {
    if (gate.len != rows * x.len or shared_gate.len != x.len or logits.len != rows + 1) return error.Length;
    for (0..rows) |r| {
        var sum: f64 = 0;
        for (x, 0..) |v, i| sum += v * @as(f64, gate[r * x.len + i]);
        logits[r] = sum;
    }
    var s: f64 = 0;
    for (x, 0..) |v, i| s += v * @as(f64, shared_gate[i]);
    logits[rows] = s;
}

test "full softmax then renormalise is the softmax over the selected logits" {
    const route = @import("exl3_route.zig");
    var rng = std.Random.DefaultPrng.init(11);
    const r = rng.random();
    for (0..20) |_| {
        var logits: [65]f64 = undefined;
        var logits32: [65]f32 = undefined;
        for (&logits, &logits32) |*a, *b| {
            const v: f32 = (r.float(f32) - 0.5) * 8;
            a.* = v;
            b.* = v;
        }
        var picks: [6]u16 = undefined;
        var w: [6]f64 = undefined;
        var sw: f64 = 0;
        try routeFromLogits(&logits, 6, &picks, &w, &sw);
        var picks32: [7]u16 = undefined;
        var w32: [7]f32 = undefined;
        var scratch: [64]f32 = undefined;
        try route.select(&logits32, 6, &picks32, &w32, &scratch);
        try std.testing.expectEqualSlices(u16, &picks, picks32[0..6]);
        var sum: f64 = 0;
        for (w, w32[0..6]) |a, b| {
            sum += a;
            // select rounds through bf16: 8 bits of mantissa, relative 2^-8
            try std.testing.expectApproxEqRel(a, @as(f64, b), 1.0 / 128.0);
        }
        try std.testing.expectApproxEqAbs(@as(f64, 1), sum, 1e-12);
        try std.testing.expectApproxEqRel(sw, @as(f64, w32[6]), 1.0 / 128.0);
        try std.testing.expectEqual(@as(u16, 64), picks32[6]);
    }
}

test "ties take the lower id and bad lengths store nothing" {
    const logits = [_]f64{ 1, 3, 3, 0, 0 };
    var picks = [_]u16{ 9, 9 };
    var w = [_]f64{ 0, 0 };
    var sw: f64 = -1;
    try routeFromLogits(&logits, 2, &picks, &w, &sw);
    try std.testing.expectEqualSlices(u16, &.{ 1, 2 }, &picks);
    try std.testing.expectEqual(@as(f64, 0.5), w[0]);
    try std.testing.expectEqual(@as(f64, 0.5), sw);
    var one = [_]u16{7};
    try std.testing.expectError(error.Length, routeFromLogits(&logits, 2, &one, &w, &sw));
    try std.testing.expectEqual(@as(u16, 7), one[0]);
}

test "logitsF16 is a row dot and the shared gate is the last logit" {
    const gate = [_]f16{ 1, 0, 0, 2, 0, 0 };
    const shared = [_]f16{ 0, 0, 3 };
    var logits: [3]f64 = undefined;
    try logitsF16(&gate, 2, &shared, &.{ 1, 5, 7 }, &logits);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 21 }, &logits);
    try std.testing.expectError(error.Length, logitsF16(&gate, 2, shared[0..2], &.{ 1, 5, 7 }, &logits));
}

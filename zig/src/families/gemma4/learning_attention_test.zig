//! An independent binary64 objective freezes each native high/low operand store, including its online-max owner.
const std = @import("std");
const core = @import("core");
const fixture = @import("../qwen4/learning_fixture.zig");
const data = @import("../qwen4/learning_contract_data.zig");
fn bf(x: f64) f64 {
    return @as(f32, @bitCast(@as(u32, core.quant.convert.bf16(@floatCast(x))) << 16));
}
fn objective(scores: []const f32, dp: []const f32, offsets: []const f64, direction: []const f64, eps: f64, groups: usize) f64 {
    var high: f64 = -std.math.inf(f64);
    for (scores, direction) |s, v| high = @max(high, s + eps * v);
    var denominator: f64 = 0;
    for (scores, direction) |s, v| denominator += @exp(s + eps * v - high);
    var numerator: f64 = 0;
    for (scores, dp, direction, offsets, 0..) |s, dy, v, offset, i| {
        var local: f64 = -std.math.inf(f64);
        for (scores[0..@min(scores.len, (i / 32 + 1) * 32)], direction[0..@min(scores.len, (i / 32 + 1) * 32)], 0..) |other, d, j| {
            if ((j / 32) % groups == (i / 32) % groups) local = @max(local, other + eps * d);
        }
        numerator += (@exp(s + eps * v - high) + offset * @exp(local - high)) * dy;
    }
    return numerator / denominator;
}
test "Gemma4 learning: CUDA high-low exponent offsets follow interleaved tile maxima in binary64 finite differences" {
    const rows = 97;
    for ([_]usize{ 2, 4 }) |groups| {
        var p = data.params(core.living_qsa, .softmax, rows, groups * 128);
        p.u[1] = 1;
        p.u[2] = 1;
        p.u[5] = 2;
        p.u[6] = @intCast(groups);
        p.f[0] = 1;
        var f = try fixture.Case(core.living_qsa).init(.softmax, p);
        defer f.deinit();
        const scores = f.floats(0)[(rows - 1) * rows ..][0..rows];
        const dp = f.floats(1)[(rows - 1) * rows ..][0..rows];
        var offsets: [rows]f64 = undefined;
        var direction: [rows]f64 = undefined;
        for (scores, dp, &direction, 0..) |*s, *v, *d, i| {
            s.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 37 + i / 7) % 103)) - 51)) / 19;
            v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11) % 23)) - 11)) / 17;
            d.* = if (i % 3 == 0) -0.75 else 0.5;
        }
        for (scores, &offsets, 0..) |s, *offset, i| {
            var maximum: f64 = -std.math.inf(f64);
            for (scores[0..@min(rows, (i / 32 + 1) * 32)], 0..) |other, j| if ((j / 32) % groups == (i / 32) % groups) {
                maximum = @max(maximum, other);
            };
            const exp = @exp(@as(f64, s) - maximum);
            offset.* = bf(exp) + bf(exp - bf(exp)) - exp;
        }
        try f.run();
        var analytic: f64 = 0;
        for (f.floats(9)[(rows - 1) * rows ..][0..rows], direction) |g, d| analytic += g * d;
        for ([_]f64{ 0.001, 0.0005, 0.00025, 0.000125 }) |eps| {
            const slope = (objective(scores, dp, &offsets, &direction, eps, groups) - objective(scores, dp, &offsets, &direction, -eps, groups)) / (2 * eps);
            try fixture.near(@floatCast(analytic), slope);
        }
    }
}
test "Gemma4 learning: GeGLU pocket forward and input pullback use the same shared family activation" {
    var p = data.params(core.living_rows, .pocket_act, 3, 64);
    p.u[12] = 1;
    var f = try data.Fixture(core.living_rows).init(std.testing.allocator, .pocket_act, p, 734, false);
    defer f.deinit();
    try f.reference();
    try f.raw.run();
    for (f.values[0], 0..) |v, i| try std.testing.expect(@abs(f.output(8, i) - v.value) <= v.err);
    for ([_]core.living_rows.Op{ .pocket_coeff, .pocket_back, .pocket_add }) |op| {
        var g = try data.Fixture(core.living_rows).init(std.testing.allocator, op, p, 936, false);
        defer g.deinit();
        try g.reference();
        try g.raw.run();
        for (g.values, 8..) |values, i| for (values, 0..) |v, j| try std.testing.expect(@abs(g.output(i, j) - v.value) <= v.err);
    }
}

fn pocketObjective(x: []const f64, w: []const f32, dy: []const f32, scale: f64) f64 {
    var out: f64 = 0;
    for (0..2) |j| {
        var g: f64 = 0;
        var u: f64 = 0;
        var d: f64 = 0;
        for (x, dy, 0..) |v, grad, i| {
            g += v * w[j * 64 + i];
            u += v * w[64 * 64 + j * 64 + i];
            d += grad * w[2 * 64 * 64 + j * 64 + i];
        }
        out += (if (j == 0) @as(f64, 1) else scale) * d * 0.5 * g * (1 + std.math.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))) * u;
    }
    return out;
}
test "Gemma4 learning: native GeGLU pocket input adjoint matches independent binary64 central slopes" {
    var p = data.params(core.living_rows, .pocket_coeff, 1, 64);
    p.u[12] = 1;
    p.u[3] = 1;
    p.f[0] = 0.75;
    var coeff = try data.Fixture(core.living_rows).init(std.testing.allocator, .pocket_coeff, p, 0x2148, false);
    defer coeff.deinit();
    var pull = try fixture.Case(core.living_rows).init(.pocket_back, p);
    defer pull.deinit();
    @memset(std.mem.bytesAsSlice(u16, coeff.raw.data[0]), 0);
    std.mem.writeInt(u16, coeff.raw.data[0][0..2], core.quant.convert.bf16(0.75), .little);
    std.mem.writeInt(u16, coeff.raw.data[0][2..4], core.quant.convert.bf16(-0.5), .little);
    std.mem.writeInt(u16, coeff.raw.data[0][4..6], core.quant.convert.bf16(1), .little);
    @memset(coeff.raw.floats(1), 0);
    const w = coeff.raw.floats(1);
    w[0] = 1;
    w[65] = -1;
    w[4096 + 2] = 0.5;
    w[4096 + 64] = -0.5;
    w[8192] = 1;
    w[8192 + 65] = 0.75;
    @memset(coeff.raw.floats(2), 0);
    coeff.raw.floats(2)[0] = 0.75;
    coeff.raw.floats(2)[1] = -0.5;
    try coeff.raw.run();
    try pull.put(0, coeff.raw.data[8][0..(try core.living_rows.needs(.pocket_coeff, p)).bytes[8]]);
    try pull.put(1, coeff.raw.data[1]);
    try pull.run();
    var x: [64]f64 = undefined;
    for (&x, 0..) |*v, i| v.* = coeff.input(0, i);
    const weights = coeff.raw.floats(1);
    const dy = coeff.raw.floats(2)[0..64];
    for (0..64) |i| {
        const original = x[i];
        for ([_]f64{ 0.0001, 0.00005 }) |eps| {
            x[i] = original + eps;
            const high = pocketObjective(&x, weights, dy, 0.75);
            x[i] = original - eps;
            const low = pocketObjective(&x, weights, dy, 0.75);
            x[i] = original;
            try fixture.near(pull.floats(8)[i], (high - low) / (2 * eps));
        }
    }
}

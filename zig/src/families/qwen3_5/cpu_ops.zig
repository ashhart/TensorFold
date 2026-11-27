//! The Qwen3.5-2B rows between the registered kernels: each output computed in float64, rounded to bf16 via fp32.
const std = @import("std");
const c = @import("config.zig");
const k = @import("cpu_kernels.zig");

/// The registered kernels these rows sit between.
pub const kernels = k;
pub const f32of = k.f32of;
pub const bf16of = k.bf16of;

/// 1 / (1 + e^-x) in float64.
pub fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

/// x · sigmoid(x) in float64.
pub fn silu(x: f64) f64 {
    return x / (1.0 + @exp(-x));
}

/// A float64 value rounded to fp32, then to bf16 bits: at a bf16 tie this can sit one ulp from a direct rounding.
pub fn bf16of64(x: f64) u16 {
    return bf16of(@floatCast(x));
}

/// An embedding row's codes dequantized: bf16(scale · code + bias), the group's table entries exact in float64.
pub fn embed(e: k.Linear, id: u32, out: []u16) void {
    for (0..c.hidden / k.group) |g| {
        const s: f64 = f32of(e.scale(id, g));
        const z: f64 = f32of(e.bias(id, g));
        for (0..k.group) |i| out[g * k.group + i] = bf16of64(s * @as(f64, @floatFromInt(e.code(id, g * k.group + i))) + z);
    }
}

/// The residual stream's add: h = bf16(h + r), the sum exact in float64 before its rounding through fp32.
pub fn residual(h: []u16, r: []const u16) void {
    for (h, r) |*hb, rb| hb.* = bf16of64(@as(f64, f32of(hb.*)) + @as(f64, f32of(rb)));
}

/// The MLP's activation over a row: bf16(silu(gate) · up).
pub fn mlpAct(gate: []const u16, up: []const u16, out: []u16) void {
    for (out, gate, up) |*o, g, u| o.* = bf16of64(silu(f32of(g)) * @as(f64, f32of(u)));
}

/// The attention output gate over a row: each head's values times the sigmoid of its gate half of q_proj's row.
pub fn attentionGate(y: []const u16, qraw: []const u16, out: []u16) void {
    for (0..c.hidden) |d| {
        const head = d / c.head_dim;
        const gate: f64 = f32of(qraw[head * 2 * c.head_dim + c.head_dim + d % c.head_dim]);
        out[d] = bf16of64(@as(f64, f32of(y[d])) * sigmoid(gate));
    }
}

/// theta^(-2i/rotary_dim) in float64: the rotary's inverse frequency for pair i.
pub fn inverseFrequency(i: usize) f64 {
    return std.math.pow(f64, c.theta, -@as(f64, @floatFromInt(2 * i)) / @as(f64, c.rotary_dim));
}

pub const SinCos = struct { cs: f64, sn: f64 };

/// cos and sin of position · inverse frequency i, the angle in float64.
pub fn rotary(position: u64, i: usize) SinCos {
    const angle = @as(f64, @floatFromInt(position)) * inverseFrequency(i);
    return .{ .cs = @cos(angle), .sn = @sin(angle) };
}

/// The first `rotary_dim` of a head's 256 rotated as (i, i + rotary_dim / 2) pairs at `position`, the rest copied.
pub fn rotate(x: []const u16, position: u64, out: []u16) void {
    const half = c.rotary_dim / 2;
    for (0..half) |i| {
        const r = rotary(position, i);
        const a: f64 = f32of(x[i]);
        const b: f64 = f32of(x[i + half]);
        out[i] = bf16of64(a * r.cs - b * r.sn);
        out[i + half] = bf16of64(b * r.cs + a * r.sn);
    }
    @memcpy(out[c.rotary_dim..c.head_dim], x[c.rotary_dim..c.head_dim]);
}

test "a bf16 embedding row dequantizes each nibble with its group's scale and bias" {
    var words: [8 * c.hidden / 8]u32 = undefined;
    for (&words) |*w| w.* = 0x76543210;
    var scales: [8 * c.hidden / k.group]u16 = undefined;
    var biases: [8 * c.hidden / k.group]u16 = undefined;
    for (&scales, &biases, 0..) |*s, *b, i| {
        s.* = bf16of(0.5);
        b.* = bf16of(@as(f32, @floatFromInt(i % (c.hidden / k.group))));
    }
    const e = try k.pack(std.testing.allocator, &words, &scales, &biases, 8, c.hidden, 8);
    defer k.free(std.testing.allocator, e);
    var out: [c.hidden]u16 = undefined;
    embed(e, 5, &out);
    try std.testing.expectEqual(bf16of(0.5 * 0 + 0), out[0]);
    try std.testing.expectEqual(bf16of(0.5 * 1 + 0), out[1]);
    try std.testing.expectEqual(bf16of(0.5 * 7 + 0), out[7]);
    try std.testing.expectEqual(bf16of(0.5 * 1 + 31), out[2047 - 6]);
}

test "the pointwise rows round through fp32: residual, activation and gate agree with float64 to half a bf16 ulp" {
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    for (0..10_000) |_| {
        const a = bf16of(rnd.floatNorm(f32) * 4);
        const b = bf16of(rnd.floatNorm(f32) * 4);
        var h = [_]u16{a};
        residual(&h, &.{b});
        try std.testing.expectEqual(bf16of64(@as(f64, f32of(a)) + @as(f64, f32of(b))), h[0]);
        var act: [1]u16 = undefined;
        mlpAct(&.{a}, &.{b}, &act);
        const want = silu(f32of(a)) * @as(f64, f32of(b));
        try std.testing.expect(@abs(@as(f64, f32of(act[0])) - want) <= 0x1p-8 * @abs(want) + 0x1p-133);
    }
    try std.testing.expectEqual(@as(f64, 0.5), sigmoid(0));
    try std.testing.expectEqual(@as(f64, 0), silu(0));
    try std.testing.expectEqual(@as(f64, 1), inverseFrequency(0));
    const r = rotary(0, 5);
    try std.testing.expectEqual(@as(f64, 1), r.cs);
    try std.testing.expectEqual(@as(f64, 0), r.sn);
}

test "the rotation keeps a head's norm and leaves the unrotated dims as they were" {
    var x: [c.head_dim]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(4);
    const rnd = prng.random();
    for (&x) |*v| v.* = bf16of(rnd.floatNorm(f32));
    var out: [c.head_dim]u16 = undefined;
    rotate(&x, 1234, &out);
    try std.testing.expectEqualSlices(u16, x[c.rotary_dim..], out[c.rotary_dim..]);
    var before: f64 = 0;
    var after: f64 = 0;
    for (x[0..c.rotary_dim], out[0..c.rotary_dim]) |a, b| {
        before += @as(f64, f32of(a)) * f32of(a);
        after += @as(f64, f32of(b)) * f32of(b);
    }
    try std.testing.expect(@abs(before - after) <= 0x1p-6 * before);
    rotate(&x, 0, &out);
    try std.testing.expectEqualSlices(u16, &x, &out);
}

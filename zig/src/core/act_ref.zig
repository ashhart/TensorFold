//! Activation rows in float64, the reference every backend's activation kernels are checked against.
const std = @import("std");
const bf = @import("bf16_round.zig");

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// silu(gate) * up.
pub fn swiglu(gate: f64, up: f64) f64 {
    return gate * sigmoid(gate) * up;
}

/// DFlash2's dynamic conv at one value: x k0 + prev k1 with the taps as rounded (k_t = bf16(base_t + dyn_t)).
pub fn dconv(x: f64, prev: f64, k0: f64, k1: f64) f64 {
    return x * k0 + prev * k1;
}

/// A 4-bit affine table's value: q * scale + bias.
pub fn affine4(q: u4, scale: f64, bias: f64) f64 {
    return @as(f64, @floatFromInt(q)) * scale + bias;
}

/// mlx_lm's gelu_approx constants, sqrt(2 / pi) and 0.044715, as the fp32 values the kernels round to bf16.
pub const gelu_c0: f32 = 0.7978845608028654;
pub const gelu_c1: f32 = 0.044715;

/// MLX's geglu at its bf16 rounding points (Metal gemma4/glue.metal's gemma_geglu), tanh's value times (1 + t).
pub fn geglu(x: f64, up: f64, t: f64) f64 {
    const cube = bf.bf16(x * x * x); // exact before its rounding for a bf16 x
    const inner = bf.bf16(x + bf.bf16(bf.bf16(gelu_c1) * cube));
    const th = bf.bf16(std.math.tanh(bf.bf16(bf.bf16(gelu_c0) * inner)) * (1 + t));
    return bf.bf16(bf.bf16(bf.bf16(0.5 * x) * bf.bf16(1 + th)) * up);
}

/// tf_draft_swiglu's bf16 steps (Metal core/draft_ops.metal), exp's value times (1 + te), the reciprocal's (1 + td).
pub fn swigluRound(g: f64, up: f64, te: f64, td: f64) f64 {
    const e = bf.bf16(@exp(@abs(g)) * (1 + te));
    const inv = bf.bf16(1 / bf.bf16(1 + e) * (1 + td));
    const sig = if (g < 0) inv else bf.bf16(1 - inv);
    return bf.bf16(bf.bf16(g * sig) * up);
}

/// MLX's soft cap at its points: bf16(fp32(bf16(tanh(bf16(fp32(x / cap))) (1 + t)) cap)), fp32 first as the kernel.
pub fn softcap(x: f64, cap: f64, t: f64) f64 {
    const q = bf.bf16(@as(f32, @floatCast(x / cap)));
    const th = bf.bf16(std.math.tanh(q) * (1 + t));
    return bf.bf16(@as(f32, @floatCast(th * cap)));
}

/// softcap's float64 ends with tanhf within `spread` either way (monotonic in t).
pub fn softcapRange(x: f64, cap: f64, spread: f64) [2]f64 {
    const a = softcap(x, cap, -spread);
    const b = softcap(x, cap, spread);
    if (std.math.isNan(a) or std.math.isNan(b)) return @splat(std.math.nan(f64));
    return .{ @min(a, b), @max(a, b) };
}

/// One of the two at its float64 ends: every fp32 libm value (tanh, exp) and division within `spread` either way.
pub fn mlxRange(geglu_: bool, x: f64, up: f64, spread: f64) [2]f64 {
    var lo = std.math.inf(f64);
    var hi = -std.math.inf(f64);
    for ([_]f64{ -spread, spread }) |a| for ([_]f64{ -spread, spread }) |b| {
        const v = if (geglu_) geglu(x, up, a) else swigluRound(x, up, a, b);
        if (std.math.isNan(v)) return @splat(std.math.nan(f64));
        lo = @min(lo, v);
        hi = @max(hi, v);
    };
    return .{ lo, hi };
}

/// A bf16 output of fp32 arithmetic through expf: half a bf16 ulp plus 2^-18 of |want|.
pub fn tolerance(want: f64) f64 {
    return @abs(want) * (std.math.pow(f64, 2, -8) + std.math.pow(f64, 2, -18)) + 1e-30;
}

test "activations against hand values" {
    try std.testing.expectEqual(@as(f64, 0.5), sigmoid(0));
    try std.testing.expectEqual(@as(f64, 7 * 0.5 - 1), affine4(7, 0.5, -1));
    try std.testing.expectApproxEqRel(@as(f64, 2 * sigmoid(2) * 3), swiglu(2, 3), 1e-15);
    try std.testing.expect(tolerance(1) > tolerance(0.5));
}

test "MLX's steps: rounding points against hand values, the ranges ordered, and both near the once-rounded forms" {
    try std.testing.expectEqual(@as(f64, 0), geglu(0, 3, 0));
    try std.testing.expectEqual(@as(f64, 0), swigluRound(0, 1, 0, 0));
    try std.testing.expectEqual(bf.bf16(-2 * bf.bf16(1 / bf.bf16(1 + bf.bf16(@exp(2.0))))), swigluRound(-2, 1, 0, 0)); // g < 0: 1 / (1 + e)
    try std.testing.expectEqual(bf.bf16(2 * bf.bf16(1 - bf.bf16(1 / bf.bf16(1 + bf.bf16(@exp(2.0)))))), swigluRound(2, 1, 0, 0));
    var prng = std.Random.DefaultPrng.init(9);
    for (0..2000) |_| {
        const x = bf.bf16(prng.random().float(f64) * 8 - 4);
        const up = bf.bf16(prng.random().float(f64) * 4 - 2);
        for ([_]bool{ true, false }) |g| {
            const r = mlxRange(g, x, up, 0x1p-21);
            try std.testing.expect(r[0] <= r[1]);
            const once = if (g) 0.5 * x * (1 + std.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))) * up else swiglu(x, up);
            try std.testing.expect(@abs(r[0] - once) <= 0x1p-5 * @abs(x * up) + 0x1p-10); // a step of 1 + tanh: 2^-9 of x up
        }
    }
}

test "the soft cap: rounding points against hand values, ends ordered, within a step of cap tanh(x / cap)" {
    try std.testing.expectEqual(@as(f64, 0), softcap(0, 30, 0));
    try std.testing.expectEqual(@as(f64, 30), softcap(1e6, 30, 0)); // tanh rounds to 1
    try std.testing.expectEqual(-softcap(7.25, 30, 0), softcap(-7.25, 30, 0));
    var prng = std.Random.DefaultPrng.init(3);
    for (0..4000) |_| {
        const x = bf.bf16(prng.random().float(f64) * 120 - 60);
        const r = softcapRange(x, 30, 0x1p-21);
        try std.testing.expect(r[0] <= r[1]);
        try std.testing.expect(@abs(r[0] - 30 * std.math.tanh(x / 30)) <= 0x1p-6 * 30);
    }
}

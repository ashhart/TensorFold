//! fn_int4.cu's output allowance: its sums through the gate/up epilogue, that epilogue's bf16 roundings and the store.
const std = @import("std");
const core = @import("core");

const c = core.contract.spec;

/// bf16's unit roundoff: half an ulp at 1, the most one rounding moves a value.
const unit_bf16 = 0x1p-8;
/// fp32's unit roundoff.
const unit_f32 = 0x1p-24;
/// expf overflows fp32 above 88.72: there the kernel's silu is 0 where float64 keeps a denormal-sized value.
const exp_overflow = 88.0;
/// The least and the most gate or up sum gate 4's stress rows give the gate/up form: eight codes of 1 to 3, nudge 4.
const stress_least = 32;
const stress_step = 4;
const stress_count = 17;

/// silu as float64 keeps it.
pub fn silu(g: f64) f64 {
    return g / (1 + @exp(-g));
}

/// silu's slope, sigmoid(x) (1 + x (1 - sigmoid(x))): at most 1.0998, at x = 2.3994.
fn slopeAt(x: f64) f64 {
    const sg = 1 / (1 + @exp(-x));
    return sg * (1 + x * (1 - sg));
}

/// The largest |silu'| on [lo, hi]: its ends, and its two turning points where they lie inside.
fn slopeOn(lo: f64, hi: f64) f64 {
    var m = @max(@abs(slopeAt(lo)), @abs(slopeAt(hi)));
    for ([_]f64{ 2.3993572805, -2.3993572805 }) |t| {
        if (lo <= t and t <= hi) m = @max(m, @abs(slopeAt(t)));
    }
    return m;
}

/// bf16 round to nearest even, as the kernel's bf() does through fp32.
fn bf(x: f64) f64 {
    return @as(f64, @floatCast(c.fromBf16(c.toBf16(@floatCast(x)))));
}

/// fp32 to bf16 toward zero: the kernel's bf() with its rounding mode wrong, the epilogue the control `cut` runs.
fn cut(x: f64) f64 {
    const bits: u32 = @bitCast(@as(f32, @floatCast(x)));
    return @as(f64, @floatCast(c.fromBf16(@truncate(bits >> 16))));
}

/// One gate/up allowance from its own terms: both sums, bf(g) through silu's slope, bf(u), bf(silu), the store.
pub fn gateUp(depth: usize, ops: f64, g: f64, u: f64, abs_g: f64, abs_u: f64, want: f64, exact: bool) f64 {
    const gam: f64 = if (exact) 0 else c.gamma(@floatFromInt(depth), unit_f32);
    // |bf(x + e) - x|: the sum's own error, then the rounding of the value it landed on
    const dg = unit_bf16 * @abs(g) + gam * abs_g * (1 + unit_bf16);
    const du = unit_bf16 * @abs(u) + gam * abs_u * (1 + unit_bf16);
    const s = @abs(silu(g));
    const d = slopeOn(g - dg, g + dg);
    const span = @abs(u) + du;
    // through the slope: the sum's error and bf(g) together, silu's rounding, then the up's error and rounding
    var epi = (d * dg + unit_bf16 * s) * (1 + 2 * unit_bf16) * span + (s + d * dg) * du;
    if (g - dg < -exp_overflow) epi += (s + d * dg) * span;
    const work = epi + ops * unit_f32 * @abs(want);
    return work + unit_bf16 * (@abs(want) + work) + 0x1p-133;
}

/// One down or head output's allowance: the sum's error and the store's rounding, nothing else.
pub fn rows(depth: usize, abs: f64, want: f64, out: c.Elem, exact: bool) f64 {
    const gam: f64 = if (exact) 0 else c.gamma(@floatFromInt(depth), unit_f32);
    const unit: f64 = if (out == .bf16) unit_bf16 else unit_f32;
    const work = gam * abs;
    return work + unit * (@abs(want) + work) + (if (out == .bf16) @as(f64, 0x1p-133) else @as(f64, 0x1p-149));
}

/// The kernel's own arithmetic on one output: bf(g), bf(u), bf(silu), the product, the store.
fn epilogue(g: f64, u: f64) f64 {
    const gv = bf(g);
    return bf(bf(gv / (1 + @exp(-gv))) * bf(u));
}

/// The epilogue the control `cut` runs: the same with gate, up and silu cut toward zero.
fn epilogueCut(g: f64, u: f64) f64 {
    const gv = cut(g);
    return bf(cut(gv / (1 + @exp(-gv))) * cut(u));
}

fn fixedDepth(_: c.Shape) usize {
    return 45;
}

const gu_spec: c.Spec = .{ .name = "t", .accum = .f32, .depth = fixedDepth, .ops = 5, .mids = 3, .out = .bf16, .served = &.{} };

test "the derived allowance covers every epilogue rounding the generic bound does not" {
    const spec: c.Spec = .{ .name = "t", .accum = .f32, .depth = fixedDepth, .ops = 5, .mids = 3, .out = .bf16, .served = &.{} };
    var prng = std.Random.DefaultPrng.init(0x696e7434);
    const rng = prng.random();
    var short: usize = 0;
    for (0..4000) |_| {
        const g = rng.float(f64) * 60 - 30;
        const u = rng.float(f64) * 60 - 30;
        const want = silu(g) * u;
        const got = epilogue(g, u);
        const allow = gateUp(45, 5, g, u, @abs(g), @abs(u), want, true);
        try std.testing.expect(@abs(got - want) <= allow);
        if (@abs(got - want) > c.boundExact(spec, want)) short += 1;
    }
    try std.testing.expect(short > 0);
}

test "the generic bound is no bound for the epilogue: one stated gate with exact sums lands twice outside it" {
    const g: f64 = -10.53;
    const u: f64 = -28.75;
    const want = silu(g) * u;
    const generic = c.boundExact(gu_spec, want);
    // bf(g) rounds the gate at the gate's own size, and that rounding alone carries the output past the bound
    try std.testing.expect(bf(g) == -10.5);
    try std.testing.expect(@abs(silu(bf(g)) * u - want) > generic);
    try std.testing.expect(@abs(epilogue(g, u) - want) > 2 * generic);
    try std.testing.expect(@abs(epilogue(g, u) - want) <= gateUp(45, 5, g, u, @abs(g), @abs(u), want, true));
}

test "gate 4's bf16 proof of a bf16 SwiGLU output stops at 1 / (4.0393 u): past every stress sum for 100, inside for 60" {
    var least: f64 = std.math.inf(f64);
    var most: f64 = 0;
    for (0..stress_count) |i| {
        const g: f64 = @floatFromInt(stress_least + stress_step * i);
        for (0..stress_count) |j| {
            const u: f64 = @floatFromInt(stress_least + stress_step * j);
            const want = silu(g) * u;
            const allow = gateUp(216, 5, g, u, g * 128, u * 128, want, true);
            // a bf16 sum that returns 0 misses by want itself: the most a mirrored row can prove of that output
            least = @min(least, want / allow);
            most = @max(most, want / allow);
            try std.testing.expect(want / c.boundExact(gu_spec, want) < 64);
        }
    }
    try std.testing.expect(most - least < 0.01);
    try std.testing.expect(least > 60 and most < 64);
}

test "the planted gate/up faults sit outside the allowance by arithmetic: a cut epilogue on some draws, a swap on most" {
    var prng = std.Random.DefaultPrng.init(0x696e7434);
    const rng = prng.random();
    var cut_out: usize = 0;
    var swap_out: usize = 0;
    for (0..4000) |_| {
        const g = rng.float(f64) * 60 - 30;
        const u = rng.float(f64) * 60 - 30;
        const want = silu(g) * u;
        const allow = gateUp(45, 5, g, u, @abs(g), @abs(u), want, true);
        if (@abs(epilogueCut(g, u) - want) > allow) cut_out += 1;
        if (@abs(epilogue(u, g) - want) > allow) swap_out += 1;
    }
    try std.testing.expect(cut_out > 20);
    try std.testing.expect(swap_out > 1500);
}

test "a down's allowance is the store's rounding of an exact sum" {
    const a = rows(45, 96, 96, .bf16, true);
    try std.testing.expect(a > 0.9 * unit_bf16 * 96);
    try std.testing.expect(a < 2 * unit_bf16 * 96);
}

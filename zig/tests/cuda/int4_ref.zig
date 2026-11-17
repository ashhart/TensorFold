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

/// One gate/up output's allowance: bf(g), bf(u), bf(silu) and the store at their own sizes, and both sums' error.
pub fn gateUp(depth: usize, ops: f64, g: f64, u: f64, abs_g: f64, abs_u: f64, want: f64, exact: bool) f64 {
    const gam: f64 = if (exact) 0 else c.gamma(@floatFromInt(depth), unit_f32);
    // |bf(x + e) - x|: the sum's own error, then the rounding of the value it landed on
    const dg = unit_bf16 * @abs(g) + gam * abs_g * (1 + unit_bf16);
    const du = unit_bf16 * @abs(u) + gam * abs_u * (1 + unit_bf16);
    const s = @abs(silu(g));
    const d = slopeOn(g - dg, g + dg);
    const span = @abs(u) + du;
    const mag = d * abs_g * span + abs_u * (s + d * dg);
    var epi = (d * dg + unit_bf16 * s) * (1 + 2 * unit_bf16) * span + s * du;
    if (g - dg < -exp_overflow) epi += (s + d * dg) * span;
    const work = gam * mag + epi + ops * unit_f32 * @abs(want);
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

fn fixedDepth(_: c.Shape) usize {
    return 45;
}

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

test "a down's allowance is the store's rounding of an exact sum" {
    const a = rows(45, 96, 96, .bf16, true);
    try std.testing.expect(a > 0.9 * unit_bf16 * 96);
    try std.testing.expect(a < 2 * unit_bf16 * 96);
}

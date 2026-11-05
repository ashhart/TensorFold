// The f32 activation pipelines' want formulas: no store rounding inside the wide layers, one RNE bf16 store
// in each transition. Same case tables and bf16 inputs as stage2_fixture_ops; GPU probes compare against these.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const ops = @import("stage2_fixture_ops.zig");

const testing = std.testing;

/// add_f32: the exact f32 sum, stored unrounded.
fn addF32Want(a_bits: u16, b_bits: u16) f32 {
    return c.fromBf16(a_bits) + c.fromBf16(b_bits);
}

/// add_wide_bf16: the f32 stream plus the widened bf16 residual, rounded once to bf16 on the store.
fn addWideWant(a_f32: f32, b_bits: u16) u16 {
    return c.toBf16(a_f32 + c.fromBf16(b_bits));
}

/// add_bf16_f32: the widened bf16 stream joins the f32 residual, the f32 sum stays unrounded.
fn addWideF32Want(a_bits: u16, b_f32: f32) f32 {
    return c.fromBf16(a_bits) + b_f32;
}

test "add f32 stores the f32 sum; the narrow pipeline is its one-store rounding" {
    for (ops.class_pairs) |pair| {
        const f64_sum = @as(f64, c.fromBf16(pair[0])) + @as(f64, c.fromBf16(pair[1]));
        const wide = addF32Want(pair[0], pair[1]);
        if (!std.math.isFinite(wide)) {
            // inf - inf and nan classes agree by class, payloads vary
            try testing.expect(!std.math.isFinite(@as(f32, @floatCast(f64_sum))));
            continue;
        }
        // the f32 add is the f64-exact sum rounded once to f32 (both operands are bf16, hence f32-exact)
        try testing.expectEqual(@as(f32, @floatCast(f64_sum)), wide);
        // the bf16 pipeline's want is exactly the f32 sum's store rounding
        try testing.expectEqual(c.toBf16(wide), c.toBf16(@floatCast(f64_sum)));
    }
}

test "cancel pairs stay exact under the f32 store" {
    var prng = std.Random.DefaultPrng.init(0x5702);
    const rng = prng.random();
    for (0..32) |i| {
        const vbits = ops.sampleBits(rng, 1);
        const v = c.fromBf16(vbits);
        const k: f32 = @floatFromInt(i % 3);
        const ulp = std.math.pow(f32, 2, @floor(std.math.log2(@abs(v))) - 7);
        const nbits = c.toBf16(v - k * ulp);
        // the near-cancelling sum is bf16-representable, so both pipelines are bit-exact on it
        try testing.expectEqual(c.fromBf16(nbits) + v, addF32Want(vbits, nbits));
    }
}

test "the mixed transitions round exactly once, on the side whose store holds bf16" {
    const a: f32 = 1.00390625; // not bf16-representable
    const b_bits = c.toBf16(0.0625);
    // wide_bf16: the f32 sum rounds on the bf16 store
    try testing.expectEqual(c.toBf16(a + 0.0625), addWideWant(a, b_bits));
    // wide_f32: the bf16 side widens losslessly and the f32 store keeps the sum
    const a_bits = c.toBf16(1.0078125);
    const b: f32 = 1.0 / 3.0;
    try testing.expectEqual(c.fromBf16(a_bits) + b, addWideF32Want(a_bits, b));
}

test "swiglu f32 keeps the ASH single-round formula in f32 without a store rounding" {
    for (ops.swiglu_gate_controls, 0..) |g_bits, i| {
        const u_bits = ops.swiglu_up_controls[i];
        const g32: f32 = c.fromBf16(g_bits);
        const u32v: f32 = c.fromBf16(u_bits);
        if (!std.math.isFinite(g32) or !std.math.isFinite(u32v)) continue;
        const want32 = g32 / (1 + @exp(-g32)) * u32v;
        // the f32 result re-rounds onto the bf16 pipeline's want within one bf16 ulp of the value
        try testing.expectApproxEqAbs(@as(f64, c.fromBf16(c.toBf16(want32))), @as(f64, want32), 0.004 * @abs(want32));
    }
}

test "rms f32 stores the unrounded norm and f32_in rounds once at the store" {
    var prng = std.Random.DefaultPrng.init(0x5708);
    const rng = prng.random();
    const width = 64;
    var x: [width]f32 = undefined;
    var sum_sq: f32 = 0;
    for (&x) |*v| {
        v.* = c.fromBf16(ops.sampleBits(rng, 1));
        sum_sq += v.* * v.*;
    }
    const inv = 1.0 / std.math.sqrt(sum_sq / @as(f32, width) + ops.rms_eps);
    const gamma: f32 = c.fromBf16(c.toBf16(0.75));
    for (x) |v| {
        const wide = gamma * (v * inv);
        // the f32 store keeps the value; the f32_in variant's bf16 store rounds it once
        try testing.expectApproxEqAbs(@as(f64, c.fromBf16(c.toBf16(wide))), @as(f64, wide), 0.004 * @abs(wide));
    }
}

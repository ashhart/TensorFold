//! MultiMax's host forms: the fp32 steps unfused, the rules for NaN and zero, the bf16 coefficients, float64.
const std = @import("std");
const mm = @import("multimax.zig");

const k: mm.Coeffs = .{ .ranges = .{ -12.5, 15.25, -20, 25 }, .ts = .{ 0.375, -0.5, 0.01953125, -0.0078125 } };

fn fused(x: f32, c: mm.Coeffs) f32 {
    const a = mm.relu(c.ranges[0] - x);
    const b = mm.relu(x - c.ranges[1]);
    const lo = mm.relu(c.ranges[2] - x);
    const hi = mm.relu(x - c.ranges[3]);
    var y = @mulAdd(f32, c.ts[0], a, x);
    y = @mulAdd(f32, c.ts[1], b, y);
    y = @mulAdd(f32, c.ts[2], lo * lo, y);
    return @mulAdd(f32, c.ts[3], hi * hi, y);
}

test "each step rounds once in the vendor's order: an fma per term gives another fp32 here" {
    const x: f32 = @bitCast(@as(u32, 0xc23dfa66));
    try std.testing.expectEqual(@as(u32, 0xc19cdb08), @as(u32, @bitCast(mm.apply(x, k))));
    try std.testing.expect(@as(u32, @bitCast(fused(x, k))) != 0xc19cdb08);
}

test "relu keeps NaN and turns -0 into +0, as torch.relu" {
    try std.testing.expect(std.math.isNan(mm.relu(std.math.nan(f32))));
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(mm.relu(@as(f32, -0.0)))));
    try std.testing.expectEqual(@as(f32, 0), mm.relu(-3));
    try std.testing.expectEqual(@as(f32, 2.5), mm.relu(2.5));
}

test "non-finite logits take float64's classes" {
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |x| {
        const got = mm.apply(x, k);
        const want = mm.reference(x, k).value;
        try std.testing.expectEqual(std.math.isNan(want), std.math.isNan(got));
        if (!std.math.isNan(want)) try std.testing.expectEqual(want, @as(f64, got));
    }
    // +Inf meets t1 < 0: Inf + -Inf
    try std.testing.expect(std.math.isNan(mm.apply(std.math.inf(f32), k)));
}

test "bf16 coefficients widen to fp32 exactly" {
    const c = mm.Coeffs.fromBf16(.{ 0xc148, 0x4174, 0xc1a0, 0x41c8 }, .{ 0x3ec0, 0xbf00, 0x3ca0, 0xbc00 });
    try std.testing.expectEqualSlices(f32, &k.ranges, &c.ranges);
    try std.testing.expectEqualSlices(f32, &k.ts, &c.ts);
}

test "the fp32 steps stay within float64's value by gamma(8) of the terms' magnitude" {
    var prng = std.Random.DefaultPrng.init(0x6d6d);
    const g8 = 8 * 0x1p-24 / (1 - 8 * 0x1p-24);
    for (0..100_000) |_| {
        const x: f32 = (prng.random().float(f32) * 2 - 1) * 48;
        const r = mm.reference(x, k);
        try std.testing.expect(@abs(@as(f64, mm.apply(x, k)) - r.value) <= g8 * r.mag);
    }
}

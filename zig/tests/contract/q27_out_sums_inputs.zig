//! Stored-BF16 sum fixtures prove reduced-accumulator discrimination before any GPU run.
const std = @import("std");
const ct = @import("contract");
const c = ct.spec;
pub fn stress(seed: u64) ![64]u16 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (0..256) |_| {
        var values: [64]f64 = undefined;
        for (values[0..32]) |*v| v.* = @as(f64, @floatFromInt(1 + rng.uintLessThan(u32, 64))) * 8192;
        std.mem.sort(f64, values[0..32], {}, std.sort.desc(f64));
        for (0..32) |i| values[32 + i] = -values[31 - i] + (if (i == 0) @as(f64, 8192) else 0);
        var good = true;
        for ([_]ct.rows.Order{ .sequential, .pairwise, .strided }) |order| {
            if (ct.rows.sumIn(.f32, order, &values) != 8192) good = false;
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                const v = ct.rows.sumIn(acc, order, &values);
                if (std.math.isFinite(v) and @abs(v - 8192) < 1) good = false;
            }
        }
        if (good) {
            var bits: [64]u16 = undefined;
            for (&bits, values) |*b, v| b.* = c.toBf16(@floatCast(v));
            return bits;
        }
    }
    return error.NoDiscriminatingOutSumsFixture;
}

test "q27 out sums: stored BF16 terms distinguish FP16 and BF16 in all accumulator orders" {
    for ([_]u64{ 0, 1, 7, 17, 91, 2026 }) |seed| {
        const bits = try stress(seed);
        var values: [64]f64 = undefined;
        for (&values, bits) |*v, b| v.* = c.fromBf16(b);
        for ([_]ct.rows.Order{ .sequential, .pairwise, .strided }) |order| {
            try std.testing.expectEqual(@as(f64, 8192), ct.rows.sumIn(.f32, order, &values));
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                const v = ct.rows.sumIn(acc, order, &values);
                try std.testing.expect(!std.math.isFinite(v) or @abs(v - 8192) >= 1);
            }
        }
    }
}

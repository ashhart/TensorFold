//! Independent FP64 logsumexp and exact selected/top/rank oracle for the native score-row ABI.
const std = @import("std");
const c = @import("core").contract.spec;
pub const words = 43;
pub const top = 20;
pub const Row = struct { bits: [words]u32, lse: f64 };
pub fn reductionDepth(shape: c.Shape) usize {
    return (shape.n + 255) / 256 + 8;
}
pub fn stress(logits: []f32) void {
    for (logits, 0..) |*x, i| x.* = if (i % 2 == 0) 0 else -@log(@as(f32, 2048));
}
pub fn row(logits: []const f32, pick: u32, k: usize) Row {
    var out: Row = .{ .bits = @splat(0), .lse = 0 };
    var max: f64 = -std.math.inf(f64);
    var nan = false;
    for (logits) |x| {
        nan = nan or std.math.isNan(x);
        if (x > max) max = x;
    }
    var sum: f64 = 0;
    if (std.math.isFinite(max)) for (logits) |x| {
        sum += @exp(@as(f64, x) - max);
    };
    out.lse = if (nan) std.math.nan(f64) else if (std.math.isInf(max)) max else max + @log(sum);
    out.bits[0] = @bitCast(@as(f32, @floatCast(out.lse)));
    const selected: f32 = if (pick < logits.len) logits[pick] else std.math.nan(f32);
    out.bits[1] = @bitCast(selected);
    for (logits) |x| {
        out.bits[42] += @intFromBool(x >= selected);
    }
    var prior: f32 = std.math.inf(f32);
    var prior_id: u32 = 0;
    for (0..k) |j| {
        var best: f32 = -std.math.inf(f32);
        var id: u32 = std.math.maxInt(u32);
        for (logits, 0..) |x, i| if ((j == 0 or x < prior or (x == prior and i > prior_id)) and (x > best or (x == best and i < id))) {
            best = x;
            id = @intCast(i);
        };
        out.bits[2 + j] = id;
        out.bits[22 + j] = @bitCast(best);
        prior = best;
        prior_id = id;
    }
    return out;
}
pub fn check(got: []const u32, expected: Row, spec: c.Spec, shape: c.Shape) ?[]const u8 {
    if (!std.mem.eql(u32, got[1..words], expected.bits[1..])) return "selected/top/rank/reserved words differ";
    const value: f32 = @bitCast(got[0]);
    if (std.math.isNan(expected.lse)) return if (std.math.isNan(value)) null else "NaN normalizer lost";
    if (std.math.isInf(expected.lse)) return if (value == expected.lse) null else "infinite normalizer changed";
    if (!std.math.isFinite(value) or @abs(@as(f64, value) - expected.lse) > c.bound(spec, shape, expected.lse, 1)) return "normalizer exceeds its FP64 bound";
    return null;
}
test "Qwen score reference rejects last-row corruption, selected-logit loss and normalized-subset errors" {
    const values = [_]f32{ 0, 0, 1, -4, 8 };
    const expected = row(&values, 3, 2);
    try std.testing.expectEqualSlices(u32, &.{ 4, 2 }, expected.bits[2..4]);
    try std.testing.expectEqual(@as(u32, 5), expected.bits[42]);
    const spec: c.Spec = .{ .name = "test", .out = .f32, .accum = .f32, .depth = struct {
        fn depth(_: c.Shape) usize {
            return 16;
        }
    }.depth, .tiles = .{ .{}, .{}, .{} }, .served = &.{.{ .rows = 1, .n = 5, .k = 2 }} };
    try std.testing.expect(check(&expected.bits, expected, spec, spec.served[0]) == null);
    var bad = expected.bits;
    bad[0] = @bitCast(@as(f32, 8));
    try std.testing.expect(check(&bad, expected, spec, spec.served[0]) != null);
    bad = expected.bits;
    bad[1] = 0;
    try std.testing.expect(check(&bad, expected, spec, spec.served[0]) != null);
    var rows: [128 * words]u32 = undefined;
    for (0..128) |r| @memcpy(rows[r * words ..][0..words], &expected.bits);
    rows[127 * words + 2] = 0;
    try std.testing.expect(check(rows[127 * words ..][0..words], expected, spec, spec.served[0]) != null);
    try std.testing.expect(check(rows[0..words], expected, spec, spec.served[0]) == null);
}
test "Qwen score reference keeps NaN, infinities, ties and selected-token ranks explicit" {
    const xs = [_]f32{ std.math.inf(f32), 1, std.math.inf(f32), -std.math.inf(f32) };
    const inf = row(&xs, 0, 2);
    try std.testing.expect(std.math.isPositiveInf(inf.lse));
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, inf.bits[2..4]);
    try std.testing.expectEqual(@as(u32, 2), inf.bits[42]);
    const bad = [_]f32{ 1, std.math.nan(f32) };
    try std.testing.expect(std.math.isNan(row(&bad, 0, 1).lse));
}

test "Qwen score stress: smaller accumulators miss every summation order by the required margin" {
    var logits: [257]f32 = undefined;
    stress(&logits);
    const expected = row(&logits, 0, 20).lse;
    const shape: c.Shape = .{ .rows = 1, .n = logits.len, .k = 20 };
    const spec: c.Spec = .{ .name = "stats stress", .accum = .f32, .depth = reductionDepth, .ops = 12, .out = .f32, .served = &.{shape}, .sums = .{ .exact = false } };
    const bound = c.bound(spec, shape, expected, 1);
    var terms: [257]f64 = undefined;
    for (&terms, logits) |*term, x| term.* = @exp(@as(f64, x));
    const contract = @import("core").contract;
    for ([_]contract.gates.Order{ .sequential, .pairwise, .strided }) |order| {
        const fp32 = @log(contract.rows.sumIn(.f32, order, &terms));
        try std.testing.expect(@abs(fp32 - expected) <= bound);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            const reduced = @log(contract.rows.sumIn(acc, order, &terms));
            try std.testing.expect(@abs(reduced - expected) / bound >= 100);
        }
    }
}

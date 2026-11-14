//! Directional-check diagnostics. Richardson is reported separately from the acceptance test.
const std = @import("std");

pub const Fit = struct {
    numerator: f64 = 0,
    denominator: f64 = 0,

    pub fn add(f: *Fit, x: f64, loss: f64) void {
        f.numerator += x * loss;
        f.denominator += x * x;
    }

    pub fn slope(f: Fit) f64 {
        return f.numerator / f.denominator;
    }
};

pub fn richardson(fine: f64, coarse: f64) f64 {
    return (4 * fine - coarse) / 3;
}

pub const Residual = struct {
    norm: f64,
    largest: f64,
    trimmed_rms: f64,
    width: usize,

    pub fn normStep(r: Residual, fraction: f64) f64 {
        return fraction * r.norm;
    }

    pub fn step(r: Residual, fraction: f64) f64 {
        return fraction * r.trimmed_rms * @sqrt(@as(f64, @floatFromInt(r.width)));
    }
};

/// Remove the largest ceil(width/100) magnitudes for the reported and accepted trimmed perturbation scale.
pub fn residual(values: []const u16, scratch: []f64) !Residual {
    if (values.len < 2 or scratch.len < values.len) return error.BadResidual;
    var sum: f64 = 0;
    for (values, scratch[0..values.len]) |bits, *out| {
        const v: f64 = @as(f32, @bitCast(@as(u32, bits) << 16));
        if (!std.math.isFinite(v)) return error.BadResidual;
        out.* = @abs(v);
        sum += v * v;
    }
    const magnitudes = scratch[0..values.len];
    std.mem.sort(f64, magnitudes, {}, std.sort.asc(f64));
    const kept = values.len - (values.len + 99) / 100;
    var trimmed: f64 = 0;
    for (magnitudes[0..kept]) |v| trimmed += v * v;
    const rms = @sqrt(trimmed / @as(f64, @floatFromInt(kept)));
    if (rms <= 0 or !std.math.isFinite(rms)) return error.BadResidual;
    return .{ .norm = @sqrt(sum), .largest = magnitudes[values.len - 1], .trimmed_rms = rms, .width = values.len };
}

test "directional slope: cubic bias is 25 h squared and Richardson removes it" {
    const a = 1.75;
    const c = -0.3;
    var slopes: [2]f64 = undefined;
    for ([_]f64{ 0.125, 0.25 }, &slopes) |h, *s| {
        var fit: Fit = .{};
        for (1..7) |m| for ([_]f64{ -1, 1 }) |sign| {
            const x = sign * @as(f64, @floatFromInt(m)) * h;
            fit.add(x, 9 + a * x + c * x * x * x);
        };
        s.* = fit.slope();
        // sum(m^4)/sum(m^2) = 2275/91 = 25 for m=1..6.
        try std.testing.expectApproxEqAbs(a + 25 * c * h * h, s.*, 1e-12);
    }
    try std.testing.expectApproxEqAbs(a, richardson(slopes[0], slopes[1]), 1e-12);
}

test "directional slope: a sink changes the reported norm but not the trimmed step" {
    var row: [200]u16 = @splat(0x3f80);
    var scratch: [200]f64 = undefined;
    const plain = try residual(&row, &scratch);
    row[0] = 0x4780;
    row[1] = 0xc600;
    const sink = try residual(&row, &scratch);
    try std.testing.expectEqual(plain.step(0.002), sink.step(0.002));
    try std.testing.expect(sink.norm > 1000 * plain.norm);
    try std.testing.expect(sink.normStep(0.002) > 1000 * plain.normStep(0.002));
    try std.testing.expectApproxEqAbs(0.002 * sink.norm, sink.normStep(0.002), 1e-12);
    try std.testing.expectEqual(@as(f64, 65536), sink.largest);
    row[10] = 0x7fc0;
    try std.testing.expectError(error.BadResidual, residual(&row, &scratch));
    @memset(&row, 0);
    try std.testing.expectError(error.BadResidual, residual(&row, &scratch));
}

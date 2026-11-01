//! Host numerical seam checks. No CUDA library, device, checkpoint or image is loaded.
const std = @import("std");
const steps = @import("core").directional;
test {
    _ = @import("tests/cuda/q27_boundary_ref.zig");
    _ = @import("tests/cuda/gdn_back_point.zig");
}

test "dense gradient: both runner grids report every mathematical mismatch" {
    const Fake = struct {
        calls: [12][2]usize = undefined,
        n: usize = 0,
        pub fn check(f: *@This(), row: usize, layer: usize) !steps.Verdict {
            f.calls[f.n] = .{ row, layer };
            f.n += 1;
            return if (f.n == 1) .mismatch else .pass;
        }
    };
    for ([_][4]usize{ .{ 12, 23, 31, 62 }, .{ 8, 9, 10, 38 } }) |layers| {
        var f: Fake = .{};
        const result = try steps.run(&f, &.{ 0, 14, 18 }, &layers);
        try std.testing.expectEqual(@as(usize, 12), result.checks);
        try std.testing.expectEqual([2]usize{ 18, layers[3] }, f.calls[11]);
        try std.testing.expectError(error.GradientMismatch, result.finish());
    }
}

test "dense gradient: driver failures remain fatal after mathematical mismatches" {
    const Fake = struct {
        n: usize = 0,
        pub fn check(f: *@This(), _: usize, _: usize) !steps.Verdict {
            f.n += 1;
            if (f.n == 2) return error.DeviceLost;
            return .mismatch;
        }
    };
    var f: Fake = .{};
    try std.testing.expectError(error.DeviceLost, steps.run(&f, &.{ 0, 14, 18 }, &.{ 12, 23, 31, 62 }));
    try std.testing.expectEqual(@as(usize, 2), f.n);
}

fn cubicSlopes(derivative: f64, cubic: f64) [6]f64 {
    var slopes: [6]f64 = undefined;
    for (steps.fractions, &slopes) |h, *s| {
        s.* = steps.central(.{ 2 - derivative * h - cubic * h * h * h, 2 + derivative * h + cubic * h * h * h }, h);
    }
    return slopes;
}

test "dense gradient: v2 uses six pairs and the median of only the four coarsest slopes" {
    for (steps.fractions, 0..) |h, k| try std.testing.expectEqual(@as(f64, 0.03) / @as(f64, @floatFromInt(@as(u32, 1) << @intCast(k))), h);
    try std.testing.expectEqual(@as(f64, 3), steps.central(.{ 1, 7 }, 1));
    const f = steps.evaluate(.{ 100, 1, 3, 2, 700, 900 }, 2.5);
    try std.testing.expectEqual(@as(f64, 2.5), f.estimate);
    try std.testing.expectEqual(steps.Verdict.pass, f.verdict);
    // Fine-step noise cannot move the coarsest-four estimate.
    const fine = steps.evaluate(.{ 100, 1, 3, 2, std.math.nan(f64), std.math.inf(f64) }, 2.5);
    try std.testing.expectEqual(f.estimate, fine.estimate);
    for ([_]f64{ 1, -1, 0.002 }) |g| try std.testing.expectEqual(steps.Verdict.pass, steps.evaluate(cubicSlopes(g, 0), g).verdict);
}

test "dense gradient: v2 has only pass or failure including invalid data and the exact bound" {
    for ([_]f64{ 0, std.math.nan(f64), std.math.inf(f64) }) |g| try std.testing.expectEqual(steps.Verdict.mismatch, steps.evaluate(@splat(1), g).verdict);
    for (0..4) |i| {
        var slopes: [6]f64 = @splat(1);
        slopes[i] = std.math.nan(f64);
        try std.testing.expectEqual(steps.Verdict.mismatch, steps.evaluate(slopes, 1).verdict);
    }
    try std.testing.expectEqual(steps.Verdict.mismatch, steps.evaluate(@splat(0), 1).verdict);
    try std.testing.expectEqual(steps.Verdict.pass, steps.evaluate(@splat(17), 20).verdict);
    try std.testing.expectEqual(steps.Verdict.pass, steps.evaluate(@splat(-17), -20).verdict);
    try std.testing.expectEqual(steps.Verdict.mismatch, steps.evaluate(@splat(16.999999999), 20).verdict);
}

test "dense gradient: reported 1.2x layer fails in both complete runner grids" {
    const Fake = struct {
        bad_layer: usize,
        pub fn check(f: *@This(), _: usize, layer: usize) !steps.Verdict {
            return steps.evaluate(cubicSlopes(1, 0), if (layer == f.bad_layer) 1.2 else 1).verdict;
        }
    };
    for ([_][4]usize{ .{ 12, 23, 31, 62 }, .{ 8, 9, 10, 38 } }) |layers| {
        var f: Fake = .{ .bad_layer = layers[1] };
        const result = try steps.run(&f, &.{ 0, 14, 18 }, &layers);
        try std.testing.expectEqual(@as(usize, 12), result.checks);
        try std.testing.expectEqual(@as(usize, 3), result.failed);
        try std.testing.expectError(error.GradientMismatch, result.finish());
        std.debug.print("CONTROL reported-layer grid {d}: changed 3, caught {d}\n", .{ layers[0], result.failed });
    }
}

test "dense gradient: v2 reports the older curved controls without hiding a missed 1.2x gradient" {
    var caught: usize = 0;
    for ([_]f64{ 1.2, 1.67 }) |scale| {
        const result = steps.evaluate(cubicSlopes(1, (scale - 1) / (0.03 * 0.03)), scale);
        try std.testing.expectApproxEqAbs(1 + (scale - 1) * 0.15625, result.estimate, 1e-12);
        if (result.verdict == .mismatch) caught += 1;
        std.debug.print("CONTROL curved scale {d:.2}: {s}, estimate {d:.8}, relative {d:.8}\n", .{ scale, @tagName(result.verdict), result.estimate, result.relative });
    }
    try std.testing.expectEqual(@as(usize, 1), caught);
}

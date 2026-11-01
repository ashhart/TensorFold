//! Host numerical seam checks. No CUDA library, device, checkpoint or image is loaded.
const std = @import("std");
const steps = @import("core").directional;
test {
    _ = @import("tests/cuda/q27_boundary_ref.zig");
    _ = @import("tests/cuda/gdn_back_point.zig");
}
test "dense gradient: shared slope retains the exact legacy accumulation" {
    for (@import("src/cli/qwen35_gradient.zig").fractions(18)) |step| {
        var losses: [6][2]f64 = undefined;
        var num: f64 = 0;
        var den: f64 = 0;
        for (&losses, 1..) |*pair, m| for ([_]f64{ -1, 1 }, pair) |sign, *loss| {
            const x = sign * @as(f64, @floatFromInt(m)) * step;
            const y = 2 + 0.057 * x + 8 * x * x * x;
            loss.* = @round(y / 0.000001) * 0.000001;
            num += x * loss.*;
            den += x * x;
        };
        try std.testing.expectEqual(@as(u64, @bitCast(num / den)), @as(u64, @bitCast(steps.slope(losses, step))));
    }
}

test "dense gradient: both runner grids report every check after a first mismatch or unresolved fit" {
    const Fake = struct {
        calls: [12][2]usize = undefined,
        n: usize = 0,
        first: steps.Verdict,
        pub fn check(f: *@This(), row: usize, layer: usize) !steps.Verdict {
            f.calls[f.n] = .{ row, layer };
            f.n += 1;
            return if (f.n == 1) f.first else .pass;
        }
    };
    for ([_][4]usize{ .{ 12, 23, 31, 62 }, .{ 8, 9, 10, 38 } }) |layers| {
        for ([_]steps.Verdict{ .mismatch, .unresolved }) |first| {
            var f: Fake = .{ .first = first };
            const result = try steps.run(&f, &.{ 0, 14, 18 }, &layers);
            try std.testing.expectEqual(@as(usize, 12), result.checks);
            try std.testing.expectEqual(@as(usize, 12), f.n);
            try std.testing.expectEqual([2]usize{ 18, layers[3] }, f.calls[11]);
            if (first == .mismatch) try std.testing.expectError(error.GradientMismatch, result.finish()) else try std.testing.expectError(error.NoiseLimitedGradient, result.finish());
        }
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

fn cubicSlope(step: f64) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (1..7) |m| for ([_]f64{ -1, 1 }) |sign| {
        const x = sign * @as(f64, @floatFromInt(m)) * step;
        const loss = 2 + x + 120 * x * x * x;
        num += x * loss;
        den += x * x;
    };
    return num / den;
}
test "dense gradient: local sink steps remove curvature without widening any gradient bound" {
    const t = std.testing;
    try t.expectEqual([3]f64{ 0.002, 0.004, 0.008 }, @import("src/cli/qwen35_gradient.zig").fractions(14));
    try t.expectEqual(@import("src/cli/qwen35_gradient.zig").fractions(14), @import("src/cli/qwen35_gradient.zig").fractions(18));
    try t.expect(@abs(cubicSlope(0.008) - 1) > 0.15);
    for (@import("src/cli/qwen35_gradient.zig").fractions(0)) |step| try t.expect(@abs(cubicSlope(step) - 1) < 0.15);
}

//! Six central differences qualify against the median of the four coarsest slopes.
const std = @import("std");
pub const fractions = [6]f64{ 0.03, 0.015, 0.0075, 0.00375, 0.001875, 0.0009375 };
pub const Verdict = enum { pass, mismatch };
pub const Check = struct {
    verdict: Verdict = .mismatch,
    estimate: f64 = std.math.nan(f64),
    relative: f64 = std.math.nan(f64),
};

pub fn central(losses: [2]f64, step: f64) f64 {
    return (losses[1] - losses[0]) / (2 * step);
}

/// The finest two slopes are diagnostics; the four coarsest alone determine the estimate.
pub fn evaluate(slopes: [6]f64, gradient: f64) Check {
    var coarse = slopes[0..4].*;
    for (coarse) |s| if (!std.math.isFinite(s)) return .{};
    std.mem.sort(f64, &coarse, {}, std.sort.asc(f64));
    const estimate = (coarse[1] + coarse[2]) / 2;
    const relative = @abs(estimate - gradient) / @abs(gradient);
    return .{ .estimate = estimate, .relative = relative, .verdict = if (std.math.isFinite(relative) and relative <= 0.15) .pass else .mismatch };
}

pub const Summary = struct {
    checks: usize = 0,
    failed: usize = 0,

    pub fn finish(s: Summary) !void {
        if (s.failed != 0) return error.GradientMismatch;
    }
};

/// Mathematical failures are values, so every layer/row is reported before qualification returns an error.
pub fn run(ctx: anytype, rows: []const usize, layers: []const usize) !Summary {
    var out: Summary = .{};
    for (rows) |row| for (layers) |layer| {
        const verdict = try ctx.check(row, layer);
        out.checks += 1;
        if (verdict == .mismatch) out.failed += 1;
    };
    return out;
}

//! Apply the existing 0.15 gradient tolerance to both paths, both steps and every worst-case sample.
const std = @import("std");
pub const Stats = [2][2][2]f64;
pub fn accepts(stats: Stats) bool {
    for (stats) |path| for (path) |step| for (step) |error_value| {
        if (!std.math.isFinite(error_value) or error_value < 0 or error_value >= 0.15) return false;
    };
    return true;
}
test "Learning persistence: gradient gate refuses cross-row, worst, boundary-step and nonfinite errors" {
    const valid: Stats = @splat(@splat(.{ 0.05, 0.1 }));
    try std.testing.expect(accepts(valid));
    for (0..2) |path| for (0..2) |step| for (0..2) |kind| {
        var invalid = valid;
        invalid[path][step][kind] = 0.15;
        try std.testing.expect(!accepts(invalid));
        invalid[path][step][kind] = std.math.nan(f64);
        try std.testing.expect(!accepts(invalid));
    };
}

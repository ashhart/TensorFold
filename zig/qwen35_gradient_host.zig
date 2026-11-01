//! Host numerical seam checks. No CUDA library, device, checkpoint or image is loaded.
const std = @import("std");
const steps = @import("src/cli/qwen35_gradient.zig");
test {
    _ = @import("tests/cuda/q27_boundary_ref.zig");
    _ = @import("tests/cuda/gdn_back_point.zig");
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
    try t.expectEqual([3]f64{ 0.002, 0.004, 0.008 }, steps.fractions(14));
    try t.expectEqual(steps.fractions(14), steps.fractions(18));
    try t.expect(@abs(cubicSlope(0.008) - 1) > 0.15);
    for (steps.fractions(0)) |step| try t.expect(@abs(cubicSlope(step) - 1) < 0.15);
}

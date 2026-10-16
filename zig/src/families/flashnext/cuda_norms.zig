//! Classify the checkpoint-wide centered or direct RMSNorm convention, rejecting ambiguous mixtures.
const std = @import("std");

pub fn offset(comptime layers_n: usize, input: [layers_n]f64) !f32 {
    comptime std.debug.assert(layers_n >= 2 and layers_n % 2 == 0);
    var means = input;
    var above: usize = 0;
    for (means) |mean| {
        if (!std.math.isFinite(mean)) return error.InvalidNormConvention;
        if (mean > 0.5) above += 1;
    }
    std.mem.sort(f64, &means, {}, std.sort.asc(f64));
    const median = (means[layers_n / 2 - 1] + means[layers_n / 2]) / 2;
    if (above * 10 >= layers_n * 9 and median >= 0.75 and median <= 1.5) return 0;
    if (above * 10 <= layers_n and median >= -0.5 and median <= 0.25) return 1;
    return error.InvalidNormConvention;
}

test "centered and direct norms have distinct offsets; mixed and nonfinite conventions fail" {
    try std.testing.expectEqual(@as(f32, 1), try offset(48, @splat(-0.1)));
    try std.testing.expectEqual(@as(f32, 0), try offset(48, @splat(1.05)));
    var mixed: [48]f64 = @splat(1.0);
    @memset(mixed[0 .. 48 / 2], 0);
    try std.testing.expectError(error.InvalidNormConvention, offset(48, mixed));
    mixed[0] = std.math.nan(f64);
    try std.testing.expectError(error.InvalidNormConvention, offset(48, mixed));
}

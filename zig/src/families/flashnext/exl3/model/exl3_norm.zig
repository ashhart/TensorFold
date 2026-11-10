//! Centred RMSNorm apply.

const std = @import("std");
const scan = @import("../pack/exl3_scan.zig");

/// `rms = sqrt(mean(x²) + eps)`, then `out[i] = x[i] / rms * (weight[i] + offset)`.
pub fn apply(x: []const f32, weight: []const f32, means: []const f32, eps: f32, out: []f32) error{ Length, NormAmbiguous }!void {
    const offset = try scan.centredOffset(means);
    if (!std.math.isFinite(eps) or eps < 0) return error.Length;
    if (x.len == 0 or x.len != weight.len or x.len != out.len) return error.Length;

    var sum_sq: f32 = 0;
    for (x) |v| sum_sq += v * v;
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    for (x, weight, out) |xi, wi, *yi| yi.* = xi / rms * (wi + offset);
}

test "a unit vector changes by exactly the weight difference" {
    // Same means `centredOffset` classifies as gamma-1 (1) and gamma (0).
    const near_zero = [_]f32{ -0.02, 0.01, 0.0, 0.04 };
    const near_one = [_]f32{ 1.0, 0.98, 1.02, 1.01 };
    // x = [1], eps = 0: mean(x²)+eps is 1, so the RMS is 1 and out = weight + offset.
    const x = [_]f32{1};
    const light = [_]f32{2};
    const heavy = [_]f32{5};
    var zero_light = [_]f32{0};
    var zero_heavy = [_]f32{0};
    var one_light = [_]f32{0};
    var one_heavy = [_]f32{0};
    try apply(&x, &light, &near_zero, 0, &zero_light);
    try apply(&x, &heavy, &near_zero, 0, &zero_heavy);
    try apply(&x, &light, &near_one, 0, &one_light);
    try apply(&x, &heavy, &near_one, 0, &one_heavy);
    try std.testing.expectEqual(@as(f32, 3), zero_light[0]);
    try std.testing.expectEqual(@as(f32, 6), zero_heavy[0]);
    try std.testing.expectEqual(@as(f32, 2), one_light[0]);
    try std.testing.expectEqual(@as(f32, 5), one_heavy[0]);
    try std.testing.expectEqual(heavy[0] - light[0], zero_heavy[0] - zero_light[0]);
    try std.testing.expectEqual(heavy[0] - light[0], one_heavy[0] - one_light[0]);

    // Two ones still have mean 1, so a forgotten division by length cannot pass.
    const pair_x = [_]f32{ 1, 1 };
    const pair_w = [_]f32{ 2, 7 };
    var pair_out = [_]f32{ 0, 0 };
    try apply(&pair_x, &pair_w, &near_zero, 0, &pair_out);
    try std.testing.expectEqual(@as(f32, 3), pair_out[0]);
    try std.testing.expectEqual(@as(f32, 8), pair_out[1]);
}

test "an ambiguous norm leaves out untouched" {
    var out = [_]f32{ 9, 8 };
    const many: [65]f32 = @splat(0);
    try std.testing.expectError(error.NormAmbiguous, apply(&.{ 1, 1 }, &.{ 1, 1 }, &many, 0, &out));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
    try std.testing.expectEqual(@as(f32, 8), out[1]);

    // Length also disagrees. The locator runs first, so this is not Length.
    try std.testing.expectError(error.NormAmbiguous, apply(&.{1}, &.{}, &.{ 0.4, 0.9 }, -1, &out));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
    try std.testing.expectEqual(@as(f32, 8), out[1]);
}

test "a length mismatch leaves out untouched" {
    const near_zero = [_]f32{ -0.02, 0.01, 0.0, 0.04 };
    const near_one = [_]f32{ 1.0, 0.98, 1.02, 1.01 };
    var out = [_]f32{ 9, 8 };
    try std.testing.expectError(error.Length, apply(&.{}, &.{}, &near_zero, 0, &out));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
    try std.testing.expectEqual(@as(f32, 8), out[1]);

    try std.testing.expectError(error.Length, apply(&.{ 1, 1 }, &.{1}, &near_zero, 0, &out));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
    try std.testing.expectEqual(@as(f32, 8), out[1]);

    try std.testing.expectError(error.Length, apply(&.{1}, &.{1}, &near_one, 0, &out));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
    try std.testing.expectEqual(@as(f32, 8), out[1]);
}

test "a negative eps leaves out untouched" {
    const near_zero = [_]f32{ -0.02, 0.01, 0.0, 0.04 };
    const near_one = [_]f32{ 1.0, 0.98, 1.02, 1.01 };
    var out = [_]f32{9};
    try std.testing.expectError(error.Length, apply(&.{1}, &.{2}, &near_zero, -1, &out));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
    try std.testing.expectError(error.Length, apply(&.{1}, &.{2}, &near_one, std.math.nan(f32), &out));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
    try std.testing.expectError(error.Length, apply(&.{1}, &.{2}, &near_zero, std.math.inf(f32), &out));
    try std.testing.expectEqual(@as(f32, 9), out[0]);
}

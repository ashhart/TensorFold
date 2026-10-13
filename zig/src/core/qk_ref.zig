//! QK norm and RoPE of one head in float64, the reference for every backend's attention inputs (cuda/qk_rows.zig).
const std = @import("std");

/// One head: RMS-normed by w (none: as is), then rotate-half RoPE on its first `rot` dims (no cos/sin: none).
pub fn head(x: []const f64, w: ?[]const f64, eps: f64, cos: ?[]const f64, sin: ?[]const f64, rot: usize, y: []f64) void {
    const n = x.len;
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = if (w != null) 1 / @sqrt(ss / @as(f64, @floatFromInt(n)) + eps) else 1;
    var z: [256]f64 = undefined;
    for (x, 0..) |v, i| z[i] = v * inv * (if (w) |ww| ww[i] else 1);
    const half = rot / 2;
    for (y, 0..) |*o, i| {
        if (cos == null or i >= rot) {
            o.* = z[i];
            continue;
        }
        const p = if (i < half) i + half else i - half;
        const rotated = if (i < half) -z[p] else z[p];
        o.* = z[i] * cos.?[i] + rotated * sin.?[i];
    }
}

test "rotate-half pairs (d, d + rot / 2) and leave the rest" {
    var y: [4]f64 = undefined;
    head(&.{ 1, 2, 3, 4 }, null, 0, &.{ 0, 0 }, &.{ 1, 1 }, 2, &y); // angle 90 degrees on the first pair
    try std.testing.expectEqualSlices(f64, &.{ -2, 1, 3, 4 }, &y);
    head(&.{ 2, 2, 2, 2 }, &.{ 1, 1, 1, 3 }, 0, null, null, 0, &y);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 3 }, &y);
}

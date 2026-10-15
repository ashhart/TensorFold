//! RADIO's position rows on the host: tf_vit_position_add's reference (the table resized to the longer side, cropped).
const std = @import("std");

/// An output index's two source indices and weights (bilinear, half-pixel centres, clamped at the edges), float32.
const Tap = struct { lo: usize, hi: usize, l0: f32, l1: f32 };

fn tap(in: usize, out: usize, i: usize) Tap {
    if (in == out) return .{ .lo = i, .hi = i, .l0 = 1, .l1 = 0 };
    const scale = @as(f32, @floatFromInt(in)) / @as(f32, @floatFromInt(out));
    const src = @max(scale * (@as(f32, @floatFromInt(i)) + 0.5) - 0.5, 0);
    const lo = @min(@as(usize, @intFromFloat(@floor(src))), in - 1);
    const l1 = @min(@max(src - @as(f32, @floatFromInt(lo)), 0), 1);
    return .{ .lo = lo, .hi = lo + @intFromBool(lo < in - 1), .l0 = 1 - l1, .l1 = l1 };
}

/// Rows [gh * gw, c] in grid order: `table` resized to max(gh, gw) a side in float32, its first gh rows and gw columns.
pub fn rows(table: []const f32, side: usize, c: usize, gh: usize, gw: usize, out: []f32) void {
    std.debug.assert(table.len == side * side * c and out.len == gh * gw * c);
    const m = @max(gh, gw);
    for (0..gh) |y| {
        const ty = tap(side, m, y);
        for (0..gw) |x| {
            const tx = tap(side, m, x);
            const w = [4]f32{ ty.l0 * tx.l0, ty.l0 * tx.l1, ty.l1 * tx.l0, ty.l1 * tx.l1 };
            const t00 = table[(ty.lo * side + tx.lo) * c ..][0..c];
            const t01 = table[(ty.lo * side + tx.hi) * c ..][0..c];
            const t10 = table[(ty.hi * side + tx.lo) * c ..][0..c];
            const t11 = table[(ty.hi * side + tx.hi) * c ..][0..c];
            for (out[(y * gw + x) * c ..][0..c], t00, t01, t10, t11) |*o, a, b, d, e| {
                o.* = a * w[0] + (b * w[1] + (d * w[2] + e * w[3]));
            }
        }
    }
}

test "the table's own size copies it; a grid is the first rows and columns of its square's resize" {
    const side = 4;
    const c = 2;
    var table: [side * side * c]f32 = undefined;
    for (&table, 0..) |*t, i| t.* = @floatFromInt(i);
    var same: [side * side * c]f32 = undefined;
    rows(&table, side, c, side, side, &same);
    try std.testing.expectEqualSlices(f32, &table, &same);
    var big: [8 * 8 * c]f32 = undefined;
    var crop: [8 * 3 * c]f32 = undefined;
    rows(&table, side, c, 8, 8, &big);
    rows(&table, side, c, 8, 3, &crop);
    for (0..8) |y| for (0..3) |x| for (0..c) |k| {
        try std.testing.expectEqual(big[(y * 8 + x) * c + k], crop[(y * 3 + x) * c + k]);
    };
    // output row 1 of 8 sits a quarter of the way from source row 0 to row 1 (value 8 at column 0)
    try std.testing.expectEqual(@as(f32, 0.25 * 8.0), big[8 * c]);
}

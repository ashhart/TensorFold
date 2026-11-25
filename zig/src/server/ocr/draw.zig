//! Region outlines on a page's RGB pixels for the visualisation images (PaddleX ``ocrImage``).
const std = @import("std");

/// The one outline colour the visualisations use.
pub const red: [3]u8 = .{ 255, 0, 0 };

/// Draws the closed polygon ``poly`` on ``w`` x ``h`` RGB rows, lines ``width_px`` thick, every pixel clipped.
pub fn outline(rgb: []u8, w: u32, h: u32, poly: [4][2]i32, rgb_color: [3]u8, width_px: u32) void {
    std.debug.assert(rgb.len >= @as(usize, w) * h * 3);
    if (w == 0 or h == 0 or width_px == 0) return;
    for (0..4) |i| segment(rgb, w, h, poly[i], poly[(i + 1) % 4], rgb_color, width_px);
}

/// One Bresenham line, first clipped to the image grown by the line width so a far-off corner costs no steps.
fn segment(rgb: []u8, w: u32, h: u32, from: [2]i32, to: [2]i32, color: [3]u8, n: u32) void {
    const pad: f64 = @floatFromInt(n);
    var p0: [2]f64 = .{ @floatFromInt(from[0]), @floatFromInt(from[1]) };
    var p1: [2]f64 = .{ @floatFromInt(to[0]), @floatFromInt(to[1]) };
    if (!clip(&p0, &p1, .{ -pad, -pad, @as(f64, @floatFromInt(w)) + pad, @as(f64, @floatFromInt(h)) + pad })) return;
    var x: i64 = @intFromFloat(@round(p0[0]));
    var y: i64 = @intFromFloat(@round(p0[1]));
    const x1: i64 = @intFromFloat(@round(p1[0]));
    const y1: i64 = @intFromFloat(@round(p1[1]));
    const dx: i64 = @intCast(@abs(x1 - x));
    const dy: i64 = -@as(i64, @intCast(@abs(y1 - y)));
    const sx: i64 = if (x < x1) 1 else -1;
    const sy: i64 = if (y < y1) 1 else -1;
    var err = dx + dy;
    while (true) {
        dot(rgb, w, h, x, y, color, n);
        if (x == x1 and y == y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
}

/// An ``n`` x ``n`` square centred on (x, y) (2 px: x-1..x), so an edge on the image border stays visible.
fn dot(rgb: []u8, w: u32, h: u32, x: i64, y: i64, color: [3]u8, n: u32) void {
    const back: i64 = @intCast(n / 2);
    for (0..n) |j| for (0..n) |i| {
        const px = x - back + @as(i64, @intCast(i));
        const py = y - back + @as(i64, @intCast(j));
        if (px < 0 or py < 0 or px >= w or py >= h) continue;
        const at: usize = (@as(usize, @intCast(py)) * w + @as(usize, @intCast(px))) * 3;
        rgb[at..][0..3].* = color;
    };
}

/// Liang-Barsky: trims the segment to the box ``[xmin, ymin, xmax, ymax]``; false when nothing is left.
fn clip(p0: *[2]f64, p1: *[2]f64, b: [4]f64) bool {
    const d: [2]f64 = .{ p1[0] - p0[0], p1[1] - p0[1] };
    const ps = [4]f64{ -d[0], d[0], -d[1], d[1] };
    const qs = [4]f64{ p0[0] - b[0], b[2] - p0[0], p0[1] - b[1], b[3] - p0[1] };
    var t0: f64 = 0;
    var t1: f64 = 1;
    for (ps, qs) |p, q| {
        if (p == 0) {
            if (q < 0) return false;
            continue;
        }
        const r = q / p;
        if (p < 0) {
            if (r > t1) return false;
            t0 = @max(t0, r);
        } else {
            if (r < t0) return false;
            t1 = @min(t1, r);
        }
    }
    const start = p0.*;
    p0.* = .{ start[0] + t0 * d[0], start[1] + t0 * d[1] };
    p1.* = .{ start[0] + t1 * d[0], start[1] + t1 * d[1] };
    return true;
}

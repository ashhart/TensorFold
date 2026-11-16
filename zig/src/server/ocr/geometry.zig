//! Quad validation and the integer forms encoders send.
const std = @import("std");
const api = @import("engine_api");
const Quad = api.ocr.Quad;

/// ``q`` clamped into the frame, corners kept; a degenerate quad becomes its min/max box, sides widened to 1 px.
pub fn validate(q: Quad, w: u32, h: u32) error{Unreadable}!Quad {
    const fw: f32 = @floatFromInt(w);
    const fh: f32 = @floatFromInt(h);
    var out: Quad = undefined;
    for (q, &out) |p, *o| {
        if (!std.math.isFinite(p[0]) or !std.math.isFinite(p[1])) return error.Unreadable;
        o.* = .{ std.math.clamp(p[0], 0, fw), std.math.clamp(p[1], 0, fh) };
    }
    if (isSimpleConvex(out)) return out;
    const b = minMax(out);
    const x = widen(b[0], b[2], fw);
    const y = widen(b[1], b[3], fh);
    return box(x[0], y[0], x[1], y[1]);
}

/// The quad's ``[xmin, ymin, xmax, ymax]``.
pub fn minMax(q: Quad) [4]f32 {
    var b: [4]f32 = .{ q[0][0], q[0][1], q[0][0], q[0][1] };
    for (q[1..]) |p| {
        b[0] = @min(b[0], p[0]);
        b[1] = @min(b[1], p[1]);
        b[2] = @max(b[2], p[0]);
        b[3] = @max(b[3], p[1]);
    }
    return b;
}

/// Whether the corners, in order, form a simple convex polygon: every turn the same strict sign, shoelace area ≥ 1 px².
pub fn isSimpleConvex(q: Quad) bool {
    var twice_area: f64 = 0;
    var sign: f64 = 0;
    for (0..4) |i| {
        const a = q[i];
        const b = q[(i + 1) % 4];
        const c = q[(i + 2) % 4];
        twice_area += @as(f64, a[0]) * b[1] - @as(f64, b[0]) * a[1];
        const cross = (@as(f64, b[0]) - a[0]) * (@as(f64, c[1]) - b[1]) - (@as(f64, b[1]) - a[1]) * (@as(f64, c[0]) - b[0]);
        if (cross == 0) return false;
        if (sign == 0) sign = cross else if ((cross > 0) != (sign > 0)) return false;
    }
    return @abs(twice_area) >= 2;
}

/// Mistral's integer box ``[x0, y0, x1, y1]``: the minimum floored, the maximum ceiled, clamped to the i32 range.
pub fn mistralBox(q: Quad) [4]i32 {
    const b = minMax(q);
    return .{ toInt(@floor(b[0])), toInt(@floor(b[1])), toInt(@ceil(b[2])), toInt(@ceil(b[3])) };
}

fn box(x0: f32, y0: f32, x1: f32, y1: f32) Quad {
    return .{ .{ x0, y0 }, .{ x1, y0 }, .{ x1, y1 }, .{ x0, y1 } };
}

/// ``[lo, hi]`` grown to at least 1 px inside ``[0, limit]`` (as far as the limit allows).
fn widen(lo: f32, hi: f32, limit: f32) [2]f32 {
    if (hi - lo >= 1) return .{ lo, hi };
    if (lo + 1 <= limit) return .{ lo, lo + 1 };
    return .{ @max(0, limit - 1), limit };
}

fn toInt(x: f32) i32 {
    return std.math.lossyCast(i32, x);
}

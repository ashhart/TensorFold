//! The image tower's per-image index tables: bilinear position-grid taps and axial rotary rows, built on the host.
const std = @import("std");

/// The vision rotary base (Inkling2VisionConfig's rope_parameters carry none, so the library default 10000 holds).
pub const rope_theta: f64 = 10000;

/// One axis' two taps into the position grid and their weights.
pub const Axis = struct { taps: [2]u32, weights: [2]f32 };

/// One axis' two taps into a `side`-long table and weights for `index` of `size` (half-pixel, border-clamped bilinear).
pub fn axis(index: usize, size: usize, side: usize) Axis {
    const src: f32 = (@as(f32, @floatFromInt(index)) + 0.5) * @as(f32, @floatFromInt(side)) / @as(f32, @floatFromInt(size)) - 0.5;
    const floor = @floor(src);
    var out: Axis = undefined;
    inline for (0..2) |k| {
        const raw: i64 = @as(i64, @intFromFloat(floor)) + k;
        out.taps[k] = @intCast(std.math.clamp(raw, 0, @as(i64, @intCast(side)) - 1));
        out.weights[k] = @max(1.0 - @abs(src - floor - @as(f32, @floatFromInt(k))), 0.0);
    }
    return out;
}

/// Where patch `i` of a `gh` x `gw` token image sits on the patch grid: tokens in raster order, patches (row, col).
pub fn place(i: usize, gw: usize, merge: usize) [2]usize {
    const token = i / (merge * merge);
    const q = i % (merge * merge);
    return .{ (token / gw) * merge + q / merge, (token % gw) * merge + q % merge };
}

/// Table sizes for an image: patches = tokens * merge^2, sequence rows = patches + 1 (the class row first).
pub fn patches(gh: u32, gw: u32, merge: u32) usize {
    return @as(usize, gh) * gw * merge * merge;
}

/// Fills taps and weights ([patches * 4]) and cos / sin ([(patches + 1) * head_dim], class row first).
pub fn fill(gh: u32, gw: u32, merge: u32, side: u32, head_dim: u32, taps: []u32, weights: []f32, cos: []f32, sin: []f32) void {
    const n = patches(gh, gw, merge);
    const hh: usize = @as(usize, gh) * merge;
    const ww: usize = @as(usize, gw) * merge;
    const nf = head_dim / 4;
    var inv: [64]f32 = undefined;
    for (0..nf) |j| inv[j] = @floatCast(1.0 / std.math.pow(f64, rope_theta, @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(nf))));
    for (0..head_dim) |d| {
        cos[d] = 1.0;
        sin[d] = 0.0;
    }
    for (0..n) |i| {
        const at = place(i, gw, merge);
        const r = axis(at[0], hh, side);
        const c = axis(at[1], ww, side);
        inline for (0..2) |a| inline for (0..2) |b| {
            taps[i * 4 + a * 2 + b] = r.taps[a] * side + c.taps[b];
            weights[i * 4 + a * 2 + b] = r.weights[a] * c.weights[b];
        };
        const pos = [2]f32{ @floatFromInt(at[1] + 1), @floatFromInt(at[0] + 1) }; // column first (recomposition_frequencies)
        for (0..head_dim / 2) |p| {
            const on_row = p >= nf;
            const angle = pos[@intFromBool(on_row)] * inv[p % nf];
            const row = (i + 1) * head_dim;
            cos[row + 2 * p] = @cos(angle);
            cos[row + 2 * p + 1] = @cos(angle);
            sin[row + 2 * p] = @sin(angle);
            sin[row + 2 * p + 1] = @sin(angle);
        }
    }
}

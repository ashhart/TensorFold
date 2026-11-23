//! The image tower's per-image index tables: the processor's bilinear position taps and axial rotary rows.
const std = @import("std");
const base = @import("../image.zig");

/// The vision rotary base (Inkling2VisionConfig's rope_parameters carry none, so the library default 10000 holds).
pub const rope_theta: f64 = 10000;

/// Table sizes for an image: patches = tokens * merge^2, sequence rows = patches + 1 (the class row first).
pub fn patches(gh: u32, gw: u32) usize {
    return @as(usize, gh) * gw * base.merge * base.merge;
}

/// Fills taps and weights ([patches * 4]) and cos / sin ([(patches + 1) * head_dim], class row first).
pub fn fill(gh: u32, gw: u32, head_dim: u32, taps: []u32, weights: []f32, cos: []f32, sin: []f32) void {
    const n = patches(gh, gw);
    var none: [0]f32 = .{};
    base.positionTaps(.{ .pixels = &none, .rows = gh, .cols = gw }, taps[0 .. n * 4], weights[0 .. n * 4]);
    const nf = head_dim / 4;
    var inv: [64]f32 = undefined;
    for (0..nf) |j| inv[j] = @floatCast(1.0 / std.math.pow(f64, rope_theta, @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(nf))));
    for (0..head_dim) |d| {
        cos[d] = 1.0;
        sin[d] = 0.0;
    }
    for (0..n) |i| {
        const at = base.place(i, gw);
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

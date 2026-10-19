//! Static rotary tables for one loaded engine.
const std = @import("std");

/// transformers' _compute_yarn_parameters for the 64 rotary dims (beta 32 and 1, truncated ends), in torch's fp32.
pub fn yarn(base: f64, factor: f64, trained: usize) struct { freq: [32]f32, scale: f32 } {
    const dim: f64 = 64;
    const tp: f64 = @floatFromInt(trained);
    const dimOf = struct {
        fn f(rot: f64, b: f64, max_pos: f64) f64 {
            return 64 * @log(max_pos / (rot * 2 * std.math.pi)) / (2 * @log(b));
        }
    }.f;
    const low = @max(@floor(dimOf(32, base, tp)), 0);
    var high = @min(@ceil(dimOf(1, base, tp)), dim - 1);
    if (low == high) high += 0.001;
    var out: [32]f32 = undefined;
    for (0..32) |k| {
        const kf: f64 = @floatFromInt(k);
        const pos_freq: f32 = @floatCast(std.math.pow(f64, base, @as(f64, @as(f32, @floatCast(2 * kf / dim)))));
        const extra: f32 = 1.0 / pos_freq;
        const inter: f32 = 1.0 / (@as(f32, @floatCast(factor)) * pos_freq);
        const ef: f32 = 1 - @as(f32, @floatCast(std.math.clamp((kf - low) / (high - low), 0, 1)));
        out[k] = inter * (1 - ef) + extra * ef;
    }
    return .{ .freq = out, .scale = @floatCast(0.1 * @log(factor) + 1) };
}

pub const words: usize = 35;
pub fn table(log2base: f32, factor: f64, trained: usize) [words]f32 {
    var out: [words]f32 = @splat(0);
    out[0] = log2base;
    out[2] = 1;
    if (factor > 1) {
        const y = yarn(std.math.pow(f64, 2, log2base), factor, trained);
        out[1] = 1;
        out[2] = y.scale;
        @memcpy(out[3..35], &y.freq);
    }
    return out;
}

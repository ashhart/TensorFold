//! Scalar reference for cross-pass attention and BF16-weighted message controllers.
const std = @import("std");
const u = @import("inkling2_util.zig");

const eps = 1e-6;

fn invRms(x: []const f64) f64 {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    return 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
}

/// Sources in depth order, already widened; the same RMS-normalized rows are keys and values.
pub fn mix(src: []const f64, q: []const f32, out: []f64) void {
    const h = q.len;
    const n = src.len / h;
    var inv: [9]f64 = undefined;
    var score: [9]f64 = undefined;
    var hi = -std.math.inf(f64);
    for (0..n) |j| {
        const x = src[j * h ..][0..h];
        inv[j] = invRms(x);
        var dot: f64 = 0;
        for (x, q) |v, w| dot += v * inv[j] * @as(f64, w);
        score[j] = dot / @sqrt(@as(f64, @floatFromInt(h)));
        if (std.math.isNan(score[j])) {
            @memset(out, std.math.nan(f64));
            return;
        }
        hi = @max(hi, score[j]);
    }
    var total: f64 = 0;
    for (score[0..n]) |*s| {
        s.* = @exp(s.* - hi);
        total += s.*;
    }
    @memset(out, 0);
    for (0..n) |j| for (out, src[j * h ..][0..h]) |*o, x| {
        o.* += score[j] / total * (x * inv[j]);
    };
}

pub fn down(x: []const f64, m: []const f64, w: []const u16, out: []f64) void {
    const h = x.len;
    const ix = invRms(x);
    const im = invRms(m);
    for (out, 0..) |*d, r| {
        const wr = w[r * 2 * h ..][0 .. 2 * h];
        var sum: f64 = 0;
        for (x, m, wr[0..h], wr[h..]) |v, z, a, b| {
            sum += u.bf(v * ix) * u.val(a) + u.bf(z * im) * u.val(b);
        }
        d.* = u.bf(sum);
    }
}

pub fn gate(x: []const f64, m: []const f64, d: []const f64, forget: []const u16, gain: []const u16, norm: []const u16, out: []f64) void {
    const rank = d.len;
    for (x, m, out, 0..) |v, msg, *o, j| {
        var f: f64 = 0;
        var g: f64 = 0;
        for (d, forget[j * rank ..][0..rank], gain[j * rank ..][0..rank]) |z, fw, gw| {
            f += z * u.val(fw);
            g += z * u.val(gw);
        }
        o.* = (1 - std.math.tanh(f)) * v + std.math.tanh(g) * msg;
    }
    const inv = invRms(out);
    for (out, norm) |*o, w| o.* = u.bf(o.* * inv * u.val(w));
}

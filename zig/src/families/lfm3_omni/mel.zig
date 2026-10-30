//! LFM3-Omni's host log-mel: 400-point Hann STFT every 160 samples, 128 Slaney bins, (max(log10, -12) + 4) / 4.
const std = @import("std");

pub const sample_rate = 16000;
pub const n_fft = 400;
pub const hop = 160;
pub const bins = n_fft / 2 + 1;
pub const mels = 128;
const log_floor = -12.0;

/// Frames for `n` samples once padded to whole hops (at least one window).
pub fn frames(n: usize) usize {
    return 1 + (padded(n) - n_fft) / hop;
}

fn padded(n: usize) usize {
    if (n < n_fft) return n_fft;
    return n + (hop - (n - n_fft) % hop) % hop;
}

/// The speech encoder's tokens for `f` frames: one per 8 frames, rounded up.
pub fn tokens(f: usize) usize {
    return (f + 7) / 8;
}

fn hzToMel(f: f64) f64 {
    const logstep = 27.0 / @log(6.4);
    return if (f >= 1000.0) 15.0 + @log(f / 1000.0) * logstep else 3.0 * f / 200.0;
}

fn melToHz(m: f64) f64 {
    const logstep = @log(6.4) / 27.0;
    return if (m >= 15.0) 1000.0 * @exp(logstep * (m - 15.0)) else 200.0 * m / 3.0;
}

/// transformers' mel_filter_bank(201, 128, 0, 8000, 16000, norm="slaney", mel_scale="slaney"), [bins][mels].
pub fn filterBank(out: *[bins][mels]f64) void {
    var hz: [mels + 2]f64 = undefined;
    const lo = hzToMel(0);
    const hi = hzToMel(sample_rate / 2);
    for (&hz, 0..) |*h, i| h.* = melToHz(lo + (hi - lo) * @as(f64, @floatFromInt(i)) / @as(f64, mels + 1));
    for (out, 0..) |*row, k| {
        const fk = @as(f64, @floatFromInt(k)) * (sample_rate / 2) / @as(f64, bins - 1);
        for (row, 0..) |*v, m| {
            const down = -(hz[m] - fk) / (hz[m + 1] - hz[m]);
            const up = (hz[m + 2] - fk) / (hz[m + 2] - hz[m + 1]);
            v.* = @max(0.0, @min(down, up)) * (2.0 / (hz[m + 2] - hz[m]));
        }
    }
}

pub const Mel = struct {
    filters: [bins][mels]f32,
    cos: [n_fft][bins]f64,
    sin: [n_fft][bins]f64,
    window: [n_fft]f64,

    pub fn init(m: *Mel) void {
        var fb: [bins][mels]f64 = undefined;
        filterBank(&fb);
        for (&m.filters, fb) |*r, s| for (r, s) |*v, x| {
            v.* = @floatCast(x);
        };
        for (&m.window, 0..) |*w, i| w.* = 0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / n_fft);
        for (0..n_fft) |n| for (0..bins) |k| {
            const a = 2.0 * std.math.pi * @as(f64, @floatFromInt((n * k) % n_fft)) / n_fft;
            m.cos[n][k] = @cos(a);
            m.sin[n][k] = @sin(a);
        };
    }

    /// Features of `x` (16 kHz mono) into `out` [mels, frames(x.len)], row-major.
    pub fn compute(m: *const Mel, x: []const f32, out: []f32) void {
        const nf = frames(x.len);
        std.debug.assert(out.len == mels * nf);
        var power: [bins]f32 = undefined;
        for (0..nf) |t| {
            var re: [bins]f64 = @splat(0);
            var im: [bins]f64 = @splat(0);
            for (0..n_fft) |n| {
                const i = t * hop + n;
                const v = (if (i < x.len) @as(f64, x[i]) else 0.0) * m.window[n];
                if (v == 0) continue;
                for (0..bins) |k| {
                    re[k] += v * m.cos[n][k];
                    im[k] -= v * m.sin[n][k];
                }
            }
            for (&power, re, im) |*p, r, i| {
                const a: f32 = @floatCast(@sqrt(r * r + i * i));
                p.* = a * a;
            }
            for (0..mels) |j| {
                var s: f32 = 0;
                for (0..bins) |k| s += m.filters[k][j] * power[k];
                const l = @max(@log10(s), log_floor);
                out[j * nf + t] = (l + 4.0) / 4.0;
            }
        }
    }
};

test "frames and tokens" {
    try std.testing.expectEqual(@as(usize, 1), frames(10));
    try std.testing.expectEqual(@as(usize, 2), frames(401));
    try std.testing.expectEqual(@as(usize, 422), frames(400 + 421 * 160));
    try std.testing.expectEqual(@as(usize, 53), tokens(422));
}

//! WhisperFeatureExtractor's log-mel rows in float64: a centered, reflect-padded STFT, slaney mel filters, log10.
const std = @import("std");
const cfg = @import("config.zig");

const bins = cfg.n_fft / 2 + 1;

fn hzToMel(f: f64) f64 {
    const logstep = 27.0 / @log(6.4);
    return if (f >= 1000.0) 15.0 + @log(f / 1000.0) * logstep else 3.0 * f / 200.0;
}

fn melToHz(m: f64) f64 {
    const logstep = @log(6.4) / 27.0;
    return if (m >= 15.0) 1000.0 * @exp(logstep * (m - 15.0)) else 200.0 * m / 3.0;
}

/// transformers' mel_filter_bank(201, mels, 0, 8000, 16000, "slaney", "slaney"): [bins][mels], float64.
pub fn filters(gpa: std.mem.Allocator, mels: usize) ![]f64 {
    const out = try gpa.alloc(f64, bins * mels);
    errdefer gpa.free(out);
    const edges = try gpa.alloc(f64, mels + 2);
    defer gpa.free(edges);
    const lo = hzToMel(0);
    const hi = hzToMel(8000);
    for (edges, 0..) |*e, i| e.* = melToHz(lo + (hi - lo) * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(mels + 1)));
    for (0..bins) |b| {
        const f = @as(f64, @floatFromInt(cfg.sample_rate / 2)) * @as(f64, @floatFromInt(b)) / @as(f64, @floatFromInt(bins - 1));
        for (0..mels) |m| {
            const down = -(edges[m] - f) / (edges[m + 1] - edges[m]);
            const up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            const enorm = 2.0 / (edges[m + 2] - edges[m]);
            out[b * mels + m] = @max(0.0, @min(down, up)) * enorm;
        }
    }
    return out;
}

/// The features of `x` (16 kHz samples): [mels][frames] float32, frames = len / 160.
pub const Features = struct {
    data: []f32,
    mels: usize,
    frames: usize,

    pub fn deinit(f: *Features, gpa: std.mem.Allocator) void {
        gpa.free(f.data);
        f.* = undefined;
    }

    pub fn at(f: Features, mel: usize, frame: usize) f32 {
        return f.data[mel * f.frames + frame];
    }
};

/// Sample `i` of x reflect-padded by n_fft / 2 on each side (torch.stft's center=True, pad_mode="reflect").
fn padded(x: []const f32, i: isize) f64 {
    const n: isize = @intCast(x.len);
    var j = i - cfg.n_fft / 2;
    if (j < 0) j = -j;
    if (j >= n) j = 2 * (n - 1) - j;
    return x[@intCast(j)];
}

/// Log-mel features of `x`, as the package's processor gives them for one clip (no padding to 30 s).
pub fn features(gpa: std.mem.Allocator, x: []const f32, mels: usize) !Features {
    if (x.len <= cfg.n_fft / 2) return error.AudioTooShort;
    const n_frames = x.len / cfg.hop;
    const fb = try filters(gpa, mels);
    defer gpa.free(fb);
    const window = try gpa.alloc(f64, cfg.n_fft);
    defer gpa.free(window);
    // torch.hann_window(400): periodic
    for (window, 0..) |*w, i| w.* = 0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / cfg.n_fft);
    const twiddle = try gpa.alloc([2]f64, cfg.n_fft);
    defer gpa.free(twiddle);
    for (twiddle, 0..) |*t, k| {
        const a = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) / cfg.n_fft;
        t.* = .{ @cos(a), @sin(a) };
    }
    const logs = try gpa.alloc(f64, mels * n_frames);
    defer gpa.free(logs);
    var frame: [cfg.n_fft]f64 = undefined;
    var power: [bins]f64 = undefined;
    var top: f64 = -std.math.inf(f64);
    for (0..n_frames) |t| {
        for (&frame, 0..) |*v, i| v.* = padded(x, @intCast(t * cfg.hop + i)) * window[i];
        for (&power, 0..) |*p, b| {
            var re: f64 = 0;
            var im: f64 = 0;
            for (frame, 0..) |v, i| {
                const w = twiddle[(b * i) % cfg.n_fft];
                re += v * w[0];
                im += v * w[1];
            }
            p.* = re * re + im * im;
        }
        for (0..mels) |m| {
            var s: f64 = 0;
            for (power, 0..) |p, b| s += fb[b * mels + m] * p;
            const l = std.math.log10(@max(s, 1e-10));
            logs[m * n_frames + t] = l;
            top = @max(top, l);
        }
    }
    const out = try gpa.alloc(f32, mels * n_frames);
    for (out, logs) |*o, l| o.* = @floatCast((@max(l, top - 8.0) + 4.0) / 4.0);
    return .{ .data = out, .mels = mels, .frames = n_frames };
}

test "filters: slaney triangles, each mel's area-normalized peak" {
    const fb = try filters(std.testing.allocator, 128);
    defer std.testing.allocator.free(fb);
    // bin 0 (0 Hz) is only in mel 0's lower edge, at its start: zero
    try std.testing.expectEqual(@as(f64, 0), fb[0]);
    var nonzero: usize = 0;
    for (0..128) |m| {
        var any = false;
        for (0..bins) |b| any = any or fb[b * 128 + m] > 0;
        nonzero += @intFromBool(any);
    }
    // 128 slaney mels over 201 bins: the lowest are narrower than a bin, so a few see no bin centre
    try std.testing.expect(nonzero > 100);
}

test "features: a 440 Hz tone peaks in the mel band that holds 440 Hz, frames = samples / 160" {
    const a = std.testing.allocator;
    const x = try a.alloc(f32, 16000);
    defer a.free(x);
    for (x, 0..) |*v, i| v.* = @floatCast(0.5 * @sin(2.0 * std.math.pi * 440.0 * @as(f64, @floatFromInt(i)) / 16000.0));
    var f = try features(a, x, 128);
    defer f.deinit(a);
    try std.testing.expectEqual(@as(usize, 100), f.frames);
    var best: usize = 0;
    for (0..128) |m| if (f.at(m, 50) > f.at(best, 50)) {
        best = m;
    };
    const lo = melToHz(hzToMel(0) + (hzToMel(8000) - hzToMel(0)) * @as(f64, @floatFromInt(best)) / 129.0);
    const hi = melToHz(hzToMel(0) + (hzToMel(8000) - hzToMel(0)) * @as(f64, @floatFromInt(best + 2)) / 129.0);
    try std.testing.expect(lo < 440.0 and 440.0 < hi);
    try std.testing.expectError(error.AudioTooShort, features(a, x[0..200], 128));
}

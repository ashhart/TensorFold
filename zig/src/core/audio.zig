//! Audio shared by families and the server: torchaudio's sinc_interp_hann resample in fp32.
const std = @import("std");

/// torchaudio.functional.resample(x, from, to) for a float32 waveform.
pub fn resample(gpa: std.mem.Allocator, x: []const f32, from: u32, to: u32) ![]f32 {
    if (from == to) return gpa.dupe(f32, x);
    const g = std.math.gcd(from, to);
    const orig: usize = from / g;
    const new: usize = to / g;
    const lpw: f64 = 6;
    const base_freq: f64 = @as(f64, @floatFromInt(@min(orig, new))) * 0.99;
    const width: usize = @intFromFloat(@ceil(lpw * @as(f64, @floatFromInt(orig)) / base_freq));
    const klen = 2 * width + orig;
    const kern = try gpa.alloc(f32, new * klen);
    defer gpa.free(kern);
    const bf: f32 = @floatCast(base_freq);
    const pi: f32 = @floatCast(std.math.pi);
    const scale: f32 = @floatCast(base_freq / @as(f64, @floatFromInt(orig)));
    for (0..new) |k| for (0..klen) |m| {
        const idx: f32 = @as(f32, @floatFromInt(@as(i64, @intCast(m)) - @as(i64, @intCast(width)))) / @as(f32, @floatFromInt(orig));
        var t: f32 = -@as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(new)) + idx;
        t *= bf;
        t = std.math.clamp(t, -6.0, 6.0);
        const c = @cos(t * pi / 6.0 / 2.0);
        const window = c * c;
        t *= pi;
        const sinc: f32 = if (t == 0) 1.0 else @sin(t) / t;
        kern[k * klen + m] = sinc * (window * scale);
    };
    const len = x.len;
    const target: usize = @intFromFloat(@ceil(@as(f64, @floatFromInt(new * len)) / @as(f64, @floatFromInt(orig))));
    const out = try gpa.alloc(f32, target);
    for (out, 0..) |*o, n| {
        const i = n / new;
        const k = n % new;
        var acc: f32 = 0;
        const kr = kern[k * klen ..][0..klen];
        for (kr, 0..) |kv, m| {
            const p = @as(i64, @intCast(i * orig + m)) - @as(i64, @intCast(width)); // padded index -> x index
            if (p >= 0 and p < len) acc += kv * x[@intCast(p)];
        }
        o.* = acc;
    }
    return out;
}

test "resampling keeps torchaudio's output length and a constant signal's level" {
    const gpa = std.testing.allocator;
    const x = try gpa.alloc(f32, 2400);
    defer gpa.free(x);
    @memset(x, 0.25);
    const y = try resample(gpa, x, 24000, 44100);
    defer gpa.free(y);
    try std.testing.expectEqual(@as(usize, 4410), y.len);
    for (y[200 .. y.len - 200]) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.25), v, 2e-3);
}

test "resampling preserves sine frequency and amplitude when upsampling and downsampling" {
    const gpa = std.testing.allocator;
    const f0: f64 = 440.0;
    const in_rate: u32 = 44100;
    const out_rate: u32 = 16000;
    const x = try gpa.alloc(f32, in_rate / 2);
    defer gpa.free(x);
    for (x, 0..) |*s, i| s.* = @floatCast(0.5 * @sin(2.0 * std.math.pi * f0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(in_rate))));

    const down = try resample(gpa, x, in_rate, out_rate);
    defer gpa.free(down);
    const expect_len = (out_rate / 2);
    try std.testing.expectEqual(@as(usize, expect_len), down.len);

    var max_err: f32 = 0;
    for (down[400 .. down.len - 400], 400..) |got, i| {
        const ref: f32 = @floatCast(0.5 * @sin(2.0 * std.math.pi * f0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(out_rate))));
        max_err = @max(max_err, @abs(got - ref));
    }
    try std.testing.expect(max_err < 0.01);

    const up = try resample(gpa, down, out_rate, in_rate);
    defer gpa.free(up);
    var up_err: f32 = 0;
    for (up[1000 .. up.len - 1000], 1000..) |got, i| {
        const ref: f32 = @floatCast(0.5 * @sin(2.0 * std.math.pi * f0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(in_rate))));
        up_err = @max(up_err, @abs(got - ref));
    }
    try std.testing.expect(up_err < 0.015);
}

test "resampling maps an impulse to a localized symmetric sinc kernel with expected DC gain" {
    const gpa = std.testing.allocator;
    var x: [101]f32 = @splat(0);
    x[50] = 1.0;
    const y = try resample(gpa, &x, 24000, 48000);
    defer gpa.free(y);
    try std.testing.expectEqual(@as(usize, 202), y.len);
    var sum: f64 = 0;
    var peak_idx: usize = 0;
    var peak_val: f32 = -1;
    for (y, 0..) |v, i| {
        sum += v;
        if (v > peak_val) {
            peak_val = v;
            peak_idx = i;
        }
    }
    try std.testing.expectEqual(@as(usize, 100), peak_idx);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), peak_val, 0.05);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), sum, 0.02);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[y.len - 1], 1e-6);
}

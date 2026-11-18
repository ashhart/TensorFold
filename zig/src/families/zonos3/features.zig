//! The SSL teacher's input features on the host: 16 kHz audio, 80 Kaldi log-mel bins a frame, normalised, in pairs.
const std = @import("std");

pub const source_rate = 44100;
pub const rate = 16000;
pub const mel_bins = 80;
pub const width = 2 * mel_bins; // a stacked row
const frame = 400;
const hop = 160;
const fft = 512;
const bins = fft / 2 + 1;
const lowpass = 6;
const rolloff = 0.99;

/// torchaudio.functional.resample's kernels (sinc_interp_hann) over the rates' gcd steps; a phase's at phase * taps.
const Kernels = struct {
    orig: usize,
    target: usize,
    pad: usize,
    taps: usize,
    k: []f32,

    fn init(gpa: std.mem.Allocator, from: u32) !Kernels {
        const g = std.math.gcd(from, rate);
        const orig: usize = from / g;
        const target: usize = rate / g;
        const base: f64 = @as(f64, @floatFromInt(@min(orig, target))) * rolloff;
        const half: usize = @intFromFloat(@ceil(lowpass * @as(f64, @floatFromInt(orig)) / base));
        const taps = 2 * half + orig;
        const k = try gpa.alloc(f32, target * taps);
        for (0..target) |phase| {
            const shift: f64 = -@as(f32, @floatFromInt(phase)) / @as(f32, @floatFromInt(target));
            for (k[phase * taps ..][0..taps], 0..) |*w, j| {
                const idx = (@as(f64, @floatFromInt(j)) - @as(f64, @floatFromInt(half))) / @as(f64, @floatFromInt(orig));
                var t = (shift + idx) * base;
                t = @min(@max(t, -lowpass), lowpass);
                const c = @cos(t * std.math.pi / lowpass / 2.0);
                t *= std.math.pi;
                const sinc = if (t == 0) 1.0 else @sin(t) / t;
                w.* = @floatCast(sinc * c * c * (base / @as(f64, @floatFromInt(orig))));
            }
        }
        return .{ .orig = orig, .target = target, .pad = half, .taps = taps, .k = k };
    }
};

/// 44.1 kHz samples to 16 kHz (caller frees): ceil(160 n / 441) samples.
pub fn resample(gpa: std.mem.Allocator, x: []const f32) ![]f32 {
    return resampleFrom(gpa, x, source_rate);
}

/// Samples at `from` Hz to 16 kHz as torchaudio.functional.resample makes them (caller frees).
pub fn resampleFrom(gpa: std.mem.Allocator, x: []const f32, from: u32) ![]f32 {
    if (from == rate) return gpa.dupe(f32, x);
    const ks = try Kernels.init(gpa, from);
    defer gpa.free(ks.k);
    // ceil(target n / orig), and no more than the strided convolution's frames hold
    const frames = (ks.pad + x.len + ks.pad + ks.orig - ks.taps) / ks.orig + 1;
    const n = @min((x.len * ks.target + ks.orig - 1) / ks.orig, frames * ks.target);
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    // each output sample is its own sum, so the cores share them
    const Part = struct {
        fn run(y: []f32, first: usize, in: []const f32, kk: *const Kernels) void {
            for (y, first..) |*o, m| {
                const q = m / kk.target;
                const k = kk.k[(m % kk.target) * kk.taps ..][0..kk.taps];
                var acc: f32 = 0;
                for (k, 0..) |w, j| {
                    const p = q * kk.orig + j;
                    if (p >= kk.pad and p - kk.pad < in.len) acc = @mulAdd(f32, w, in[p - kk.pad], acc);
                }
                o.* = acc;
            }
        }
    };
    const cores = @min(16, std.Thread.getCpuCount() catch 1, @max(1, n / 16000));
    var threads: [16]?std.Thread = @splat(null);
    const share = (n + cores - 1) / cores;
    for (0..cores) |i| {
        const lo = @min(n, i * share);
        const hi = @min(n, lo + share);
        if (i + 1 < cores) threads[i] = std.Thread.spawn(.{}, Part.run, .{ out[lo..hi], lo, x, &ks }) catch null;
        if (threads[i] == null) Part.run(out[lo..hi], lo, x, &ks);
    }
    for (threads) |th| if (th) |t| t.join();
    return out;
}

fn mel(hz: f64) f64 {
    return 1127.0 * @log(1.0 + hz / 700.0);
}

/// Triangles in mel space from 20 Hz to Nyquist, over the FFT bins' mel frequencies: [bins][mel_bins].
fn filters(out: *[bins][mel_bins]f64) void {
    const lo = mel(20.0);
    const hi = mel(rate / 2);
    var centers: [mel_bins + 2]f64 = undefined;
    for (&centers, 0..) |*c, i| c.* = lo + (hi - lo) * @as(f64, @floatFromInt(i)) / (mel_bins + 1);
    const bin_width: f64 = @as(f64, rate) / fft;
    for (out, 0..) |*row, b| {
        const f = mel(bin_width * @as(f64, @floatFromInt(b)));
        for (row, 0..) |*v, m| {
            const down = (f - centers[m]) / (centers[m + 1] - centers[m]);
            const up = (centers[m + 2] - f) / (centers[m + 2] - centers[m + 1]);
            v.* = @max(0.0, @min(down, up));
        }
    }
}

/// The FFT's rotations, a stage after another: cos and sin of -2 pi k / len for len = 2, 4, .. fft and k < len / 2.
const Twiddles = struct {
    wr: [fft - 1]f64,
    wi: [fft - 1]f64,

    fn init() Twiddles {
        var t: Twiddles = undefined;
        var at: usize = 0;
        var len: usize = 2;
        while (len <= fft) : (len <<= 1) {
            const angle = -2.0 * std.math.pi / @as(f64, @floatFromInt(len));
            for (0..len / 2) |k| {
                t.wr[at] = @cos(angle * @as(f64, @floatFromInt(k)));
                t.wi[at] = @sin(angle * @as(f64, @floatFromInt(k)));
                at += 1;
            }
        }
        return t;
    }
};

/// In-place radix-2 FFT of `fft` points.
fn transform(re: *[fft]f64, im: *[fft]f64, tw: *const Twiddles) void {
    var j: usize = 0;
    for (1..fft) |i| {
        var bit: usize = fft >> 1;
        while (j & bit != 0) : (bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std.mem.swap(f64, &re[i], &re[j]);
            std.mem.swap(f64, &im[i], &im[j]);
        }
    }
    var len: usize = 2;
    var first: usize = 0;
    while (len <= fft) : (len <<= 1) {
        var start: usize = 0;
        while (start < fft) : (start += len) {
            for (0..len / 2) |k| {
                const wr = tw.wr[first + k];
                const wi = tw.wi[first + k];
                const a = start + k;
                const b = a + len / 2;
                const tr = re[b] * wr - im[b] * wi;
                const ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        first += len / 2;
    }
}

/// Kaldi log-mel rows [frames][mel_bins] of 16 kHz audio (caller frees); `hamming`: that window, else Povey's.
pub fn logMel(gpa: std.mem.Allocator, audio: []const f32, hamming: bool) ![]f32 {
    if (audio.len < frame) return error.AudioTooShort;
    const frames = 1 + (audio.len - frame) / hop;
    const bank = try gpa.create([bins][mel_bins]f64);
    defer gpa.destroy(bank);
    filters(bank);
    var window: [frame]f64 = undefined;
    for (&window, 0..) |*w, n| {
        const c = @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(n)) / (frame - 1));
        w.* = if (hamming) 0.54 - 0.46 * c else std.math.pow(f64, 0.5 - 0.5 * c, 0.85);
    }
    const logmel = try gpa.alloc(f32, frames * mel_bins);
    errdefer gpa.free(logmel);
    var re: [fft]f64 = undefined;
    var im: [fft]f64 = undefined;
    var power: [bins]f64 = undefined;
    const tw: Twiddles = .init();
    for (0..frames) |f| {
        var mean: f64 = 0;
        for (re[0..frame], audio[f * hop ..][0..frame]) |*r, x| {
            r.* = x * @as(f32, 32768.0);
            mean += r.*;
        }
        mean /= frame;
        for (re[0..frame]) |*r| r.* -= mean;
        var n: usize = frame - 1;
        while (n > 0) : (n -= 1) re[n] -= 0.97 * re[n - 1];
        re[0] *= 1.0 - 0.97;
        for (re[0..frame], window) |*r, w| r.* *= w;
        @memset(re[frame..], 0);
        @memset(&im, 0);
        transform(&re, &im, &tw);
        // the publisher's spectrogram holds single-precision complex values before the power
        for (&power, re[0..bins], im[0..bins]) |*p, r, i| {
            const mag = std.math.hypot(@as(f64, @as(f32, @floatCast(r))), @as(f64, @as(f32, @floatCast(i))));
            p.* = mag * mag;
        }
        for (0..mel_bins) |m| {
            var acc: f64 = 0;
            for (power, bank) |p, row| acc += row[m] * p;
            logmel[f * mel_bins + m] = @floatCast(@log(@max(acc, 1.192092955078125e-07)));
        }
    }
    return logmel;
}

/// Stacked feature rows [frames / 2][width] of 16 kHz audio (caller frees); at least two frames of audio.
pub fn rows(gpa: std.mem.Allocator, audio: []const f32) ![]f32 {
    if (audio.len < frame + hop) return error.AudioTooShort;
    const frames = (1 + (audio.len - frame) / hop) & ~@as(usize, 1); // an odd last frame has no pair
    const all = try logMel(gpa, audio, false);
    defer gpa.free(all);
    const logmel = all[0 .. frames * mel_bins];
    // each bin to zero mean and unit variance over the frames (n - 1 in the variance)
    for (0..mel_bins) |m| {
        var mean: f64 = 0;
        for (0..frames) |f| mean += logmel[f * mel_bins + m];
        mean /= @floatFromInt(frames);
        var ss: f64 = 0;
        for (0..frames) |f| {
            const d = logmel[f * mel_bins + m] - mean;
            ss += d * d;
        }
        const scale = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(frames - 1)) + 1e-7);
        for (0..frames) |f| logmel[f * mel_bins + m] = @floatCast((logmel[f * mel_bins + m] - mean) * scale);
    }
    // two frames a row: the buffer already holds them side by side
    return gpa.dupe(f32, logmel);
}

test "resample: a constant stays a constant away from the edges" {
    const t = std.testing;
    const x = try t.allocator.alloc(f32, 4410);
    defer t.allocator.free(x);
    @memset(x, 0.5);
    const y = try resample(t.allocator, x);
    defer t.allocator.free(y);
    try t.expectEqual(@as(usize, 1600), y.len);
    for (y[100..1500]) |v| try t.expectApproxEqAbs(@as(f32, 0.5), v, 2e-3);
}

test "transform: a cosine lands in its own bin" {
    const t = std.testing;
    var re: [fft]f64 = undefined;
    var im: [fft]f64 = @splat(0);
    for (&re, 0..) |*r, n| r.* = @cos(2.0 * std.math.pi * 5.0 * @as(f64, @floatFromInt(n)) / fft);
    transform(&re, &im, &Twiddles.init());
    try t.expectApproxEqAbs(@as(f64, fft / 2), re[5], 1e-9);
    try t.expectApproxEqAbs(@as(f64, 0), re[6], 1e-9);
    try t.expectApproxEqAbs(@as(f64, 0), im[5], 1e-9);
}

test "rows: the frames pair up, and each bin of a noisy lane is normalised" {
    const t = std.testing;
    const x = try t.allocator.alloc(f32, rate);
    defer t.allocator.free(x);
    var seed: u32 = 1;
    for (x) |*v| {
        seed = seed *% 1664525 +% 1013904223;
        v.* = (@as(f32, @floatFromInt(seed >> 8)) / 16777216.0 - 0.5) * 0.2;
    }
    const got = try rows(t.allocator, x);
    defer t.allocator.free(got);
    try t.expectEqual(@as(usize, 98 * mel_bins), got.len); // 98 frames, 49 rows
    var mean: f64 = 0;
    var i: usize = 3;
    while (i < got.len) : (i += mel_bins) mean += got[i];
    try t.expectApproxEqAbs(@as(f64, 0), mean / 98.0, 1e-4);
}

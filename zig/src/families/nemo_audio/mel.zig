//! NeMo's log-mel frontend on the host (FilterbankFeatures in eval), rounding where torch's bf16 path does.
const std = @import("std");
const Config = @import("config.zig");
const Allocator = std.mem.Allocator;

/// bf16 bits of x, round to nearest even (x finite or infinite).
pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn widen(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// x rounded to bf16 and back.
pub fn rbf(x: f32) f32 {
    return widen(bf16(x));
}

/// A 16 kHz mono WAV's samples in [-1, 1): 16-bit PCM over 32768, as soundfile reads it, or 32-bit float.
pub fn decodeWav(gpa: Allocator, bytes: []const u8) ![]f32 {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.NotWav;
    var at: usize = 12;
    var format: u16 = 0;
    var bits: u16 = 0;
    while (at + 8 <= bytes.len) {
        const id = bytes[at..][0..4];
        const len = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
        const body = bytes[at + 8 ..][0..@min(len, bytes.len - at - 8)];
        if (std.mem.eql(u8, id, "fmt ")) {
            if (body.len < 16) return error.BadWav;
            format = std.mem.readInt(u16, body[0..2], .little);
            const channels = std.mem.readInt(u16, body[2..4], .little);
            const rate = std.mem.readInt(u32, body[4..8], .little);
            bits = std.mem.readInt(u16, body[14..16], .little);
            if (format == 0xfffe and body.len >= 26) format = std.mem.readInt(u16, body[24..26], .little); // WAVE_FORMAT_EXTENSIBLE
            if (channels != 1) return error.WavNotMono;
            if (rate != Config.sample_rate) return error.WavRate;
        } else if (std.mem.eql(u8, id, "data")) {
            if (format == 1 and bits == 16) {
                const out = try gpa.alloc(f32, body.len / 2);
                for (out, 0..) |*o, i| o.* = @as(f32, @floatFromInt(std.mem.readInt(i16, body[2 * i ..][0..2], .little))) / 32768.0;
                return out;
            }
            if (format == 3 and bits == 32) {
                const out = try gpa.alloc(f32, body.len / 4);
                for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, body[4 * i ..][0..4], .little));
                return out;
            }
            return error.WavFormat;
        }
        at += 8 + len + (len & 1);
    }
    return error.BadWav;
}

pub const Features = struct {
    frames: usize, // 1 + samples / hop (center padding), the last past the valid length
    valid: usize, // samples / hop: frames from here on are zero
    mel: []u16, // [frames, 128] bf16: the Sortformer reads these as they are
    asr: []u16, // [frames, 128] bf16: normalized per feature over the valid frames, for the ASR encoder

    pub fn deinit(f: *Features, gpa: Allocator) void {
        gpa.free(f.mel);
        gpa.free(f.asr);
    }
};

/// Log-mel features of `wav` with the checkpoint's window [400] and filterbank [128, 257] (bf16 values as f32).
pub fn features(gpa: Allocator, wav: []const f32, window: []const f32, fb: []const f32) !Features {
    if (window.len != Config.win or fb.len != Config.mels * Config.bins) return error.BadFilterbank;
    if (wav.len < Config.hop) return error.AudioTooShort;
    const n = wav.len;
    const frames = n / Config.hop + 1;
    const valid = n / Config.hop;
    // preemphasis in fp32: x[i] - 0.97 x[i - 1]
    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    x[0] = wav[0];
    for (1..n) |i| x[i] = wav[i] - Config.preemph * wav[i - 1];
    // the 400-sample window centred in 512, as torch.stft pads a shorter window
    var w: [Config.n_fft]f32 = @splat(0);
    const left = (Config.n_fft - Config.win) / 2;
    @memcpy(w[left..][0..Config.win], window);
    var fft = Fft.init();
    const mel = try gpa.alloc(u16, frames * Config.mels);
    errdefer gpa.free(mel);
    var re: [Config.n_fft]f64 = undefined;
    var im: [Config.n_fft]f64 = undefined;
    var power: [Config.bins]f32 = undefined;
    for (0..frames) |t| {
        // frame t covers samples [t hop - 256, t hop + 256) of the zero-padded signal
        for (0..Config.n_fft, &re, &im) |i, *r, *m| {
            const s = @as(isize, @intCast(t * Config.hop + i)) - Config.n_fft / 2;
            const v: f32 = if (s >= 0 and s < n) x[@intCast(s)] else 0;
            r.* = @floatCast(v * w[i]);
            m.* = 0;
        }
        fft.run(&re, &im);
        for (&power, 0..) |*p, b| {
            const fr: f32 = @floatCast(re[b]);
            const fi: f32 = @floatCast(im[b]);
            const mag = @sqrt(fr * fr + fi * fi);
            p.* = mag * mag;
        }
        for (0..Config.mels) |m| {
            var acc: f64 = 0;
            for (fb[m * Config.bins ..][0..Config.bins], power) |f, p| acc += @as(f64, f) * p;
            const v: f32 = if (t < valid) @log(@as(f32, @floatCast(acc)) + Config.log_guard) else 0;
            mel[t * Config.mels + m] = bf16(v);
        }
    }
    const asr = try gpa.alloc(u16, frames * Config.mels);
    errdefer gpa.free(asr);
    normalize(mel, asr, frames, valid);
    return .{ .frames = frames, .valid = valid, .mel = mel, .asr = asr };
}

/// normalize_batch(per_feature) on bf16 features as torch rounds it (variance and std in fp32); later frames zero.
fn normalize(mel: []const u16, out: []u16, frames: usize, valid: usize) void {
    @memset(out, 0);
    if (valid == 0) return;
    const den: f32 = @floatFromInt(valid);
    for (0..Config.mels) |c| {
        const ref = widen(mel[c]);
        var sum: f64 = 0;
        for (0..valid) |t| sum += rbf(widen(mel[t * Config.mels + c]) - ref);
        const mean = rbf(ref + rbf(rbf(@floatCast(sum)) / den));
        var sq: f64 = 0;
        for (0..valid) |t| {
            const d = rbf(widen(mel[t * Config.mels + c]) - mean);
            sq += rbf(d * d);
        }
        var std_ = @sqrt(rbf(@floatCast(sq)) / (den - 1.0));
        if (std.math.isNan(std_)) std_ = 0;
        std_ += Config.norm_eps;
        for (0..@min(valid, frames)) |t| out[t * Config.mels + c] = bf16(rbf(widen(mel[t * Config.mels + c]) - mean) / std_);
    }
}

/// A 512-point complex radix-2 FFT in f64.
const Fft = struct {
    cos: [Config.n_fft / 2]f64,
    sin: [Config.n_fft / 2]f64,

    fn init() Fft {
        var f: Fft = undefined;
        for (0..Config.n_fft / 2) |k| {
            const a = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) / Config.n_fft;
            f.cos[k] = @cos(a);
            f.sin[k] = @sin(a);
        }
        return f;
    }

    fn run(f: *const Fft, re: *[Config.n_fft]f64, im: *[Config.n_fft]f64) void {
        const n = Config.n_fft;
        const bits = std.math.log2_int(usize, n);
        for (0..n) |i| {
            const j = @bitReverse(@as(u9, @intCast(i))) >> @intCast(9 - bits);
            if (j > i) {
                std.mem.swap(f64, &re[i], &re[j]);
                std.mem.swap(f64, &im[i], &im[j]);
            }
        }
        var len: usize = 2;
        while (len <= n) : (len *= 2) {
            const step = n / len;
            var i: usize = 0;
            while (i < n) : (i += len) for (0..len / 2) |k| {
                const wr = f.cos[k * step];
                const wi = f.sin[k * step];
                const a = i + k;
                const b = a + len / 2;
                const tr = re[b] * wr - im[b] * wi;
                const ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            };
        }
    }
};

test "the FFT of an impulse is flat and a cosine lands in its bin" {
    var f = Fft.init();
    var re: [Config.n_fft]f64 = @splat(0);
    var im: [Config.n_fft]f64 = @splat(0);
    re[0] = 1;
    f.run(&re, &im);
    for (re, im) |r, i| {
        try std.testing.expectApproxEqAbs(@as(f64, 1), r, 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, 0), i, 1e-12);
    }
    for (&re, &im, 0..) |*r, *i, t| {
        r.* = @cos(2.0 * std.math.pi * 5.0 * @as(f64, @floatFromInt(t)) / Config.n_fft);
        i.* = 0;
    }
    f.run(&re, &im);
    try std.testing.expectApproxEqAbs(@as(f64, Config.n_fft / 2), re[5], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0), re[6], 1e-9);
}

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // a tie stays even
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000)))); // a tie rounds up to even
}

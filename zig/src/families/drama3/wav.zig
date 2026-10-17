//! WAV in and out for the codec, and torchaudio's sinc_interp_hann resample in fp32 for reference clips.
const std = @import("std");

pub const Audio = struct {
    rate: u32,
    samples: []f32, // mono

    pub fn deinit(a: Audio, gpa: std.mem.Allocator) void {
        gpa.free(a.samples);
    }
};

/// RIFF WAVE: PCM 8 / 16 / 24 / 32-bit or IEEE float32, any channel count (averaged to mono).
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Audio {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.NotWav;
    var at: usize = 12;
    var fmt_tag: u16 = 0;
    var channels: u16 = 0;
    var rate: u32 = 0;
    var bits: u16 = 0;
    var data: ?[]const u8 = null;
    while (at + 8 <= bytes.len) {
        const id = bytes[at..][0..4];
        const len = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
        const body = bytes[at + 8 .. @min(bytes.len, at + 8 + len)];
        if (std.mem.eql(u8, id, "fmt ")) {
            if (body.len < 16) return error.BadWav;
            fmt_tag = std.mem.readInt(u16, body[0..2], .little);
            channels = std.mem.readInt(u16, body[2..4], .little);
            rate = std.mem.readInt(u32, body[4..8], .little);
            bits = std.mem.readInt(u16, body[14..16], .little);
            // extensible
            if (fmt_tag == 0xFFFE and body.len >= 26) fmt_tag = std.mem.readInt(u16, body[24..26], .little);
        } else if (std.mem.eql(u8, id, "data")) {
            data = body;
        }
        at += 8 + len + (len & 1);
    }
    const d = data orelse return error.BadWav;
    if (channels == 0 or rate == 0) return error.BadWav;
    const bps: usize = bits / 8;
    if (!((fmt_tag == 1 and (bits == 8 or bits == 16 or bits == 24 or bits == 32)) or (fmt_tag == 3 and bits == 32))) return error.UnsupportedWav;
    const frames = d.len / (bps * channels);
    const out = try gpa.alloc(f32, frames);
    for (out, 0..) |*o, i| {
        var sum: f32 = 0;
        for (0..channels) |c| {
            const p = d[(i * channels + c) * bps ..];
            const v: f32 = switch (fmt_tag) {
                3 => @bitCast(std.mem.readInt(u32, p[0..4], .little)),
                else => switch (bits) {
                    8 => (@as(f32, @floatFromInt(p[0])) - 128.0) / 128.0,
                    16 => @as(f32, @floatFromInt(std.mem.readInt(i16, p[0..2], .little))) / 32768.0,
                    24 => @as(f32, @floatFromInt(std.mem.readInt(i24, p[0..3], .little))) / 8388608.0,
                    else => @as(f32, @floatFromInt(std.mem.readInt(i32, p[0..4], .little))) / 2147483648.0,
                },
            };
            sum += v;
        }
        o.* = sum / @as(f32, @floatFromInt(channels));
    }
    return .{ .rate = rate, .samples = out };
}

/// 16-bit PCM mono: round(clamp(x, -1, 1) * 32767) (the oracle's write_wav).
pub fn encode16(gpa: std.mem.Allocator, samples: []const f32, rate: u32) ![]u8 {
    const n = samples.len;
    const out = try gpa.alloc(u8, 44 + n * 2);
    header(out[0..44], rate, n);
    for (samples, 0..) |s, i| {
        const v: i16 = @intFromFloat(@round(std.math.clamp(s, -1.0, 1.0) * 32767.0));
        std.mem.writeInt(i16, out[44 + 2 * i ..][0..2], v, .little);
    }
    return out;
}

/// A 16-bit mono header for n samples (streaming writers patch the sizes at the end; n = 0x7FFFFFF0 for unknown).
pub fn header(h: *[44]u8, rate: u32, n: usize) void {
    const data_len: u32 = @intCast(@min(n * 2, 0xFFFFFFF0));
    @memcpy(h[0..4], "RIFF");
    std.mem.writeInt(u32, h[4..8], data_len +% 36, .little);
    @memcpy(h[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, h[16..20], 16, .little);
    std.mem.writeInt(u16, h[20..22], 1, .little);
    std.mem.writeInt(u16, h[22..24], 1, .little);
    std.mem.writeInt(u32, h[24..28], rate, .little);
    std.mem.writeInt(u32, h[28..32], rate * 2, .little);
    std.mem.writeInt(u16, h[32..34], 2, .little);
    std.mem.writeInt(u16, h[34..36], 16, .little);
    @memcpy(h[36..40], "data");
    std.mem.writeInt(u32, h[40..44], data_len, .little);
}

pub fn pcm16(samples: []const f32, out: []u8) void {
    for (samples, 0..) |s, i| {
        const v: i16 = @intFromFloat(@round(std.math.clamp(s, -1.0, 1.0) * 32767.0));
        std.mem.writeInt(i16, out[2 * i ..][0..2], v, .little);
    }
}

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

test "a written 16-bit WAV parses back to its samples, clamped and rounded" {
    const gpa = std.testing.allocator;
    const x = [_]f32{ 0, 0.5, -0.5, 1.5, -2.0, 1.0 / 32767.0 };
    const bytes = try encode16(gpa, &x, 22050);
    defer gpa.free(bytes);
    const a = try parse(gpa, bytes);
    defer a.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 22050), a.rate);
    const want = [_]i16{ 0, 16384, -16384, 32767, -32767, 1 };
    for (a.samples, want) |s, w| try std.testing.expectEqual(@as(f32, @floatFromInt(w)) / 32768.0, s);
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

//! A whole speech take as the reply asks: 16 or 24-bit or float WAV, raw PCM or FLAC, at its rate and channel count.
const std = @import("std");
const audio = @import("core").audio;
const flac = @import("flac.zig");
const Allocator = std.mem.Allocator;

pub const Format = enum {
    wav,
    wav24,
    wav32f,
    pcm,
    flac,

    pub fn contentType(f: Format) []const u8 {
        return switch (f) {
            .wav, .wav24, .wav32f => "audio/wav",
            .pcm => "audio/pcm",
            .flac => "audio/flac",
        };
    }
};

/// The rates a reply may ask for: each resamples from 24 or 44.1 kHz with a kernel of a few hundred taps a phase.
pub const rates = [_]u32{ 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000 };

pub const Error = error{ NonFinite, TooLarge, OutOfMemory };

/// The speech route's quantization, (clip(x, -1, 1) * 32767) truncated toward zero.
pub fn quantize16(x: f32) i16 {
    return @intFromFloat(std.math.clamp(x, -1.0, 1.0) * 32767.0);
}

/// The same at 24 bits: clip(x, -1, 1) * 8388607, truncated toward zero.
pub fn quantize24(x: f32) i24 {
    return @intFromFloat(std.math.clamp(x, -1.0, 1.0) * 8388607.0);
}

/// A 44-byte RIFF/WAVE header: integer PCM (tag 1) or IEEE float (tag 3), samples interleaved by channel.
pub fn wavHeader(rate: u32, channels: u16, bits: u16, float: bool, data_bytes: u32) [44]u8 {
    const block: u16 = channels * (bits / 8);
    var h: [44]u8 = undefined;
    @memcpy(h[0..4], "RIFF");
    std.mem.writeInt(u32, h[4..8], 36 + data_bytes, .little);
    @memcpy(h[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, h[16..20], 16, .little); // fmt chunk size
    std.mem.writeInt(u16, h[20..22], if (float) 3 else 1, .little);
    std.mem.writeInt(u16, h[22..24], channels, .little);
    std.mem.writeInt(u32, h[24..28], rate, .little);
    std.mem.writeInt(u32, h[28..32], rate * block, .little); // byte rate
    std.mem.writeInt(u16, h[32..34], block, .little);
    std.mem.writeInt(u16, h[34..36], bits, .little);
    @memcpy(h[36..40], "data");
    std.mem.writeInt(u32, h[40..44], data_bytes, .little);
    return h;
}

/// The take resampled from `from` to `to` (one of `rates`), each sample repeated on every channel, in `format`.
pub fn encode(a: Allocator, take: []const f32, from: u32, to: u32, channels: u8, format: Format) Error![]u8 {
    std.debug.assert(channels == 1 or channels == 2);
    for (take) |x| if (!std.math.isFinite(x)) return error.NonFinite;
    const pcm = if (from == to) take else try audio.resample(a, take, from, to);
    defer if (from != to) a.free(pcm);
    for (pcm) |x| if (!std.math.isFinite(x)) return error.NonFinite;
    if (format == .flac) return flac.encode(a, pcm, to, channels);
    const width: usize = switch (format) {
        .wav, .pcm => 2,
        .wav24 => 3,
        .wav32f, .flac => 4,
    };
    const bytes = std.math.mul(usize, pcm.len, width * channels) catch return error.TooLarge;
    if (bytes > std.math.maxInt(u32) - 36) return error.TooLarge;
    const head: usize = if (format == .pcm) 0 else 44;
    const out = try a.alloc(u8, head + bytes);
    if (head > 0) out[0..44].* = wavHeader(to, channels, @intCast(8 * width), format == .wav32f, @intCast(bytes));
    var at = head;
    for (pcm) |x| for (0..channels) |_| {
        switch (format) {
            .wav, .pcm => std.mem.writeInt(i16, out[at..][0..2], quantize16(x), .little),
            .wav24 => std.mem.writeInt(i24, out[at..][0..3], quantize24(x), .little),
            .wav32f, .flac => std.mem.writeInt(u32, out[at..][0..4], @bitCast(x), .little),
        }
        at += width;
    };
    return out;
}

test "audio out: float WAV carries the take's bits, 24-bit WAV its quantization, stereo each sample twice" {
    const a = std.testing.allocator;
    const take = [_]f32{ 0.0, -0.0, 0.12345679, -0.9876543, 1.25, -1.5, 1.0 / 32768.0, std.math.floatMin(f32) };
    const float = try encode(a, &take, 44100, 44100, 2, .wav32f);
    defer a.free(float);
    try std.testing.expectEqualSlices(u8, &wavHeader(44100, 2, 32, true, take.len * 8), float[0..44]);
    try std.testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, float[20..22], .little));
    try std.testing.expectEqual(@as(u32, 44100 * 8), std.mem.readInt(u32, float[28..32], .little));
    const deep = try encode(a, &take, 44100, 44100, 1, .wav24);
    defer a.free(deep);
    try std.testing.expectEqual(@as(usize, 44 + take.len * 3), deep.len);
    try std.testing.expectEqual(@as(u16, 24), std.mem.readInt(u16, deep[34..36], .little));
    for (take, 0..) |x, i| {
        for (0..2) |c| try std.testing.expectEqual(@as(u32, @bitCast(x)), std.mem.readInt(u32, float[44 + 8 * i + 4 * c ..][0..4], .little));
        try std.testing.expectEqual(quantize24(x), std.mem.readInt(i24, deep[44 + 3 * i ..][0..3], .little));
    }
    try std.testing.expectEqual(@as(i24, 8388607), quantize24(1.25));
    try std.testing.expectEqual(@as(i24, -8388607), quantize24(-1.5));
    var bad = take;
    bad[3] = std.math.nan(f32);
    try std.testing.expectError(error.NonFinite, encode(a, &bad, 44100, 44100, 1, .wav24));
}

test "audio out: raw PCM is the 16-bit quantization, resampled to the asked rate, header-free" {
    const a = std.testing.allocator;
    var take: [4410]f32 = undefined;
    for (&take, 0..) |*x, i| x.* = @floatCast(0.6 * @sin(2.0 * std.math.pi * 440.0 * @as(f64, @floatFromInt(i)) / 44100.0));
    const pcm = try encode(a, &take, 44100, 22050, 2, .pcm);
    defer a.free(pcm);
    const half = try audio.resample(a, &take, 44100, 22050);
    defer a.free(half);
    try std.testing.expectEqual(@as(usize, 2205 * 4), pcm.len);
    for (half, 0..) |x, i| for (0..2) |c| try std.testing.expectEqual(quantize16(x), std.mem.readInt(i16, pcm[4 * i + 2 * c ..][0..2], .little));
}

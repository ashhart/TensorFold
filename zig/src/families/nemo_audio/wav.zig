//! A WAV file's samples as the encoder reads them: mono 16 kHz, each sample a float in [-1, 1).
const std = @import("std");

pub const rate = 16000;

/// RIFF WAVE, one channel at 16 kHz: PCM 16, 24 or 32-bit (scaled by 2^-(bits-1)) or IEEE float32. Caller frees.
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8) ![]f32 {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.NotWav;
    var at: usize = 12;
    var format: ?Format = null;
    var data: ?[]const u8 = null;
    while (at + 8 <= bytes.len) {
        const len = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
        if (len > bytes.len - at - 8) return error.BadWav;
        const body = bytes[at + 8 ..][0..len];
        if (std.mem.eql(u8, bytes[at..][0..4], "fmt ")) format = try Format.parse(body);
        if (std.mem.eql(u8, bytes[at..][0..4], "data")) data = body;
        at += 8 + len + (len & 1);
    }
    const f = format orelse return error.BadWav;
    const d = data orelse return error.BadWav;
    if (f.channels != 1) return error.WavNotMono;
    if (f.rate != rate) return error.WavNot16k;
    const width: usize = f.bits / 8;
    const out = try gpa.alloc(f32, d.len / width);
    errdefer gpa.free(out);
    for (out, 0..) |*o, i| {
        o.* = f.sample(d[i * width ..][0..width]);
        if (!std.math.isFinite(o.*)) return error.WavNotFinite;
    }
    return out;
}

const Format = struct {
    float: bool,
    channels: u16,
    rate: u32,
    bits: u16,

    fn parse(body: []const u8) !Format {
        if (body.len < 16) return error.BadWav;
        var tag = std.mem.readInt(u16, body[0..2], .little);
        // WAVE_FORMAT_EXTENSIBLE: the subformat GUID's first two bytes are the format tag
        if (tag == 0xFFFE and body.len >= 26) tag = std.mem.readInt(u16, body[24..26], .little);
        const f: Format = .{ .float = tag == 3, .channels = std.mem.readInt(u16, body[2..4], .little), .rate = std.mem.readInt(u32, body[4..8], .little), .bits = std.mem.readInt(u16, body[14..16], .little) };
        const pcm = tag == 1 and (f.bits == 16 or f.bits == 24 or f.bits == 32);
        if (!pcm and !(f.float and f.bits == 32)) return error.UnsupportedWav;
        return f;
    }

    fn sample(f: Format, p: []const u8) f32 {
        if (f.float) return @bitCast(std.mem.readInt(u32, p[0..4], .little));
        return switch (f.bits) {
            16 => @as(f32, @floatFromInt(std.mem.readInt(i16, p[0..2], .little))) / 32768.0,
            24 => @as(f32, @floatFromInt(std.mem.readInt(i24, p[0..3], .little))) / 8388608.0,
            else => @as(f32, @floatFromInt(std.mem.readInt(i32, p[0..4], .little))) / 2147483648.0,
        };
    }
};

fn fixture(buf: []u8, tag: u16, channels: u16, hz: u32, bits: u16, payload: []const u8) []const u8 {
    return fixtureFmt(buf, tag, null, channels, hz, bits, payload);
}

/// A WAV file; `sub`: WAVE_FORMAT_EXTENSIBLE (tag 0xFFFE) with this subformat tag in a 40-byte fmt chunk.
fn fixtureFmt(buf: []u8, tag: u16, sub: ?u16, channels: u16, hz: u32, bits: u16, payload: []const u8) []const u8 {
    const fmt_len: u32 = if (sub == null) 16 else 40;
    @memcpy(buf[0..4], "RIFF");
    std.mem.writeInt(u32, buf[4..8], @intCast(20 + fmt_len + payload.len), .little);
    @memcpy(buf[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, buf[16..20], fmt_len, .little);
    std.mem.writeInt(u16, buf[20..22], tag, .little);
    std.mem.writeInt(u16, buf[22..24], channels, .little);
    std.mem.writeInt(u32, buf[24..28], hz, .little);
    std.mem.writeInt(u32, buf[28..32], hz * channels * bits / 8, .little);
    std.mem.writeInt(u16, buf[32..34], channels * bits / 8, .little);
    std.mem.writeInt(u16, buf[34..36], bits, .little);
    if (sub) |t| {
        std.mem.writeInt(u16, buf[36..38], 22, .little); // cbSize
        std.mem.writeInt(u16, buf[38..40], bits, .little); // valid bits
        std.mem.writeInt(u32, buf[40..44], 4, .little); // channel mask: front centre
        @memcpy(buf[44..60], &[_]u8{ 0, 0, 0, 0, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 });
        std.mem.writeInt(u16, buf[44..46], t, .little);
    }
    const at = 20 + fmt_len;
    @memcpy(buf[at..][0..4], "data");
    std.mem.writeInt(u32, buf[at + 4 ..][0..4], @intCast(payload.len), .little);
    @memcpy(buf[at + 8 ..][0..payload.len], payload);
    return buf[0 .. at + 8 + payload.len];
}

test "nemo audio wav: PCM16 and float32 samples, scaled as stored" {
    const gpa = std.testing.allocator;
    var buf: [64]u8 = undefined;
    const pcm = [_]u8{ 0x00, 0x80, 0xff, 0x7f, 0x00, 0x40 };
    const s = try decode(gpa, fixture(&buf, 1, 1, rate, 16, &pcm));
    defer gpa.free(s);
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 32767.0 / 32768.0, 0.5 }, s);
    const floats = [_]f32{ 0.25, -0.125 };
    const g = try decode(gpa, fixture(&buf, 3, 1, rate, 32, std.mem.sliceAsBytes(&floats)));
    defer gpa.free(g);
    try std.testing.expectEqualSlices(f32, &floats, g);
}

test "nemo audio wav: other rates, stereo, 8-bit and truncated chunks are refused" {
    const gpa = std.testing.allocator;
    var buf: [64]u8 = undefined;
    const pcm = [_]u8{ 0, 0, 0, 0 };
    try std.testing.expectError(error.WavNot16k, decode(gpa, fixture(&buf, 1, 1, 44100, 16, &pcm)));
    try std.testing.expectError(error.WavNotMono, decode(gpa, fixture(&buf, 1, 2, rate, 16, &pcm)));
    try std.testing.expectError(error.UnsupportedWav, decode(gpa, fixture(&buf, 1, 1, rate, 8, &pcm)));
    const whole = fixture(&buf, 1, 1, rate, 16, &pcm);
    try std.testing.expectError(error.BadWav, decode(gpa, whole[0 .. whole.len - 2]));
    try std.testing.expectError(error.NotWav, decode(gpa, "RIFX"));
}

test "nemo audio wav: PCM24, PCM32 and WAVE_FORMAT_EXTENSIBLE, scaled as stored" {
    const gpa = std.testing.allocator;
    var buf: [96]u8 = undefined;
    const p24 = [_]u8{ 0x00, 0x00, 0x80, 0xff, 0xff, 0x7f, 0x00, 0x00, 0x40 };
    const a = try decode(gpa, fixture(&buf, 1, 1, rate, 24, &p24));
    defer gpa.free(a);
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 8388607.0 / 8388608.0, 0.5 }, a);
    const p32 = [_]u8{ 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0xc0 };
    const b = try decode(gpa, fixture(&buf, 1, 1, rate, 32, &p32));
    defer gpa.free(b);
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 0.5, -0.5 }, b);
    const p16 = [_]u8{ 0x00, 0x40, 0x00, 0xc0 };
    const c = try decode(gpa, fixtureFmt(&buf, 0xFFFE, 1, 1, rate, 16, &p16));
    defer gpa.free(c);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, -0.5 }, c);
    const floats = [_]f32{ 0.25, -0.125 };
    const d = try decode(gpa, fixtureFmt(&buf, 0xFFFE, 3, 1, rate, 32, std.mem.sliceAsBytes(&floats)));
    defer gpa.free(d);
    try std.testing.expectEqualSlices(f32, &floats, d);
    try std.testing.expectError(error.WavNotMono, decode(gpa, fixtureFmt(&buf, 0xFFFE, 1, 2, rate, 16, &p16)));
}

test "nemo audio wav: a float sample that is NaN or infinite is refused" {
    const gpa = std.testing.allocator;
    var buf: [64]u8 = undefined;
    inline for (.{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        const floats = [_]f32{ 0.25, bad, -0.125 };
        try std.testing.expectError(error.WavNotFinite, decode(gpa, fixture(&buf, 3, 1, rate, 32, std.mem.sliceAsBytes(&floats))));
    }
}

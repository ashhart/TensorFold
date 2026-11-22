//! A RIFF WAV clip's samples as the package reads them (soundfile, float32): 16 kHz mono, PCM16 or float32 data.
const std = @import("std");

fn le16(b: []const u8) u16 {
    return std.mem.readInt(u16, b[0..2], .little);
}

fn le32(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .little);
}

/// The samples of `bytes`, each PCM16 value v as v / 32768; refuses other rates, channel counts and encodings.
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8) ![]f32 {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.NotWav;
    var at: usize = 12;
    var format: ?struct { tag: u16, channels: u16, rate: u32, bits: u16 } = null;
    while (at + 8 <= bytes.len) {
        const id = bytes[at..][0..4];
        const len = le32(bytes[at + 4 ..]);
        const body_at = at + 8;
        if (len > bytes.len - body_at) return error.BadWav;
        const body = bytes[body_at..][0..len];
        if (std.mem.eql(u8, id, "fmt ")) {
            if (len < 16) return error.BadWav;
            format = .{ .tag = le16(body), .channels = le16(body[2..]), .rate = le32(body[4..]), .bits = le16(body[14..]) };
        } else if (std.mem.eql(u8, id, "data")) {
            const f = format orelse return error.BadWav;
            if (f.channels != 1) return error.WavNotMono;
            if (f.rate != 16000) return error.WavNot16kHz;
            const tag = f.tag; // WAVE_FORMAT_EXTENSIBLE is refused with the other encodings
            if (tag == 1 and f.bits == 16) {
                const out = try gpa.alloc(f32, len / 2);
                for (out, 0..) |*o, i| o.* = @as(f32, @floatFromInt(@as(i16, @bitCast(le16(body[2 * i ..]))))) / 32768.0;
                return out;
            }
            if (tag == 3 and f.bits == 32) {
                const out = try gpa.alloc(f32, len / 4);
                for (out, 0..) |*o, i| o.* = @bitCast(le32(body[4 * i ..]));
                return out;
            }
            return error.WavEncoding;
        }
        at = body_at + len + (len & 1);
    }
    return error.BadWav;
}

test "decode: PCM16 mono 16 kHz, other rates refused" {
    var buf: [48]u8 = undefined;
    @memcpy(buf[0..4], "RIFF");
    std.mem.writeInt(u32, buf[4..8], 40, .little);
    @memcpy(buf[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, buf[16..20], 16, .little);
    std.mem.writeInt(u16, buf[20..22], 1, .little);
    std.mem.writeInt(u16, buf[22..24], 1, .little);
    std.mem.writeInt(u32, buf[24..28], 16000, .little);
    std.mem.writeInt(u32, buf[28..32], 32000, .little);
    std.mem.writeInt(u16, buf[32..34], 2, .little);
    std.mem.writeInt(u16, buf[34..36], 16, .little);
    @memcpy(buf[36..40], "data");
    std.mem.writeInt(u32, buf[40..44], 4, .little);
    std.mem.writeInt(i16, buf[44..46], -16384, .little);
    std.mem.writeInt(i16, buf[46..48], 32767, .little);
    const x = try decode(std.testing.allocator, &buf);
    defer std.testing.allocator.free(x);
    try std.testing.expectEqualSlices(f32, &.{ -0.5, 32767.0 / 32768.0 }, x);
    std.mem.writeInt(u32, buf[24..28], 44100, .little);
    try std.testing.expectError(error.WavNot16kHz, decode(std.testing.allocator, &buf));
}

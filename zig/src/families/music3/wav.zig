//! 16-bit PCM WAV out: channels [C, N] float audio, round(clamp(x, -1, 1) 32767), interleaved.
const std = @import("std");

pub fn encode(gpa: std.mem.Allocator, audio: []const f32, channels: usize, rate: u32) ![]u8 {
    const n = audio.len / channels;
    const data_len = n * channels * 2;
    const out = try gpa.alloc(u8, 44 + data_len);
    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], @intCast(36 + data_len), .little);
    @memcpy(out[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, out[16..20], 16, .little);
    std.mem.writeInt(u16, out[20..22], 1, .little);
    std.mem.writeInt(u16, out[22..24], @intCast(channels), .little);
    std.mem.writeInt(u32, out[24..28], rate, .little);
    std.mem.writeInt(u32, out[28..32], rate * @as(u32, @intCast(channels)) * 2, .little);
    std.mem.writeInt(u16, out[32..34], @intCast(channels * 2), .little);
    std.mem.writeInt(u16, out[34..36], 16, .little);
    @memcpy(out[36..40], "data");
    std.mem.writeInt(u32, out[40..44], @intCast(data_len), .little);
    for (0..n) |i| for (0..channels) |c| {
        const v: i16 = @intFromFloat(@round(std.math.clamp(audio[c * n + i], -1.0, 1.0) * 32767.0));
        std.mem.writeInt(i16, out[44 + (i * channels + c) * 2 ..][0..2], v, .little);
    };
    return out;
}

test "a stereo header and samples" {
    const gpa = std.testing.allocator;
    const b = try encode(gpa, &.{ 0.0, 1.0, -1.0, 0.5 }, 2, 44100);
    defer gpa.free(b);
    try std.testing.expectEqual(@as(usize, 44 + 8), b.len);
    try std.testing.expectEqual(@as(i16, 0), std.mem.readInt(i16, b[44..46], .little));
    try std.testing.expectEqual(@as(i16, -32767), std.mem.readInt(i16, b[46..48], .little));
    try std.testing.expectEqual(@as(i16, 32767), std.mem.readInt(i16, b[48..50], .little));
}

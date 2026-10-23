const std = @import("std");
const speech = @import("speech.zig");
const quantizeInPlace = speech.quantizeInPlace;
const wavHeader = speech.wavHeader;

test "quantization clips, truncates toward zero and stays in place" {
    var pcm = [_]f32{ 0, 1, -1, 1.5, -7, 0.5, -0.5, 0.25, 1e-6, -1e-6 };
    const want = [_]i16{ 0, 32767, -32767, 32767, -32767, 16383, -16383, 8191, 0, 0 };
    const bytes = try quantizeInPlace(&pcm);
    try std.testing.expectEqual(@as(usize, 2 * want.len), bytes.len);
    try std.testing.expectEqual(@intFromPtr(&pcm), @intFromPtr(bytes.ptr));
    for (want, 0..) |w, i| try std.testing.expectEqual(w, std.mem.readInt(i16, bytes[2 * i ..][0..2], .little));
}

test "non-finite audio is refused" {
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        var pcm = [_]f32{ 0.1, bad, 0.2 };
        try std.testing.expectError(error.NonFinite, quantizeInPlace(&pcm));
    }
}

test "the WAV header describes mono 16-bit PCM at the advertised rate" {
    const h = wavHeader(44100, 4096);
    try std.testing.expectEqualStrings("RIFF", h[0..4]);
    try std.testing.expectEqual(@as(u32, 36 + 4096), std.mem.readInt(u32, h[4..8], .little));
    try std.testing.expectEqualStrings("WAVEfmt ", h[8..16]);
    try std.testing.expectEqual(@as(u32, 16), std.mem.readInt(u32, h[16..20], .little));
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, h[20..22], .little));
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, h[22..24], .little));
    try std.testing.expectEqual(@as(u32, 44100), std.mem.readInt(u32, h[24..28], .little));
    try std.testing.expectEqual(@as(u32, 88200), std.mem.readInt(u32, h[28..32], .little));
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, h[32..34], .little));
    try std.testing.expectEqual(@as(u16, 16), std.mem.readInt(u16, h[34..36], .little));
    try std.testing.expectEqualStrings("data", h[36..40]);
    try std.testing.expectEqual(@as(u32, 4096), std.mem.readInt(u32, h[40..44], .little));
}

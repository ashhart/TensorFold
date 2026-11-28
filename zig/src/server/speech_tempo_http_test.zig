//! The speech route's `speed` on the wire: today's WAV at 1, refusals without --speech-speed, stretched takes with it.
const std = @import("std");
const speech = @import("speech.zig");
const tempo = @import("speech_tempo.zig");
const wire = @import("speech_stream_test.zig");
const gpa = std.testing.allocator;

fn tone(n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    for (out, 0..) |*s, i| s.* = @floatCast(0.4 * @sin(2.0 * std.math.pi * 220.0 * @as(f64, @floatFromInt(i)) / 44100.0));
    return out;
}

/// Serves one request and returns its status line and body; the caller frees the body.
fn ask(audio: []const f32, speed_flag: bool, body: []const u8, status: *[12]u8) ![]u8 {
    var script: wire.Script = .{ .steps = &.{audio.len}, .audio = &.{audio}, .max_frames = 32, .inline_ = true };
    var s: wire.Served = undefined;
    try s.open(&script, .{ .served_name = "m", .model_ids = &.{"m"}, .speech = .{ .speed = speed_flag } }, "", "HTTP/1.1", body, 2000);
    defer s.deinit();
    const head = try s.readHead();
    status.* = head[0..12].*;
    const at = std.mem.indexOf(u8, head, "Content-Length: ") orelse return error.NoLength;
    const end = std.mem.indexOfScalarPos(u8, head, at, '\r').?;
    const len = try std.fmt.parseInt(usize, head[at + "Content-Length: ".len .. end], 10);
    const start = head.len;
    while (s.got.items.len < start + len) if (!try s.more(2000)) return error.Truncated;
    s.join(&script);
    return gpa.dupe(u8, s.got.items[start..][0..len]);
}

/// The WAV the route sends for `samples`, as speech.zig's header and quantization write it.
fn wav(samples: []const f32) ![]u8 {
    const copy = try gpa.dupe(f32, samples);
    defer gpa.free(copy);
    const data = try speech.quantizeInPlace(copy);
    const out = try gpa.alloc(u8, speech.wav_header_len + data.len);
    out[0..speech.wav_header_len].* = speech.wavHeader(44100, @intCast(data.len));
    @memcpy(out[speech.wav_header_len..], data);
    return out;
}

test "speech speed: absent, null and 1 send the take's WAV unchanged, flag or not" {
    const audio = try tone(3 * wire.frame);
    defer gpa.free(audio);
    const want = try wav(audio);
    defer gpa.free(want);
    for ([_]bool{ false, true }) |flag| for ([_][]const u8{
        "{\"model\":\"m\",\"input\":\"hi\"}",
        "{\"model\":\"m\",\"input\":\"hi\",\"speed\":null}",
        "{\"model\":\"m\",\"input\":\"hi\",\"speed\":1.0}",
        "{\"model\":\"m\",\"input\":\"hi\",\"speed\":1}",
    }) |body| {
        var status: [12]u8 = undefined;
        const got = try ask(audio, flag, body, &status);
        defer gpa.free(got);
        try std.testing.expectEqualStrings("HTTP/1.1 200", &status);
        try std.testing.expectEqualSlices(u8, want, got);
    };
}

test "speech speed: other values need --speech-speed, a range and a whole reply" {
    const audio = try tone(wire.frame);
    defer gpa.free(audio);
    for ([_]struct { flag: bool, body: []const u8, message: []const u8 }{
        .{ .flag = false, .body = "{\"model\":\"m\",\"input\":\"hi\",\"speed\":1.25}", .message = "speed other than 1 needs the server's --speech-speed flag" },
        .{ .flag = true, .body = "{\"model\":\"m\",\"input\":\"hi\",\"speed\":4.5}", .message = "speed must be a number from 0.25 to 4" },
        .{ .flag = true, .body = "{\"model\":\"m\",\"input\":\"hi\",\"speed\":\"fast\"}", .message = "speed must be a number from 0.25 to 4" },
        .{ .flag = true, .body = "{\"model\":\"m\",\"input\":\"hi\",\"speed\":true}", .message = "speed must be a number from 0.25 to 4" },
        .{ .flag = true, .body = "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"stream\":true,\"speed\":0.5}", .message = "stream=true requires speed 1" },
    }) |case| {
        var status: [12]u8 = undefined;
        const got = try ask(audio, case.flag, case.body, &status);
        defer gpa.free(got);
        try std.testing.expectEqualStrings("HTTP/1.1 400", &status);
        try std.testing.expect(std.mem.indexOf(u8, got, case.message) != null);
    }
}

test "speech speed: with --speech-speed the reply is the stretched take, its length over speed" {
    const audio = try tone(8 * wire.frame);
    defer gpa.free(audio);
    for ([_]f64{ 0.75, 1.25 }) |speed| {
        const stretched = try tempo.stretch(gpa, audio, 44100, speed);
        defer gpa.free(stretched);
        const want = try wav(stretched);
        defer gpa.free(want);
        var body: [96]u8 = undefined;
        var status: [12]u8 = undefined;
        const got = try ask(audio, true, try std.fmt.bufPrint(&body, "{{\"model\":\"m\",\"input\":\"hi\",\"speed\":{d}}}", .{speed}), &status);
        defer gpa.free(got);
        try std.testing.expectEqualStrings("HTTP/1.1 200", &status);
        try std.testing.expectEqualSlices(u8, want, got);
        const ratio = @as(f64, @floatFromInt(audio.len)) / speed / @as(f64, @floatFromInt(stretched.len));
        try std.testing.expectApproxEqAbs(@as(f64, 1), ratio, 0.01);
    }
}

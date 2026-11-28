//! ZONOS3 speech tempo on the wire: speed defaults retain bytes and non-1 stretches before the output encoder.
const std = @import("std");
const zonos = @import("zonos_speech.zig");
const tempo = @import("speech_tempo.zig");
const audio_out = @import("audio_out.zig");
const wire = @import("speech_stream_test.zig");
const gpa = std.testing.allocator;

const reference = "\"speaker_reference_codes\":[[0,0,0,0,0,0,0,0,0,0]]";

fn tone(n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    for (out, 0..) |*s, i| s.* = @floatCast(0.4 * @sin(2.0 * std.math.pi * 220.0 * @as(f64, @floatFromInt(i)) / 44100.0));
    return out;
}

fn fakeState() zonos.State {
    return .{ .voices = &.{}, .silence = @splat(0), .default = null };
}

/// Serves one ZONOS3 request and returns its status line and body; the caller frees the body.
fn ask(audio: []const f32, state: *zonos.State, body: []const u8, status: *[12]u8) ![]u8 {
    var script: wire.Script = .{ .steps = &.{audio.len}, .audio = &.{audio}, .max_frames = 32, .inline_ = true, .zonos = state };
    var s: wire.Served = undefined;
    try s.open(&script, .{ .served_name = "m", .model_ids = &.{"m"}, .speech = .{ .speed = true } }, "", "HTTP/1.1", body, 2000);
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

test "ZONOS3 speech speed: defaults retain encoded bytes and non-1 stretches before resampling" {
    const audio = try tone(8 * wire.frame);
    defer gpa.free(audio);
    var state = fakeState();
    const stereo_rate = "\"response_format\":\"wav32f\",\"sample_rate\":22050,\"channels\":2";
    for ([_]struct { body: []const u8, speed: f64 }{
        .{ .body = "{\"model\":\"m\",\"input\":\"hi\"," ++ stereo_rate ++ "," ++ reference ++ "}", .speed = 1 },
        .{ .body = "{\"model\":\"m\",\"input\":\"hi\",\"speed\":null," ++ stereo_rate ++ "," ++ reference ++ "}", .speed = 1 },
        .{ .body = "{\"model\":\"m\",\"input\":\"hi\",\"speed\":1.0," ++ stereo_rate ++ "," ++ reference ++ "}", .speed = 1 },
        .{ .body = "{\"model\":\"m\",\"input\":\"hi\",\"speed\":1.25," ++ stereo_rate ++ "," ++ reference ++ "}", .speed = 1.25 },
    }) |case| {
        const pcm = if (case.speed == 1) audio else try tempo.stretch(gpa, audio, 44100, case.speed);
        defer if (case.speed != 1) gpa.free(pcm);
        const want = try audio_out.encode(gpa, pcm, 44100, 22050, 2, .wav32f);
        defer gpa.free(want);
        var status: [12]u8 = undefined;
        const got = try ask(audio, &state, case.body, &status);
        defer gpa.free(got);
        try std.testing.expectEqualStrings("HTTP/1.1 200", &status);
        try std.testing.expectEqualSlices(u8, want, got);
    }
}

test "ZONOS3 speech speed: a non-1 stream is refused" {
    const audio = try tone(wire.frame);
    defer gpa.free(audio);
    var state = fakeState();
    var status: [12]u8 = undefined;
    const got = try ask(audio, &state, "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"stream\":true,\"speed\":0.5," ++ reference ++ "}", &status);
    defer gpa.free(got);
    try std.testing.expectEqualStrings("HTTP/1.1 400", &status);
    try std.testing.expect(std.mem.indexOf(u8, got, "stream=true requires speed 1") != null);
}

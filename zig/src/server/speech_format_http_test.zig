//! The speech route's output formats on the wire: today's WAV and PCM unchanged, the new formats opt-in per request.
const std = @import("std");
const speech = @import("speech.zig");
const audio_out = @import("audio_out.zig");
const flac = @import("flac.zig");
const audio = @import("core").audio;
const wire = @import("speech_stream_test.zig");
const gpa = std.testing.allocator;

const Reply = struct { status: [12]u8, content_type: []u8, body: []u8 };

fn tone() ![]f32 {
    const out = try gpa.alloc(f32, 2 * wire.frame + 100);
    for (out, 0..) |*s, i| s.* = @floatCast(0.6 * @sin(2.0 * std.math.pi * 440.0 * @as(f64, @floatFromInt(i)) / 44100.0));
    return out;
}

fn header(head: []const u8, name: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, head, name) orelse return error.NoHeader;
    return head[at + name.len .. std.mem.indexOfScalarPos(u8, head, at, '\r').?];
}

/// One whole reply to `body` from a scripted take of `take`; the caller frees it.
fn ask(take: []const f32, body: []const u8) !Reply {
    var script: wire.Script = .{ .steps = &.{take.len}, .audio = &.{take}, .inline_ = true };
    var s: wire.Served = undefined;
    try s.open(&script, .{ .served_name = "m", .model_ids = &.{"m"} }, "", "HTTP/1.1", body, 2000);
    defer s.deinit();
    const head = try s.readHead();
    const status = head[0..12].*;
    const content_type = try gpa.dupe(u8, try header(head, "Content-Type: "));
    errdefer gpa.free(content_type);
    const len = try std.fmt.parseInt(usize, try header(head, "Content-Length: "), 10);
    const start = head.len;
    while (s.got.items.len < start + len) if (!try s.more(2000)) return error.Truncated;
    s.join(&script);
    return .{ .status = status, .content_type = content_type, .body = try gpa.dupe(u8, s.got.items[start..][0..len]) };
}

fn free(r: Reply) void {
    gpa.free(r.content_type);
    gpa.free(r.body);
}

fn quantized(take: []const f32) ![]u8 {
    const copy = try gpa.dupe(f32, take);
    defer gpa.free(copy);
    return gpa.dupe(u8, try speech.quantizeInPlace(copy));
}

test "speech formats: wav and pcm stay today's bytes, with or without the new fields at their defaults" {
    const take = try tone();
    defer gpa.free(take);
    const data = try quantized(take);
    defer gpa.free(data);
    const head = speech.wavHeader(44100, @intCast(data.len));
    for ([_][]const u8{
        "{\"model\":\"m\",\"input\":\"hi\"}",
        "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"wav\",\"sample_rate\":44100,\"channels\":1}",
        "{\"model\":\"m\",\"input\":\"hi\",\"sample_rate\":null,\"channels\":null}",
        "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\"}",
    }) |body| {
        const r = try ask(take, body);
        defer free(r);
        try std.testing.expectEqualStrings("HTTP/1.1 200", &r.status);
        const pcm = std.mem.indexOf(u8, body, "pcm") != null;
        try std.testing.expectEqualStrings(if (pcm) "audio/pcm" else "audio/wav", r.content_type);
        if (!pcm) try std.testing.expectEqualSlices(u8, &head, r.body[0..44]);
        try std.testing.expectEqualSlices(u8, data, r.body[if (pcm) 0 else 44..]);
    }
}

test "speech formats: flac, wav24, wav32f, a sample_rate and stereo each carry the take" {
    const take = try tone();
    defer gpa.free(take);
    {
        const r = try ask(take, "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"flac\",\"channels\":2}");
        defer free(r);
        try std.testing.expectEqualStrings("audio/flac", r.content_type);
        const want = try flac.encode(gpa, take, 44100, 2);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u8, want, r.body);
    }
    {
        const r = try ask(take, "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"wav32f\"}");
        defer free(r);
        try std.testing.expectEqualStrings("audio/wav", r.content_type);
        try std.testing.expectEqualSlices(u8, &audio_out.wavHeader(44100, 1, 32, true, @intCast(4 * take.len)), r.body[0..44]);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(take), r.body[44..]);
    }
    {
        const r = try ask(take, "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"wav24\",\"sample_rate\":22050,\"channels\":2}");
        defer free(r);
        const half = try audio.resample(gpa, take, 44100, 22050);
        defer gpa.free(half);
        try std.testing.expectEqualSlices(u8, &audio_out.wavHeader(22050, 2, 24, false, @intCast(6 * half.len)), r.body[0..44]);
        for (half, 0..) |x, i| for (0..2) |c|
            try std.testing.expectEqual(audio_out.quantize24(x), std.mem.readInt(i24, r.body[44 + 6 * i + 3 * c ..][0..3], .little));
    }
}

test "speech formats: other codecs, rates, channel counts and streamed options are refused" {
    const take = try tone();
    defer gpa.free(take);
    for ([_][2][]const u8{
        .{ "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"mp3\"}", "response_format mp3 is not supported: use flac, pcm, wav, wav24 or wav32f" },
        .{ "{\"model\":\"m\",\"input\":\"hi\",\"sample_rate\":44101}", "sample_rate must be one of: 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000" },
        .{ "{\"model\":\"m\",\"input\":\"hi\",\"sample_rate\":\"48000\"}", "sample_rate must be one of: " },
        .{ "{\"model\":\"m\",\"input\":\"hi\",\"channels\":3}", "channels must be 1 or 2" },
        .{ "{\"model\":\"m\",\"input\":\"hi\",\"channels\":true}", "channels must be 1 or 2" },
        .{ "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"stream\":true,\"channels\":2}", "stream=true requires the model's sample_rate and channels 1" },
        .{ "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"stream\":true,\"sample_rate\":24000}", "stream=true requires the model's sample_rate and channels 1" },
        .{ "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"flac\",\"stream\":true}", "stream=true requires response_format='pcm'" },
    }) |case| {
        const r = try ask(take, case[0]);
        defer free(r);
        try std.testing.expectEqualStrings("HTTP/1.1 400", &r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, case[1]) != null);
    }
}

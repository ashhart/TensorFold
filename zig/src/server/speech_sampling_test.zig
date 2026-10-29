//! Speech sampling controls through the HTTP route and its scripted engine.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const harness = @import("speech_stream_test.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_allow = @import("drama_allow.zig");
const gpa = std.testing.allocator;
const config: server.Config = .{ .served_name = "m", .model_ids = &.{"m"}, .seed_salt = 17 };
const defaults: harness.Sampling = .{ .temperature = 1, .top_p = 0.9, .top_k = 30 };

fn exchange(script: *harness.Script, cfg: server.Config, body: []const u8) ![]u8 {
    var s: harness.Served = undefined;
    try s.open(script, cfg, "", "HTTP/1.1", body, 2000);
    defer s.deinit();
    defer s.join(script);
    _ = try s.readHead();
    _ = try s.readRest();
    return gpa.dupe(u8, s.got.items);
}

fn promptSeed(text: []const u8, attempt: i64) !u64 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var bytes: harness.Bytes = .{};
    const prompt = try drama_prompt.encode(arena.allocator(), bytes.text(), text);
    return api.seedFor(prompt, config.seed_salt +% attempt);
}

fn expectDraw(script: *const harness.Script, index: usize, sampling: harness.Sampling, seed: u64) !void {
    try std.testing.expectEqualDeep(sampling, script.sampling[index]);
    try std.testing.expectEqual(seed, script.seeds[index]);
}

test "speech sampling: omitted and null controls retain default parameters and PCM" {
    const bodies = [_][]const u8{
        "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\"}",
        "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"temperature\":null,\"top_p\":null,\"top_k\":null,\"seed\":null}",
    };
    var first: ?[]u8 = null;
    defer if (first) |pcm| gpa.free(pcm);
    for (bodies) |body| {
        var script: harness.Script = .{ .steps = &.{harness.frame}, .inline_ = true };
        const reply = try exchange(&script, config, body);
        defer gpa.free(reply);
        try harness.expectHeader(reply, "HTTP/1.1 200");
        try std.testing.expectEqual(@as(usize, 1), script.takes);
        try expectDraw(&script, 0, defaults, try promptSeed("hi", 0));
        const pcm = reply[(std.mem.indexOf(u8, reply, "\r\n\r\n").? + 4)..];
        if (first) |before| try std.testing.expectEqualSlices(u8, before, pcm) else first = try gpa.dupe(u8, pcm);
    }
}

test "speech sampling: custom parameters and exact integer seeds reach whole and streamed requests" {
    const cases = [_]struct { fields: []const u8, sampling: harness.Sampling, seed: u64 }{
        .{ .fields = "\"temperature\":1.25,\"top_p\":0.8,\"top_k\":64,\"seed\":9007199254740993", .sampling = .{ .temperature = 1.25, .top_p = 0.8, .top_k = 64 }, .seed = 9007199254740993 },
        .{ .fields = "\"temperature\":0,\"top_p\":1,\"top_k\":1,\"seed\":0", .sampling = .{ .temperature = 0, .top_p = 1, .top_k = 1 }, .seed = 0 },
        .{ .fields = "\"seed\":18446744073709551615", .sampling = defaults, .seed = std.math.maxInt(u64) },
        .{ .fields = "\"temperature\":0.5", .sampling = .{ .temperature = 0.5, .top_p = 0.9, .top_k = 30 }, .seed = try promptSeed("hi", 0) },
        .{ .fields = "\"temperature\":1e-400", .sampling = .{ .temperature = 0, .top_p = 0.9, .top_k = 30 }, .seed = try promptSeed("hi", 0) },
        .{ .fields = "\"temperature\":-1e-400", .sampling = .{ .temperature = 0, .top_p = 0.9, .top_k = 30 }, .seed = try promptSeed("hi", 0) },
        .{ .fields = "\"temperature\":-0", .sampling = .{ .temperature = 0, .top_p = 0.9, .top_k = 30 }, .seed = try promptSeed("hi", 0) },
        .{ .fields = "\"temperature\":3.4028234663852886e38,\"top_p\":1e-45", .sampling = .{ .temperature = std.math.floatMax(f32), .top_p = 1e-45, .top_k = 30 }, .seed = try promptSeed("hi", 0) },
    };
    for (cases) |case| for ([_]bool{ false, true }) |stream| {
        const body = try std.fmt.allocPrint(gpa, "{{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"stream\":{s},{s}}}", .{ if (stream) "true" else "false", case.fields });
        defer gpa.free(body);
        var script: harness.Script = .{ .steps = &.{harness.frame}, .inline_ = true };
        const reply = try exchange(&script, config, body);
        defer gpa.free(reply);
        try harness.expectHeader(reply, "HTTP/1.1 200");
        if (stream) try harness.expectHeader(reply, "Transfer-Encoding: chunked");
        try std.testing.expectEqual(@as(usize, 1), script.takes);
        try expectDraw(&script, 0, case.sampling, case.seed);
        try std.testing.expect(!std.math.signbit(script.sampling[0].temperature));
    };
}

test "speech sampling: invalid controls fail before any engine submission" {
    const fields = [_][]const u8{
        "\"top_p\":1e-400",               "\"top_p\":-1e-400",
        "\"temperature\":true",           "\"temperature\":\"1\"",
        "\"temperature\":[]",             "\"temperature\":{}",
        "\"temperature\":-1",             "\"temperature\":NaN",
        "\"temperature\":Infinity",       "\"temperature\":1e39",
        "\"temperature\":1e-100",         "\"top_p\":false",
        "\"top_p\":\"0.9\"",              "\"top_p\":0",
        "\"top_p\":-0.1",                 "\"top_p\":1.00000001",
        "\"top_p\":NaN",                  "\"top_p\":1e-100",
        "\"top_p\":Infinity",             "\"top_k\":true",
        "\"top_k\":\"30\"",               "\"top_k\":1.5",
        "\"top_k\":30.0",                 "\"top_k\":0",
        "\"top_k\":-1",                   "\"top_k\":65",
        "\"top_k\":18446744073709551616", "\"seed\":true",
        "\"seed\":\"1\"",                 "\"seed\":1.5",
        "\"seed\":1.0",                   "\"seed\":-1",
        "\"seed\":18446744073709551616",  "\"seed\":NaN",
    };
    for (fields) |field| {
        const body = try std.fmt.allocPrint(gpa, "{{\"model\":\"m\",\"input\":\"hi\",{s}}}", .{field});
        defer gpa.free(body);
        var script: harness.Script = .{ .steps = &.{harness.frame}, .inline_ = true };
        const reply = try exchange(&script, config, body);
        defer gpa.free(reply);
        try harness.expectHeader(reply, "HTTP/1.1 400");
        try std.testing.expectEqual(@as(usize, 0), script.takes);
    }
}

test "speech sampling: references and re-prompts retain overrides and wrap explicit retry seeds" {
    const codes = [_][10]i32{.{ 7, 1, 2, 3, 4, 5, 6, 7, 8, 9 }};
    const refs = [_]@import("drama_prompt").Reference{.{ .transcript = "A small cloud passes.", .codes = &codes }};
    const voices = [_]@import("drama_voice.zig").Voice{.{ .name = "tone", .references = &refs }};
    var cfg = config;
    cfg.redraw = .{ .max = 2, .controls = try drama_allow.parse("sighs") };
    for ([_]bool{ false, true }) |referenced| {
        var script: harness.Script = .{ .steps = &.{harness.frame}, .inline_ = true, .reasons = &.{ .silence, .silence, .stop }, .voices = &voices };
        const body = try std.fmt.allocPrint(gpa, "{{\"model\":\"m\",\"input\":\"hi [sighs] [laughs] there\",\"voice\":\"{s}\",\"temperature\":0.5,\"top_p\":0.75,\"top_k\":8,\"seed\":18446744073709551615}}", .{if (referenced) "tone" else "default"});
        defer gpa.free(body);
        const reply = try exchange(&script, cfg, body);
        defer gpa.free(reply);
        try harness.expectHeader(reply, "HTTP/1.1 200");
        try std.testing.expectEqual(@as(usize, 3), script.takes);
        for ([_]u64{ std.math.maxInt(u64), 0, 1 }, 0..) |seed, k|
            try expectDraw(&script, k, .{ .temperature = 0.5, .top_p = 0.75, .top_k = 8 }, seed);
        try std.testing.expectEqual(referenced, script.reference_rows[0] > 0);
    }
}

test "speech sampling: no seed keeps prompt-keyed redraws after stripping controls" {
    var script: harness.Script = .{ .steps = &.{harness.frame}, .inline_ = true, .reasons = &.{ .silence, .stop } };
    var cfg = config;
    cfg.redraw = .{ .max = 1, .controls = try drama_allow.parse("sighs") };
    const reply = try exchange(&script, cfg, "{\"model\":\"m\",\"input\":\"hi [sighs] [laughs] there\",\"temperature\":0.5}");
    defer gpa.free(reply);
    try harness.expectHeader(reply, "HTTP/1.1 200");
    try std.testing.expectEqual(@as(usize, 2), script.takes);
    const sampling: harness.Sampling = .{ .temperature = 0.5, .top_p = 0.9, .top_k = 30 };
    try expectDraw(&script, 0, sampling, try promptSeed("hi [sighs] [laughs] there", 0));
    try expectDraw(&script, 1, sampling, try promptSeed("hi [sighs] there", 1));
}

test "speech sampling: a silenced stream retains controls and does not redraw" {
    var script: harness.Script = .{ .steps = &.{harness.frame}, .inline_ = true, .reason = .silence };
    var cfg = config;
    cfg.redraw = .{ .max = 2 };
    const reply = try exchange(&script, cfg, "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"stream\":true,\"temperature\":0.5,\"seed\":2}");
    defer gpa.free(reply);
    try harness.expectHeader(reply, "HTTP/1.1 200");
    try harness.expectHeader(reply, "X-Finish-Reason: silence");
    try std.testing.expectEqual(@as(usize, 1), script.takes);
    try expectDraw(&script, 0, .{ .temperature = 0.5, .top_p = 0.9, .top_k = 30 }, 2);
}

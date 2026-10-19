//! SSE framing and every status and error code without sockets.
const std = @import("std");
const suite = @import("benchmark_suite.zig");
const sse = @import("benchmark_sse.zig");

const ms = std.time.ns_per_ms;

/// Lines of ``bytes`` through one stream, the i-th line read at ``at_lines[i]`` (or 0), then the post-loop checks.
fn play(a: std.mem.Allocator, bytes: []const u8, at_lines: []const i96, tokens: u64) !suite.Sample {
    return playAt(a, bytes, at_lines, tokens, 1.0);
}

fn playAt(a: std.mem.Allocator, bytes: []const u8, at_lines: []const i96, tokens: u64, temperature: f64) !suite.Sample {
    var sample: suite.Sample = .{ .fixture_id = "code", .temperature = temperature, .repeat = 0, .seed = 1234 };
    var s = try sse.Stream.init(a);
    var failure: ?suite.ErrorCode = null;
    var rest = bytes;
    var i: usize = 0;
    while (std.mem.indexOfScalar(u8, rest, '\n')) |nl| : (i += 1) {
        const at = if (i < at_lines.len) at_lines[i] else if (at_lines.len > 0) at_lines[at_lines.len - 1] else 0;
        const at_done = s.line(rest[0 .. nl + 1], at) catch |f| {
            failure = sse.code(f);
            break;
        };
        rest = rest[nl + 1 ..];
        if (at_done) break;
    }
    if (failure == null) s.finish(tokens, &sample) catch |f| {
        failure = sse.code(f);
    };
    s.timings(0, 100 * ms, &sample);
    if (failure) |c| {
        sample.status = .@"error";
        sample.error_code = c;
    }
    return sample;
}

const text1 = "data: {\"choices\": [{\"index\": 0, \"text\": \"synthetic\"}]}\n\n";
const text2 = "data: {\"choices\": [{\"index\": 0, \"text\": \" output\"}]}\n\n";
const final = "data: {\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"length\"}], \"tensorfold\": {\"tokens_per_second\": 42.0, \"prefill_seconds\": 0.03, \"token_sha\": \"0123456789AB\"}}\n\n";
const usage4 = "data: {\"choices\": [], \"usage\": {\"prompt_tokens\": 24, \"completion_tokens\": 4, \"prompt_tokens_details\": {\"cached_tokens\": 0}}}\n\n";
const done = "data: [DONE]\n\n";
/// Stamps: the first text at line 0, the second at line 2, the rest later.
const stamps = [_]i96{ 10 * ms, 10 * ms, 30 * ms, 30 * ms, 40 * ms };

fn joined(a: std.mem.Allocator, parts: []const []const u8) ![]const u8 {
    return std.mem.concat(a, u8, parts);
}

test "CRLF data lines of one event and a keepalive comment make one event" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try sse.Stream.init(a);
    const event = ": keepalive\r\n\r\ndata: {\"choices\": [{\"index\": 0,\r\ndata:  \"text\": \"x\"}]}\r\n\r\n";
    var rest: []const u8 = event;
    while (std.mem.indexOfScalar(u8, rest, '\n')) |nl| : (rest = rest[nl + 1 ..]) _ = try s.line(rest[0 .. nl + 1], 5);
    try std.testing.expectEqual(@as(u32, 1), s.deliveries);
}

test "one stamp is unmeasured, two are ok, and missing telemetry stays null" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const all = try joined(a, &.{ text1, text2, final, usage4, done });
    const one = try play(a, all, &.{}, 4);
    try std.testing.expectEqual(suite.Status.unmeasured, one.status);
    try std.testing.expectEqual(suite.ErrorCode.unmeasured_delivery, one.error_code.?);
    try std.testing.expect(one.delivery_tps == null and one.server_decode_tps.? == 42.0);
    const ok = try play(a, all, &stamps, 4);
    try std.testing.expectEqual(suite.Status.ok, ok.status);
    try std.testing.expectEqualStrings("0123456789ab", ok.token_sha.?);
    try std.testing.expectEqual(@as(f64, 3) / ok.delivery_seconds.?, ok.delivery_tps.?);
    try std.testing.expect(ok.server_decode_seconds == null and ok.prefill_seconds.? == 0.03 and ok.ttft_seconds.? == 0.01);
    const bare = try play(a, try joined(a, &.{ text1, text2, "data: {\"choices\": [{\"index\": 0, \"finish_reason\": \"stop\"}]}\n\n", usage4, done }), &stamps, 4);
    try std.testing.expect(bare.status == .ok and bare.server_decode_tps == null and bare.token_sha == null);
}

test "usage and the stream end decide the error codes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, suite.ErrorCode }{
        .{ try joined(a, &.{ text1, text2, final, done }), .missing_usage },
        .{ try joined(a, &.{ text1, text2, final, usage4 }), .truncated_stream },
        .{ try joined(a, &.{ text1, text2, final, "data: {\"usage\": {\"prompt_tokens\": 24, \"completion_tokens\": -1}}\n\n", done }), .invalid_usage },
        .{ try joined(a, &.{ text1, text2, final, "data: {\"usage\": {\"prompt_tokens\": 24, \"completion_tokens\": true}}\n\n", done }), .invalid_usage },
        .{ try joined(a, &.{ text1, text2, final, "data: {\"usage\": {\"prompt_tokens\": 24, \"completion_tokens\": 0}}\n\n", done }), .empty_output },
        .{ try joined(a, &.{ text1, text2, final, "data: {\"usage\": {\"prompt_tokens\": 24, \"completion_tokens\": 5}}\n\n", done }), .token_count_mismatch },
        .{ try joined(a, &.{ text1, text2, final, "data: {\"usage\": {\"prompt_tokens\": 2, \"completion_tokens\": 4, \"prompt_tokens_details\": {\"cached_tokens\": 3}}}\n\n", done }), .invalid_usage },
    };
    for (cases) |c| {
        const s = try play(a, c[0], &stamps, 4);
        try std.testing.expectEqual(suite.Status.@"error", s.status);
        try std.testing.expectEqual(c[1], s.error_code.?);
    }
    const mismatch = try play(a, cases[5][0], &stamps, 4);
    try std.testing.expect(mismatch.completion_tokens.? == 5 and mismatch.delivery_tps != null);
}

test "fewer tokens than asked is early_eos with a rate" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = try play(a, try joined(a, &.{ text1, text2, final, "data: {\"usage\": {\"prompt_tokens\": 24, \"completion_tokens\": 2}}\n\n", done }), &stamps, 4);
    try std.testing.expect(s.status == .early_eos and s.error_code == null and s.delivery_tps.? > 0);
}

test "server errors, bad JSON, a missing blank line and an empty reply keep fixed codes and no server text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, suite.ErrorCode }{
        .{ "data: {\"error\": {\"message\": \"private upstream URL must not escape\"}}\n\n", .server_error },
        .{ "data: {invalid JSON}\n\n", .invalid_stream },
        .{ "data: [DONE]", .truncated_stream },
        .{ done, .unexpected_finish },
        .{ "data: {\"choices\": [{\"delta\": {}, \"finish_reason\": \"stop\"}], \"usage\": {\"prompt_tokens\": 2, \"completion_tokens\": 4}}\n\n" ++ done, .empty_output },
        .{ try joined(a, &.{ text1, text2, "data: {\"choices\": [{\"index\": 0, \"delta\": {\"tool_calls\": []}, \"finish_reason\": \"tool_calls\"}]}\n\n", usage4, done }), .unexpected_finish },
    };
    for (cases) |c| {
        const s = try play(a, c[0], &stamps, 4);
        try std.testing.expectEqual(c[1], s.error_code.?);
        try std.testing.expect(s.token_sha == null);
    }
}

test "bad telemetry values are null, a null tensorfold block hides runtime, and hashes are lowercased" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bad = "data: {\"choices\": [{\"index\": 0, \"finish_reason\": \"length\"}], \"tensorfold\": {\"token_sha\": \"synthetic output\", \"decode_s\": -1, \"decode_tps\": true, \"prefill_s\": \"0.1\"}}\n\n";
    const s = try play(a, try joined(a, &.{ text1, text2, bad, usage4, done }), &stamps, 4);
    try std.testing.expect(s.server_decode_tps == null and s.server_decode_seconds == null and s.prefill_seconds == null and s.token_sha == null);
    const hidden = "data: {\"choices\": [{\"index\": 0, \"finish_reason\": \"length\"}], \"tensorfold\": null, \"runtime\": {\"decode_tps\": 9.0}}\n\n";
    const h = try play(a, try joined(a, &.{ text1, text2, hidden, usage4, done }), &stamps, 4);
    try std.testing.expect(h.status == .ok and h.server_decode_tps == null);
    const old = "data: {\"choices\": [{\"index\": 0, \"finish_reason\": \"length\"}], \"runtime\": {\"decode_tps\": 9.0, \"decode_s\": 0.2, \"cached\": 0}}\n\n";
    const o = try play(a, try joined(a, &.{ text1, text2, old, "data: {\"usage\": {\"prompt_tokens\": 24, \"completion_tokens\": 4}}\n\n", done }), &stamps, 4);
    try std.testing.expect(o.server_decode_tps.? == 9.0 and o.server_decode_seconds.? == 0.2 and o.cached_tokens.? == 0);
}

test "choice shapes and finish order are checked, and reasoning counts as delivery" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const invalid = [_][]const u8{
        "data: {\"choices\": [{}, {}]}\n\n",
        "data: {\"choices\": [{\"index\": 1, \"text\": \"x\"}]}\n\n",
        "data: {\"choices\": [{\"index\": 0, \"delta\": null}]}\n\n",
        "data: {\"choices\": [{\"index\": 0, \"text\": 3}]}\n\n",
        "data: {\"choices\": null}\n\n",
        "data: [1]\n\n",
        try joined(a, &.{ final, text1 }),
        try joined(a, &.{ final, final }),
    };
    for (invalid) |bytes| try std.testing.expectEqual(suite.ErrorCode.invalid_stream, (try play(a, bytes, &stamps, 4)).error_code.?);
    const reasoning = "data: {\"choices\": [{\"index\": 0.0, \"delta\": {\"reasoning_content\": \"r\"}}]}\n\n";
    const role = "data: {\"choices\": [{\"index\": false, \"delta\": {\"role\": \"assistant\"}}]}\n\n";
    const s = try play(a, try joined(a, &.{ role, reasoning, text2, final, usage4, done }), &.{ 1 * ms, 1 * ms, 10 * ms, 10 * ms, 30 * ms }, 4);
    try std.testing.expect(s.status == .ok and s.ttft_seconds.? == 0.01);
}

test "line, event, stream and UTF-8 limits give invalid_stream" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const filler = try a.alloc(u8, 300 * 1024);
    @memset(filler, 'x');
    var s = try sse.Stream.init(a);
    try std.testing.expectError(error.InvalidStream, s.line(try joined(a, &.{ "data: ", filler, filler, "\n" }), 0));
    s = try sse.Stream.init(a);
    _ = try s.line(try joined(a, &.{ "data: ", filler, "\n" }), 0);
    try std.testing.expectError(error.InvalidStream, s.line(try joined(a, &.{ "data: ", filler, "\n" }), 0));
    s = try sse.Stream.init(a);
    var fed: usize = 0;
    const comment = try joined(a, &.{ ":", filler, "\n" });
    var over: ?sse.Failure = null;
    while (fed <= sse.max_stream and over == null) : (fed += comment.len) _ = s.line(comment, 0) catch |e| {
        over = e;
    };
    try std.testing.expectEqual(error.InvalidStream, over.?);
    s = try sse.Stream.init(a);
    try std.testing.expectError(error.InvalidStream, s.line("data: \xff\n", 0));
}

test "the delivery rate is (completion - 1) / delivery_seconds in the same f64 the receipt holds" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = try play(a, try joined(a, &.{ text1, text2, final, usage4, done }), &.{ 3, 3, 2123456792, 2123456792 }, 4);
    try std.testing.expectEqual(@as(f64, 2.123456789), s.delivery_seconds.?);
    try std.testing.expectEqual(@as(f64, 3) / 2.123456789, s.delivery_tps.?);
}

test "issue 969 v2: natural sampled stops at 128 qualify, short, greedy, length and empty do not" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { n: u32, temp: f64, reason: []const u8, status: suite.Status }{
        .{ .n = 127, .temp = 1, .reason = "stop", .status = .early_eos },
        .{ .n = 128, .temp = 1, .reason = "stop", .status = .ok },
        .{ .n = 177, .temp = 1, .reason = "stop", .status = .ok },
        .{ .n = 128, .temp = 0, .reason = "stop", .status = .early_eos },
        .{ .n = 177, .temp = 1, .reason = "length", .status = .early_eos },
        .{ .n = 256, .temp = 0, .reason = "length", .status = .ok },
    };
    for (cases) |c| {
        const body = try std.fmt.allocPrint(a, "data: {{\"choices\":[{{\"text\":\"a\"}}]}}\n\ndata: {{\"choices\":[{{\"text\":\"b\"}}]}}\n\ndata: {{\"choices\":[{{\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":24,\"completion_tokens\":{d}}}}}\n\ndata: [DONE]\n\n", .{ c.reason, c.n });
        const sample = try playAt(a, body, &.{ 0, 0, 100 * ms, 100 * ms }, 256, c.temp);
        try std.testing.expectEqual(c.status, sample.status);
        try std.testing.expectEqualStrings(c.reason, sample.finish_reason.?);
        try std.testing.expectEqual(@as(f64, @floatFromInt(c.n - 1)) / 0.1, sample.delivery_tps.?);
    }
    const empty = "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":24,\"completion_tokens\":177}}\n\ndata: [DONE]\n\n";
    const sample = try playAt(a, empty, &.{}, 256, 1);
    try std.testing.expectEqual(suite.ErrorCode.empty_output, sample.error_code.?);
}

//! Requests against the fake: bodies, timing, deadlines, Ctrl-C and run order.
const std = @import("std");
const suite = @import("benchmark_suite.zig");
const stream = @import("benchmark_stream.zig");
const stop = @import("benchmark_stop.zig");
const fake = @import("benchmark_fake.zig");
const Write = fake.Write;

const golden_chat = "{\"model\": \".tensorfold-benchmark\", \"max_tokens\": 256, \"temperature\": 1.0, \"seed\": 1234, \"stream\": true, \"stream_options\": {\"include_usage\": true}, \"top_k\": 20, \"top_p\": 0.95, \"draft\": false, \"messages\": [{\"role\": \"user\", \"content\": \"Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.\"}], \"chat_template_kwargs\": {\"enable_thinking\": false}}";
const golden_code = "{\"model\": \".tensorfold-benchmark\", \"max_tokens\": 256, \"temperature\": 0.0, \"seed\": 1238, \"stream\": true, \"stream_options\": {\"include_usage\": true}, \"prompt\": \"Write a short Python function that computes the Fibonacci sequence and explain it.\"}";

fn opts(tokens: u32, serial: bool, timeout_s: f64) stream.Opts {
    return .{ .tokens = tokens, .reps = 1, .temperatures = &.{ 1.0, 0.0 }, .serial = serial, .timeout_s = timeout_s };
}

fn one(f: *fake.Fake, a: std.mem.Allocator, fixture: suite.Fixture, o: stream.Opts, t: f64, seed: u32) !suite.Sample {
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = std.testing.io, .read_buffer_size = 576 << 10 };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    return stream.attempt(a, std.testing.io, &client, .{ .base = try f.url(a, ""), .served = ".tensorfold-benchmark" }, o, fixture, t, seed, 0, null);
}

fn sha(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

fn batch(a: std.mem.Allocator, writes: []const Write) ![]const Write {
    var all: std.ArrayList(u8) = .empty;
    for (writes) |w| try all.appendSlice(a, w.bytes);
    const out = try a.alloc(Write, 1);
    out[0] = .{ .bytes = all.items };
    return out;
}

test "request bodies equal the golden bytes, routes and headers, and a chat reply is measured" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("d85d948f99e8c87c578e103e3e4e584e5d10fc53144a7585cfadd5f8c493fd5b", &sha(golden_chat));
    try std.testing.expectEqualStrings("fb516b455e574eba1c065ba914a8b7707d3839331156cb4defdcd766c97f4318", &sha(golden_code));
    const f = try fake.Fake.start(std.testing.allocator, &.{
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = &.{try fake.reply(a, true, 256, 12)} },
        .{ .method = "POST", .path = "/v1/completions", .replies = &.{try fake.reply(a, false, 256, 12)} },
    });
    defer f.stop();
    const s = try one(f, a, suite.fixtures[1], opts(256, true, 5), 1.0, 1234);
    try std.testing.expectEqual(suite.Status.ok, s.status);
    try std.testing.expectEqualStrings(golden_chat, f.request(0).body);
    try std.testing.expectEqualStrings("text/event-stream", f.header(0, "accept").?);
    try std.testing.expect(s.prompt_tokens.? == 24 and s.completion_tokens.? == 256 and s.cached_tokens.? == 0);
    try std.testing.expectEqual(@as(f64, 255) / s.delivery_seconds.?, s.delivery_tps.?);
    try std.testing.expect(s.ttft_seconds.? < s.end_to_end_seconds.?);
    try std.testing.expect(s.server_decode_tps.? == 42 and s.server_decode_seconds == null and s.prefill_seconds.? == 0.03);
    try std.testing.expectEqualStrings("0123456789ab", s.token_sha.?);
    _ = try one(f, a, suite.fixtures[0], opts(256, false, 5), 0.0, 1238);
    try std.testing.expectEqualStrings("/v1/completions", f.request(1).path);
    try std.testing.expectEqualStrings(golden_code, f.request(1).body);
}

test "writes apart are measured, one write is unmeasured, and the head's bytes take its time" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const replies = try fake.reply(a, false, 4, 12);
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "POST", .path = "/v1/completions", .replies = &.{ replies, try batch(a, replies) } }});
    defer f.stop();
    const apart = try one(f, a, suite.fixtures[0], opts(4, false, 5), 1.0, 1234);
    try std.testing.expect(apart.status == .ok and apart.delivery_seconds.? >= 0.010);
    try std.testing.expect(apart.ttft_seconds.? < 0.5 and apart.delivery_seconds.? < 0.5);
    const together = try one(f, a, suite.fixtures[0], opts(4, false, 5), 1.0, 1234);
    try std.testing.expect(together.status == .unmeasured and together.delivery_seconds == null);
}

test "bytes one at a time keep UTF-8 and CRLF, and a final carriage return without a newline never dispatches" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = (try batch(a, try fake.reply(a, false, 4, 0)))[0].bytes;
    const at = std.mem.indexOf(u8, whole, "\r\n\r\n").? + 4;
    const lambda = try std.mem.replaceOwned(u8, a, whole[at..], "\"synthetic\"", "\"\xce\xbb\"");
    const crlf = try std.mem.replaceOwned(u8, a, lambda, "\n", "\r\n");
    const second = std.mem.indexOf(u8, crlf, "\" output\"").?;
    var bytes: std.ArrayList(Write) = .empty;
    try bytes.append(a, .{ .bytes = whole[0..at] });
    for (crlf, 0..) |_, i| try bytes.append(a, .{ .bytes = crlf[i .. i + 1], .pause_ms = if (i + 1 == second) 15 else 0 });
    const tail = [_]Write{
        .{ .bytes = try std.mem.concat(a, u8, &.{ try fake.headOf(a, "200 OK", "text/event-stream"), "data: {\"choices\": [{\"index\": 0, \"text\": \"a\"}]}\n\n" }), .pause_ms = 20 },
        .{ .bytes = "data: {\"choices\": [{\"index\": 0, \"text\": \"b\"}]}\n\r" },
    };
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "POST", .path = "/v1/completions", .replies = &.{ bytes.items, &tail } }});
    defer f.stop();
    const s = try one(f, a, suite.fixtures[0], opts(4, false, 5), 1.0, 1234);
    try std.testing.expectEqual(suite.Status.ok, s.status);
    const t = try one(f, a, suite.fixtures[0], opts(4, false, 5), 1.0, 1234);
    try std.testing.expect(t.error_code.? == .truncated_stream and t.delivery_seconds == null and t.ttft_seconds != null);
}

test "status, media type and redirects decide before any line is read" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const charset = (try batch(a, try fake.reply(a, false, 4, 0)))[0].bytes;
    const with_charset = try std.mem.replaceOwned(u8, a, charset, "text/event-stream", "text/event-stream; charset=utf-8");
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "POST", .path = "/v1/completions", .replies = &.{
        try fake.jsonReply(a, "503 Service Unavailable", "not public"),
        try fake.jsonReply(a, "200 OK", "{}"),
        &.{.{ .bytes = "HTTP/1.1 307 Temporary Redirect\r\nLocation: /elsewhere\r\nContent-Length: 0\r\nConnection: close\r\n\r\n" }},
        &.{.{ .bytes = with_charset }},
    } }});
    defer f.stop();
    for ([_]suite.ErrorCode{ .http_error, .invalid_stream, .http_error }) |want| try std.testing.expectEqual(want, (try one(f, a, suite.fixtures[0], opts(4, false, 5), 0.0, 1234)).error_code.?);
    try std.testing.expectEqual(@as(usize, 3), f.count());
    const s = try one(f, a, suite.fixtures[0], opts(4, false, 5), 0.0, 1234);
    try std.testing.expect(s.error_code.? == .unmeasured_delivery and s.status == .unmeasured);
}

test "a trickling reply times out, and Ctrl-C cancels the attempt and ends the run" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var trickle: [60]Write = undefined;
    trickle[0] = .{ .bytes = try fake.headOf(a, "200 OK", "text/event-stream"), .pause_ms = 4 };
    for (trickle[1..]) |*w| w.* = .{ .bytes = "x", .pause_ms = 4 };
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "POST", .path = "/v1/completions", .replies = &.{&trickle} }});
    defer f.stop();
    const io = std.testing.io;
    var started = stream.now(io);
    const s = try one(f, a, suite.fixtures[0], opts(4, false, 0.04), 1.0, 1234);
    try std.testing.expectEqual(suite.ErrorCode.timeout, s.error_code.?);
    try std.testing.expect(stream.now(io) - started < 300 * std.time.ns_per_ms);
    stop.reset();
    defer stop.reset();
    started = stream.now(io);
    const samples = blk: {
        const t = try std.Thread.spawn(.{}, struct {
            fn later() void {
                fake.nap(50);
                stop.request(.INT);
            }
        }.later, .{});
        defer t.join();
        break :blk try stream.run(a, io, .{ .base = try f.url(a, ""), .served = "m" }, .{ .tokens = 4, .reps = 2, .temperatures = &.{1.0}, .serial = false, .timeout_s = 30 }, null);
    };
    try std.testing.expect(stream.now(io) - started < 300 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(usize, 1), samples.len);
    try std.testing.expect(samples[0].repeat == -1 and samples[0].error_code.? == .cancelled);
}

test "a closed port is a connection error, chunked SSE is read, and a cut chunk is truncated" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const closed = try fake.Fake.start(std.testing.allocator, &.{});
    const dead = try closed.url(a, "");
    closed.stop();
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = std.testing.io };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    const s = stream.attempt(a, std.testing.io, &client, .{ .base = dead, .served = "m" }, opts(4, false, 5), suite.fixtures[0], 0.0, 1234, 0, null);
    try std.testing.expectEqual(suite.ErrorCode.connection_error, s.error_code.?);
    const whole = (try batch(a, try fake.reply(a, false, 4, 0)))[0].bytes;
    const at = std.mem.indexOf(u8, whole, "\r\n\r\n").? + 4;
    const chunked_head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    const half = std.mem.indexOf(u8, whole[at..], "data: {\"choices\": [{\"index\": 0, \"text\": \" output").?;
    const ok = [_]Write{
        .{ .bytes = try std.fmt.allocPrint(a, "{s}{x}\r\n{s}\r\n", .{ chunked_head, half, whole[at .. at + half] }), .pause_ms = 15 },
        .{ .bytes = try std.fmt.allocPrint(a, "{x}\r\n{s}\r\n0\r\n\r\n", .{ whole.len - at - half, whole[at + half ..] }) },
    };
    const cut = [_]Write{.{ .bytes = try std.fmt.allocPrint(a, "{s}{x}\r\n{s}", .{ chunked_head, whole.len - at, whole[at .. at + half] }) }};
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "POST", .path = "/v1/completions", .replies = &.{ &ok, &cut } }});
    defer f.stop();
    try std.testing.expectEqual(suite.Status.ok, (try one(f, a, suite.fixtures[0], opts(4, false, 5), 0.0, 1234)).status);
    try std.testing.expectEqual(suite.ErrorCode.truncated_stream, (try one(f, a, suite.fixtures[0], opts(4, false, 5), 0.0, 1234)).error_code.?);
}

const Counter = struct {
    n: usize = 0,
    fn call(ctx: *anyopaque, _: *const suite.Sample) void {
        const c: *Counter = @ptrCast(@alignCast(ctx));
        c.n += 1;
    }
};

test "the run sends a warm-up and the repeats per cell, keeping only a failed warm-up" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const err_event = [_]Write{.{ .bytes = try std.mem.concat(a, u8, &.{ try fake.headOf(a, "200 OK", "text/event-stream"), try fake.event(a, "{\"error\": {\"message\": \"warm-up failed\"}}") }) }};
    {
        const f = try fake.Fake.start(std.testing.allocator, &.{
            .{ .method = "POST", .path = "/v1/completions", .replies = &.{try fake.reply(a, false, 4, 2)} },
            .{ .method = "POST", .path = "/v1/chat/completions", .replies = &.{try fake.reply(a, true, 4, 2)} },
        });
        defer f.stop();
        var counter: Counter = .{};
        const samples = try stream.run(a, std.testing.io, .{ .base = try f.url(a, "/v1"), .served = "example/model" }, .{ .tokens = 4, .reps = 2, .temperatures = &.{ 1.0, 0.0 }, .serial = false, .timeout_s = 5 }, .{ .ctx = &counter, .call = Counter.call });
        try std.testing.expect(samples.len == 8 and f.count() == 12 and counter.n == 12);
        for (0..4) |cell| for ([_]u32{ 1234, 1234, 1235 }, 0..) |seed, k| {
            const b = f.request(cell * 3 + k).body;
            try std.testing.expect(std.mem.indexOf(u8, b, try std.fmt.allocPrint(a, "\"seed\": {d},", .{seed})) != null);
            try std.testing.expectEqual(cell < 2, std.mem.indexOf(u8, b, "\"top_k\": 20") != null);
        };
        for (samples, 0..) |s, i| try std.testing.expect(s.temperature == @as(f64, if (i < 4) 1.0 else 0.0) and s.repeat >= 0);
    }
    const g = try fake.Fake.start(std.testing.allocator, &.{
        .{ .method = "POST", .path = "/v1/completions", .replies = &.{ &err_event, try fake.reply(a, false, 4, 2) } },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = &.{try fake.reply(a, true, 4, 2)} },
    });
    defer g.stop();
    const kept = try stream.run(a, std.testing.io, .{ .base = try g.url(a, ""), .served = "example/model" }, .{ .tokens = 4, .reps = 5, .temperatures = &.{0.0}, .serial = false, .timeout_s = 5 }, null);
    var code_rows: usize = 0;
    for (kept) |s| code_rows += @intFromBool(std.mem.eql(u8, s.fixture_id, "code"));
    try std.testing.expect(code_rows == 6 and kept[0].repeat == -1 and kept[0].error_code.? == .server_error);
}

test "the endpoint keeps a prefix, adds /v1 once and refuses other schemes, credentials and queries" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    for ([_]struct { []const u8, bool, []const u8 }{
        .{ "http://127.0.0.1:8080", true, "http://127.0.0.1:8080/v1/chat/completions" },
        .{ "http://127.0.0.1:8080/v1/", false, "http://127.0.0.1:8080/v1/completions" },
        .{ "HTTPS://example.test/proxy/", false, "https://example.test/proxy/v1/completions" },
    }) |c| try std.testing.expectEqualStrings(c[2], (try stream.endpoint(a, c[0], c[1], &problem)).?);
    for ([_][]const u8{ "file:///tmp/model", "http://test:password@example.test", "https://example.test?key=x", "https://example.test#frag", "ftp://example.test" }) |bad|
        try std.testing.expect(try stream.endpoint(a, bad, false, &problem) == null);
}

test "with no task to run it on, a request is a connection error and is never sent" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var single: std.Io.Threaded = .init_single_threaded;
    const io = single.io();
    const f = try fake.Fake.start(std.testing.allocator, &.{
        .{ .method = "POST", .path = "/v1/completions", .replies = &.{try fake.reply(a, false, 4, 0)} },
        .{ .method = "GET", .path = "/health", .replies = &.{try fake.jsonReply(a, "200 OK", "{}")} },
    });
    defer f.stop();
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = io };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    const s = stream.attempt(a, io, &client, .{ .base = try f.url(a, ""), .served = "m" }, opts(4, false, 5), suite.fixtures[0], 0.0, 1234, 0, null);
    try std.testing.expectEqual(suite.ErrorCode.connection_error, s.error_code.?);
    const got = stream.fetch(a, io, &client, .{ .url = try f.url(a, "/health"), .limit = 64, .timeout_s = 5 });
    try std.testing.expect(got.failed == .network and f.count() == 0);
}

test "a reply of exactly the limit is read, and one byte more is too large" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "GET", .path = "/x", .replies = &.{
        try fake.jsonReply(a, "200 OK", "[1234]"),
        try fake.jsonReply(a, "200 OK", "[12345]"),
    } }});
    defer f.stop();
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = std.testing.io };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    const at = stream.fetch(a, std.testing.io, &client, .{ .url = try f.url(a, "/x"), .limit = 6, .timeout_s = 5 });
    try std.testing.expect(at.failed == .none and std.mem.eql(u8, at.body, "[1234]"));
    try std.testing.expect(stream.fetch(a, std.testing.io, &client, .{ .url = try f.url(a, "/x"), .limit = 6, .timeout_s = 5 }).failed == .too_large);
}

test "an https exchange is never sent through a proxy that refuses to tunnel" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const refused = [_]Write{.{ .bytes = "HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\nConnection: close\r\n\r\n" }};
    const f = try fake.Fake.start(std.testing.allocator, &.{
        .{ .method = "CONNECT", .path = "receiver.example.test:443", .replies = &.{&refused} },
        .{ .method = "GET", .path = "https://receiver.example.test/api/me", .replies = &.{try fake.jsonReply(a, "200 OK", "{}")} },
    });
    defer f.stop();
    var proxy: std.http.Client.Proxy = .{ .protocol = .plain, .host = .{ .bytes = "127.0.0.1" }, .authorization = null, .port = f.port, .supports_connect = true };
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = std.testing.io, .https_proxy = &proxy };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    const got = stream.fetch(a, std.testing.io, &client, .{ .url = "https://receiver.example.test/api/me", .token = "synthetic-contributor-token-0001", .limit = 64, .timeout_s = 5 });
    try std.testing.expect(got.failed == .network);
    for (0..f.count()) |i| try std.testing.expectEqualStrings("CONNECT", f.request(i).method);
}

test "an https exchange is never sent in clear text through a tunnel the proxy opened" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const opened = [_]Write{.{ .bytes = "HTTP/1.1 200 Connection established\r\n\r\n" }};
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "CONNECT", .path = "receiver.example.test:443", .replies = &.{&opened}, .tunnel = true }});
    defer f.stop();
    var proxy: std.http.Client.Proxy = .{ .protocol = .plain, .host = .{ .bytes = "127.0.0.1" }, .authorization = null, .port = f.port, .supports_connect = true };
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = std.testing.io, .https_proxy = &proxy };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    const got = stream.fetch(a, std.testing.io, &client, .{ .url = "https://receiver.example.test/api/me", .token = "synthetic-contributor-token-0001", .limit = 64, .timeout_s = 1 });
    try std.testing.expect(got.failed == .network);
    fake.nap(100);
    try std.testing.expectEqual(@as(usize, 1), f.count());
    try std.testing.expectEqualStrings("CONNECT", f.request(0).method);
}

test "a refused attempt keeps its fixed sample and code and hands back the server's reason, cleaned, waiting 100 ms at most" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const greedy = "{\"error\": {\"message\": \"this engine decodes greedily only: send temperature 0 or leave it out\", \"type\": \"invalid_request_error\"}}";
    const long: [400]u8 = @splat('y');
    const late_body = "{\"error\": {\"message\": \"late\"}}";
    const late_head = try std.fmt.allocPrint(a, "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{late_body.len});
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "POST", .path = "/v1/completions", .replies = &.{
        try fake.jsonReply(a, "400 Bad Request", greedy),
        try fake.jsonReply(a, "400 Bad Request", greedy),
        try fake.jsonReply(a, "503 Service Unavailable", "not public\nsecond line"),
        &.{.{ .bytes = "HTTP/1.1 404 Not Found\r\nContent-Type: text/html\r\nContent-Length: 15\r\nConnection: close\r\n\r\n<html>x</html>\n" }},
        try fake.jsonReply(a, "400 Bad Request", "{\"error\": \"bad\\u001b[31m red\\u0007\\nline\"}"),
        try fake.jsonReply(a, "400 Bad Request", try std.mem.concat(a, u8, &.{ "{\"detail\": \"", &long, "\"}" })),
        &.{ .{ .bytes = late_head, .pause_ms = 400 }, .{ .bytes = late_body } },
        &.{ .{ .bytes = late_head, .pause_ms = 400 }, .{ .bytes = late_body } },
    } }});
    defer f.stop();
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = io, .read_buffer_size = 576 << 10 };
    client.connection_pool.free_size = 0;
    defer client.deinit();
    const target: stream.Target = .{ .base = try f.url(a, ""), .served = "m" };
    var said: []const u8 = "stale";
    const heard = stream.attempt(a, io, &client, target, opts(4, false, 5), suite.fixtures[0], 1.0, 1234, 0, &said);
    try std.testing.expectEqualStrings("HTTP 400: this engine decodes greedily only: send temperature 0 or leave it out", said);
    var unheard = stream.attempt(a, io, &client, target, opts(4, false, 5), suite.fixtures[0], 1.0, 1234, 0, null);
    try std.testing.expect(heard.error_code.? == .http_error and heard.status == .@"error" and heard.end_to_end_seconds != null);
    unheard.end_to_end_seconds = heard.end_to_end_seconds;
    try std.testing.expectEqualDeep(heard, unheard);
    for ([_][]const u8{ "HTTP 503: not public", "HTTP 404", "HTTP 400: bad red line" }) |want| {
        _ = stream.attempt(a, io, &client, target, opts(4, false, 5), suite.fixtures[0], 0.0, 1234, 0, &said);
        try std.testing.expectEqualStrings(want, said);
    }
    _ = stream.attempt(a, io, &client, target, opts(4, false, 5), suite.fixtures[0], 0.0, 1234, 0, &said);
    try std.testing.expect(said.len == "HTTP 400: ".len + 303 and std.mem.endsWith(u8, said, "y..."));
    const started = stream.now(io);
    const late = stream.attempt(a, io, &client, target, opts(4, false, 5), suite.fixtures[0], 0.0, 1234, 0, &said);
    const took: f64 = @floatFromInt(@as(i64, @intCast(stream.now(io) - started)));
    try std.testing.expect(took >= 100e6 and took < 300e6 and late.error_code.? == .http_error);
    try std.testing.expect(late.end_to_end_seconds.? * 1e9 < took - 50e6);
    try std.testing.expectEqualStrings("HTTP 400", said);
    const cut = stream.attempt(a, io, &client, target, opts(4, false, 0.05), suite.fixtures[0], 0.0, 1234, 0, &said);
    try std.testing.expect(cut.error_code.? == .http_error and cut.status == .@"error");
}

test "an unbuffered refusal body is read after its head without a timed sender pause" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var gate: std.Io.Event = .unset;
    const refusal_body = "{\"error\": {\"message\": \"soon\"}}";
    const head = try std.fmt.allocPrint(a, "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{refusal_body.len});
    const f = try fake.Fake.start(std.testing.allocator, &.{.{ .method = "GET", .path = "/reason", .replies = &.{&.{ .{ .bytes = head, .after = &gate }, .{ .bytes = refusal_body } }} }});
    defer {
        gate.set(io);
        f.stop();
    }
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = io };
    defer client.deinit();
    var req = try client.request(.GET, try std.Uri.parse(try f.url(a, "/reason")), .{ .keep_alive = false });
    defer {
        gate.set(io);
        req.deinit();
    }
    try req.sendBodiless();
    const resp = try req.receiveHead(&.{});
    try std.testing.expectEqual(@as(usize, 0), resp.request.reader.in.buffered().len);
    gate.set(io);
    try std.testing.expectEqualStrings(refusal_body, stream.errorBody(a, io, &resp));
}

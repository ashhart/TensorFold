//! --confidential over real sockets: whole responses, whole error frames and a persistent WebSocket's later turns.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const output = @import("confidential_output.zig");
const fixture = @import("witness_fixture.zig");
const ws_test = @import("responses_ws_test.zig");
const client = @import("responses_ws_client_test.zig");
const posix = @import("sys").posix;
const t = std.testing;
const Allocator = std.mem.Allocator;

const marker = "MARKER-7f3a";

/// `bytes` as sent, then everything the server writes until it closes.
fn raw(a: Allocator, h: *fixture.Harness, bytes: []const u8) ![]const u8 {
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.listener.port()) };
    var storage: std.Io.Threaded.PosixAddress = undefined;
    const len = std.Io.Threaded.addressToPosix(&address, &storage);
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.Socket;
    const fd: posix.socket_t = @intCast(rc);
    defer _ = posix.system.close(fd);
    if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.Connect;
    if (posix.errno(posix.system.write(fd, bytes.ptr, bytes.len)) != .SUCCESS) return error.Write;
    var out: std.ArrayList(u8) = .empty;
    var buffer: [4096]u8 = undefined;
    while (true) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        if (try posix.poll(&fds, 5000) == 0) return error.Timeout;
        const n = posix.system.read(fd, &buffer, buffer.len);
        if (posix.errno(n) != .SUCCESS or n == 0) return out.items;
        try out.appendSlice(a, buffer[0..@intCast(n)]);
    }
}

/// `bytes` as sent, then the server's head only: a 101 keeps its socket open.
fn rawHead(a: Allocator, h: *fixture.Harness, bytes: []const u8) ![]const u8 {
    const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.listener.port()) };
    var storage: std.Io.Threaded.PosixAddress = undefined;
    const len = std.Io.Threaded.addressToPosix(&address, &storage);
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.Socket;
    const fd: posix.socket_t = @intCast(rc);
    defer _ = posix.system.close(fd);
    if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.Connect;
    if (posix.errno(posix.system.write(fd, bytes.ptr, bytes.len)) != .SUCCESS) return error.Write;
    var out: std.ArrayList(u8) = .empty;
    var buffer: [4096]u8 = undefined;
    while (std.mem.indexOf(u8, out.items, "\r\n\r\n") == null) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        if (try posix.poll(&fds, 5000) == 0) return error.Timeout;
        const n = posix.system.read(fd, &buffer, buffer.len);
        if (posix.errno(n) != .SUCCESS or n == 0) break;
        try out.appendSlice(a, buffer[0..@intCast(n)]);
    }
    return out.items;
}

fn has(text: []const u8, part: []const u8) bool {
    return std.mem.indexOf(u8, text, part) != null;
}

test "confidential: a whole HTTP reply, status line and HTTP/0.9 body included, never carries the request's words" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: fixture.Engine = .{ .hooked = false };
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try h.start(&engine, null, null, .{});
    defer h.deinit();
    const requests = [_][]const u8{
        marker ++ " /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n", // 501: its phrase quotes the method
        "GET / HTTP/" ++ marker ++ "\r\n\r\n", // an unreadable version: answered as HTTP/0.9
        marker ++ " /v1/models\r\n", // an HTTP/0.9 request: a body and no head
        "GET /v1/responses/" ++ marker ++ " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    };
    // the control: each whole reply quotes the marker without the flag
    for (requests) |r| try t.expect(has(try raw(a, &h, r), marker));
    output.confine();
    defer output.release();
    for (requests) |r| {
        const reply = try raw(a, &h, r);
        if (has(reply, marker)) std.debug.print("quoted: {s}\n", .{reply});
        try t.expect(!has(reply, marker));
        try t.expect(has(reply, output.refused_message) or has(reply, output.failed_message)); // 501 is a 5xx
        try t.expect(has(reply, "\"request_id\": \"req-"));
    }
}

fn errorFrames(a: Allocator, body: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var frames = std.mem.splitSequence(u8, body, "\n\n");
    while (frames.next()) |f| if (std.mem.startsWith(u8, f, "event: error\n") or std.mem.startsWith(u8, f, "event: response.failed\n"))
        try out.append(a, f);
    return out.items;
}

test "confidential: a failed Responses stream's error frames carry no echoed request field, whole" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: fixture.Engine = .{ .hooked = false, .fail_message = "the engine failed on " ++ marker };
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try h.start(&engine, null, null, .{});
    defer h.deinit();
    const body = "{\"model\":\"m\",\"stream\":true,\"store\":false,\"instructions\":\"" ++ marker ++ "\"," ++
        "\"metadata\":{\"k\":\"" ++ marker ++ "\"},\"input\":\"hi\"}";
    const before = try errorFrames(a, try h.ask(a, "POST", "/v1/responses", "", "localhost", body));
    try t.expect(before.len > 0);
    var quoted = false; // the control: the failed response echoes the request's instructions and metadata
    for (before) |f| quoted = quoted or has(f, marker);
    try t.expect(quoted);
    output.confine();
    defer output.release();
    const after = try errorFrames(a, try h.ask(a, "POST", "/v1/responses", "", "localhost", body));
    try t.expect(after.len > 0);
    for (after) |f| {
        if (has(f, marker)) std.debug.print("quoted: {s}\n", .{f});
        try t.expect(!has(f, marker));
        try t.expect(has(f, "\"request_id\""));
    }
}

test "confidential: a WebSocket's later refusal names the connection's id, not the earlier turn's freed one" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const log_level = t.log_level;
    t.log_level = .err;
    defer t.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    output.confine();
    defer output.release();
    var f: ws_test.Fixture = undefined;
    try f.start(t.allocator, t.io, .{});
    errdefer f.stop() catch {};
    var c: client.Client = undefined;
    try c.open(a, t.io, f.port, null);
    const done = try c.turn(a, "{\"type\":\"response.create\",\"input\":\"hello\",\"store\":false}");
    try t.expectEqualStrings("response.completed", client.last(done).strField("type").?);
    const refused = try c.turn(a, "{\"type\":\"response.create\",\"store\":false,\"input\":[{\"type\":\"" ++ marker ++ "\"}]}");
    const e = client.last(refused);
    const text = try json.stringify(a, e, .{});
    try t.expect(!has(text, marker));
    try t.expectEqualStrings("error", e.strField("type").?);
    const id = e.get("error").?.strField("request_id").?;
    try t.expect(std.mem.startsWith(u8, id, "req-"));
    try t.expect(std.fmt.parseInt(u64, id["req-".len..], 10) catch null != null);
    c.close();
    try f.stop();
}

test "confidential: a WebSocket upgrade and its version refusal keep their standard reason phrases" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: fixture.Engine = .{ .hooked = false };
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try h.start(&engine, null, null, .{});
    defer h.deinit();
    output.confine();
    defer output.release();
    const upgrade = "GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n" ++
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: ";
    try t.expect(std.mem.startsWith(u8, try rawHead(a, &h, upgrade ++ "13\r\n\r\n"), "HTTP/1.1 101 Switching Protocols\r\n"));
    try t.expect(std.mem.startsWith(u8, try rawHead(a, &h, upgrade ++ "12\r\n\r\n"), "HTTP/1.1 426 Upgrade Required\r\n"));
}

test "confidential: a HEAD error names the fixed body's length and sends no body" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: fixture.Engine = .{ .hooked = false };
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try h.start(&engine, null, null, .{});
    defer h.deinit();
    output.confine();
    defer output.release();
    // the GET first, for the request number: the HEAD that follows is the next one
    const get = try raw(a, &h, "GET /v1/responses/" ++ marker ++ " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    const fixed = try fixture.body(get);
    const id = (try fixture.payload(a, get)).get("error").?.strField("request_id").?;
    const next = try std.fmt.allocPrint(a, "req-{d}", .{1 + try std.fmt.parseInt(u64, id["req-".len..], 10)});
    const reply = try raw(a, &h, "HEAD /v1/responses/" ++ marker ++ " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    const split = std.mem.indexOf(u8, reply, "\r\n\r\n").?;
    try t.expectEqual(reply.len, split + 4); // no body bytes
    const at = std.mem.indexOf(u8, reply, "Content-Length: ").? + "Content-Length: ".len;
    const n = try std.fmt.parseInt(usize, reply[at..std.mem.indexOfScalarPos(u8, reply, at, '\r').?], 10);
    const status = try std.fmt.parseInt(u16, reply["HTTP/1.1 ".len .. "HTTP/1.1 ".len + 3], 10);
    try t.expectEqual((try output.httpBody(a, status, "", next)).len, n); // HEAD is a 501 here: the same fixed body
    try t.expect(fixed.len > 0);
}

test "confidential: Ollama's NDJSON error line keeps its shape and carries the fixed words, streamed or whole" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: fixture.Engine = .{ .hooked = false, .fail_message = "the engine failed on " ++ marker };
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try h.start(&engine, null, null, .{});
    defer h.deinit();
    const asks = [_][2][]const u8{
        .{ "/api/chat", "{\"model\":\"m\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
        .{ "/api/chat", "{\"model\":\"m\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
        .{ "/api/generate", "{\"model\":\"m\",\"stream\":true,\"prompt\":\"hi\"}" },
        .{ "/api/generate", "{\"model\":\"m\",\"stream\":false,\"prompt\":\"hi\"}" },
    };
    // the control: each error line carries the engine's words without the flag
    for (asks) |q| try t.expect(has(try h.ask(a, "POST", q[0], "", "localhost", q[1]), marker));
    output.confine();
    defer output.release();
    for (asks) |q| {
        const reply = try h.ask(a, "POST", q[0], "", "localhost", q[1]);
        if (has(reply, marker)) std.debug.print("quoted: {s}\n", .{reply});
        try t.expect(!has(reply, marker));
        try t.expect(has(reply, "{\"error\":\"" ++ output.failed_message ++ "\"}\n"));
    }
}

test "confidential: a failed background response read back by GET keeps only generated ids and the fixed error" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: fixture.Engine = .{ .hooked = false, .fail_message = "the engine failed on " ++ marker };
    var h: fixture.Harness = .{ .model = fixture.ChatText.model() };
    try h.start(&engine, null, null, .{});
    defer h.deinit();
    const body = "{\"model\":\"m\",\"background\":true,\"store\":true,\"instructions\":\"" ++ marker ++ "\"," ++
        "\"metadata\":{\"k\":\"" ++ marker ++ "\"},\"input\":\"hi\"}";
    const read = struct {
        fn failed(arena_a: Allocator, hh: *fixture.Harness, post: []const u8) ![]const u8 {
            const created = try fixture.payload(arena_a, try hh.ask(arena_a, "POST", "/v1/responses", "", "localhost", post));
            const path = try std.fmt.allocPrint(arena_a, "/v1/responses/{s}", .{created.strField("id").?});
            for (0..500) |_| {
                const got = try hh.ask(arena_a, "GET", path, "", "localhost", "");
                if (has(got, "\"failed\"")) return got;
                std.Io.sleep(t.io, .fromMilliseconds(10), .awake) catch {};
            }
            return error.NeverFailed;
        }
    };
    try t.expect(has(try read.failed(a, &h, body), marker)); // the control: the stored failure echoes the request
    output.confine();
    defer output.release();
    const got = try read.failed(a, &h, body);
    try fixture.expectStatus(got, 200);
    try t.expect(!has(got, marker));
    const v = try fixture.payload(a, got);
    try t.expect(std.mem.startsWith(u8, v.strField("id").?, "resp_"));
    try t.expectEqualStrings(output.failed_message, v.get("error").?.strField("message").?);
}

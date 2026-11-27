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

//! X-Request-Id: a client's valid id echoed and in its log lines, else the id the server holds the request by.
const std = @import("std");
const http = @import("http_conn.zig");
const log = @import("log.zig");
const auth = @import("auth.zig");
const scoped = @import("auth_scope_http_test.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;

/// A connection whose socket holds `bytes`, and whose replies go to `out`.
const Wire = struct {
    bytes: []const u8,
    at: usize = 0,
    out: std.ArrayList(u8) = .empty,
    a: Allocator,

    fn cast(ctx: *anyopaque) *Wire {
        return @ptrCast(@alignCast(ctx));
    }
    fn now(_: *anyopaque) i64 {
        return 0;
    }
    fn read(ctx: *anyopaque, buf: []u8, _: i32) http.IoError!usize {
        const w = cast(ctx);
        const n = @min(buf.len, w.bytes.len - w.at);
        @memcpy(buf[0..n], w.bytes[w.at..][0..n]);
        w.at += n;
        return n;
    }
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w = cast(ctx);
        w.out.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

/// The log lines this test thread writes, kept whole.
const Lines = struct {
    a: Allocator,
    text: std.ArrayList(u8) = .empty,
    fn call(ctx: *anyopaque, line: []const u8) void {
        const l: *Lines = @ptrCast(@alignCast(ctx));
        l.text.appendSlice(l.a, line) catch {};
    }
};

/// The requests in `raw` read in turn on one connection, each answered with `{}`; the reply bytes of each.
fn answer(a: Allocator, raw: []const u8, comptime requests: usize) ![requests][]const u8 {
    var w: Wire = .{ .bytes = raw, .a = a };
    var c = try http.Conn.init(t.allocator, -1, "127.0.0.1");
    defer c.deinit();
    c.read_hooks = .{ .ctx = &w, .now = Wire.now, .read = Wire.read };
    c.write_hooks = .{ .ctx = &w, .write = Wire.write };
    var replies: [requests][]const u8 = undefined;
    for (&replies) |*reply| {
        const from = w.out.items.len;
        try t.expectEqual(http.Outcome.ready, try c.readRequest(a, true));
        c.sendJson(200, "{}");
        reply.* = try a.dupe(u8, w.out.items[from..]);
    }
    return replies;
}

/// The X-Request-Id value in the reply head that `reply` starts with.
fn idOf(reply: []const u8) ![]const u8 {
    const head = reply[0 .. std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.NoHead];
    const at = std.mem.indexOf(u8, head, "\r\nX-Request-Id: ") orelse return error.NoRequestId;
    const rest = head[at + "\r\nX-Request-Id: ".len ..];
    return rest[0 .. std.mem.indexOf(u8, rest, "\r\n") orelse rest.len];
}

fn expectFresh(id: []const u8) !void {
    try t.expectEqual(@as(usize, 36), id.len);
    try t.expectEqualStrings("req_", id[0..4]);
    for (id[4..]) |c| try t.expect(std.ascii.isDigit(c) or (c >= 'a' and c <= 'f'));
}

test "request id: a client's valid id is echoed and ends each of the request's log lines" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lines: Lines = .{ .a = a };
    log.Test.sink = .{ .ctx = &lines, .call = Lines.call };
    defer log.Test.sink = null;
    defer log.requestId("");
    const longest: [128]u8 = @splat('k');
    for ([_][]const u8{ "client-trace-1", "a", "x-Y_z.9:/=%", &longest }) |id| {
        for ([_][]const u8{ "X-Request-Id", "x-request-id" }) |name| {
            lines.text.clearRetainingCapacity();
            const raw = try std.fmt.allocPrint(a, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n{s}:  {s} \r\n\r\n", .{ name, id });
            const reply = (try answer(a, raw, 1))[0];
            try t.expectEqualStrings(id, try idOf(reply));
            try t.expect(std.mem.indexOf(u8, reply, "\r\nDate: ").? < std.mem.indexOf(u8, reply, "\r\nX-Request-Id: ").?);
            try t.expectEqualStrings(try std.fmt.allocPrint(a, "[tensorfold] 127.0.0.1 \"GET /v1/models HTTP/1.1\" 200 - request_id={s}\n", .{id}), lines.text.items);
            // a line a handler writes later in the request, such as its refusal, carries it too
            lines.text.clearRetainingCapacity();
            var buf: [256]u8 = undefined;
            log.line("{s}", .{log.refused(&buf, "chatcmpl-1", "why")});
            try t.expectEqualStrings(try std.fmt.allocPrint(a, "[tensorfold] refused chatcmpl-1: why request_id={s}\n", .{id}), lines.text.items);
        }
    }
}

test "request id: an empty, oversized, spaced, non-ASCII or repeated id is ignored for a new req_ id" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lines: Lines = .{ .a = a };
    log.Test.sink = .{ .ctx = &lines, .call = Lines.call };
    defer log.Test.sink = null;
    defer log.requestId("");
    const over: [129]u8 = @splat('k');
    const heads = [_][]const u8{
        "X-Request-Id: \r\n",
        try std.fmt.allocPrint(a, "X-Request-Id: {s}\r\n", .{&over}),
        "X-Request-Id: two words\r\n",
        "X-Request-Id: caf\xe9\r\n",
        "X-Request-Id: one\r\nX-Request-Id: one\r\n",
        "X-Request-Ids: near-miss\r\n",
    };
    var seen: [heads.len][]const u8 = undefined;
    for (heads, &seen) |extra, *slot| {
        lines.text.clearRetainingCapacity();
        const raw = try std.fmt.allocPrint(a, "GET /health HTTP/1.1\r\nHost: localhost\r\n{s}\r\n", .{extra});
        slot.* = try idOf((try answer(a, raw, 1))[0]);
        try expectFresh(slot.*);
        try t.expect(std.mem.indexOf(u8, lines.text.items, "request_id=") == null);
        try t.expect(std.mem.endsWith(u8, lines.text.items, "\" 200 -\n"));
    }
    for (seen[1..]) |id| try t.expect(!std.mem.eql(u8, seen[0], id));
}

test "request id: the next request on a kept-alive connection forgets the last one's id" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lines: Lines = .{ .a = a };
    log.Test.sink = .{ .ctx = &lines, .call = Lines.call };
    defer log.Test.sink = null;
    defer log.requestId("");
    const raw = "GET /health HTTP/1.1\r\nHost: localhost\r\nX-Request-Id: first-1\r\n\r\n" ++
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    const replies = try answer(a, raw, 2);
    try t.expectEqualStrings("first-1", try idOf(replies[0]));
    try expectFresh(try idOf(replies[1]));
    const want = "[tensorfold] 127.0.0.1 \"GET /health HTTP/1.1\" 200 - request_id=first-1\n" ++
        "[tensorfold] 127.0.0.1 \"GET /health HTTP/1.1\" 200 -\n";
    try t.expectEqualStrings(want, lines.text.items);
}

/// One request to the harness's server with the full key, `extra` header lines first; the whole reply.
fn ask(h: *scoped.Harness, a: Allocator, method: []const u8, path: []const u8, extra: []const u8, body: []const u8) ![]const u8 {
    const raw = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: localhost\r\n{s}Authorization: Bearer sk-test\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ method, path, extra, body.len, body });
    return h.exchange(a, raw);
}

/// The reply's body, after its head.
fn bodyOf(reply: []const u8) []const u8 {
    return reply[(std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return "") + 4 ..];
}

fn start(engine: *scoped.Engine, keys: *auth.Store, h: *scoped.Harness) !void {
    var problem: []const u8 = "";
    keys.* = try auth.Store.init(t.allocator, &.{"sk-test"}, "", null, false, &problem);
    errdefer keys.deinit();
    try h.start(engine, keys);
}

test "request id: a reply the cancel route can stop carries its id, streamed or whole; other replies a req_ id" {
    var engine: scoped.Engine = .{};
    var keys: auth.Store = undefined;
    var h: scoped.Harness = .{};
    try start(&engine, &keys, &h);
    defer keys.deinit();
    defer h.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const say = "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1";
    const Case = struct { path: []const u8, body: []const u8, prefix: []const u8 };
    const cases = [_]Case{
        .{ .path = "/v1/chat/completions", .body = "{\"model\":\"m\"," ++ say ++ "}", .prefix = "chatcmpl-" },
        .{ .path = "/v1/chat/completions", .body = "{\"model\":\"m\"," ++ say ++ ",\"stream\":true}", .prefix = "chatcmpl-" },
        .{ .path = "/v1/completions", .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":1}", .prefix = "cmpl-" },
        .{ .path = "/v1/completions", .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"max_tokens\":1,\"stream\":true}", .prefix = "cmpl-" },
        .{ .path = "/v1/responses", .body = "{\"input\":\"hi\",\"max_output_tokens\":1,\"store\":false}", .prefix = "resp_" },
        .{ .path = "/v1/responses", .body = "{\"input\":\"hi\",\"max_output_tokens\":1,\"store\":false,\"stream\":true}", .prefix = "resp_" },
        // a chat completion's 400: the id its refusal line names
        .{ .path = "/v1/chat/completions", .body = "{\"model\":\"m\"}", .prefix = "chatcmpl-" },
    };
    for (cases) |c| {
        const reply = try ask(&h, a, "POST", c.path, "", c.body);
        const id = try idOf(reply);
        try t.expect(std.mem.startsWith(u8, id, c.prefix) and id.len == c.prefix.len + 32);
        if (std.mem.startsWith(u8, reply, "HTTP/1.1 200 ")) try t.expect(std.mem.indexOf(u8, bodyOf(reply), id) != null);
    }
    // Messages run as a chat completion: the id is the one the cancel route takes, not the reply's msg_ id
    const messages = try ask(&h, a, "POST", "/v1/messages", "", "{\"model\":\"m\"," ++ say ++ "}");
    const held = try idOf(messages);
    try t.expect(std.mem.startsWith(u8, held, "chatcmpl-"));
    try t.expect(std.mem.indexOf(u8, bodyOf(messages), held) == null and std.mem.indexOf(u8, bodyOf(messages), "\"msg_") != null);
    const first = try idOf(try ask(&h, a, "GET", "/v1/models", "", ""));
    const second = try idOf(try ask(&h, a, "GET", "/v1/models", "", ""));
    try expectFresh(first);
    try expectFresh(second);
    try t.expect(!std.mem.eql(u8, first, second));
}

test "request id: a client's id is echoed on streams, errors, refusals and unknown paths, the body unchanged" {
    var engine: scoped.Engine = .{};
    var keys: auth.Store = undefined;
    var h: scoped.Harness = .{};
    try start(&engine, &keys, &h);
    defer keys.deinit();
    defer h.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const mine = "X-Request-Id: trace-42\r\n";
    const stream = try ask(&h, a, "POST", "/v1/chat/completions", mine, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":true}");
    try t.expect(std.mem.indexOf(u8, stream, "text/event-stream") != null);
    try t.expectEqualStrings("trace-42", try idOf(stream));
    try t.expect(std.mem.indexOf(u8, bodyOf(stream), "trace-42") == null and std.mem.indexOf(u8, bodyOf(stream), "\"chatcmpl-") != null);
    for ([_][3][]const u8{
        .{ "POST", "/v1/chat/completions", "{}" },
        .{ "GET", "/future/private", "" },
        .{ "DELETE", "/v1/responses/resp_missing", "" },
    }) |r| try t.expectEqualStrings("trace-42", try idOf(try ask(&h, a, r[0], r[1], mine, r[2])));
    // refused before any route, at the key gate
    const refused = try h.exchange(a, "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n" ++ mine ++ "Connection: close\r\n\r\n");
    try t.expect(std.mem.startsWith(u8, refused, "HTTP/1.1 401 "));
    try t.expectEqualStrings("trace-42", try idOf(refused));
}

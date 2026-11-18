//! Tests for the text and --log-json request lines in log.zig.
const std = @import("std");
const log = @import("log.zig");
const Conn = @import("http_conn.zig").Conn;
const Test = log.Test;
const setJson = log.setJson;
const request = log.request;
const ended = log.ended;
const refused = log.refused;
const line = log.line;

const JsonLogTest = struct {
    const Capture = struct {
        bytes: [4 * 8192 + 1]u8 = undefined,
        len: usize = 0,
        fn call(ctx: *anyopaque, bytes: []const u8) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(c.bytes[0..bytes.len], bytes);
            c.len = bytes.len;
        }
        fn line(c: *const @This()) []const u8 {
            return c.bytes[0..c.len];
        }
        fn payload(c: *const @This()) []const u8 {
            const prefix = "[tensorfold] ";
            return c.bytes[prefix.len .. c.len - 1];
        }
    };

    fn sink(c: *Capture) void {
        Test.sink = .{ .ctx = c, .call = Capture.call };
    }
};

test "default request log: access, ended and refused lines stay text" {
    var c: JsonLogTest.Capture = .{};
    JsonLogTest.sink(&c);
    defer Test.sink = null;
    setJson(false);
    var conn = try Conn.init(std.testing.allocator, 0, "127.0.0.1");
    defer conn.deinit();
    conn.requestline = "GET / HTTP/1.1";
    conn.id = "chatcmpl-1";
    request(&conn, 200);
    try std.testing.expectEqualStrings("[tensorfold] 127.0.0.1 \"GET / HTTP/1.1\" 200 -\n", c.line());
    var buf: [1400]u8 = undefined;
    line("{s}", .{ended(&buf, "chatcmpl-1", null, 12, 3, 0.5)});
    try std.testing.expectEqualStrings("[tensorfold] ended chatcmpl-1 reason=client-left prompt=12 tokens=3 after=0.50s\n", c.line());
    line("{s}", .{refused(&buf, "cmpl-1", "short")});
    try std.testing.expectEqualStrings("[tensorfold] refused cmpl-1: short\n", c.line());
}

test "json request log: access includes the id when known" {
    var c: JsonLogTest.Capture = .{};
    JsonLogTest.sink(&c);
    defer Test.sink = null;
    setJson(true);
    defer setJson(false);
    var conn = try Conn.init(std.testing.allocator, 0, "127.0.0.1");
    defer conn.deinit();
    conn.requestline = "POST /v1/chat/completions HTTP/1.1";
    conn.id = "chatcmpl-1";
    request(&conn, 200);
    try std.testing.expectEqual(@as(u8, '\n'), c.line()[c.len - 1]);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, c.line(), "\n"));
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("access", o.get("event").?.string);
    try std.testing.expectEqualStrings("chatcmpl-1", o.get("id").?.string);
    try std.testing.expectEqual(@as(i64, 200), o.get("status").?.integer);
    try std.testing.expectEqualStrings("127.0.0.1", o.get("peer").?.string);
    try std.testing.expectEqualStrings("POST /v1/chat/completions HTTP/1.1", o.get("request").?.string);
    conn.id = null;
    request(&conn, 503);
    const parsed2 = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer parsed2.deinit();
    try std.testing.expectEqualStrings("access", parsed2.value.object.get("event").?.string);
    try std.testing.expect(parsed2.value.object.get("id") == null);
    try std.testing.expectEqual(@as(i64, 503), parsed2.value.object.get("status").?.integer);
}

test "json request log: ended includes id and tokens; refused invents none" {
    var c: JsonLogTest.Capture = .{};
    JsonLogTest.sink(&c);
    defer Test.sink = null;
    setJson(true);
    defer setJson(false);
    var buf: [1400]u8 = undefined;
    line("{s}", .{ended(&buf, "chatcmpl-1", null, 12, 3, 0.5)});
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, c.line(), "\n"));
    const ended_json = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer ended_json.deinit();
    const e = ended_json.value.object;
    try std.testing.expectEqualStrings("ended", e.get("event").?.string);
    try std.testing.expectEqualStrings("chatcmpl-1", e.get("id").?.string);
    try std.testing.expectEqual(@as(i64, 12), e.get("prompt").?.integer);
    try std.testing.expectEqual(@as(i64, 3), e.get("tokens").?.integer);
    try std.testing.expectEqualStrings("client-left", e.get("reason").?.string);
    try std.testing.expect(e.get("status") == null);
    line("{s}", .{ended(&buf, "chatcmpl-1", "OutOfMemory", 12, 0, 1.25)});
    const err_json = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer err_json.deinit();
    try std.testing.expectEqualStrings("error (OutOfMemory)", err_json.value.object.get("reason").?.string);
    try std.testing.expectEqual(@as(i64, 0), err_json.value.object.get("tokens").?.integer);
    line("{s}", .{refused(&buf, "cmpl-1", "short")});
    const refused_json = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer refused_json.deinit();
    const r = refused_json.value.object;
    try std.testing.expectEqualStrings("refused", r.get("event").?.string);
    try std.testing.expectEqualStrings("cmpl-1", r.get("id").?.string);
    try std.testing.expectEqualStrings("short", r.get("reason").?.string);
    try std.testing.expect(r.get("prompt") == null);
    try std.testing.expect(r.get("tokens") == null);
}

test "json request log: secrets stay out of the line" {
    var c: JsonLogTest.Capture = .{};
    JsonLogTest.sink(&c);
    defer Test.sink = null;
    setJson(true);
    defer setJson(false);
    var conn = try Conn.init(std.testing.allocator, 0, "127.0.0.1");
    defer conn.deinit();
    conn.requestline = "GET /health HTTP/1.1";
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Authorization", .value = "Bearer sk-secret-value" }};
    conn.headers = &headers;
    conn.auth_enabled = true;
    conn.key_label = "cli-1";
    request(&conn, 200);
    const line_text = c.line();
    try std.testing.expect(std.mem.indexOf(u8, line_text, "sk-secret-value") == null);
    try std.testing.expect(std.mem.indexOf(u8, line_text, "Bearer") == null);
    try std.testing.expect(std.mem.indexOf(u8, line_text, "Authorization") == null);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("cli-1", parsed.value.object.get("key").?.string);
    conn.requestline = "GET / \"quoted\" HTTP/1.1";
    request(&conn, 400);
    const quoted = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer quoted.deinit();
    try std.testing.expectEqualStrings("GET / \"quoted\" HTTP/1.1", quoted.value.object.get("request").?.string);
}

test "json request log: a request line over 8 KiB still parses" {
    var c: JsonLogTest.Capture = .{};
    JsonLogTest.sink(&c);
    defer Test.sink = null;
    setJson(true);
    defer setJson(false);
    var conn = try Conn.init(std.testing.allocator, 0, "127.0.0.1");
    defer conn.deinit();
    const long = try std.testing.allocator.alloc(u8, "GET /".len + 9000 + " HTTP/1.1".len);
    defer std.testing.allocator.free(long);
    @memcpy(long[0..5], "GET /");
    @memset(long[5 .. 5 + 9000], 'a');
    @memcpy(long[5 + 9000 ..], " HTTP/1.1");
    conn.requestline = long;
    conn.auth_enabled = true;
    request(&conn, 401);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, c.line(), "\n"));
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("access", o.get("event").?.string);
    try std.testing.expect(std.mem.startsWith(u8, long, o.get("request").?.string));
    try std.testing.expectEqualStrings("unauthenticated", o.get("key").?.string);
    @memset(long, 0x01);
    conn.requestline = long;
    request(&conn, 400);
    const controls = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer controls.deinit();
    try std.testing.expect(controls.value.object.get("request").?.string.len > 0);
}

test "json request log: a refusal of control bytes still parses" {
    var c: JsonLogTest.Capture = .{};
    JsonLogTest.sink(&c);
    defer Test.sink = null;
    setJson(true);
    defer setJson(false);
    var message: [300]u8 = undefined;
    @memset(&message, 0x01);
    var buf: [1024]u8 = undefined;
    line("{s}", .{refused(&buf, "cmpl-1", &message)});
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.payload(), .{});
    defer parsed.deinit();
    const r = parsed.value.object;
    try std.testing.expectEqualStrings("refused", r.get("event").?.string);
    try std.testing.expectEqualStrings("cmpl-1", r.get("id").?.string);
    const reason = r.get("reason").?.string;
    try std.testing.expect(reason.len > 0 and reason.len < message.len);
    for (reason) |b| try std.testing.expectEqual(@as(u8, 0x01), b);
}

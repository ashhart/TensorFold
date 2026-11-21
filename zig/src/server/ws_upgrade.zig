//! The RFC 6455 opening handshake: whether a request asks for a WebSocket, and the 101, 400 or 426 that answers it.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Conn = @import("http_conn.zig").Conn;

const guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Whether the request asks to upgrade: ``Upgrade`` names ``websocket`` and ``Connection`` names ``upgrade``.
pub fn requested(conn: *const Conn) bool {
    return hasToken(conn, "Upgrade", "websocket") and hasToken(conn, "Connection", "upgrade");
}

/// Whether any ``name`` header holds ``token`` in its comma list, any case.
fn hasToken(conn: *const Conn, name: []const u8, token: []const u8) bool {
    for (conn.headers) |h| {
        if (!std.ascii.eqlIgnoreCase(h.name, name)) continue;
        var it = std.mem.splitScalar(u8, h.value, ',');
        while (it.next()) |part| if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, part, " \t"), token)) return true;
    }
    return false;
}

/// ``Sec-WebSocket-Accept``: base64 of the SHA-1 of the client's key and the RFC 6455 GUID.
pub fn accept(key: []const u8) [28]u8 {
    var sha = std.crypto.hash.Sha1.init(.{});
    sha.update(key);
    sha.update(guid);
    var out: [28]u8 = undefined;
    _ = std.base64.standard.Encoder.encode(&out, &sha.finalResult());
    return out;
}

/// Answers the handshake: 101 when it is valid, else 400 or 426. The connection closes to HTTP on every path.
pub fn answer(conn: *Conn, a: Allocator) bool {
    conn.close = true;
    if (problem(conn)) |p| {
        refuse(conn, a, p) catch {};
        return false;
    }
    const key = std.mem.trim(u8, conn.header("Sec-WebSocket-Key").?, " \t");
    switchProtocols(conn, &accept(key)) catch return false;
    return !conn.broken;
}

const Problem = struct { status: u16, message: []const u8 };

/// Why the handshake is refused, or null when it is valid.
fn problem(conn: *const Conn) ?Problem {
    if (!std.mem.eql(u8, conn.method, "GET")) return .{ .status = 400, .message = "a WebSocket upgrade must use GET" };
    if (!std.mem.eql(u8, conn.version, "HTTP/1.1")) return .{ .status = 400, .message = "a WebSocket upgrade must use HTTP/1.1" };
    if (!requested(conn)) return .{ .status = 400, .message = "the request does not ask for a WebSocket upgrade" };
    if (conn.headerCount("Sec-WebSocket-Key") != 1) return .{ .status = 400, .message = "a WebSocket upgrade needs one Sec-WebSocket-Key header" };
    if (!validKey(std.mem.trim(u8, conn.header("Sec-WebSocket-Key").?, " \t"))) return .{ .status = 400, .message = "Sec-WebSocket-Key must be 16 bytes in base64" };
    const version = conn.header("Sec-WebSocket-Version") orelse return .{ .status = 400, .message = "a WebSocket upgrade needs Sec-WebSocket-Version" };
    if (conn.headerCount("Sec-WebSocket-Version") != 1 or !std.mem.eql(u8, std.mem.trim(u8, version, " \t"), "13"))
        return .{ .status = 426, .message = "this server speaks WebSocket version 13 only" };
    return null;
}

fn validKey(key: []const u8) bool {
    const decoder = std.base64.standard.Decoder;
    if ((decoder.calcSizeForSlice(key) catch return false) != 16) return false;
    var out: [16]u8 = undefined;
    decoder.decode(&out, key) catch return false;
    return true;
}

fn switchProtocols(conn: *Conn, key: []const u8) error{ Closed, OutOfMemory }!void {
    try conn.startResponse(101, "Switching Protocols");
    try conn.addHeader("Upgrade", "websocket");
    try conn.addHeader("Connection", "Upgrade");
    try conn.addHeader("Sec-WebSocket-Accept", key);
    try conn.finish("");
}

/// A JSON error reply; a 426 also names the version this server speaks.
fn refuse(conn: *Conn, a: Allocator, p: Problem) error{ Closed, OutOfMemory }!void {
    const body = try std.fmt.allocPrint(a, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"invalid_request_error\"}}}}", .{p.message});
    try conn.startResponse(p.status, if (p.status == 426) "Upgrade Required" else null);
    try conn.addHeader("Content-Type", "application/json");
    if (p.status == 426) try conn.addHeader("Sec-WebSocket-Version", "13");
    var len: [24]u8 = undefined;
    try conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch unreachable);
    try conn.addHeader("Connection", "close");
    try conn.finish(body);
}

const testing = std.testing;
const Header = @import("http_conn.zig").Header;

const Capture = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, data: []const u8) error{Closed}!void {
        const w: *Capture = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(testing.allocator, data) catch return error.Closed;
    }
};

/// Runs ``answer`` on a request with ``headers`` and returns whether it upgraded, with the reply in ``cap``.
fn handshake(cap: *Capture, method: []const u8, headers: []Header) !bool {
    var c = try Conn.init(testing.allocator, -1, "fixture");
    defer c.deinit();
    c.quiet_log = true;
    c.close = false;
    c.method = method;
    c.version = "HTTP/1.1";
    c.path = "/v1/responses";
    c.headers = headers;
    c.write_hooks = .{ .ctx = cap, .write = Capture.write };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const ok = answer(&c, arena.allocator());
    try testing.expect(c.close);
    return ok;
}

fn upgradeHeaders(key: []const u8, version: []const u8) [4]Header {
    return .{
        .{ .name = "Upgrade", .value = "WebSocket" },
        .{ .name = "Connection", .value = "keep-alive, Upgrade" },
        .{ .name = "Sec-WebSocket-Key", .value = key },
        .{ .name = "Sec-WebSocket-Version", .value = version },
    };
}

test "ws_upgrade: the RFC 6455 accept example" {
    try testing.expectEqualStrings("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", &accept("dGhlIHNhbXBsZSBub25jZQ=="));
}

test "ws_upgrade: a valid handshake gets 101 with the accept key and no body" {
    var cap: Capture = .{};
    defer cap.bytes.deinit(testing.allocator);
    var headers = upgradeHeaders("dGhlIHNhbXBsZSBub25jZQ==", "13");
    try testing.expect(try handshake(&cap, "GET", &headers));
    const reply = cap.bytes.items;
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 101 Switching Protocols\r\n"));
    try testing.expect(std.mem.indexOf(u8, reply, "\r\nUpgrade: websocket\r\n") != null);
    try testing.expect(std.mem.indexOf(u8, reply, "\r\nConnection: Upgrade\r\n") != null);
    try testing.expect(std.mem.indexOf(u8, reply, "\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n") != null);
    try testing.expect(std.mem.indexOf(u8, reply, "Content-Length") == null);
    try testing.expect(std.mem.endsWith(u8, reply, "\r\n\r\n"));
}

test "ws_upgrade: requested reads token lists in any case" {
    var c = try Conn.init(testing.allocator, -1, "fixture");
    defer c.deinit();
    var headers = upgradeHeaders("k", "13");
    c.headers = &headers;
    try testing.expect(requested(&c));
    headers[1].value = "keep-alive";
    try testing.expect(!requested(&c));
    headers[1].value = "upgradex, keep-alive";
    try testing.expect(!requested(&c));
    headers[1].value = " UPGRADE ";
    try testing.expect(requested(&c));
    headers[0].value = "h2c";
    try testing.expect(!requested(&c));
}

test "ws_upgrade: a missing or bad key and a non-GET get 400, a wrong version 426" {
    const key = "dGhlIHNhbXBsZSBub25jZQ==";
    var cap: Capture = .{};
    defer cap.bytes.deinit(testing.allocator);
    var headers = upgradeHeaders(key, "13");
    try testing.expect(!try handshake(&cap, "POST", &headers));
    try testing.expect(std.mem.startsWith(u8, cap.bytes.items, "HTTP/1.1 400 Bad Request\r\n"));
    try testing.expect(std.mem.indexOf(u8, cap.bytes.items, "\r\nConnection: close\r\n") != null);
    try testing.expect(std.mem.endsWith(u8, cap.bytes.items, "\"type\":\"invalid_request_error\"}}"));
    for ([_][]const u8{ "", "c2hvcnQ=", "not base64!!" }) |bad| {
        cap.bytes.clearRetainingCapacity();
        headers = upgradeHeaders(bad, "13");
        try testing.expect(!try handshake(&cap, "GET", &headers));
        try testing.expect(std.mem.startsWith(u8, cap.bytes.items, "HTTP/1.1 400 Bad Request\r\n"));
    }
    cap.bytes.clearRetainingCapacity();
    headers = upgradeHeaders(key, "13");
    var keyless = [_]Header{ headers[0], headers[1], headers[3] };
    try testing.expect(!try handshake(&cap, "GET", &keyless));
    try testing.expect(std.mem.startsWith(u8, cap.bytes.items, "HTTP/1.1 400 Bad Request\r\n"));
    cap.bytes.clearRetainingCapacity();
    headers = upgradeHeaders(key, "8");
    try testing.expect(!try handshake(&cap, "GET", &headers));
    try testing.expect(std.mem.startsWith(u8, cap.bytes.items, "HTTP/1.1 426 Upgrade Required\r\n"));
    try testing.expect(std.mem.indexOf(u8, cap.bytes.items, "\r\nSec-WebSocket-Version: 13\r\n") != null);
}

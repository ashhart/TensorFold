//! Exact CORS admission and the shared response writer, including an SSE head and per-request reset.
const std = @import("std");
const http = @import("http_conn.zig");
const guard = @import("request_guard.zig");
const cli = @import("cli.zig");
const t = std.testing;
const Wire = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(t.allocator, bytes) catch return error.Closed;
    }
};

test "CORS: repeatable exact origins imply admission and never accept wildcard, userinfo or paths" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    const args = try cli.parse(a, &.{ "m", "--cors-origin", "https://ui.invalid", "--cors-origin=http://other.invalid:3000" }, &usage);
    try t.expectEqual(@as(usize, 2), args.cors_origin.len);
    const policy: guard.Policy = .{ .cors_origins = args.cors_origin };
    const headers = [_]http.Header{ .{ .name = "Host", .value = "localhost:8000" }, .{ .name = "Origin", .value = "https://UI.invalid:443" } };
    try t.expectEqual(@as(?guard.Reason, null), policy.blocked(&headers));
    try t.expectEqualStrings("https://UI.invalid:443", policy.corsOrigin(&headers).?);
    for ([_][]const u8{ "*", "null", "https://user@ui.invalid", "https://ui.invalid/", "https://ui.invalid/path", "https://ui.invalid\r\nX: x" }) |v|
        try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--cors-origin", v }, &usage));
}

test "CORS: preflights and streams echo only listed origins, and disabled policy has no CORS headers" {
    var wire: Wire = .{};
    defer wire.bytes.deinit(t.allocator);
    var conn = try http.Conn.init(t.allocator, -1, "fixture");
    defer conn.deinit();
    conn.version = "HTTP/1.1";
    conn.method = "OPTIONS";
    conn.path = "/v1/chat/completions";
    conn.quiet_log = true;
    conn.write_hooks = .{ .ctx = &wire, .write = Wire.write };
    const policy: guard.Policy = .{ .cors_origins = &.{"https://ui.invalid"} };
    var headers = [_]http.Header{
        .{ .name = "Host", .value = "localhost" },
        .{ .name = "Origin", .value = "https://ui.invalid" },
        .{ .name = "Access-Control-Request-Method", .value = "POST" },
        .{ .name = "Access-Control-Request-Headers", .value = "Authorization, x-api-key, Content-Type, anthropic-version, anthropic-beta" },
    };
    conn.headers = &headers;
    try t.expect(!guard.allow(&conn, policy, t.allocator));
    try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 204 "));
    for ([_][]const u8{ "Access-Control-Allow-Origin: https://ui.invalid\r\n", "Vary: Origin\r\n", "Access-Control-Allow-Methods:", "Access-Control-Allow-Headers:", "Access-Control-Max-Age: 600" }) |s|
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, s) != null);
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Content-Length") == null and std.mem.endsWith(u8, wire.bytes.items, "\r\n\r\n"));
    wire.bytes.clearRetainingCapacity();
    conn.method = "POST";
    conn.quiet_log = true;
    try t.expect(guard.allow(&conn, policy, t.allocator));
    try conn.startResponse(200, null);
    try conn.addHeader("Content-Type", "text/event-stream");
    try conn.finish("data: [DONE]\n\n");
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Access-Control-Allow-Origin: https://ui.invalid") != null);
    try t.expect(std.mem.endsWith(u8, wire.bytes.items, "data: [DONE]\n\n"));
    wire.bytes.clearRetainingCapacity();
    headers[1].value = "https://unlisted.invalid";
    conn.method = "OPTIONS";
    conn.quiet_log = true;
    try t.expect(!guard.allow(&conn, policy, t.allocator));
    try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 403 "));
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Access-Control-Allow-Origin") == null);
    wire.bytes.clearRetainingCapacity();
    conn.headers = headers[0..1];
    conn.method = "GET";
    conn.quiet_log = true;
    try t.expect(guard.allow(&conn, .{}, t.allocator));
    try conn.startResponse(200, null);
    try conn.finish("");
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Access-Control-") == null);
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Vary: Origin") == null);
}

test "CORS: the OpenAI and Anthropic browser SDKs' headers pass a listed origin's preflight; unknown ones do not" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator); // a request's arena: error replies allocate from it
    defer arena.deinit();
    const a = arena.allocator();
    var wire: Wire = .{};
    defer wire.bytes.deinit(t.allocator);
    var conn = try http.Conn.init(t.allocator, -1, "fixture");
    defer conn.deinit();
    conn.version = "HTTP/1.1";
    conn.path = "/v1/messages";
    conn.write_hooks = .{ .ctx = &wire, .write = Wire.write };
    const policy: guard.Policy = .{ .cors_origins = &.{"https://ui.invalid"} };
    var headers = [_]http.Header{
        .{ .name = "Host", .value = "localhost" },
        .{ .name = "Origin", .value = "https://ui.invalid" },
        .{ .name = "Access-Control-Request-Method", .value = "POST" },
        .{ .name = "Access-Control-Request-Headers", .value = "x-stainless-os, anthropic-version, anthropic-dangerous-direct-browser-access" },
    };
    conn.headers = &headers;
    conn.method = "OPTIONS";
    conn.quiet_log = true;
    try t.expect(!guard.allow(&conn, policy, a));
    try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 204 "));
    const allowed = "Access-Control-Allow-Headers: authorization, x-api-key, content-type, anthropic-version, anthropic-beta, " ++
        "anthropic-dangerous-direct-browser-access, x-stainless-os\r\n";
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, allowed) != null);
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Access-Control-Allow-Origin: https://ui.invalid\r\n") != null);
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Access-Control-Allow-Credentials") == null);
    // an unknown header, a bare or malformed prefix: the whole preflight is refused, and nothing is allowed
    for ([_][]const u8{ "x-stainless-os, x-evil", "x-evil", "x-stainless-", "x-stainless-a b", "x-stainless-os;x" }) |asked| {
        wire.bytes.clearRetainingCapacity();
        headers[3].value = asked;
        conn.quiet_log = true;
        try t.expect(!guard.allow(&conn, policy, a));
        try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 403 "));
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Access-Control-Allow-Headers") == null);
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "x-evil") == null);
    }
    // every SDK platform header, any case, is echoed lowercased after the fixed list
    wire.bytes.clearRetainingCapacity();
    headers[3].value = "X-Stainless-Package-Version, x-stainless-retry-count, Content-Type";
    conn.quiet_log = true;
    try t.expect(!guard.allow(&conn, policy, a));
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "browser-access, x-stainless-package-version, x-stainless-retry-count\r\n") != null);
    // the same SDK headers from an unlisted origin get the origin 403
    wire.bytes.clearRetainingCapacity();
    headers[1].value = "https://unlisted.invalid";
    conn.quiet_log = true;
    try t.expect(!guard.allow(&conn, policy, a));
    try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 403 "));
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Access-Control-") == null);
}

test "CORS: requested headers and methods are bounded to the API set" {
    var conn = try http.Conn.init(t.allocator, -1, "fixture");
    defer conn.deinit();
    conn.cors_origin = "https://ui.invalid";
    var headers = [_]http.Header{ .{ .name = "Access-Control-Request-Method", .value = "POST" }, .{ .name = "Access-Control-Request-Headers", .value = "X-Not-Allowed" } };
    conn.headers = &headers;
    try t.expect(!@import("cors.zig").valid(&conn));
    headers[1].value = "authorization, content-type";
    try t.expect(@import("cors.zig").valid(&conn));
    headers[0].value = "CONNECT";
    try t.expect(!@import("cors.zig").valid(&conn));
    headers[0].value = "POST";
    headers[1] = headers[0];
    try t.expect(!@import("cors.zig").valid(&conn));
}

test "CORS: repeat count is bounded and readRequest clears the previous origin before parsing" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.append(a, "m");
    for (0..64) |_| try argv.appendSlice(a, &.{ "--cors-origin", "https://ui.invalid" });
    var usage: cli.Usage = .{};
    try t.expectEqual(@as(usize, 64), (try cli.parse(a, argv.items, &usage)).cors_origin.len);
    try argv.appendSlice(a, &.{ "--cors-origin", "https://other.invalid" });
    try t.expectError(error.Usage, cli.parse(a, argv.items, &usage));
    var conn = try http.Conn.init(t.allocator, -1, "fixture");
    defer conn.deinit();
    conn.cors_origin = "https://previous.invalid";
    conn.cors_vary = true;
    const next = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    @memcpy(conn.buf[0..next.len], next);
    conn.end = next.len;
    try t.expectEqual(http.Outcome.ready, try conn.readRequest(a, true));
    try t.expect(conn.cors_origin == null and !conn.cors_vary);
}

test "CORS: an --allowed-origin alone admits the origin but adds no CORS header and answers no preflight" {
    var wire: Wire = .{};
    defer wire.bytes.deinit(t.allocator);
    var conn = try http.Conn.init(t.allocator, -1, "fixture");
    defer conn.deinit();
    conn.version = "HTTP/1.1";
    conn.path = "/v1/chat/completions";
    conn.write_hooks = .{ .ctx = &wire, .write = Wire.write };
    var headers = [_]http.Header{
        .{ .name = "Host", .value = "localhost" },
        .{ .name = "Origin", .value = "https://ui.invalid" },
        .{ .name = "Access-Control-Request-Method", .value = "POST" },
    };
    conn.headers = &headers;
    const policy: guard.Policy = .{ .allowed_origins = &.{"https://ui.invalid"} };
    for ([_][]const u8{ "OPTIONS", "POST" }) |method| {
        conn.method = method;
        conn.quiet_log = true;
        try t.expect(guard.allow(&conn, policy, t.allocator)); // routed as before, no preflight reply
        try t.expect(conn.cors_origin == null and !conn.cors_vary);
        try t.expectEqual(@as(usize, 0), wire.bytes.items.len);
        conn.quiet_log = true;
        try conn.startResponse(200, null);
        try conn.finish("");
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Access-Control-") == null);
        try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Vary") == null);
        wire.bytes.clearRetainingCapacity();
    }
}

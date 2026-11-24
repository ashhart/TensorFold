//! The key gate before any route: a server's and a router's, with 401 for a gated path without a valid key.
const std = @import("std");
const auth = @import("auth.zig");
const scope = @import("auth_scope.zig");
const http_conn = @import("http_conn.zig");
const Allocator = std.mem.Allocator;

/// Whether ``conn`` may go on to its route; a refusal is sent here (``Authenticated._authenticate``).
pub fn admit(store: ?*auth.Store, conn: *http_conn.Conn, a: Allocator) bool {
    const keys = store orelse return true;
    if (std.mem.eql(u8, conn.method, "OPTIONS") or !keys.enabled()) return true;
    const matched = keys.matchIdentity(a, single(conn, "Authorization"), single(conn, "x-api-key"));
    if (matched) |m| {
        conn.key_label = m.label;
        conn.key_digest = m.digest;
        conn.key_scopes = m.scopes;
    }
    if (!auth.gatedRequest(conn.method, conn.path, keys.metrics_open)) return true;
    const path = auth.routePath(conn.path);
    // a Responses WebSocket upgrade opens with an inference key; the socket checks each response.create again
    const upgrade = @import("ws_upgrade.zig").requested(conn) and std.mem.eql(u8, @import("responses.zig").route(path) orelse "-", "");
    if (matched != null and (scope.allows(conn.key_scopes, conn.method, conn.path) or (upgrade and scope.allowsUpgrade(conn.key_scopes)))) return true;
    const messages = for ([_][]const u8{ "/v1/messages", "/v1/messages/count_tokens", "/messages", "/messages/count_tokens" }) |p| {
        if (std.mem.eql(u8, path, p)) break true;
    } else false;
    if (matched != null) {
        // a key without this route's scope is refused before its body is read or a parked model wakes
        @import("auth_scope_reply.zig").deny(conn, messages, .route);
        return false;
    }
    const body = if (messages)
        "{\"type\": \"error\", \"error\": {\"type\": \"authentication_error\", \"message\": \"A valid API key is required\"}}"
    else
        "{\"error\": {\"message\": \"A valid API key is required\", \"type\": \"invalid_request_error\", \"param\": null, \"code\": \"invalid_api_key\"}}";
    conn.close = true;
    conn.startResponse(401, null) catch return false;
    conn.addHeader("WWW-Authenticate", "Bearer") catch return false;
    conn.addHeader("Content-Type", "application/json") catch return false;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch unreachable) catch return false;
    conn.addHeader("Connection", "close") catch return false;
    conn.finish(if (std.mem.eql(u8, conn.method, "HEAD")) "" else body) catch {};
    return false;
}

/// A header that appears exactly once; none or several read as absent (as the key match reads them).
fn single(conn: *const http_conn.Conn, name: []const u8) ?[]const u8 {
    if (conn.headerCount(name) != 1) return null;
    return conn.header(name);
}

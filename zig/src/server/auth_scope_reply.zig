//! A restricted key's 403 in its route's error shape; it names no key, label or scope, and closes the connection.
const std = @import("std");
const scope = @import("auth_scope.zig");
const Conn = @import("http_conn.zig").Conn;

pub fn deny(conn: *Conn, anthropic_shape: bool, reason: scope.Denial) void {
    const body = if (anthropic_shape)
        "{\"type\": \"error\", \"error\": {\"type\": \"permission_error\", \"message\": \"This API key does not permit this operation\"}}"
    else switch (reason) {
        .route => "{\"error\": {\"message\": \"This API key does not permit this operation\", \"type\": \"permission_error\", \"param\": null, \"code\": \"insufficient_scope\"}}",
        .response_history => "{\"error\": {\"message\": \"This API key cannot read stored responses: send the whole conversation without previous_response_id\", \"type\": \"permission_error\", \"param\": \"previous_response_id\", \"code\": \"insufficient_scope\"}}",
        .response_storage => "{\"error\": {\"message\": \"This API key cannot store responses: send \\\"store\\\": false\", \"type\": \"permission_error\", \"param\": \"store\", \"code\": \"insufficient_scope\"}}",
    };
    conn.close = true;
    conn.startResponse(403, null) catch return;
    conn.addHeader("Content-Type", "application/json") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch unreachable) catch return;
    conn.addHeader("Connection", "close") catch return;
    conn.finish(if (std.mem.eql(u8, conn.method, "HEAD")) "" else body) catch {};
}

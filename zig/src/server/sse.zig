//! Server-sent events as the Python server frames them: the head, ``data:`` frames, typed ``event:`` frames, ``[DONE]``.
const std = @import("std");
const json = @import("json.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// 200 with text/event-stream; the connection closes after the stream.
pub fn open(conn: *Conn) error{Closed}!void {
    conn.startResponse(200, null) catch return error.Closed;
    conn.addHeader("Content-Type", "text/event-stream") catch return error.Closed;
    conn.addHeader("Cache-Control", "no-cache") catch return error.Closed;
    conn.addHeader("Connection", "close") catch return error.Closed;
    return conn.finish("");
}

/// ``data: {json}`` and a blank line, sent at once.
pub fn data(conn: *Conn, a: Allocator, payload: json.Value) error{Closed}!void {
    const text = json.stringify(a, payload, .{}) catch return error.Closed;
    return conn.writeAll(std.mem.concat(a, u8, &.{ "data: ", text, "\n\n" }) catch return error.Closed);
}

pub fn done(conn: *Conn) error{Closed}!void {
    return conn.writeAll("data: [DONE]\n\n");
}

/// ``event: KIND`` then its data line (Responses and Messages events).
pub fn event(conn: *Conn, a: Allocator, kind: []const u8, payload: json.Value) error{Closed}!void {
    const text = json.stringify(a, payload, .{}) catch return error.Closed;
    return conn.writeAll(std.fmt.allocPrint(a, "event: {s}\ndata: {s}\n\n", .{ kind, text }) catch return error.Closed);
}

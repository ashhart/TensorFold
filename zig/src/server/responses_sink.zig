//! Where a Responses turn's typed events go: an SSE stream, or another transport with the same three calls.
const std = @import("std");
const json = @import("json");
const sse_wire = @import("sse.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// A turn's event transport.
pub const Sink = struct {
    ctx: *anyopaque,
    vt: *const VTable,

    pub const VTable = struct {
        /// The stream's head, before the first event (an SSE reply head; a socket sends nothing).
        open: *const fn (ctx: *anyopaque) error{Closed}!void,
        /// One typed event object, already carrying its ``type`` and ``sequence_number``.
        event: *const fn (ctx: *anyopaque, a: Allocator, kind: []const u8, payload: *json.Object) error{Closed}!void,
        /// Bytes that keep a silent stream alive (an SSE comment; a socket ping).
        keepalive: *const fn (ctx: *anyopaque) error{Closed}!void,
    };
};

/// The HTTP sink: ``event:`` and ``data:`` frames on the client's connection.
pub fn sse(conn: *Conn) Sink {
    return .{ .ctx = conn, .vt = &sse_vt };
}

const sse_vt: Sink.VTable = .{ .open = sseOpen, .event = sseEvent, .keepalive = sseKeepalive };

fn connOf(ctx: *anyopaque) *Conn {
    return @ptrCast(@alignCast(ctx));
}

fn sseOpen(ctx: *anyopaque) error{Closed}!void {
    return sse_wire.open(connOf(ctx));
}

fn sseEvent(ctx: *anyopaque, a: Allocator, kind: []const u8, payload: *json.Object) error{Closed}!void {
    return sse_wire.event(connOf(ctx), a, kind, .{ .object = payload });
}

fn sseKeepalive(ctx: *anyopaque) error{Closed}!void {
    return sse_wire.comment(connOf(ctx));
}

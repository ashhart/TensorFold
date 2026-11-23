//! A chat reply written straight to the client: its own Out, an SSE stream or one JSON reply (openai.zig).
const std = @import("std");
const sse = @import("sse.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;
const Value = @import("json").Value;
const Out = @import("openai.zig").Out;
const Allocator = std.mem.Allocator;

pub const ClientOut = struct {
    startup_at: ?u64 = null,
    conn: *Conn,
    a: Allocator,

    pub fn out(c: *ClientOut) Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply, .keepalive = keepalive } };
    }

    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return sse.open(c.conn);
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return if (payload) |p| sse.data(c.conn, c.a, p) else sse.done(c.conn);
    }

    fn reply(ctx: *anyopaque, status: u16, payload: Value) void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        routes.sendValue(c.conn, c.a, status, payload);
        if (status == 200 and c.conn.response_complete and !c.conn.broken) @import("startup_receipt.zig").complete(c.startup_at, payload);
    }

    fn keepalive(ctx: *anyopaque) error{Closed}!void {
        const c: *ClientOut = @ptrCast(@alignCast(ctx));
        return sse.comment(c.conn);
    }
};

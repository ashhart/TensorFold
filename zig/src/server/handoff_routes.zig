//! GET and DELETE /v1/handoff/<id>: a producer's held exports as an MKVH stream, read by a decoder with no fabric link.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const param = @import("handoff_param.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

pub fn isRoute(route: []const u8) bool {
    return std.mem.startsWith(u8, route, param.route_prefix);
}

fn fail(conn: *Conn, a: Allocator, code: u16, message: []const u8) void {
    const body = std.fmt.allocPrint(a, "{{\"error\": {{\"message\": {s}, \"type\": \"invalid_request_error\"}}}}", .{json.quote(a, message, .{}) catch return}) catch return;
    conn.sendJson(code, body);
}

/// What this route needs of the engine: the exporter's serve hook, or the reply that says there is none.
fn serving(srv: *Server, conn: *Conn, a: Allocator, route: []const u8) ?struct { id: [16]u8, serve: api.LaneHost.Export.Serve } {
    const id = param.routeId(route) orelse {
        fail(conn, a, 400, "the hand-off id is 32 hex digits: /v1/handoff/<id>");
        return null;
    };
    const serve = api.LaneHost.Export.serving(srv.engine) orelse {
        fail(conn, a, 404, "this server serves no hand-off exports (a producer starts with --mcdma-produce http)");
        return null;
    };
    return .{ .id = id, .serve = serve };
}

/// GET: streams the export under the id; 404 when none is held, 409 with the reason when its export failed.
pub fn get(srv: *Server, conn: *Conn, a: Allocator, route: []const u8) void {
    const s = serving(srv, conn, a, route) orelse return;
    var why: []const u8 = "";
    const stream = s.serve.open(s.serve.ptr, s.id, &why) catch |err| return switch (err) {
        error.NotHeld => fail(conn, a, 404, "no export is held under this id: it was never made, was deleted, or expired"),
        error.Refused => fail(conn, a, 409, why),
        error.OutOfMemory, error.Unreadable => fail(conn, a, 500, "the export could not be read"),
    };
    defer stream.close(stream.ptr);
    var length: [24]u8 = undefined;
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "application/octet-stream") catch return;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&length, "{d}", .{stream.size}) catch unreachable) catch return;
    conn.addHeader("Connection", "close") catch return; // one export a connection: a cut stream is a short body, never a reused line
    conn.finish("") catch return;
    var sent: u64 = 0;
    while (true) {
        const message = stream.next(stream.ptr) catch {
            conn.broken = true; // the body cannot be completed: the client sees a short one
            conn.close = true;
            return;
        } orelse break;
        conn.writeAll(message) catch return;
        sent += message.len;
    }
    if (sent != stream.size) { // the stream's length was promised in the head
        conn.broken = true;
        conn.close = true;
    }
}

/// DELETE: the decoder is done with the export; it is released now. 404 when none is held.
pub fn delete(srv: *Server, conn: *Conn, a: Allocator, route: []const u8) void {
    const s = serving(srv, conn, a, route) orelse return;
    if (!s.serve.release(s.serve.ptr, s.id)) return fail(conn, a, 404, "no export is held under this id: it was never made, was deleted, or expired");
    conn.sendJson(200, "{\"released\": true}");
}

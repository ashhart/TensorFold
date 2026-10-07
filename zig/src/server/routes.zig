//! Method and path dispatch, as ``do_GET``, ``do_POST`` and ``do_DELETE`` route them.
const std = @import("std");
const json = @import("json.zig");
const http_conn = @import("http_conn.zig");
const http_body = @import("http_body.zig");
const auth = @import("auth.zig");
const openai = @import("openai.zig");
const responses = @import("responses.zig");
const anthropic = @import("anthropic.zig");
const tokens = @import("tokens.zig");
const decisions = @import("decisions.zig");
const status = @import("status_routes.zig");
const Server = @import("server.zig").Server;
const Conn = http_conn.Conn;
const Allocator = std.mem.Allocator;

pub fn dispatch(srv: *Server, conn: *Conn, a: Allocator) void {
    const m = conn.method;
    if (std.mem.eql(u8, m, "GET")) return get(srv, conn, a);
    if (std.mem.eql(u8, m, "POST")) return post(srv, conn, a);
    if (std.mem.eql(u8, m, "DELETE")) return responses.delete(srv, conn, a, responses.route(auth.routePath(conn.path)));
    const message = std.fmt.allocPrint(a, "Unsupported method ({s})", .{http_conn.pyRepr(a, m) catch m}) catch return;
    conn.sendError(a, 501, message, null) catch {};
}

/// Python's ``json.dumps(payload)`` sent as a JSON reply.
pub fn sendValue(conn: *Conn, a: Allocator, code: u16, payload: json.Value) void {
    const body = json.stringify(a, payload, .{}) catch return;
    conn.sendJson(code, body);
}

pub fn unknown(conn: *Conn, a: Allocator) void {
    const message = std.fmt.allocPrint(a, "unknown path {s}", .{conn.requestPath(a)}) catch return;
    const body = std.fmt.allocPrint(a, "{{\"error\": {{\"message\": {s}}}}}", .{json.quote(a, message, .{}) catch return}) catch return;
    conn.sendJson(404, body);
}

fn get(srv: *Server, conn: *Conn, a: Allocator) void {
    const route = auth.routePath(conn.path);
    if (responses.route(route)) |rid| if (rid.len > 0) return responses.get(srv, conn, a, rid);
    if (std.mem.eql(u8, route, "/metrics") or std.mem.eql(u8, route, "/v1/metrics")) return status.metrics(srv, conn, a);
    if (std.mem.eql(u8, route, "/dashboard")) return status.dashboard(srv, conn, a);
    if (std.mem.eql(u8, route, "/stats")) return status.stats(srv, conn, a);
    if (route.len == 0 or std.mem.eql(u8, route, "/health")) return status.health(srv, conn, a) catch {};
    if (std.mem.endsWith(u8, route, "/models")) return status.models(srv, conn, a) catch {};
    unknown(conn, a);
}

/// A refused route's body is read and dropped, so it cannot reach the next request on this connection.
fn discardBody(conn: *Conn, a: Allocator) void {
    _ = http_body.read(conn, a, http_body.limit) catch {};
}

fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    const route = auth.routePath(conn.path);
    if (std.mem.endsWith(u8, route, "/decisions")) return decisions.post(srv, conn, a);
    if (anthropic.route(conn.path)) return anthropic.post(srv, conn, a);
    if (responses.route(route)) |rid| if (rid.len == 0) return responses.post(srv, conn, a);
    for (tokens.paths) |r| if (std.mem.eql(u8, route, r)) return tokens.post(srv, conn, a, std.mem.endsWith(u8, route, "/detokenize"));
    const is_chat = std.mem.endsWith(u8, route, "/chat/completions");
    if (!is_chat and !std.mem.endsWith(u8, route, "/completions")) {
        discardBody(conn, a);
        return unknown(conn, a);
    }
    openai.post(srv, conn, a, is_chat);
}

//! Method and path dispatch, as ``do_GET``, ``do_POST`` and ``do_DELETE`` route them.
const speech = @import("speech.zig");
const embeddings = @import("embeddings.zig");
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const http_conn = @import("http_conn.zig");
const http_body = @import("http_body.zig");
const auth = @import("auth.zig");
const openai = @import("openai.zig");
const responses = @import("responses.zig");
const anthropic = @import("anthropic.zig");
const tokens = @import("tokens.zig");
const decisions = @import("decisions.zig");
const systemone = @import("systemone.zig");
const status = @import("status_routes.zig");
const memory = @import("memory_routes.zig");
const slide = @import("slide.zig");
const recall = @import("recall_routes.zig");
const imprint = @import("imprint_routes.zig");
const slide_forget = @import("slide_forget.zig");
const Server = @import("server.zig").Server;
const Conn = http_conn.Conn;
const Allocator = std.mem.Allocator;

/// DELETE's per-fact route: the fact's id follows it.
const fact_path = "/v1/slide/fact/";

pub fn dispatch(srv: *Server, conn: *Conn, a: Allocator) void {
    const m = conn.method;
    if (!std.mem.eql(u8, m, "POST") and !consumeBody(conn, a)) return;
    if (std.mem.eql(u8, m, "GET")) return get(srv, conn, a);
    if (std.mem.eql(u8, m, "POST")) return post(srv, conn, a);
    if (std.mem.eql(u8, m, "DELETE")) {
        const route = auth.routePath(conn.path);
        if (std.mem.eql(u8, route, "/v1/slide/graph")) return slide.forget(srv, conn, a);
        if (std.mem.startsWith(u8, route, fact_path)) return slide_forget.route(srv, conn, a, route[fact_path.len..]);
        if (std.mem.eql(u8, route, recall.path)) return recall.empty(srv, conn, a);
        if (std.mem.eql(u8, route, "/v1/imprint")) return imprint.forget(srv, conn, a);
        return responses.delete(srv, conn, a, responses.route(route));
    }
    const message = std.fmt.allocPrint(a, "Unsupported method ({s})", .{http_conn.pyRepr(a, m) catch m}) catch return;
    conn.sendError(a, 501, message, null) catch {};
}

fn consumeBody(conn: *Conn, a: Allocator) bool {
    const body = http_body.read(conn, a, http_body.limit) catch {
        conn.close = true;
        conn.sendError(a, 500, "out of memory", null) catch {};
        return false;
    };
    switch (body) {
        .ok => return true,
        .refused => |message| {
            conn.sendError(a, 400, message, null) catch {};
            return false;
        },
    }
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
    if (std.mem.eql(u8, route, "/v1/voices")) {
        if (srv.info.speech == null) return unknown(conn, a);
        const value = @import("drama_voice.zig").listing(a, srv.voices) catch return;
        return sendValue(conn, a, 200, value);
    }
    if (std.mem.eql(u8, route, "/memory") or std.mem.eql(u8, route, "/v1/memory"))
        return memory.get(srv, conn, a) catch {};
    if (responses.route(route)) |rid| if (rid.len > 0) return responses.get(srv, conn, a, rid);
    if (std.mem.eql(u8, route, "/metrics") or std.mem.eql(u8, route, "/v1/metrics")) return status.metrics(srv, conn, a);
    if (std.mem.eql(u8, route, "/dashboard")) return status.dashboard(srv, conn, a);
    if (std.mem.eql(u8, route, "/stats")) return status.stats(srv, conn, a);
    if (std.mem.eql(u8, route, "/slide")) return slide.view(srv, conn, a);
    if (std.mem.startsWith(u8, route, "/slide/")) return slide.script(srv, conn, a, route["/slide/".len..]);
    if (std.mem.eql(u8, route, "/v1/slide/graph")) return slide.graph(srv, conn, a);
    if (std.mem.eql(u8, route, recall.path)) return recall.get(srv, conn, a);
    if (std.mem.eql(u8, route, "/v1/benchmark/manifest")) return status.manifestRoute(srv, conn, a) catch {};
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
    for (tokens.paths) |r| if (std.mem.eql(u8, route, r)) return tokens.post(srv, conn, a, std.mem.endsWith(u8, route, "/detokenize"));
    const is_speech = std.mem.eql(u8, route, "/v1/audio/speech");
    const is_embedding = std.mem.eql(u8, route, "/v1/embeddings");
    const is_anthropic = anthropic.route(conn.path);
    const is_responses = if (responses.route(route)) |rid| rid.len == 0 else false;
    const is_decisions = std.mem.endsWith(u8, route, "/decisions");
    const is_systemone = std.mem.endsWith(u8, route, "/systemone");
    const is_slide = std.mem.eql(u8, route, "/v1/slide/learn");
    const imprint_op: ?api.ImprintOp = if (std.mem.eql(u8, route, "/v1/imprint/resume")) .where else if (std.mem.eql(u8, route, "/v1/imprint/load")) .load else null;
    const is_chat = std.mem.endsWith(u8, route, "/chat/completions");
    if (!is_embedding and !is_speech and !is_slide and imprint_op == null and !is_anthropic and !is_responses and !is_decisions and !is_systemone and !is_chat and !std.mem.endsWith(u8, route, "/completions")) {
        discardBody(conn, a);
        return unknown(conn, a);
    }
    // counted before the drain flag is read: a stop that sees no request in progress sees every later one refused
    _ = srv.generating.fetchAdd(1, .seq_cst);
    defer _ = srv.generating.fetchSub(1, .seq_cst);
    if (@import("builtin").is_test) if (counted_hook) |h| h();
    if (srv.draining.load(.seq_cst)) {
        discardBody(conn, a);
        return draining(conn, is_anthropic);
    }
    if (srv.info.embeddings != null and (is_speech or is_anthropic or is_responses or is_decisions or is_systemone or is_chat or std.mem.endsWith(u8, route, "/completions"))) {
        discardBody(conn, a);
        conn.sendJson(400, "{\"error\":{\"message\":\"this model produces embeddings; use POST /v1/embeddings\",\"type\":\"invalid_request_error\"}}");
        return;
    }
    if (std.mem.eql(u8, route, "/v1/audio/speech")) return speech.post(srv, conn, a) catch {};
    if (is_embedding) return embeddings.post(srv, conn, a);
    if (is_slide) return slide.learn(srv, conn, a);
    if (imprint_op) |op| return imprint.post(srv, conn, a, op);
    if (is_decisions) return decisions.post(srv, conn, a);
    if (is_systemone) return systemone.post(srv, conn, a);
    if (is_anthropic) return anthropic.post(srv, conn, a);
    if (is_responses) return responses.post(srv, conn, a);
    openai.post(srv, conn, a, is_chat);
}

/// Tests only: the gap between a request's count and its drain check, where a stop must still see it counted.
pub var counted_hook: ?*const fn () void = null;

/// Seconds a refused client is told to wait while the server drains for a restart (Retry-After).
pub const retry_after_s = 30;
pub const restarting = "server restarting: retry shortly";

/// A request that arrives while a stop drains: 503, Retry-After, the API's own error shape; the connection closes.
pub fn draining(conn: *Conn, anthropic_shape: bool) void {
    const body = if (anthropic_shape)
        "{\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\", \"message\": \"" ++ restarting ++ "\"}}"
    else
        "{\"error\": {\"message\": \"" ++ restarting ++ "\", \"type\": \"server_error\", \"param\": null, \"code\": \"server_restarting\"}}";
    sendDraining(conn, body);
}

/// 503 with Retry-After and Connection: close, `body` as JSON.
pub fn sendDraining(conn: *Conn, body: []const u8) void {
    conn.close = true;
    conn.startResponse(503, null) catch return;
    conn.addHeader("Content-Type", "application/json") catch return;
    conn.addHeader("Retry-After", std.fmt.comptimePrint("{d}", .{retry_after_s})) catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch unreachable) catch return;
    conn.addHeader("Connection", "close") catch return;
    conn.finish(body) catch {};
}

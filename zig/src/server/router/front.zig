//! The router's HTTP side: a thread a connection behind the server's guards, then a model route or the router's own.
const std = @import("std");
const auth = @import("../auth.zig");
const auth_scope = @import("../auth_scope.zig");
const http_conn = @import("../http_conn.zig");
const http_body = @import("../http_body.zig");
const admission = @import("../http_admission.zig");
const listener_mod = @import("../listener.zig");
const request_guard = @import("../request_guard.zig");
const catalog = @import("catalog.zig");
const pool_mod = @import("pool.zig");
const relay = @import("relay.zig");
const replies = @import("replies.zig");
const listing = @import("listing.zig");
const model_route = @import("model_route.zig");
const Conn = http_conn.Conn;
const Allocator = std.mem.Allocator;

/// The most a kept-alive connection's arena keeps between requests, as the server keeps it.
const arena_limit: usize = 1 << 20;

/// A POST route that names its model: its error shape, and whether it is Responses (whose storage keys are checked).
pub const ModelRoute = struct { shape: replies.Shape, responses: bool = false };

pub const Front = struct {
    gpa: Allocator,
    io: std.Io,
    pool: *pool_mod.Pool,
    entries: []const catalog.Entry,
    keys: ?*auth.Store = null,
    browser: request_guard.Policy = .{},
    version: []const u8 = "",
    timeouts: http_conn.Timeouts = .{},
    limits: relay.Limits = .{},
    max_connections: u32 = admission.default_limit,
    open: std.atomic.Value(u32) = .init(0),
    peers: admission.Peers = .{},
    /// Set once the router is stopping: model requests get 503 and /health says draining.
    draining: std.atomic.Value(bool) = .init(false),

    /// Accepts connections until ``stop`` is set, each on its own thread.
    pub fn serve(f: *Front, lis: listener_mod.Listener, stop: *const std.atomic.Value(bool)) void {
        const loopback_bind = if (lis.boundAddress()) |addr| auth.loopbackAddress(addr) else true;
        relay.closeOnExec(lis.fd);
        while (lis.accept(stop)) |c| {
            relay.closeOnExec(c.fd);
            if (!admission.acquire(&f.open, f.max_connections)) {
                admission.reject(c.fd);
                continue;
            }
            const slot = admission.peerSlot(&f.peers, f.io, &f.open, f.max_connections, .{}, c.fd, c.peer) orelse continue;
            const t = std.Thread.spawn(.{ .stack_size = 8 << 20 }, connection, .{ f, c, loopback_bind, slot }) catch {
                admission.release(&f.peers, f.io, &f.open, slot);
                _ = std.posix.system.close(c.fd);
                continue;
            };
            t.detach();
        }
    }

    fn connection(f: *Front, accepted: listener_mod.Listener.Accepted, loopback_bind: bool, slot: u16) void {
        defer admission.release(&f.peers, f.io, &f.open, slot);
        defer _ = std.posix.system.close(accepted.fd);
        var peer_buf: [64]u8 = undefined;
        var conn = Conn.init(f.gpa, accepted.fd, listener_mod.peerText(&peer_buf, accepted.peer)) catch return;
        defer conn.deinit();
        conn.timeouts = f.timeouts;
        conn.shed = .{ .open = &f.open, .limit = f.max_connections };
        conn.auth_enabled = f.keys != null and f.keys.?.enabled();
        var browser = f.browser;
        browser.loopback_bind = loopback_bind;
        var arena: std.heap.ArenaAllocator = .init(f.gpa);
        defer arena.deinit();
        var first = true;
        while (true) {
            _ = arena.reset(.{ .retain_with_limit = arena_limit });
            const a = arena.allocator();
            const outcome = conn.readRequest(a, !first) catch break;
            first = false;
            switch (outcome) {
                .closed => break,
                .handled => if (conn.close) break else continue,
                .ready => {},
            }
            if (!request_guard.allow(&conn, browser, a)) break;
            if (!@import("../auth_gate.zig").admit(f.keys, &conn, a)) break;
            if (conn.header("Expect")) |expect| if (std.ascii.eqlIgnoreCase(expect, "100-continue") and std.mem.order(u8, conn.version, "HTTP/1.1") != .lt) conn.sendContinue();
            f.dispatch(&conn, a);
            if (!conn.response_complete) conn.close = true;
            if (conn.close or conn.broken) break;
        }
    }

    fn dispatch(f: *Front, conn: *Conn, a: Allocator) void {
        const route = auth.routePath(conn.path);
        if (modelRoute(conn.method, conn.path)) |r| return model_route.serve(f, conn, a, r);
        const shape = shapeOf(conn.path);
        if (std.mem.eql(u8, conn.method, "POST") or std.mem.eql(u8, conn.method, "PUT") or std.mem.eql(u8, conn.method, "DELETE")) {
            _ = http_body.read(conn, a, http_body.limit) catch {};
            const what = std.fmt.allocPrint(a, "{s} {s}", .{ conn.method, route }) catch "this route";
            return replies.send(conn, a, shape, replies.unsupported(a, what, "the router serves model routes, whose body names a model, and lists its models itself"));
        }
        if (!std.mem.eql(u8, conn.method, "GET")) {
            if (!consumeBody(conn, a)) return;
            const message = std.fmt.allocPrint(a, "Unsupported method ({s})", .{http_conn.pyRepr(a, conn.method) catch conn.method}) catch return;
            return conn.sendError(a, 501, message, null) catch {};
        }
        if (!consumeBody(conn, a)) return;
        if (route.len == 0 or std.mem.eql(u8, route, "/health")) return listing.health(f, conn, a);
        if (std.mem.eql(u8, route, "/api/tags")) return listing.tags(f, conn, a);
        if (std.mem.eql(u8, route, "/api/ps")) return listing.ps(f, conn, a);
        if (std.mem.eql(u8, route, "/api/version")) return listing.version(f, conn, a);
        if (std.mem.endsWith(u8, route, "/models")) return listing.models(f, conn, a);
        if (@import("../responses.zig").route(route)) |rid| {
            if (rid.len > 0) return replies.send(conn, a, shape, replies.unsupported(a, "GET /v1/responses/{id}", "stored responses stay in the model server that made them; send the whole conversation, or store: false"));
            if (@import("../ws_upgrade.zig").requested(conn)) return replies.send(conn, a, shape, replies.unsupported(a, "The Responses WebSocket mode", "send POST /v1/responses over HTTP"));
        }
        if (std.mem.eql(u8, route, "/metrics") or std.mem.eql(u8, route, "/v1/metrics"))
            return replies.send(conn, a, shape, replies.unsupported(a, "GET /metrics", "metrics are per model, and each model's server keeps its own; /health lists the loaded models"));
        const what = std.fmt.allocPrint(a, "GET {s}", .{route}) catch "this route";
        replies.send(conn, a, shape, replies.unsupported(a, what, "the router answers /health, /v1/models, /api/tags, /api/ps and /api/version itself"));
    }
};

/// The model routes the router forwards: POST (and PUT for rerank) with a JSON body that names a model.
pub fn modelRoute(method: []const u8, raw_path: []const u8) ?ModelRoute {
    const route = auth.routePath(raw_path);
    if (std.mem.eql(u8, method, "PUT") and auth_scope.putRoute(route)) return .{ .shape = .openai };
    if (!std.mem.eql(u8, method, "POST")) return null;
    if (std.mem.endsWith(u8, route, "/completions")) return .{ .shape = .openai };
    if (@import("../responses.zig").route(route)) |rid| return if (rid.len == 0) .{ .shape = .openai, .responses = true } else null;
    if (@import("../anthropic.zig").route(raw_path)) return .{ .shape = .anthropic };
    for ([_][]const u8{ "/v1/embeddings", "/v1/rerank", "/rerank" } ++ @import("../tokens.zig").paths) |p| if (std.mem.eql(u8, route, p)) return .{ .shape = .openai };
    for ([_][]const u8{ "/api/chat", "/api/generate", "/api/embed", "/api/show" }) |p| if (std.mem.eql(u8, route, p)) return .{ .shape = .ollama };
    return null;
}

/// The error shape for any path: Ollama's under /api/, Anthropic's for Messages, else OpenAI's.
pub fn shapeOf(raw_path: []const u8) replies.Shape {
    if (std.mem.startsWith(u8, auth.routePath(raw_path), "/api/")) return .ollama;
    if (@import("../anthropic.zig").route(raw_path)) return .anthropic;
    return .openai;
}

/// A body no route reads, taken to keep the connection in step, as the server takes it.
fn consumeBody(conn: *Conn, a: Allocator) bool {
    const body = http_body.read(conn, a, http_body.unread_limit) catch {
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

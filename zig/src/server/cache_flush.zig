//! POST /v1/prompt_cache/flush: every kept prompt state dropped, in RAM and on disk, after the requests in flight end.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const cancel_ids = @import("cancel_ids.zig");
const log = @import("log.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
const Ask = api.ImprintAsk;

pub const path = "/v1/prompt_cache/flush";

/// The engine answers once the requests in flight have ended; a long reply can hold it this long.
pub const wait_ms = 10 * 60 * 1000;

pub const Reply = struct { status: u16, body: json.Value };

/// The flush asked of `engine` and waited for: 200 with what it dropped, 501 where no store keeps states, else 503.
pub fn answer(gpa: Allocator, io: std.Io, a: Allocator, engine: api.Engine, ms: u32) !Reply {
    // the ask lives in the server's memory: once the engine holds it past a 503, it leaks rather than be freed under it
    const ask = try gpa.create(Ask);
    ask.* = .{ .op = .flush };
    const t0 = std.Io.Clock.awake.now(io);
    if (try hold(io, a, engine, ask, ms, "flush")) |refused| {
        if (refused.status == 501) gpa.destroy(ask);
        return refused;
    }
    defer gpa.destroy(ask);
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "prompt_cache.flush" });
    try o.put(a, "states", try json.intValue(a, ask.dropped));
    try o.put(a, "ram_bytes", try json.intValue(a, ask.ram_freed));
    try o.put(a, "disk_bytes", try json.intValue(a, ask.freed));
    try o.put(a, "disk", .{ .bool = ask.disk });
    const ns = t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    try o.put(a, "waited_seconds", .{ .float = @as(f64, @floatFromInt(ns)) / 1e9 });
    return .{ .status = 200, .body = .{ .object = o } };
}

/// A flush or drop put to `engine` and waited for: null once answered, else its 501 (untaken) or 503 (left to it).
pub fn hold(io: std.Io, a: Allocator, engine: api.Engine, ask: *Ask, ms: u32, what: []const u8) !?Reply {
    if (!engine.imprintAsk(ask)) {
        const none = "this engine keeps no prompt states it can drop";
        return .{ .status = 501, .body = try failure(a, none, "invalid_request_error", "not_supported") };
    }
    if (ask.wait(io, ms)) return null;
    const withdrawn = ask.withdraw(); // not yet taken: it never runs, and admissions go on at once
    if (!withdrawn and ask.wait(io, ms)) return null;
    log.line("prompt cache {s}: the engine did not answer in {d} s", .{ what, ms / 1000 });
    const late = "the requests in flight did not end in time: it was withdrawn and nothing was dropped";
    const slow = "the engine did not finish in time";
    const why = try std.fmt.allocPrint(a, "{s}: {s}", .{ what, if (withdrawn) late else slow });
    return .{ .status = 503, .body = try failure(a, why, "server_error", "timeout") };
}

pub fn failure(a: Allocator, message: []const u8, kind: []const u8, code: []const u8) !json.Value {
    const err = try json.newObject(a);
    try err.put(a, "message", .{ .string = message });
    try err.put(a, "type", .{ .string = kind });
    try err.put(a, "code", .{ .string = code });
    const o = try json.newObject(a);
    try o.put(a, "error", .{ .object = err });
    return .{ .object = o };
}

/// POST /v1/prompt_cache/flush under the request cancel route's rule: an admin key, or a direct loopback caller.
pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    const keyed = if (srv.keys) |k| k.enabled() else false;
    if (!cancel_ids.allowed(keyed, conn)) {
        if (keyed) return @import("auth_scope_reply.zig").deny(conn, false, .route);
        const message = "without API keys, only a direct loopback caller may flush the prompt cache";
        return routes.sendValue(conn, a, 403, failure(a, message, "permission_error", "loopback_only") catch return);
    }
    // --idle-unload: a resident engine is not parked under the flush; a parked one is not loaded for it
    const idle = srv.idle_manager;
    if (idle) |m| m.begin();
    defer if (idle) |m| m.end();
    if (idle) |m| if (m.isParked()) {
        const message = "the model is unloaded: no state is in RAM, and those on disk stay until a flush while it is loaded";
        return routes.sendValue(conn, a, 503, failure(a, message, "server_error", "model_unloaded") catch return);
    };
    @import("warm.zig").forgetting(srv); // no warm pass of a reply running now keeps a state after the flush
    const got = answer(srv.gpa, srv.io, a, srv.engine, wait_ms) catch return conn.sendError(a, 500, "out of memory", null) catch {};
    routes.sendValue(conn, a, got.status, got.body);
}

test {
    _ = @import("cache_flush_test.zig");
}

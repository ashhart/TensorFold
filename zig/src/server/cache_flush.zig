//! POST /v1/prompt_cache/flush: full by default, or retain the newest states after active requests end.
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
const http_body = @import("http_body.zig");

pub const path = "/v1/prompt_cache/flush";

/// The engine answers once the requests in flight have ended; a long reply can hold it this long.
pub const wait_ms = 10 * 60 * 1000;

pub const Reply = struct { status: u16, body: json.Value };

/// The flush asked of `engine` and waited for: 200 with what it dropped, 501 where no store keeps states, else 503.
pub fn answer(gpa: Allocator, io: std.Io, a: Allocator, engine: api.Engine, ms: u32) !Reply {
    return answerRetaining(gpa, io, a, engine, ms, 0);
}

/// A full flush at `retain == 0`; otherwise retain that many newest complete states on the store's existing clock.
pub fn answerRetaining(gpa: Allocator, io: std.Io, a: Allocator, engine: api.Engine, ms: u32, retain: u32) !Reply {
    // the ask lives in the server's memory: once the engine holds it past a 503, it leaks rather than be freed under it
    const ask = try gpa.create(Ask);
    ask.* = .{ .op = .flush, .retain = retain };
    const t0 = std.Io.Clock.awake.now(io);
    if (!engine.imprintAsk(ask)) {
        gpa.destroy(ask);
        return .{ .status = 501, .body = try failure(a, "this engine keeps no prompt states it can drop", "invalid_request_error", "not_supported") };
    }
    if (!ask.wait(io, ms)) {
        const withdrawn = ask.withdraw(); // not yet taken: it never runs, and admissions go on at once
        if (withdrawn or !ask.wait(io, ms)) {
            log.line("prompt cache flush: the engine did not answer in {d} s", .{ms / 1000});
            const why = if (withdrawn) "the requests in flight did not end in time: the flush was withdrawn and nothing was dropped" else "the engine did not finish the flush in time";
            return .{ .status = 503, .body = try failure(a, why, "server_error", "timeout") };
        }
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

fn failure(a: Allocator, message: []const u8, kind: []const u8, code: []const u8) !json.Value {
    const err = try json.newObject(a);
    try err.put(a, "message", .{ .string = message });
    try err.put(a, "type", .{ .string = kind });
    try err.put(a, "code", .{ .string = code });
    const o = try json.newObject(a);
    try o.put(a, "error", .{ .object = err });
    return .{ .object = o };
}

fn retainFromBody(a: Allocator, raw: []const u8) (Allocator.Error || error{InvalidRetain})!u32 {
    if (raw.len == 0) return 0;
    const parsed = try json.parse(a, raw);
    const body = switch (parsed) {
        .ok => |v| v,
        .err => return 0, // a body without usable retain JSON remains the old full flush
    };
    if (body != .object) return 0;
    const field = body.get("retain") orelse return 0;
    if (field != .int) return error.InvalidRetain;
    const n = field.int64() orelse return error.InvalidRetain;
    if (n < 0 or n > std.math.maxInt(u32)) return error.InvalidRetain;
    return @intCast(n);
}
/// POST /v1/prompt_cache/flush under the request cancel route's rule: an admin key, or a direct loopback caller.
pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    // main read and dropped the body here and flushed all: a body not read, or refused, is still that full flush
    const raw: []const u8 = if (http_body.read(conn, a, http_body.limit)) |body| switch (body) {
        .ok => |b| b,
        .refused => "",
    } else |_| "";
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
    const retain = retainFromBody(a, raw) catch |err| switch (err) {
        error.OutOfMemory => 0, // no retention count was read: main dropped the body and flushed all
        error.InvalidRetain => return routes.sendValue(conn, a, 400, failure(a, "retain must be a nonnegative 32-bit integer", "invalid_request_error", "invalid_request_error") catch return),
    };
    const got = answerRetaining(srv.gpa, srv.io, a, srv.engine, wait_ms, retain) catch return conn.sendError(a, 500, "out of memory", null) catch {};
    routes.sendValue(conn, a, got.status, got.body);
}

test {
    _ = @import("cache_flush_test.zig");
}

//! GET /health, /v1/models and /metrics.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const live = @import("live.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// With keys on, health names nothing about the model.
pub fn health(srv: *Server, conn: *Conn, a: Allocator) !void {
    if (srv.keys) |k| if (k.enabled()) return conn.sendJson(200, "{\"status\": \"ok\"}");
    const o = try json.newObject(a);
    try o.put(a, "status", .{ .string = "ok" });
    try o.put(a, "model", .{ .string = srv.config.served_name });
    const ids = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, ids) |id, *slot| slot.* = .{ .string = id };
    try o.put(a, "model_ids", .{ .array = ids });
    try o.put(a, "max_batch_size", try json.intValue(a, srv.info.lanes));
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    try o.put(a, "warming", .{ .bool = status.warming });
    const memory = try json.newObject(a);
    if (srv.engine.memory(std.mem.indexOf(u8, conn.path, "reset_peak=1") != null)) |mem| {
        try memory.put(a, "active", try json.intValue(a, mem.active));
        try memory.put(a, "cache", try json.intValue(a, mem.cache));
        try memory.put(a, "peak", try json.intValue(a, mem.peak));
    }
    try o.put(a, "memory", .{ .object = memory });
    try o.put(a, "live", try live.snapshot(a, srv.engine));
    routes.sendValue(conn, a, 200, .{ .object = o });
}

pub fn models(srv: *Server, conn: *Conn, a: Allocator) !void {
    const created = std.Io.Clock.real.now(srv.io).toSeconds();
    const data = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, data) |id, *slot| {
        const m = try json.newObject(a);
        try m.put(a, "id", .{ .string = id });
        try m.put(a, "object", .{ .string = "model" });
        try m.put(a, "created", try json.intValue(a, created));
        try m.put(a, "owned_by", .{ .string = "tensorfold" });
        slot.* = .{ .object = m };
    }
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "list" });
    try o.put(a, "data", .{ .array = data });
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// Prometheus text, version 0.0.4.
pub fn metrics(srv: *Server, conn: *Conn, a: Allocator) void {
    var out: std.Io.Writer.Allocating = .init(a);
    srv.metrics.render(srv.io, &out.writer, srv.engine, srv.info.context_window) catch return;
    const body = out.written();
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    conn.finish(body) catch {};
}

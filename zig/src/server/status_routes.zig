//! GET /health, /v1/models, /metrics, and the opt-in dashboard.
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
    if (srv.info.policy.len > 0) try o.put(a, "policy", .{ .string = srv.info.policy });
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
        slot.* = try modelValue(a, id, srv.info, created);
    }
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "list" });
    try o.put(a, "data", .{ .array = data });
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// Discovery extensions report the loaded engine's prompt-plus-reply limit; zero leaves it unspecified.
fn modelValue(a: Allocator, id: []const u8, info: api.Info, created: i64) Allocator.Error!Value {
    const m = try json.newObject(a);
    try m.put(a, "id", .{ .string = id });
    try m.put(a, "object", .{ .string = "model" });
    try m.put(a, "created", try json.intValue(a, created));
    try m.put(a, "owned_by", .{ .string = "tensorfold" });
    if (info.context_window > 0) {
        try m.put(a, "context_length", try json.intValue(a, info.context_window));
        try m.put(a, "max_model_len", try json.intValue(a, info.context_window));
    }
    return .{ .object = m };
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

pub fn dashboard(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/html; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{dashboard_page.len}) catch return) catch return;
    conn.finish(dashboard_page) catch {};
}

pub fn stats(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    const memory = srv.engine.memory(false);
    srv.metrics.mutex.lockUncancelable(srv.io);
    const prompt = srv.metrics.prompt;
    const generation = srv.metrics.generation;
    const drafted = srv.metrics.drafted;
    const accepted = srv.metrics.accepted;
    const rounds = srv.metrics.rounds;
    srv.metrics.mutex.unlock(srv.io);
    const body = statsSnapshot(a, status, memory, prompt, generation, drafted, accepted, rounds) catch return;
    conn.quiet_log = true;
    routes.sendValue(conn, a, 200, body);
}

pub fn statsSnapshot(a: Allocator, status: api.Status, memory: ?api.Memory, prompt: u64, generation: u64, drafted: u64, accepted: u64, rounds: u64) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "running", try json.intValue(a, status.running));
    try o.put(a, "waiting", try json.intValue(a, status.waiting));
    try o.put(a, "decode_tokens_per_second", .{ .float = status.decode_tokens_per_second });
    try o.put(a, "prefill_tokens_per_second", .{ .float = status.prefill_tokens_per_second });
    try o.put(a, "generation_tokens_running", try json.intValue(a, status.generation_tokens));
    try o.put(a, "prompt_tokens_total", try json.intValue(a, prompt));
    try o.put(a, "generation_tokens_total", try json.intValue(a, generation));
    try o.put(a, "mtp_drafted_total", try json.intValue(a, drafted));
    try o.put(a, "mtp_accepted_total", try json.intValue(a, accepted));
    try o.put(a, "decode_rounds_total", try json.intValue(a, rounds));
    const mem = try json.newObject(a);
    if (memory) |m| {
        try mem.put(a, "active", try json.intValue(a, m.active));
        try mem.put(a, "cache", try json.intValue(a, m.cache));
        try mem.put(a, "peak", try json.intValue(a, m.peak));
    }
    try o.put(a, "memory", .{ .object = mem });
    return .{ .object = o };
}

const dashboard_page = @embedFile("dashboard.html");

test "model discovery reports fitted engine context equally for its served name and aliases" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const info: api.Info = .{ .context_window = 8192, .context_fitted = true };
    for ([_][]const u8{ "served-model", "model-alias" }) |id| {
        const value = try modelValue(arena.allocator(), id, info, 123);
        try std.testing.expectEqualStrings(id, value.get("id").?.string);
        try std.testing.expectEqual(@as(i64, 8192), value.get("context_length").?.int64().?);
        try std.testing.expectEqual(value.get("context_length").?.int64(), value.get("max_model_len").?.int64());
        try std.testing.expectEqual(@as(i64, 123), value.get("created").?.int64().?);
    }
}

test "model discovery omits an unspecified context and preserves the full u32 limit" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const unlimited = try modelValue(arena.allocator(), "model", .{}, 0);
    try std.testing.expect(unlimited.get("context_length") == null);
    try std.testing.expect(unlimited.get("max_model_len") == null);
    const bounded = try modelValue(arena.allocator(), "model", .{ .context_window = std.math.maxInt(u32) }, 0);
    try std.testing.expectEqual(@as(i64, std.math.maxInt(u32)), bounded.get("max_model_len").?.int64().?);
}

test "stats snapshot exposes counters and omits unknown memory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try statsSnapshot(arena.allocator(), .{ .running = 1, .generation_tokens = 2 }, null, 3, 4, 5, 6, 7);
    const object = value.object;
    for ([_][]const u8{ "running", "waiting", "generation_tokens_running", "prompt_tokens_total", "generation_tokens_total", "mtp_drafted_total", "mtp_accepted_total", "decode_rounds_total", "memory" }) |key|
        try std.testing.expect(object.contains(key));
    try std.testing.expectEqual(@as(usize, 0), object.get("memory").?.object.count());
}

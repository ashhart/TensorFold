//! The /metrics route's open-connections gauge through the production dispatch, in every idle-manager state.
const std = @import("std");
const http = @import("http_conn.zig");
const idle = @import("idle_unload.zig");
const server_mod = @import("server.zig");
const Harness = @import("idle_unload_test.zig").Harness;
const t = std.testing;

/// What the route wrote, as the socket would have read it.
const Wire = struct {
    a: std.mem.Allocator,
    bytes: std.ArrayList(u8) = .empty,

    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

/// One GET /metrics served by the production route table with `open` admitted connections, its body.
fn scrape(srv: *server_mod.Server, a: std.mem.Allocator, open: u32) ![]const u8 {
    var conn = try http.Conn.init(a, -1, "fixture");
    defer conn.deinit();
    conn.method = "GET";
    conn.path = "/metrics";
    conn.version = "HTTP/1.1";
    var wire: Wire = .{ .a = a };
    conn.write_hooks = .{ .ctx = &wire, .write = Wire.write };
    srv.open_connections.store(open, .release); // the count the admission path leaves behind
    srv.dispatch(&conn, a);
    try t.expect(conn.response_complete);
    try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 200 "));
    const head = std.mem.indexOf(u8, wire.bytes.items, "\r\n\r\n") orelse return error.MissingBody;
    return wire.bytes.items[head + 4 ..];
}

/// The gauge exactly once: one sample line carrying `open`, and its exact TYPE line; no HELP prose pinned.
fn expectGauge(body: []const u8, open: u32) !void {
    const sample = "tensorfold:open_connections ";
    var samples: usize = 0;
    var types: usize = 0;
    var value: ?u32 = null;
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, sample)) {
            samples += 1;
            value = try std.fmt.parseInt(u32, line[sample.len..], 10);
        }
        if (std.mem.eql(u8, line, "# TYPE tensorfold:open_connections gauge")) types += 1;
    }
    try t.expectEqual(@as(usize, 1), samples);
    try t.expectEqual(@as(usize, 1), types);
    try t.expectEqual(open, value.?);
}

test "open connections route: no idle manager serves one gauge with the admitted count" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    idle.stopServer(h.srv.?); // the manager stops behind a resident engine, as shutdown does
    h.man = null;
    try t.expectEqual(@as(?*idle.Manager, null), h.srv.?.idle_manager);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const body = try scrape(h.srv.?, arena.allocator(), 7);
    try expectGauge(body, 7);
    // the resident engine's families rendered, not the parked body
    try t.expect(std.mem.indexOf(u8, body, "tensorfold:kv_cache_usage_ratio") != null);
}

test "open connections route: a healthy idle manager serves one gauge with the admitted count" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    try t.expect(h.srv.?.idle_manager != null and !h.man.?.isParked());
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const body = try scrape(h.srv.?, arena.allocator(), 12);
    try expectGauge(body, 12);
    // the resident engine's families rendered, not the parked body
    try t.expect(std.mem.indexOf(u8, body, "tensorfold:kv_cache_usage_ratio") != null);
}

test "open connections route: a parked idle manager serves one gauge with the admitted count" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.expire();
    try h.waitParked(2000);
    try t.expect(h.man.?.isParked());
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const body = try scrape(h.srv.?, arena.allocator(), 23);
    try expectGauge(body, 23);
    // parked: the request and process families only, no engine KV families
    try t.expect(std.mem.indexOf(u8, body, "tensorfold:kv_cache_usage_ratio") == null);
}

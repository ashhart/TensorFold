//! The ledger's reasons in the scrape and /v1/load: a fixed label set, the shared names the client codes use.
const std = @import("std");
const api = @import("engine_api");
const wait = @import("wait_reason.zig");
const load_metrics = @import("load_metrics.zig");
const ledger = api.load_ledger;
const reasons = ledger.reasons;

fn snapshot() ledger.Snapshot {
    return .{ .instance = @splat('a'), .start_s = 1793200000, .parked = false, .lanes = 2, .http = @splat(0) };
}

/// The label values `family` carries in `body`, each line's value checked to be 0.
fn labels(body: []const u8, family: []const u8, out: *std.ArrayList([]const u8)) !void {
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| {
        const head = "tensorfold:";
        if (!std.mem.startsWith(u8, line, head) or !std.mem.startsWith(u8, line[head.len..], family)) continue;
        const rest = line[head.len + family.len ..];
        if (rest.len == 0 or rest[0] != '{') continue; // another family that starts with the same name
        try std.testing.expect(std.mem.startsWith(u8, rest, "{reason=\""));
        const end = std.mem.indexOf(u8, rest, "\"} ").?;
        try std.testing.expectEqualStrings("0", rest[end + 3 ..]);
        try out.append(std.testing.allocator, rest["{reason=\"".len..end]);
    }
}

test "wait reason ledger: a fresh host's scrape shows every wait reason at 0 and no other label" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    var s = snapshot();
    s.requests = .{}; // what a fresh host publishes: nothing queued
    try load_metrics.render(&out.writer, s, .{ .model = "m" });
    var seen: std.ArrayList([]const u8) = .empty;
    defer seen.deinit(std.testing.allocator);
    try labels(out.written(), "requests_waiting_by_reason", &seen);
    try std.testing.expectEqual(std.enums.values(reasons.Wait).len, seen.items.len);
    for (std.enums.values(reasons.Wait), seen.items) |r, label| try std.testing.expectEqualStrings(@tagName(r), label);
    seen.clearRetainingCapacity();
    try labels(out.written(), "request_refusals_total", &seen);
    try std.testing.expectEqual(std.enums.values(reasons.Refusal).len, seen.items.len);
    for (std.enums.values(reasons.Refusal), seen.items) |r, label| try std.testing.expectEqualStrings(@tagName(r), label);
    seen.clearRetainingCapacity();
    try labels(out.written(), "request_failures_total", &seen);
    try std.testing.expectEqual(std.enums.values(reasons.Failure).len, seen.items.len);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "tensorfold:restore_fallbacks_total 0\n") != null);
}

test "wait reason ledger: the capacity codes are the internal reasons' and the refusal counter's names" {
    const shared = [_][]const u8{ wait.engine_busy, wait.engine_unavailable, wait.memory_pressure };
    for (shared, [_]reasons.Wait{ .engine_busy, .engine_unavailable, .memory_pressure }) |code, r| try std.testing.expectEqualStrings(code, @tagName(r));
    for (shared, [_]reasons.Refusal{ .engine_busy, .engine_unavailable, .memory_pressure }) |code, r| try std.testing.expectEqualStrings(code, @tagName(r));
    for (wait.all) |w| try std.testing.expect(std.meta.stringToEnum(reasons.Wait, w.code) == null); // queues nothing
}

test "wait reason ledger: /v1/load names each queued reason and each refusal" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var s = snapshot();
    s.refusals[@backingInt(reasons.Refusal.memory_pressure)] = 2;
    s.requests = .{ .waiting = 3 };
    s.requests.?.waiting_by_reason[@backingInt(reasons.Wait.kv_pool)] = 2;
    s.requests.?.waiting_by_reason[@backingInt(reasons.Wait.memory_floor)] = 1;
    const body = try @import("json").stringify(a, try load_metrics.object(a, s, 1793200001.5, "m"), .{});
    const flat = try std.mem.replaceOwned(u8, a, body, " ", "");
    for ([_][]const u8{ "\"waiting\":3", "\"kv_pool\":2", "\"memory_floor\":1", "\"memory_pressure\":2", "\"invalid_request\":0", "\"waiting_by_reason\":{", "\"refusals\":{" }) |key| {
        if (std.mem.indexOf(u8, flat, key) == null) std.debug.print("missing {s} in {s}\n", .{ key, flat });
        try std.testing.expect(std.mem.indexOf(u8, flat, key) != null);
    }
}

test "wait reason ledger: each admission flag's 429 code is a refusal label of the same name" {
    const admission = @import("request_admission.zig");
    for (std.enums.values(admission.Reason)) |r|
        try std.testing.expectEqualStrings(admission.code(r), @tagName(@import("refusals.zig").codeOf(r)));
}

/// An engine whose every submit is refused with `err`; `Server.init` reads only `info`.
const Refusing = struct {
    err: api.SubmitError,
    fn engine(r: *Refusing) api.Engine {
        return .{ .ctx = r, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        const r: *Refusing = @ptrCast(@alignCast(ctx));
        return r.err;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The route's answer, captured instead of sent to a socket.
const Wire = struct {
    a: std.mem.Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

test "wait reason ledger: a stale device's 500 counts as an engine_stale failure, not a refusal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var refusing: Refusing = .{ .err = error.Stale };
    const srv = try @import("server.zig").Server.init(a, std.testing.io, refusing.engine(), @import("embedding_protocol_test.zig").ByteText.model(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const http = @import("http_conn.zig");
    var conn = try http.Conn.init(a, -1, "fixture");
    defer conn.deinit();
    var wire: Wire = .{ .a = a };
    conn.write_hooks = .{ .ctx = &wire, .write = Wire.write };
    conn.method = "POST";
    conn.path = "/v1/completions";
    conn.version = "HTTP/1.1";
    conn.close = false;
    conn.quiet_log = true;
    const body = "{\"model\":\"m\",\"prompt\":\"x\",\"max_tokens\":1}";
    var length: [8]u8 = undefined;
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.bufPrint(&length, "{d}", .{body.len}) }};
    conn.headers = &headers;
    @memcpy(conn.buf[0..body.len], body);
    conn.end = body.len;
    srv.dispatch(&conn, a);
    try std.testing.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 500"));
    try std.testing.expect(std.mem.indexOf(u8, wire.bytes.items, api.stale_message) != null);
    const s = srv.load.read();
    try std.testing.expectEqual(@as(u64, 1), s.failed(.engine_stale));
    for (s.refusals) |n| try std.testing.expectEqual(@as(u64, 0), n);
}

//! The bounded wait reasons: one code each, and the body every wait answers with.
const std = @import("std");
const wait = @import("wait_reason.zig");
const errors = @import("errors.zig");
const openai = @import("openai.zig");
const json = @import("json");
const api = @import("engine_api");
const cli = @import("cli.zig");
const idle = @import("idle_unload.zig");
const http = @import("http_conn.zig");
const Server = @import("server.zig").Server;
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const t = std.testing;
const Allocator = std.mem.Allocator;

test "wait reasons: one distinct code each, and a body that names it" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 3), wait.all.len);
    try testing.expectEqual(@as(usize, 3), wait.capacity_codes.len);
    const codes = [_][]const u8{
        wait.restarting.code, wait.loading.code,       wait.reload_failed.code,
        wait.engine_busy,     wait.engine_unavailable, wait.memory_pressure,
    };
    for (codes, 0..) |code, i| {
        try testing.expect(code.len > 0);
        for (codes[i + 1 ..]) |other| try testing.expect(!std.mem.eql(u8, code, other));
    }
    for (wait.all) |w| {
        var buf: [512]u8 = undefined;
        const body = wait.body(&buf, w);
        try testing.expect(std.mem.indexOf(u8, body, w.code) != null);
        try testing.expect(std.mem.indexOf(u8, body, w.message) != null);
        const anthropic = wait.anthropicBody(&buf, w);
        try testing.expect(std.mem.indexOf(u8, anthropic, w.message) != null);
        try testing.expect(std.mem.indexOf(u8, anthropic, w.code) == null);
    }
    // the drain's body bytes are frozen: a client already keys on them
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        "{\"error\": {\"message\": \"server restarting: retry shortly\", \"type\": \"server_error\", \"param\": null, \"code\": \"server_restarting\"}}",
        wait.body(&buf, wait.restarting),
    );
    try testing.expectEqualStrings(
        "{\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\", \"message\": \"server restarting: retry shortly\"}}",
        wait.anthropicBody(&buf, wait.restarting),
    );
}

test "wait reasons: an idle reload's problem names loading or a failed reload" {
    try std.testing.expectEqualStrings(wait.loading.code, wait.reloadReason(wait.unloading_problem).code);
    try std.testing.expectEqualStrings(wait.loading.code, wait.reloadReason("the engine is still unloading; send the request again").code);
    try std.testing.expectEqualStrings(wait.reload_failed.code, wait.reloadReason("the engine could not reload (bad shard); send the request again").code);
    try std.testing.expectEqualStrings(wait.reload_failed.code, wait.reloadReason("").code);
}

test "wait reasons: a capacity refusal answers with its code beside its message" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const refused: errors.Refused!void = cx.wait(wait.engine_busy, "the engine is busy; retry shortly", .{});
    try std.testing.expectError(error.Refused, refused);
    try std.testing.expectEqualStrings(wait.engine_busy, cx.code.?);
    const body = try openai.errorBody(a, &cx, null);
    try std.testing.expectEqualStrings(
        "{\"message\": \"the engine is busy; retry shortly\", \"type\": \"invalid_request_error\", \"code\": \"engine_busy\"}",
        try json.stringify(a, body, .{}),
    );
    var plain: errors.Cx = .{ .a = a };
    const nope: errors.Refused!void = plain.refuse("nope");
    try std.testing.expectError(error.Refused, nope);
    try std.testing.expect(plain.code == null);
    const control = try json.stringify(a, try openai.errorBody(a, &plain, null), .{});
    try std.testing.expect(std.mem.indexOf(u8, control, "code") == null);
}

/// A scripted engine the 503 paths never submit to; `Server.init` reads only `info`.
const Parked = struct {
    fn engine() api.Engine {
        return .{ .ctx = undefined, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.InvalidRequest;
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
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

/// A real `idle_unload.Reopen` whose reload fails the way a missing checkpoint shard does.
const Broken = struct {
    attempts: usize = 0,
    const missing = "the checkpoint shard is missing";
    fn reopen(b: *@This()) idle.Reopen {
        return .{ .ctx = b, .open = open, .close_held = closeHeld, .replace = replace };
    }
    fn open(ctx: *anyopaque, _: Allocator, _: Allocator, _: std.Io, _: []const u8, _: []const u8, _: cli.Args, problem: *[]const u8) anyerror!?api.Opened {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.attempts += 1;
        problem.* = missing;
        return null;
    }
    fn closeHeld(_: *anyopaque) void {}
    fn replace(_: *anyopaque, _: api.Opened, _: *std.heap.ArenaAllocator) void {}
};

/// A server on the scripted engine whose idle manager is `manager`.
fn parkedServer(a: Allocator, manager: *idle.Manager) !*Server {
    const srv = try Server.init(a, t.io, Parked.engine(), ByteText.model(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    srv.idle_manager = manager;
    return srv;
}

/// One POST through the real route, returning the bytes it answered with.
fn post(a: Allocator, srv: *Server, path: []const u8) ![]const u8 {
    var conn = try http.Conn.init(a, -1, "fixture");
    defer conn.deinit();
    var wire: Wire = .{ .a = a };
    conn.write_hooks = .{ .ctx = &wire, .write = Wire.write };
    conn.method = "POST";
    conn.path = path;
    conn.version = "HTTP/1.1";
    conn.close = false;
    conn.quiet_log = true;
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = "2" }};
    conn.headers = &headers;
    @memcpy(conn.buf[0..2], "{}");
    conn.end = 2;
    srv.dispatch(&conn, a);
    return a.dupe(u8, wire.bytes.items);
}

test "wait reasons HTTP: a model still unloading answers 503 engine_loading, not server_restarting" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var manager: idle.Manager = .{
        .gpa = a,
        .io = t.io,
        .args = .{},
        .re = undefined,
        .dir = "m",
        .model_type = "fake",
        .idle_s = 60,
        .parked = true,
        .state = .unloading,
    };
    const srv = try parkedServer(a, &manager);
    defer srv.deinit(); // the manager outlives the server: clear it first, so deinit never sees a stack pointer
    defer srv.idle_manager = null;
    const answer = try post(a, srv, "/v1/completions");
    try t.expect(std.mem.startsWith(u8, answer, "HTTP/1.1 503"));
    try t.expect(std.mem.indexOf(u8, answer, "Retry-After: 30") != null);
    try t.expect(std.mem.indexOf(u8, answer, "\"code\": \"engine_loading\"") != null);
    try t.expect(std.mem.indexOf(u8, answer, "\"message\": \"the model is loading; retry shortly\"") != null);
    try t.expect(std.mem.indexOf(u8, answer, "server_restarting") == null);
    try t.expectEqual(@as(u32, 0), manager.active);
    try t.expectEqual(idle.State.unloading, manager.state);
}

test "wait reasons HTTP: a failed reload answers 503 engine_reload_failed and leaves the model unloaded" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var broken: Broken = .{};
    var manager: idle.Manager = .{
        .gpa = a,
        .io = t.io,
        .args = .{},
        .re = broken.reopen(),
        .dir = "m",
        .model_type = "fake",
        .idle_s = 60,
        .parked = true,
        .state = .unloaded,
    };
    const srv = try parkedServer(a, &manager);
    defer srv.deinit(); // the manager outlives the server: clear it first, so deinit never sees a stack pointer
    defer srv.idle_manager = null;
    for (0..2) |round| {
        const answer = try post(a, srv, "/v1/completions");
        try t.expect(std.mem.startsWith(u8, answer, "HTTP/1.1 503"));
        try t.expect(std.mem.indexOf(u8, answer, "Retry-After: 30") != null);
        try t.expect(std.mem.indexOf(u8, answer, "\"code\": \"engine_reload_failed\"") != null);
        try t.expect(std.mem.indexOf(u8, answer, "server_restarting") == null);
        try t.expectEqual(@as(u32, 0), manager.active);
        try t.expectEqual(idle.State.unloaded, manager.state);
        try t.expectEqual(round + 1, broken.attempts);
    }
}

test "wait reasons: HTTP capacity codes equal load and scrape codes" {
    const reasons = api.wait_reason;
    try t.expectEqualStrings(wait.engine_busy, reasons.waitCode(.engine_busy));
    try t.expectEqualStrings(wait.engine_busy, reasons.refusalCode(.engine_busy));
    try t.expectEqualStrings(wait.engine_unavailable, reasons.waitCode(.engine_unavailable));
    try t.expectEqualStrings(wait.engine_unavailable, reasons.refusalCode(.engine_unavailable));
    try t.expectEqualStrings(wait.memory_pressure, reasons.waitCode(.memory_pressure));
    try t.expectEqualStrings(wait.memory_pressure, reasons.refusalCode(.memory_pressure));
}

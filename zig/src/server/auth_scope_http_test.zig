//! Real sockets: scope refusals precede the body and engine work, and stored Responses stay with full keys.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const server = @import("server.zig");
const auth = @import("auth.zig");
const idle = @import("idle_unload.zig");
const listener = @import("listener.zig");
const model_text = @import("model_text.zig");
const KeyFile = @import("auth_scope_test.zig").KeyFile;
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const posix = @import("sys").posix;
const t = std.testing;
const Allocator = std.mem.Allocator;

const key_text = "app[inference]: infer-secret\nmonitor[read]: read-secret\nboth[inference,read]: both-secret\n" ++
    "ops[admin]: admin-secret\nlegacy: legacy-secret\n";
const restricted = [_][]const u8{ "infer-secret", "both-secret", "read-secret" };
const full_keys = [_][]const u8{ "admin-secret", "legacy-secret" };

const Text = struct {
    const table = blk: {
        var base = ByteText.model().vtable.*;
        base.render = render;
        base.decode = decode;
        break :blk base;
    };
    fn model() model_text.Text {
        return .{ .ctx = undefined, .vtable = &table };
    }
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) ![]u8 {
        const bytes = try a.alloc(u8, ids.len);
        for (bytes, ids) |*out, id| out.* = @intCast(id);
        return bytes;
    }
};

const Engine = struct {
    calls: std.atomic.Value(u32) = .init(0),
    reads: std.atomic.Value(u32) = .init(0),
    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn cast(ptr: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ptr));
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ptr: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        _ = cast(ptr).calls.fetchAdd(1, .monotonic);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{'x'} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(ptr: *anyopaque, out: *api.Status, _: []u32) void {
        _ = cast(ptr).reads.fetchAdd(1, .monotonic);
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Harness = struct {
    srv: *server.Server = undefined,
    listener: listener.Listener = undefined,
    accept: std.Thread = undefined,
    stop: std.atomic.Value(bool) = .init(false),

    fn start(h: *Harness, engine: *Engine, keys: *auth.Store) !void {
        h.srv = try server.Server.init(t.allocator, t.io, engine.engine(), Text.model(), .{
            .served_name = "m",
            .model_ids = &.{"m"},
            .enable_thinking = false,
            .use_drafts = false,
            .keep_warm_s = 0,
            .dashboard = true,
        }, keys);
        errdefer h.srv.deinit();
        h.listener = try listener.Listener.open(.{ .ip4 = .loopback(0) });
        errdefer h.listener.close();
        h.accept = try std.Thread.spawn(.{}, server.Server.serve, .{ h.srv, h.listener, &h.stop });
    }
    fn deinit(h: *Harness) void {
        h.stop.store(true, .release);
        h.accept.join();
        var waited: usize = 0;
        while (h.srv.open_connections.load(.acquire) > 0) : (waited += 1) {
            if (waited == 5000) @panic("scope fixture did not quiesce");
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        h.listener.close();
        h.srv.deinit();
    }
    fn exchange(h: *Harness, a: Allocator, raw: []const u8) ![]const u8 {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.listener.port()) };
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&address, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.Socket;
        const fd: posix.socket_t = @intCast(rc);
        defer _ = posix.system.close(fd);
        if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.Connect;
        var sent: usize = 0;
        while (sent < raw.len) {
            const n = posix.system.write(fd, raw[sent..].ptr, raw.len - sent);
            if (posix.errno(n) != .SUCCESS or n == 0) return error.Write;
            sent += @intCast(n);
        }
        var out: std.ArrayList(u8) = .empty;
        var buffer: [4096]u8 = undefined;
        while (true) {
            var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
            if (try posix.poll(&fds, 5000) == 0) return error.Timeout;
            const n = posix.system.read(fd, &buffer, buffer.len);
            if (posix.errno(n) != .SUCCESS) return error.Read;
            if (n == 0) return out.items;
            try out.appendSlice(a, buffer[0..@intCast(n)]);
        }
    }
    fn ask(h: *Harness, a: Allocator, method: []const u8, path: []const u8, key: []const u8, body: []const u8) ![]const u8 {
        const header = if (key.len > 0) try std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{key}) else "";
        const raw = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: localhost\r\n{s}Content-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ method, path, header, body.len, body });
        return h.exchange(a, raw);
    }
};

fn expectStatus(reply: []const u8, code: u16) !void {
    var prefix: [32]u8 = undefined;
    const expected = try std.fmt.bufPrint(&prefix, "HTTP/1.1 {d} ", .{code});
    if (!std.mem.startsWith(u8, reply, expected)) {
        std.debug.print("expected {d}, got: {s}\n", .{ code, reply[0..@min(reply.len, 200)] });
        return error.UnexpectedStatus;
    }
}

fn payload(a: Allocator, reply: []const u8) !json.Value {
    const split = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.NoBody;
    return switch (try json.parse(a, reply[split + 4 ..])) {
        .ok => |v| v,
        .err => error.InvalidJson,
    };
}

fn start(file: *KeyFile, keys: *auth.Store, engine: *Engine, h: *Harness) !void {
    var problem: []const u8 = "";
    keys.* = try auth.Store.init(t.allocator, &.{}, "", file.path, false, &problem);
    errdefer keys.deinit();
    try h.start(engine, keys);
}

test "auth scopes HTTP: refusals come before the body, a 100 Continue or a parked model's wake" {
    var file = try KeyFile.init(key_text);
    defer file.deinit();
    var keys: auth.Store = undefined;
    var engine: Engine = .{};
    var h: Harness = .{};
    try start(&file, &keys, &engine, &h);
    defer keys.deinit();
    defer h.deinit();
    var manager: idle.Manager = .{
        .gpa = t.allocator,
        .io = t.io,
        .args = .{},
        .re = undefined,
        .dir = "private-model",
        .model_type = "private-family",
        .idle_s = 60,
        .parked = true,
        .state = .unloaded,
        .last_activity_ns = 77,
    };
    h.srv.idle_manager = &manager;
    defer h.srv.idle_manager = null;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const private_posts = [_][]const u8{ "/v1/slide/learn", "/v1/imprint/load", "/v1/imprint/resume", "/x/completions", "/future/private" };
    for (restricted) |key| {
        for (private_posts) |path| {
            const raw = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {s}\r\nContent-Length: 100000\r\nExpect: 100-continue\r\n\r\n", .{ path, key });
            const denied = try h.exchange(a, raw);
            try expectStatus(denied, 403);
            try t.expect(std.mem.indexOf(u8, denied, "100 Continue") == null);
            try t.expect(std.mem.indexOf(u8, denied, "\"code\": \"insufficient_scope\"") != null);
            try t.expect(std.mem.indexOf(u8, denied, "Connection: close\r\n") != null);
            try t.expect(std.mem.indexOf(u8, denied, key) == null);
            try t.expect(std.mem.indexOf(u8, denied, "WWW-Authenticate") == null);
        }
        for ([_][]const u8{ "/v1/recall", "/v1/imprint", "/v1/slide/graph", "/v1/slide/fact/1", "/v1/responses/id" }) |path|
            try expectStatus(try h.ask(a, "DELETE", path, key, ""), 403);
    }
    const missing = try h.ask(a, "POST", "/v1/completions", "unknown-secret", "{}");
    try expectStatus(missing, 401);
    try t.expect(std.mem.indexOf(u8, missing, "WWW-Authenticate: Bearer") != null);
    const anthropic = try h.ask(a, "POST", "/v1/messages", "read-secret", "{}");
    try expectStatus(anthropic, 403);
    const err = (try payload(a, anthropic)).get("error").?;
    try t.expectEqualStrings("permission_error", err.strField("type").?);
    try t.expectEqual(@as(u32, 0), engine.calls.load(.acquire));
    try t.expectEqual(@as(u32, 0), engine.reads.load(.acquire));
    try t.expectEqual(idle.State.unloaded, manager.state);
    try t.expectEqual(@as(i64, 77), manager.last_activity_ns);
    try t.expectEqual(@as(u32, 0), manager.active);
    try t.expectEqual(@as(u32, 0), h.srv.generating.load(.acquire));
}

test "auth scopes HTTP: inference and read are separate grants and full keys keep the whole dispatcher" {
    var file = try KeyFile.init(key_text);
    defer file.deinit();
    var keys: auth.Store = undefined;
    var engine: Engine = .{};
    var h: Harness = .{};
    try start(&file, &keys, &engine, &h);
    defer keys.deinit();
    defer h.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "/models", "/v1/models", "/v1/models/?x=1" }) |path| for ([_][]const u8{ "read-secret", "infer-secret" }) |key|
        try expectStatus(try h.ask(a, "GET", path, key, ""), 200);
    for ([_][]const u8{ "/metrics", "/v1/metrics", "/memory", "/v1/memory", "/stats", "/dashboard" }) |path| {
        try expectStatus(try h.ask(a, "GET", path, "read-secret", ""), 200);
        try expectStatus(try h.ask(a, "GET", path, "infer-secret", ""), 403);
    }
    const body = "{\"model\":\"m\",\"prompt\":\"hello\",\"max_tokens\":1,\"temperature\":0}";
    try expectStatus(try h.ask(a, "POST", "/v1/completions", "read-secret", body), 403);
    for ([_][]const u8{ "infer-secret", "both-secret", "admin-secret", "legacy-secret" }) |key|
        try expectStatus(try h.ask(a, "POST", "/v1/completions", key, body), 200);
    try expectStatus(try h.ask(a, "POST", "/v1/completions///?x=1", "infer-secret", body), 200);
    try t.expectEqual(@as(u32, 5), engine.calls.load(.acquire));
    // a suffix alias the dispatcher serves for a full key is not a grant
    try expectStatus(try h.ask(a, "POST", "/x/completions", "both-secret", body), 403);
    for (full_keys) |key| try expectStatus(try h.ask(a, "POST", "/x/completions", key, body), 200);
    try t.expectEqual(@as(u32, 7), engine.calls.load(.acquire));
    const chat = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":true}";
    const streamed = try h.ask(a, "POST", "/v1/chat/completions", "infer-secret", chat);
    try expectStatus(streamed, 200);
    try t.expect(std.mem.indexOf(u8, streamed, "text/event-stream") != null);
    try t.expect(std.mem.indexOf(u8, streamed, "data: [DONE]") != null);
    try expectStatus(try h.ask(a, "POST", "/v1/chat/completions", "read-secret", chat), 403);
    try t.expectEqual(@as(u32, 8), engine.calls.load(.acquire));
    try expectStatus(try h.ask(a, "GET", "/future/private", "both-secret", ""), 403);
    for (full_keys) |key| try expectStatus(try h.ask(a, "GET", "/future/private", key, ""), 404);
    try expectStatus(try h.ask(a, "HEAD", "/v1/models", "read-secret", ""), 403);
    for (full_keys) |key| try expectStatus(try h.ask(a, "HEAD", "/v1/models", key, ""), 501);
    try expectStatus(try h.ask(a, "OPTIONS", "/v1/slide/learn", "both-secret", ""), 501);
    try expectStatus(try h.ask(a, "POST", "/v1/tokenize", "infer-secret", "{\"prompt\":\"hello\"}"), 200);
    try expectStatus(try h.ask(a, "POST", "/v1/tokenize", "read-secret", "{\"prompt\":\"hello\"}"), 403);
    try expectStatus(try h.ask(a, "GET", "/health", "read-secret", ""), 200);
}

test "auth scopes HTTP: restricted Responses cannot read, keep or continue the shared stored conversations" {
    var file = try KeyFile.init(key_text);
    defer file.deinit();
    var keys: auth.Store = undefined;
    var engine: Engine = .{};
    var h: Harness = .{};
    try start(&file, &keys, &engine, &h);
    defer keys.deinit();
    defer h.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const created = try h.ask(a, "POST", "/v1/responses", "admin-secret", "{\"input\":\"private history\",\"max_output_tokens\":1}");
    try expectStatus(created, 200);
    const id = (try payload(a, created)).strField("id").?;
    const path = try std.fmt.allocPrint(a, "/v1/responses/{s}", .{id});
    const before = engine.calls.load(.acquire);
    for (restricted) |key| {
        try expectStatus(try h.ask(a, "GET", path, key, ""), 403);
        try expectStatus(try h.ask(a, "DELETE", path, key, ""), 403);
    }
    const stored = [_][]const u8{ "{\"input\":\"x\"}", "{\"input\":\"x\",\"store\":true}", "{\"input\":\"x\",\"store\":null}", "{\"input\":\"x\",\"store\":0}" };
    for (stored) |body| {
        const denied = try h.ask(a, "POST", "/v1/responses", "infer-secret", body);
        try expectStatus(denied, 403);
        try t.expectEqualStrings("store", (try payload(a, denied)).get("error").?.strField("param").?);
    }
    for ([_][]const u8{ id, "missing-id" }) |previous| {
        const body = try std.fmt.allocPrint(a, "{{\"input\":\"x\",\"store\":false,\"previous_response_id\":\"{s}\"}}", .{previous});
        const denied = try h.ask(a, "POST", "/responses", "both-secret", body);
        try expectStatus(denied, 403);
        try t.expectEqualStrings("previous_response_id", (try payload(a, denied)).get("error").?.strField("param").?);
        try t.expect(std.mem.indexOf(u8, denied, "private history") == null);
    }
    try t.expectEqual(before, engine.calls.load(.acquire));
    const stateless = try h.ask(a, "POST", "/v1/responses", "infer-secret", "{\"input\":\"x\",\"store\":false,\"previous_response_id\":null,\"max_output_tokens\":1}");
    try expectStatus(stateless, 200);
    const output = try payload(a, stateless);
    try t.expectEqual(false, output.get("store").?.bool);
    const absent = try std.fmt.allocPrint(a, "/v1/responses/{s}", .{output.strField("id").?});
    try expectStatus(try h.ask(a, "GET", absent, "admin-secret", ""), 404);
    try expectStatus(try h.ask(a, "GET", path, "admin-secret", ""), 200);
    const continued = try std.fmt.allocPrint(a, "{{\"input\":\"y\",\"previous_response_id\":\"{s}\",\"max_output_tokens\":1}}", .{id});
    try expectStatus(try h.ask(a, "POST", "/v1/responses", "legacy-secret", continued), 200);
    try expectStatus(try h.ask(a, "DELETE", path, "legacy-secret", ""), 200);
}

test "auth scopes HTTP: a keep-alive request does not inherit the previous request's full key" {
    var file = try KeyFile.init(key_text);
    defer file.deinit();
    var keys: auth.Store = undefined;
    var engine: Engine = .{};
    var h: Harness = .{};
    try start(&file, &keys, &engine, &h);
    defer keys.deinit();
    defer h.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const reply = try h.exchange(arena.allocator(), "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer admin-secret\r\n\r\n" ++
        "DELETE /v1/responses/unknown HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer infer-secret\r\nConnection: close\r\n\r\n");
    try t.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 "));
    try t.expect(std.mem.indexOf(u8, reply, "HTTP/1.1 403 ") != null);
    try t.expect(std.mem.indexOf(u8, reply, "HTTP/1.1 404 ") == null);
    const unkeyed = try h.exchange(arena.allocator(), "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer admin-secret\r\n\r\n" ++
        "DELETE /v1/responses/unknown HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    try t.expect(std.mem.indexOf(u8, unkeyed, "HTTP/1.1 401 ") != null);
}

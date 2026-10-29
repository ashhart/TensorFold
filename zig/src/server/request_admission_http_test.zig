//! Admission over real sockets: refusals before the body, verified keys, cancellation, and an inert default.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const auth = @import("auth.zig");
const admission = @import("request_admission.zig");
const listener = @import("listener.zig");
const posix = @import("sys").posix;
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const t = std.testing;
const Allocator = std.mem.Allocator;

const Engine = struct {
    calls: std.atomic.Value(u32) = .init(0),
    active: std.atomic.Value(u32) = .init(0),
    hold: std.atomic.Value(bool) = .init(false),
    cancelled: std.atomic.Value(api.Id) = .init(0),

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
        const e = cast(ptr);
        _ = e.calls.fetchAdd(1, .monotonic);
        if (!e.hold.load(.acquire)) return finish(id, sink);
        _ = e.active.fetchAdd(1, .release);
        const thread = std.Thread.spawn(.{}, wait, .{ e, id, sink }) catch {
            _ = e.active.fetchSub(1, .release);
            return error.Busy;
        };
        thread.detach();
    }
    fn wait(e: *Engine, id: api.Id, sink: api.Sink) void {
        defer _ = e.active.fetchSub(1, .release);
        while (e.hold.load(.acquire)) {
            if (e.cancelled.load(.acquire) == id)
                return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        finish(id, sink);
    }
    fn finish(id: api.Id, sink: api.Sink) void {
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{'x'} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(ptr: *anyopaque, id: api.Id) void {
        cast(ptr).cancelled.store(id, .release);
    }
    fn status(ptr: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{ .running = cast(ptr).active.load(.acquire) };
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Harness = struct {
    srv: *server.Server = undefined,
    engine: *Engine = undefined,
    listener: listener.Listener = undefined,
    accept: std.Thread = undefined,
    stop: std.atomic.Value(bool) = .init(false),

    fn start(h: *Harness, engine: *Engine, keys: ?*auth.Store, limits: admission.Limits) !void {
        h.engine = engine;
        h.srv = try server.Server.init(t.allocator, t.io, engine.engine(), ByteText.model(), .{
            .served_name = "m",
            .model_ids = &.{"m"},
            .enable_thinking = false,
            .use_drafts = false,
            .keep_warm_s = 0,
            .admission = limits,
        }, keys);
        errdefer h.srv.deinit();
        h.listener = try listener.Listener.open(.{ .ip4 = .loopback(0) });
        errdefer h.listener.close();
        h.accept = try std.Thread.spawn(.{}, server.Server.serve, .{ h.srv, h.listener, &h.stop });
    }
    fn deinit(h: *Harness) void {
        h.engine.hold.store(false, .release);
        h.stop.store(true, .release);
        h.accept.join();
        var waited: usize = 0;
        while (h.srv.open_connections.load(.acquire) > 0 or h.engine.active.load(.acquire) > 0) : (waited += 1) {
            if (waited == 5000) @panic("admission fixture did not quiesce");
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        h.listener.close();
        h.srv.deinit();
    }
    fn begin(h: *Harness, raw_request: []const u8) !posix.socket_t {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.listener.port()) };
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&address, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.Socket;
        const fd: posix.socket_t = @intCast(rc);
        errdefer _ = posix.system.close(fd);
        if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.Connect;
        var sent: usize = 0;
        while (sent < raw_request.len) {
            const n = posix.system.write(fd, raw_request[sent..].ptr, raw_request.len - sent);
            if (posix.errno(n) != .SUCCESS or n == 0) return error.Write;
            sent += @intCast(n);
        }
        return fd;
    }
    fn request(h: *Harness, a: Allocator, raw_request: []const u8) ![]const u8 {
        const fd = try h.begin(raw_request);
        defer _ = posix.system.close(fd);
        return receive(a, fd);
    }
};

fn receive(a: Allocator, fd: posix.socket_t) ![]const u8 {
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

fn requestText(a: Allocator, path: []const u8, credential: []const u8, body: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: localhost\r\n{s}Content-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ path, credential, body.len, body });
}

fn expectStatus(reply: []const u8, status: []const u8) !void {
    if (!std.mem.startsWith(u8, reply, status)) {
        std.debug.print("expected {s}, got: {s}\n", .{ status, reply[0..@min(reply.len, 200)] });
        return error.UnexpectedStatus;
    }
}

fn contains(reply: []const u8, part: []const u8) bool {
    return std.mem.indexOf(u8, reply, part) != null;
}

const completion = "{\"model\":\"m\",\"prompt\":\"x\",\"max_tokens\":1,\"temperature\":0}";

test "admission HTTP: every model route refuses before its body and engine work; other routes stay open" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null, .{ .max_inflight = 1 });
    defer h.deinit();
    var lease = h.srv.admission.acquire(t.io, .{ .max_inflight = 1 }, null).accepted;
    defer lease.release();
    const paths = [_][]const u8{
        "/v1/completions",  "/v1/chat/completions", "/v1/responses",   "/v1/messages",     "/messages",
        "/v1/audio/speech", "/v1/embeddings",       "/v1/slide/learn", "/v1/imprint/load", "/v1/imprint/resume",
        "/v1/decisions",    "/v1/systemone",
    };
    for (paths) |path| {
        const req = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100000\r\n\r\n", .{path});
        const reply = try h.request(a, req);
        try expectStatus(reply, "HTTP/1.1 429 Too Many Requests\r\n");
        try t.expect(contains(reply, "Retry-After: 1\r\n") and contains(reply, "Connection: close\r\n"));
        if (std.mem.indexOf(u8, path, "messages") != null) {
            try t.expect(contains(reply, "{\"type\": \"error\", \"error\": {\"type\": \"rate_limit_error\", \"message\": "));
        } else try t.expect(contains(reply, "\"type\": \"rate_limit_error\", \"param\": null, \"code\": \"max_inflight_exceeded\"}}"));
    }
    for ([_][]const u8{ "/health", "/v1/models", "/metrics" }) |path| {
        const req = try std.fmt.allocPrint(a, "GET {s} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n", .{path});
        try expectStatus(try h.request(a, req), "HTTP/1.1 200 ");
    }
    try expectStatus(try h.request(a, try requestText(a, "/v1/tokenize", "", "{\"prompt\":\"x\"}")), "HTTP/1.1 200 ");
    try t.expectEqual(@as(u32, 0), engine.calls.load(.acquire));
    try t.expectEqual(@as(u32, 1), h.srv.admission.snapshot(t.io).inflight);
    try t.expectEqual(@as(u64, paths.len), h.srv.admission.snapshot(t.io).rejected_global);
    try t.expectEqual(@as(u32, 0), h.srv.generating.load(.acquire));
}

test "admission HTTP: verified keys get their own reservations and a parse error releases one" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(t.allocator, &.{ "first", "second" }, "", null, false, &problem);
    defer keys.deinit();
    const first = keys.matchIdentity(a, "Bearer first", null).?;
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, &keys, .{ .max_inflight_per_key = 1 });
    defer h.deinit();
    var lease = h.srv.admission.acquire(t.io, .{ .max_inflight_per_key = 1 }, first.digest).accepted;
    defer lease.release();
    const refused = try h.request(a, try requestText(a, "/v1/completions", "Authorization: Bearer first\r\n", completion));
    try expectStatus(refused, "HTTP/1.1 429 ");
    try t.expect(contains(refused, "max_inflight_per_key_exceeded"));
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "x-api-key: second\r\n", completion)), "HTTP/1.1 200 ");
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "x-api-key: second\r\n", "{")), "HTTP/1.1 400 ");
    try t.expectEqual(@as(u32, 1), h.srv.admission.snapshot(t.io).inflight);
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "x-api-key: unknown\r\n", completion)), "HTTP/1.1 401 ");
    try t.expectEqual(@as(u32, 1), engine.calls.load(.acquire));
    lease.release();
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "Authorization: Bearer first\r\n", completion)), "HTTP/1.1 200 ");
    try t.expectEqual(@as(u32, 0), h.srv.admission.snapshot(t.io).inflight);
}

test "admission HTTP: header values cannot split an unauthenticated server's shared principal" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null, .{ .max_inflight_per_key = 1 });
    defer h.deinit();
    var lease = h.srv.admission.acquire(t.io, .{ .max_inflight_per_key = 1 }, null).accepted;
    defer lease.release();
    for ([_][]const u8{ "", "x-api-key: arbitrary\r\n", "Authorization: Bearer another\r\n" }) |credential| {
        try expectStatus(try h.request(a, try requestText(a, "/v1/completions", credential, completion)), "HTTP/1.1 429 ");
    }
    try t.expectEqual(@as(u32, 1), h.srv.admission.snapshot(t.io).principals);
}

test "admission HTTP: a client that leaves frees its reservation only when the model request ends" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    engine.hold.store(true, .release);
    var h: Harness = .{};
    try h.start(&engine, null, .{ .max_inflight = 1 });
    defer h.deinit();
    const fd = try h.begin(try requestText(a, "/v1/completions", "", completion));
    var closed = false;
    defer if (!closed) {
        _ = posix.system.close(fd);
    };
    var waited: usize = 0;
    while (engine.active.load(.acquire) == 0) : (waited += 1) {
        if (waited == 5000) return error.SubmitTimeout;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "", completion)), "HTTP/1.1 429 ");
    _ = posix.system.close(fd);
    closed = true;
    waited = 0;
    while (h.srv.admission.snapshot(t.io).inflight != 0) : (waited += 1) {
        if (waited == 5000) return error.CancelTimeout;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    try t.expect(engine.cancelled.load(.acquire) != 0);
    engine.hold.store(false, .release);
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "", completion)), "HTTP/1.1 200 ");
    try t.expectEqual(@as(u32, 0), h.srv.admission.snapshot(t.io).inflight);
}

test "admission HTTP: a spent rate refuses after the request ends and the metrics carry fixed reasons" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(t.allocator, &.{ "first", "second" }, "", null, false, &problem);
    defer keys.deinit();
    const first = keys.matchIdentity(a, "Bearer first", null).?;
    var engine: Engine = .{};
    var h: Harness = .{};
    const limits: admission.Limits = .{ .requests_per_second = 1, .request_burst = 1 };
    try h.start(&engine, &keys, limits);
    defer h.deinit();
    const held_time = @import("clock.zig").nowNs(t.io) + 60 * std.time.ns_per_s;
    var lease = h.srv.admission.acquireAt(t.io, limits, first.digest, held_time).accepted;
    lease.release();
    for ([_][]const u8{ "/v1/completions", "/v1/messages" }) |path| {
        const reply = try h.request(a, try requestText(a, path, "Authorization: Bearer first\r\n", completion));
        try expectStatus(reply, "HTTP/1.1 429 ");
        try t.expect(contains(reply, "Retry-After: 1\r\n") and contains(reply, "rate_limit_error"));
        if (std.mem.endsWith(u8, path, "completions")) try t.expect(contains(reply, "requests_per_second_exceeded"));
    }
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "x-api-key: second\r\n", completion)), "HTTP/1.1 200 ");
    try t.expectEqual(@as(u32, 1), engine.calls.load(.acquire));
    try t.expectEqual(@as(u64, 2), h.srv.admission.snapshot(t.io).rejected_rate);
    const scraped = try h.request(a, "GET /metrics HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer second\r\nConnection: close\r\n\r\n");
    try expectStatus(scraped, "HTTP/1.1 200 ");
    try t.expect(contains(scraped, "\ntensorfold:admission_rejected_total{reason=\"rate_limit\"} 2\n"));
    try t.expect(contains(scraped, "\ntensorfold:admission_accepted_total 2\n"));
    var lines = std.mem.splitScalar(u8, scraped, '\n');
    while (lines.next()) |line| if (std.mem.startsWith(u8, line, "tensorfold:admission_") and contains(line, "{"))
        try t.expect(std.mem.startsWith(u8, line, "tensorfold:admission_rejected_total{reason=\"")); // no key, label or digest
}

test "admission HTTP: with no admission flag nothing is reserved, refused or added to the metrics" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    engine.hold.store(true, .release);
    var h: Harness = .{};
    try h.start(&engine, null, .{});
    defer h.deinit();
    var held: [40]posix.socket_t = undefined;
    var opened: usize = 0;
    defer for (held[0..opened]) |fd| {
        _ = posix.system.close(fd);
    };
    for (&held) |*fd| {
        fd.* = try h.begin(try requestText(a, "/v1/completions", "", completion));
        opened += 1;
    }
    var waited: usize = 0;
    while (engine.active.load(.acquire) < opened) : (waited += 1) {
        if (waited == 10000) return error.SubmitTimeout;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    const scraped = try h.request(a, "GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    try expectStatus(scraped, "HTTP/1.1 200 ");
    try t.expect(!contains(scraped, "tensorfold:admission_"));
    const snap = h.srv.admission.snapshot(t.io);
    try t.expectEqual(@as(u32, 0), snap.inflight);
    try t.expectEqual(@as(u64, 0), snap.admitted);
    engine.hold.store(false, .release);
    for (held[0..opened]) |fd| try expectStatus(try receive(a, fd), "HTTP/1.1 200 ");
}

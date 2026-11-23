//! Admission over real sockets: refusals before the body, verified keys, cancellation, and an inert default.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const auth = @import("auth.zig");
const admission = @import("request_admission.zig");
const cancel_ids = @import("cancel_ids.zig");
const listener = @import("listener.zig");
const posix = @import("sys").posix;
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const t = std.testing;
const Allocator = std.mem.Allocator;

const Engine = struct {
    calls: std.atomic.Value(u32) = .init(0),
    active: std.atomic.Value(u32) = .init(0),
    hold: std.atomic.Value(bool) = .init(false),
    // how many engine.cancel calls landed, and their ids (multi-choice requests cancel every submission).
    cancelled: std.atomic.Value(u32) = .init(0),
    cancelled_ids: [64]std.atomic.Value(api.Id) = @splat(.init(0)),
    // Submits past this count end .failed instead of .stop; maxInt means never.
    fail_after: std.atomic.Value(u32) = .init(std.math.maxInt(u32)),
    // A per-call stall so a client disconnect lands while the request runs.
    delay_ms: std.atomic.Value(u32) = .init(0),
    logprobs: bool = false,

    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn cast(ptr: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ptr));
    }
    fn info(ptr: *anyopaque) api.Info {
        return .{ .context_window = 4096, .logprobs = cast(ptr).logprobs };
    }
    fn submit(ptr: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = cast(ptr);
        if (e.calls.fetchAdd(1, .monotonic) >= e.fail_after.load(.monotonic)) {
            sink.event(sink.ctx, id, &.{ .prefilled = 0 });
            sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .failed, .message = "synthetic choice failure" } });
            return;
        }
        const delay = e.delay_ms.load(.monotonic);
        if (delay != 0) std.Io.sleep(t.io, .fromMilliseconds(delay), .awake) catch {};
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
            if (e.isCancelled(id))
                return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        finish(id, sink);
    }
    fn isCancelled(e: *Engine, id: api.Id) bool {
        const n = @min(e.cancelled.load(.acquire), e.cancelled_ids.len);
        for (e.cancelled_ids[0..n]) |*slot| if (slot.load(.acquire) == id) return true;
        return false;
    }
    fn finish(id: api.Id, sink: api.Sink) void {
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{'x'} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(ptr: *anyopaque, id: api.Id) void {
        const e = cast(ptr);
        const i = e.cancelled.fetchAdd(1, .acq_rel);
        if (i < e.cancelled_ids.len) e.cancelled_ids[i].store(id, .release);
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

test "wait reason ledger: an admission 429 counts as a refusal under the code its reply carries" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null, .{ .max_inflight = 1 });
    defer h.deinit();
    var lease = h.srv.admission.acquire(t.io, .{ .max_inflight = 1 }, null).accepted;
    defer lease.release();
    const reasons = api.load_ledger.reasons;
    for (0..2) |_| {
        const reply = try h.request(a, try requestText(a, "/v1/completions", "", completion));
        try expectStatus(reply, "HTTP/1.1 429 Too Many Requests\r\n");
        try t.expect(contains(reply, "\"code\": \"max_inflight_exceeded\""));
    }
    const refusals = h.srv.load.read().refusals;
    for (std.enums.values(reasons.Refusal)) |r|
        try t.expectEqual(@as(u64, if (r == .max_inflight_exceeded) 2 else 0), refusals[@backingInt(r)]);
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

test "admission HTTP: n choices take n in-flight places, all or none, until the reply ends" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null, .{ .max_inflight = 2 });
    defer h.deinit();
    const three = "{\"model\":\"m\",\"prompt\":\"x\",\"max_tokens\":1,\"temperature\":0,\"n\":3}";
    const refused = try h.request(a, try requestText(a, "/v1/completions", "", three));
    try expectStatus(refused, "HTTP/1.1 429 Too Many Requests\r\n");
    try t.expect(contains(refused, "Retry-After: 1\r\n") and contains(refused, "\"code\": \"max_inflight_exceeded\"}}"));
    try t.expectEqual(@as(u32, 0), engine.calls.load(.acquire)); // no choice started
    engine.hold.store(true, .release);
    const two = "{\"model\":\"m\",\"prompt\":\"x\",\"max_tokens\":1,\"temperature\":0,\"n\":2}";
    const fd = try h.begin(try requestText(a, "/v1/completions", "", two));
    defer _ = posix.system.close(fd);
    var waited: usize = 0;
    while (engine.active.load(.acquire) < 2) : (waited += 1) {
        if (waited == 5000) return error.SubmitTimeout;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    try t.expectEqual(@as(u32, 2), h.srv.admission.snapshot(t.io).inflight);
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "", completion)), "HTTP/1.1 429 ");
    engine.hold.store(false, .release);
    const reply = try receive(a, fd);
    try expectStatus(reply, "HTTP/1.1 200 ");
    try t.expect(contains(reply, "\"index\": 1"));
    try t.expectEqual(@as(u32, 0), h.srv.admission.snapshot(t.io).inflight);
    try t.expectEqual(@as(u64, 2), h.srv.admission.snapshot(t.io).rejected_global);
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "", completion)), "HTTP/1.1 200 ");
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

fn choiceBody(a: Allocator, n: u32, extra: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":\"x\",\"max_tokens\":1,\"temperature\":0,\"n\":{d}{s}}}", .{ n, extra });
}

// inflight and every occupied principal slot back at zero; polled since a reply can precede the release by a beat.
fn waitFreed(h: *Harness) !void {
    var waited: usize = 0;
    while (true) : (waited += 1) {
        const snap = h.srv.admission.snapshot(t.io);
        if (snap.inflight == 0 and snap.principals == 0) return;
        if (waited == 5000) {
            std.debug.print("places never returned: {any}\n", .{snap});
            return error.PlacesLeaked;
        }
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn waitInflight(h: *Harness, want: u32) !void {
    var waited: usize = 0;
    while (h.srv.admission.snapshot(t.io).inflight != want) : (waited += 1) {
        if (waited == 5000) return error.AdmitTimeout;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
}

// The engine has `want` submits in all: a client that walks away after this has requests the server must cancel.
fn waitSubmitted(e: *Engine, want: u32) !void {
    var waited: usize = 0;
    while (e.calls.load(.monotonic) < want) : (waited += 1) {
        if (waited == 5000) return error.SubmitTimeout;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
}

test "admission HTTP: every way an n-choices request ends returns its places" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    const limits: admission.Limits = .{ .max_inflight = 16, .max_inflight_per_key = 4 };
    try h.start(&engine, null, limits);
    defer h.deinit();
    const completions = "/v1/completions";

    // whole replies for n = 2 and n = 4, each followed by a full-n request that must still fit.
    for ([_]u32{ 2, 4 }) |n| {
        const body = try choiceBody(a, n, "");
        for (0..2) |_| {
            try expectStatus(try h.request(a, try requestText(a, completions, "", body)), "HTTP/1.1 200 ");
            try waitFreed(&h);
        }
    }
    // n = 16 cannot fit one principal under max_inflight_per_key = 4: widen refuses all or none.
    try expectStatus(try h.request(a, try requestText(a, completions, "", try choiceBody(a, 16, ""))), "HTTP/1.1 429 ");
    try waitFreed(&h);
    try expectStatus(try h.request(a, try requestText(a, completions, "", try choiceBody(a, 4, ""))), "HTTP/1.1 200 ");
    try waitFreed(&h);

    // streamed replies for n = 2, 4, and the refused 16.
    for ([_]u32{ 2, 4 }) |n| {
        const reply = try h.request(a, try requestText(a, completions, "", try choiceBody(a, n, ",\"stream\":true")));
        try expectStatus(reply, "HTTP/1.1 200 ");
        try t.expect(std.mem.indexOf(u8, reply, "[DONE]") != null);
        try waitFreed(&h);
    }
    try expectStatus(try h.request(a, try requestText(a, completions, "", try choiceBody(a, 16, ",\"stream\":true"))), "HTTP/1.1 429 ");
    try waitFreed(&h);

    // a refused body still releases the place the gate already took.
    engine.logprobs = true;
    try expectStatus(try h.request(a, try requestText(a, completions, "", try choiceBody(a, 2, ",\"logprobs\":5"))), "HTTP/1.1 400 ");
    try waitFreed(&h);

    // a failed choice still returns all four places.
    engine.fail_after.store(1, .monotonic);
    _ = try h.request(a, try requestText(a, completions, "", try choiceBody(a, 4, "")));
    try waitFreed(&h);
    engine.fail_after.store(std.math.maxInt(u32), .monotonic);

    // a global refusal while every place is externally held: four principals at four places each.
    var leases: [16]admission.Lease = undefined;
    for (&leases, 0..) |*lease, i| {
        var digest: [32]u8 = @splat(0);
        digest[0] = @intCast(i / 4);
        lease.* = h.srv.admission.acquire(t.io, limits, digest).accepted;
    }
    try expectStatus(try h.request(a, try requestText(a, completions, "", completion)), "HTTP/1.1 429 ");
    for (&leases) |*lease| lease.release();
    try waitFreed(&h);

    // a key refusal while one principal holds all four of its places; then the holder walks away mid-request.
    engine.hold.store(true, .release);
    var submits = engine.calls.load(.monotonic);
    const occupied = try h.begin(try requestText(a, completions, "", try choiceBody(a, 4, "")));
    try waitInflight(&h, 4);
    try waitSubmitted(&engine, submits + 4); // all four in the engine: a choice still preparing would leave uncancelled
    try expectStatus(try h.request(a, try requestText(a, completions, "", completion)), "HTTP/1.1 429 ");
    _ = posix.system.close(occupied);
    try waitFreed(&h);
    try t.expect(engine.cancelled.load(.acquire) != 0);

    // a streaming request whose client walks away.
    engine.cancelled.store(0, .release);
    submits = engine.calls.load(.monotonic);
    const streamed = try h.begin(try requestText(a, completions, "", try choiceBody(a, 4, ",\"stream\":true")));
    try waitInflight(&h, 4);
    try waitSubmitted(&engine, submits + 4);
    _ = posix.system.close(streamed);
    try waitFreed(&h);
    try t.expect(engine.cancelled.load(.acquire) != 0);

    // an admin cancel: the running request's id sits in the registry.
    const cancelled = try h.begin(try requestText(a, completions, "", try choiceBody(a, 4, "")));
    defer _ = posix.system.close(cancelled);
    try waitInflight(&h, 4);
    var rid: []const u8 = "";
    var waited: usize = 0;
    while (rid.len == 0) : (waited += 1) {
        if (waited == 5000) return error.RegistryTimeout;
        for (cancel_ids.registry.held(t.io, a) catch return error.RegistryTimeout) |id| {
            if (std.mem.startsWith(u8, id, "cmpl-")) rid = id;
        }
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    const cancel_req = try std.fmt.allocPrint(a, "POST /v1/requests/{s}/cancel HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{rid});
    try expectStatus(try h.request(a, cancel_req), "HTTP/1.1 200 ");
    try waitFreed(&h);
    engine.hold.store(false, .release);

    // draining is what shutdown sets: the routes refuse 503 without a place, and a full-n request fits after.
    h.srv.draining.store(true, .release);
    try expectStatus(try h.request(a, try requestText(a, completions, "", try choiceBody(a, 4, ""))), "HTTP/1.1 503 ");
    try waitFreed(&h);
    h.srv.draining.store(false, .release);
    try expectStatus(try h.request(a, try requestText(a, completions, "", try choiceBody(a, 4, ""))), "HTTP/1.1 200 ");
    try waitFreed(&h);
}

fn stressWorker(h: *Harness, key: []const u8, seed: u64, rounds: usize, done: *std.atomic.Value(u32)) void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    const cred = std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{key}) catch return;
    var i: usize = 0;
    while (i < rounds) : (i += 1) {
        // n = 2 or 4 fit a key's places; n = 5's widen must refuse without leaking.
        const n: u32 = switch (random.uintLessThan(u32, 8)) {
            0 => 5,
            1...3 => 4,
            else => 2,
        };
        const raw = requestText(a, "/v1/completions", cred, choiceBody(a, n, "") catch return) catch return;
        if (random.uintLessThan(u32, 3) == 0) {
            if (h.begin(raw)) |fd| _ = posix.system.close(fd) else |_| {}
        } else _ = h.request(a, raw) catch {};
        _ = done.fetchAdd(1, .monotonic);
    }
}

test "admission HTTP: keyed n-choices under a seeded mix of replies and disconnects return every place" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(t.allocator, &.{ "one", "two", "three", "four" }, "", null, false, &problem);
    defer keys.deinit();
    var engine: Engine = .{};
    engine.delay_ms.store(10, .monotonic);
    var h: Harness = .{};
    try h.start(&engine, &keys, .{ .max_inflight = 16, .max_inflight_per_key = 4 });
    defer h.deinit();

    // four keys at once, 50 requests each, fixed seeds: a third of the clients close mid-flight.
    var done: std.atomic.Value(u32) = .init(0);
    const names = [_][]const u8{ "one", "two", "three", "four" };
    var threads: [4]std.Thread = undefined;
    for (&threads, names, 0..) |*th, name, i|
        th.* = try std.Thread.spawn(.{}, stressWorker, .{ &h, name, 0x5eed * (i + 1), 50, &done });
    for (threads) |th| th.join();
    try t.expectEqual(@as(u32, 200), done.load(.monotonic));

    try waitFreed(&h);
    try expectStatus(try h.request(a, try requestText(a, "/v1/completions", "Authorization: Bearer one\r\n", try choiceBody(a, 4, ""))), "HTTP/1.1 200 ");
    try waitFreed(&h);
}

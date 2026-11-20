//! Durable image jobs through production HTTP dispatch and image execution, without sockets or devices.
const std = @import("std");
const h = @import("image_test_harness.zig");
const json = @import("json");
const root = @import("root.zig");
const Bytes = @import("speech_stream_test.zig").Bytes;
const Conn = @import("http_conn.zig").Conn;
const Server = root.Server;
const Allocator = std.mem.Allocator;
const t = std.testing;
var bytes: Bytes = .{};

var cancel_hook_server: ?*Server = null;
fn cancelBeforeActive(invocation: @import("media_job.zig").Invocation) void {
    _ = cancel_hook_server.?.media_jobs.registry.cancel(invocation.id, owner(6)) catch {};
}

const Reply = struct {
    status: u16,
    body: []const u8,
    value: ?json.Value,
    head: []const u8,
    unread: usize,
    connection_close: bool,
};

fn ask(a: Allocator, srv: *Server, stub: *h.Stub, method: []const u8, route_path: []const u8, body: []const u8, principal: ?[32]u8) !Reply {
    var wire: h.Wire = .{ .a = a };
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: Conn = .{
        .fd = -1,
        .peer = "",
        .buf = try a.dupe(u8, body),
        .end = body.len,
        .gpa = a,
        .method = method,
        .path = route_path,
        .version = "HTTP/1.1",
        .headers = &headers,
        .key_digest = principal,
        .write_hooks = .{ .ctx = &wire, .write = h.Wire.write },
        .gone_hook = .{ .ctx = stub, .call = h.Stub.gone },
    };
    defer conn.deinit();
    @import("routes.zig").dispatch(srv, &conn, a);
    const unread = conn.pending().len;
    const connection_close = conn.close;
    const split = std.mem.indexOf(u8, wire.bytes.items, "\r\n\r\n") orelse return error.NoReply;
    const payload = wire.bytes.items[split + 4 ..];
    const parsed = json.parse(a, payload) catch return .{
        .status = try std.fmt.parseInt(u16, wire.bytes.items[9..12], 10),
        .body = payload,
        .value = null,
        .head = wire.bytes.items[0..split],
        .unread = unread,
        .connection_close = connection_close,
    };
    return .{
        .status = try std.fmt.parseInt(u16, wire.bytes.items[9..12], 10),
        .body = payload,
        .value = switch (parsed) {
            .ok => |value| value,
            .err => null,
        },
        .head = wire.bytes.items[0..split],
        .unread = unread,
        .connection_close = connection_close,
    };
}

fn makeServer(stub: *h.Stub, directory: ?[]const u8) !*Server {
    const srv = try Server.init(t.allocator, t.io, stub.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    errdefer srv.deinit();
    if (directory) |path_name| try srv.media_jobs.init(srv, path_name, .{ .running = 1 });
    return srv;
}

fn path(a: Allocator, id: []const u8, suffix: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "/v1/images/jobs/{s}{s}", .{ id, suffix });
}

fn waitFor(a: Allocator, srv: *Server, stub: *h.Stub, id: []const u8, principal: ?[32]u8, wanted: []const u8) !Reply {
    for (0..2000) |_| {
        const reply = try ask(a, srv, stub, "GET", try path(a, id, ""), "", principal);
        if (reply.value) |value| if (value.get("status")) |status| if (status == .string and std.mem.eql(u8, status.string, wanted)) return reply;
        std.Thread.yield() catch {};
    }
    return error.MediaJobTimeout;
}

fn owner(comptime byte: u8) ?[32]u8 {
    return @splat(byte);
}

fn tempRoot(a: Allocator, tmp: *std.testing.TmpDir) ![]const u8 {
    return tmp.dir.realPathFileAlloc(t.io, ".", a);
}

test "media jobs: routed execution returns exact artifacts and fences every other owner's operation" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{};
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    defer srv.deinit();
    const body = "{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":\"fog\",\"size\":\"512x768\",\"steps\":8,\"guidance\":2,\"n\":2,\"seed\":42}";
    const submitted = try ask(a, srv, &stub, "POST", "/v1/images/jobs", body, owner(1));
    try t.expectEqual(@as(u16, 202), submitted.status);
    const id = submitted.value.?.get("id").?.string;
    const done = try waitFor(a, srv, &stub, id, owner(1), "succeeded");
    try t.expectEqual(@as(usize, 2), done.value.?.get("artifacts").?.array.len);
    try t.expectEqual(@as(usize, 2), stub.calls);
    try t.expectEqual(@as(usize, 2), stub.finished);

    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "GET", try path(a, id, ""), "", owner(2))).status);
    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "GET", try path(a, id, "/content/0"), "", owner(2))).status);
    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "POST", try path(a, id, "/cancel"), "", owner(2))).status);
    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "DELETE", try path(a, id, ""), "", owner(2))).status);

    const content = try ask(a, srv, &stub, "GET", try path(a, id, "/content/0"), "", owner(1));
    try t.expectEqual(@as(u16, 200), content.status);
    try t.expect(content.value == null);
    try t.expectEqual(@as(usize, 40), content.body.len);
    try t.expectEqualSlices(u8, "\x89PNG\r\n\x1a\n", content.body[0..8]);
    try t.expectEqual(@as(u16, 204), (try ask(a, srv, &stub, "DELETE", try path(a, id, ""), "", owner(1))).status);
    try t.expectEqual(@as(u16, 404), (try ask(a, srv, &stub, "GET", try path(a, id, ""), "", owner(1))).status);
}

test "media jobs: cancellation reaches the held engine and settles before server teardown" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{ .hold_at = 1 };
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    var open = true;
    defer if (open) srv.deinit();
    const submitted = try ask(a, srv, &stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"hold\"}", owner(3));
    const id = submitted.value.?.get("id").?.string;
    _ = try waitFor(a, srv, &stub, id, owner(3), "running");
    _ = try ask(a, srv, &stub, "POST", try path(a, id, "/cancel"), "", owner(3));
    _ = try waitFor(a, srv, &stub, id, owner(3), "cancelled");
    srv.deinit();
    open = false;
    try t.expectEqual(@as(usize, 1), stub.calls);
    try t.expectEqual(@as(usize, 1), stub.cancels);
    try t.expectEqual(@as(usize, 1), stub.finished);
    try t.expect(stub.pending == null);
}

test "media jobs: invalid input persists no execution and an unconfigured server keeps the route closed" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{};
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    var open = true;
    defer if (open) srv.deinit();
    const invalid = try ask(a, srv, &stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"\",\"n\":1}", owner(1));
    try t.expectEqual(@as(u16, 400), invalid.status);
    try t.expectEqual(@as(usize, 0), stub.calls);
    srv.deinit();
    open = false;

    var closed_stub: h.Stub = .{};
    const closed = try makeServer(&closed_stub, null);
    defer closed.deinit();
    const missing = try ask(a, closed, &closed_stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"hi\"}", owner(1));
    try t.expectEqual(@as(u16, 404), missing.status);
    try t.expectEqual(@as(usize, 0), closed_stub.calls);
}

test "media jobs: scoped keys reach only their authorized job operations" {
    const scope = @import("auth_scope.zig");
    const inference: scope.Scopes = .{ .inference = true };
    const read: scope.Scopes = .{ .read = true };
    const id = "media-0123456789abcdef0123456789abcdef";
    const job = "/v1/images/jobs/" ++ id;
    try t.expect(scope.allows(
        inference,
        "POST",
        "/v1/images/jobs",
    ));
    try t.expect(scope.allows(inference, "POST", job ++ "/cancel"));
    try t.expect(scope.allows(inference, "GET", job));
    try t.expect(scope.allows(read, "GET", job ++ "/content/0"));
    try t.expect(!scope.allows(read, "POST", "/v1/images/jobs"));
    try t.expect(!scope.allows(inference, "DELETE", job));
    try t.expect(scope.allows(scope.full, "DELETE", job));
}

test "media jobs: serve CLI carries the opt-in directory and publishes its flag" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cli = @import("cli.zig");
    var usage: cli.Usage = .{};
    const args = try cli.parse(a, &.{ "m", "--media-jobs-dir", "jobs" }, &usage);
    try t.expectEqualStrings("jobs", args.media_jobs_dir.?);
    var output: std.Io.Writer.Allocating = .init(a);
    try cli.capabilities(&output.writer, .{ .version = "test" });
    try t.expect(std.mem.indexOf(u8, output.written(), "\"--media-jobs-dir\"") != null);
}

test "media jobs: cancel before Active registration cannot run stale generation" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{};
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    defer srv.deinit();
    cancel_hook_server = srv;
    const service = @import("media_job_service.zig");
    service.before_active_hook = cancelBeforeActive;
    defer {
        service.before_active_hook = null;
        cancel_hook_server = null;
    }
    const submitted = try ask(a, srv, &stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"hi\"}", owner(6));
    try t.expectEqual(@as(u16, 202), submitted.status);
    const id = submitted.value.?.get("id").?.string;
    _ = try waitFor(a, srv, &stub, id, owner(6), "cancelled");
    try t.expectEqual(@as(usize, 0), stub.calls);
    try t.expectEqual(@as(usize, 0), stub.finished);
    for (0..2000) |_| {
        if (srv.generating.load(.seq_cst) == 0) break;
        std.Thread.yield() catch {};
    }
    try t.expectEqual(@as(u32, 0), srv.generating.load(.seq_cst));
}

test "media jobs: draining refuses submission without accepting durable work" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{};
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    defer srv.deinit();
    srv.draining.store(true, .seq_cst);
    const refused = try ask(a, srv, &stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"hi\"}", owner(1));
    try t.expectEqual(@as(u16, 503), refused.status);
    try t.expectEqual(@as(usize, 0), refused.unread);
    try t.expect(refused.connection_close);
    try t.expectEqual(@as(usize, 0), srv.media_jobs.registry.stats().active);
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "media jobs: DELETE consumes its framed body and closes on uncertain framing" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{};
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    defer srv.deinit();
    const id = "media-0123456789abcdef0123456789abcdef";
    const deleted = try ask(a, srv, &stub, "DELETE", try path(a, id, ""), "discard me", owner(1));
    try t.expectEqual(@as(u16, 404), deleted.status);
    try t.expectEqual(@as(usize, 0), deleted.unread);

    var wire: h.Wire = .{ .a = a };
    var headers = [_]@import("http_conn.zig").Header{
        .{ .name = "Content-Length", .value = "0" },
        .{ .name = "Transfer-Encoding", .value = "chunked" },
    };
    var conn: Conn = .{
        .fd = -1,
        .peer = "",
        .buf = try a.dupe(u8, ""),
        .gpa = a,
        .method = "DELETE",
        .path = try path(a, id, ""),
        .version = "HTTP/1.1",
        .headers = &headers,
        .key_digest = owner(1),
        .write_hooks = .{ .ctx = &wire, .write = h.Wire.write },
        .gone_hook = .{ .ctx = &stub, .call = h.Stub.gone },
    };
    defer conn.deinit();
    @import("routes.zig").dispatch(srv, &conn, a);
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, " 400 ") != null);
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Content-Length and Transfer-Encoding cannot be combined") != null);
    try t.expect(std.mem.indexOf(u8, wire.bytes.items, "Connection: close\r\n") != null);
    try t.expect(conn.close);
}

const Reload = struct {
    stub: *h.Stub,
    opens: std.atomic.Value(u32) = .init(0),
    closes: std.atomic.Value(u32) = .init(0),
    replacements: std.atomic.Value(u32) = .init(0),
    fail: bool = false,

    fn reopen(r: *Reload) @import("idle_unload.zig").Reopen {
        return .{ .ctx = r, .open = open, .close_held = closeHeld, .replace = replace };
    }
    fn cast(ctx: *anyopaque) *Reload {
        return @ptrCast(@alignCast(ctx));
    }
    fn open(ctx: *anyopaque, _: Allocator, _: Allocator, _: std.Io, _: []const u8, _: []const u8, _: @import("cli.zig").Args, problem: *[]const u8) anyerror!?@import("engine_api").Opened {
        const r = cast(ctx);
        _ = r.opens.fetchAdd(1, .acq_rel);
        if (r.fail) {
            problem.* = "the fake image engine could not reopen";
            return null;
        }
        return .{ .engine = r.stub.engine(), .close = closeCandidate, .ctx = r.stub };
    }
    fn closeCandidate(_: *anyopaque) void {}
    fn closeHeld(ctx: *anyopaque) void {
        _ = cast(ctx).closes.fetchAdd(1, .acq_rel);
    }
    fn replace(ctx: *anyopaque, opened: @import("engine_api").Opened, scratch: *std.heap.ArenaAllocator) void {
        _ = opened;
        _ = cast(ctx).replacements.fetchAdd(1, .acq_rel);
        @import("idle_unload_closer.zig").destroyScratch(scratch);
    }
};

fn expireIdle(srv: *Server) void {
    const m = srv.idle_manager.?;
    m.mutex.lockUncancelable(srv.io);
    const now: i64 = @intCast(std.Io.Clock.awake.now(srv.io).toNanoseconds());
    m.last_activity_ns = now -% (m.idle_s * std.time.ns_per_s) -% 1;
    m.mutex.unlock(srv.io);
    m.cond.broadcast(srv.io);
}

fn waitUnloaded(srv: *Server) !void {
    const m = srv.idle_manager.?;
    for (0..2000) |_| {
        m.mutex.lockUncancelable(srv.io);
        const unloaded = m.state == .unloaded;
        m.mutex.unlock(srv.io);
        if (unloaded) return;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.MediaJobTimeout;
}

fn waitLeaseReleased(srv: *Server) !void {
    for (0..2000) |_| {
        if (srv.generating.load(.seq_cst) == 0) return;
        std.Thread.yield() catch {};
    }
    return error.MediaJobTimeout;
}

test "media jobs: workers reload parked engines and hold both leases through failure" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stub: h.Stub = .{};
    var reload: Reload = .{ .stub = &stub };
    const srv = try makeServer(&stub, try tempRoot(a, &tmp));
    srv.load.attach(srv.engine);
    var args: @import("cli.zig").Args = .{};
    args.idle_unload = 60;
    try t.expectEqual(@import("idle_unload.zig").Startup.enabled, @import("idle_unload.zig").startServer(srv, args, reload.reopen(), "d", "image"));
    defer {
        srv.media_jobs.deinit();
        @import("idle_unload.zig").stopServer(srv);
        srv.deinit();
    }
    expireIdle(srv);
    try waitUnloaded(srv);
    try t.expectEqual(@as(u32, 1), reload.closes.load(.acquire));

    const first = try ask(a, srv, &stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"reload\"}", owner(1));
    const first_id = first.value.?.get("id").?.string;
    _ = try waitFor(a, srv, &stub, first_id, owner(1), "succeeded");
    try waitLeaseReleased(srv);
    try t.expectEqual(@as(u32, 1), reload.opens.load(.acquire));
    try t.expectEqual(@as(u32, 1), reload.replacements.load(.acquire));
    try t.expectEqual(@as(usize, 1), stub.calls);

    stub.hold_at = 2;
    const second = try ask(a, srv, &stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"hold\"}", owner(1));
    const second_id = second.value.?.get("id").?.string;
    for (0..2000) |_| {
        if (stub.pending != null) break;
        std.Thread.yield() catch {};
    }
    try t.expect(stub.pending != null);
    expireIdle(srv);
    std.Io.sleep(t.io, .fromMilliseconds(50), .awake) catch {};
    try t.expectEqual(@as(u32, 1), reload.closes.load(.acquire));
    try t.expect(!srv.idle_manager.?.isParked());
    _ = try ask(a, srv, &stub, "POST", try path(a, second_id, "/cancel"), "", owner(1));
    _ = try waitFor(a, srv, &stub, second_id, owner(1), "cancelled");
    try waitLeaseReleased(srv);

    reload.fail = true;
    expireIdle(srv);
    try waitUnloaded(srv);
    const third = try ask(a, srv, &stub, "POST", "/v1/images/jobs", "{\"model\":\"m\",\"prompt\":\"fail load\"}", owner(1));
    const third_id = third.value.?.get("id").?.string;
    _ = try waitFor(a, srv, &stub, third_id, owner(1), "failed");
    try waitLeaseReleased(srv);
    const m = srv.idle_manager.?;
    m.mutex.lockUncancelable(srv.io);
    const active = m.active;
    m.mutex.unlock(srv.io);
    try t.expectEqual(@as(u32, 0), active);
    try t.expectEqual(@as(u32, 2), reload.opens.load(.acquire));
    try t.expectEqual(@as(u32, 1), reload.replacements.load(.acquire));
    try t.expectEqual(@as(usize, 2), stub.calls);
}

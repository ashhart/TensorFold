//! GET and DELETE /v1/handoff/<id> over real sockets: Content-Length stream, key gate first, hex ids, 404 if empty.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const auth = @import("auth.zig");
const listener = @import("listener.zig");
const KeyFile = @import("auth_scope_test.zig").KeyFile;
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const posix = @import("sys").posix;
const t = std.testing;
const Allocator = std.mem.Allocator;
const Serve = api.LaneHost.Export.Serve;

const keys = "app[inference]: infer-secret\nmonitor[read]: read-secret\nops[admin]: admin-secret\n";
const held_id = "00112233445566778899aabbccddeeff";
const failed_id = "ffeeddccbbaa99887766554433221100";

/// An engine holding one export of 5000 bytes (served 700 at a time) and one failed export.
const Engine = struct {
    body: [5000]u8 = undefined,
    opened: std.atomic.Value(u32) = .init(0),
    closed: std.atomic.Value(u32) = .init(0),
    released: std.atomic.Value(bool) = .init(false),
    serves: bool = true,

    const Reading = struct { e: *Engine, at: usize = 0 };

    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .handoff = handoff } };
    }
    fn cast(ptr: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ptr));
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Busy;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn handoff(ptr: *anyopaque) ?Serve {
        const e = cast(ptr);
        return if (e.serves) .{ .ptr = e, .open = open, .release = release } else null;
    }
    fn open(ptr: *anyopaque, id: [16]u8, why: *[]const u8) Serve.OpenError!api.LaneHost.Export.Stream {
        const e = cast(ptr);
        var hex: [32]u8 = undefined;
        _ = std.fmt.bufPrint(&hex, "{x}", .{&id}) catch unreachable;
        if (std.mem.eql(u8, &hex, failed_id)) {
            why.* = "NotAHandoffPrompt";
            return error.Refused;
        }
        if (!std.mem.eql(u8, &hex, held_id) or e.released.load(.acquire)) return error.NotHeld;
        const r = try t.allocator.create(Reading);
        r.* = .{ .e = e };
        _ = e.opened.fetchAdd(1, .monotonic);
        return .{ .ptr = r, .size = e.body.len, .next = next, .close = close };
    }
    fn next(ptr: *anyopaque) anyerror!?[]const u8 {
        const r: *Reading = @ptrCast(@alignCast(ptr));
        if (r.at == r.e.body.len) return null;
        const end = @min(r.at + 700, r.e.body.len);
        defer r.at = end;
        return r.e.body[r.at..end];
    }
    fn close(ptr: *anyopaque) void {
        const r: *Reading = @ptrCast(@alignCast(ptr));
        _ = r.e.closed.fetchAdd(1, .monotonic);
        t.allocator.destroy(r);
    }
    fn release(ptr: *anyopaque, id: [16]u8) bool {
        const e = cast(ptr);
        var hex: [32]u8 = undefined;
        _ = std.fmt.bufPrint(&hex, "{x}", .{&id}) catch unreachable;
        if (!std.mem.eql(u8, &hex, held_id)) return false;
        return !e.released.swap(true, .acq_rel);
    }
};

const Harness = struct {
    srv: *server.Server = undefined,
    listener: listener.Listener = undefined,
    accept: std.Thread = undefined,
    stop: std.atomic.Value(bool) = .init(false),

    fn start(h: *Harness, e: *Engine, store: ?*auth.Store) !void {
        h.srv = try server.Server.init(t.allocator, t.io, e.engine(), ByteText.model(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .keep_warm_s = 0 }, store);
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
            if (waited == 5000) @panic("handoff fixture did not quiesce");
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        h.listener.close();
        h.srv.deinit();
    }
    fn ask(h: *Harness, a: Allocator, method: []const u8, path: []const u8, key: []const u8) ![]const u8 {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.listener.port()) };
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&address, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.Socket;
        const fd: posix.socket_t = @intCast(rc);
        defer _ = posix.system.close(fd);
        if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.Connect;
        const header = if (key.len > 0) try std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{key}) else "";
        const raw = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: localhost\r\n{s}Connection: close\r\n\r\n", .{ method, path, header });
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
};

fn expectStatus(reply: []const u8, code: u16) !void {
    var prefix: [32]u8 = undefined;
    const expected = try std.fmt.bufPrint(&prefix, "HTTP/1.1 {d} ", .{code});
    if (!std.mem.startsWith(u8, reply, expected)) {
        std.debug.print("expected {d}, got: {s}\n", .{ code, reply[0..@min(reply.len, 200)] });
        return error.UnexpectedStatus;
    }
}

fn bodyOf(reply: []const u8) []const u8 {
    return reply[(std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return &.{}) + 4 ..];
}

test "handoff routes: a held export streams whole under its Content-Length, behind the key gate and its scopes" {
    var file = try KeyFile.init(keys);
    defer file.deinit();
    var problem: []const u8 = "";
    var store = try auth.Store.init(t.allocator, &.{}, "", file.path, false, &problem);
    defer store.deinit();
    var e: Engine = .{};
    for (&e.body, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
    var h: Harness = .{};
    try h.start(&e, &store);
    defer h.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = "/v1/handoff/" ++ held_id;

    try expectStatus(try h.ask(a, "GET", path, ""), 401);
    try expectStatus(try h.ask(a, "GET", path, "wrong-secret"), 401);
    try expectStatus(try h.ask(a, "GET", path, "read-secret"), 403); // a read scope lists and measures; it does not fetch states
    try t.expectEqual(@as(u32, 0), e.opened.load(.monotonic));
    for ([_][]const u8{ "infer-secret", "admin-secret" }) |key| {
        const reply = try h.ask(a, "GET", path ++ "?x=1", key);
        try expectStatus(reply, 200);
        try t.expect(std.mem.indexOf(u8, reply, "Content-Length: 5000\r\n") != null);
        try t.expect(std.mem.indexOf(u8, reply, "Content-Type: application/octet-stream\r\n") != null);
        try t.expectEqualSlices(u8, &e.body, bodyOf(reply)); // the whole body, byte for byte
    }
    try t.expectEqual(@as(u32, 2), e.opened.load(.monotonic));
    try t.expectEqual(@as(u32, 2), e.closed.load(.monotonic)); // each read released its export
}

test "handoff routes: an id is 32 hex digits and never a path; an export not held is 404 and a failed one is 409 with its reason" {
    var e: Engine = .{};
    var h: Harness = .{};
    try h.start(&e, null);
    defer h.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "/v1/handoff/00112233", "/v1/handoff/../../etc/passwd", "/v1/handoff/" ++ held_id ++ "00", "/v1/handoff/" ++ held_id ++ "/x", "/v1/handoff/%2e%2e%2fsecret", "/v1/handoff/zz112233445566778899aabbccddeeff" }) |bad|
        try expectStatus(try h.ask(a, "GET", bad, ""), 400);
    try expectStatus(try h.ask(a, "GET", "/v1/handoff", ""), 404); // no id: not a route at all
    const missing = try h.ask(a, "GET", "/v1/handoff/0123456789abcdef0123456789abcdef", "");
    try expectStatus(missing, 404);
    try t.expect(std.mem.indexOf(u8, missing, "no export is held") != null);
    const failed = try h.ask(a, "GET", "/v1/handoff/" ++ failed_id, "");
    try expectStatus(failed, 409);
    try t.expect(std.mem.indexOf(u8, failed, "NotAHandoffPrompt") != null);
    try t.expectEqual(@as(u32, 0), e.opened.load(.monotonic));
}

test "handoff routes: DELETE releases the export once, and an engine that serves no exports answers 404" {
    var e: Engine = .{};
    var h: Harness = .{};
    try h.start(&e, null);
    defer h.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = "/v1/handoff/" ++ held_id;
    try expectStatus(try h.ask(a, "DELETE", path, ""), 200);
    try expectStatus(try h.ask(a, "DELETE", path, ""), 404);
    try expectStatus(try h.ask(a, "GET", path, ""), 404); // released: gone for readers too
    try expectStatus(try h.ask(a, "DELETE", "/v1/handoff/short", ""), 400);
    e.serves = false;
    const none = try h.ask(a, "GET", path, "");
    try expectStatus(none, 404);
    try t.expect(std.mem.indexOf(u8, none, "--mcdma-produce http") != null);
    try expectStatus(try h.ask(a, "DELETE", path, ""), 404);
}

//! The production learn route must keep saving inside the stream, including slow and failed shard writes.
const std = @import("std");
const api = @import("engine_api");
const slide = @import("slide.zig");
const Server = @import("server.zig").Server;
const ByteText = @import("slide_forget_test.zig").ByteText;
const Conn = @import("http_conn.zig").Conn;
const t = std.testing;
const Model = struct {
    entered: std.Io.Event = .unset,
    release: std.Io.Event = .unset,
    finished: std.Io.Event = .unset,
    saves: std.atomic.Value(usize) = .init(0),
    fail: bool,
    fn engine(m: *@This()) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn learn(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (request.save) {
            _ = m.saves.fetchAdd(1, .monotonic);
            m.entered.set(t.io);
            m.release.waitUncancelable(t.io);
        }
        sink.event(sink.ctx, &.{ .done = .{ .message = if (request.save and m.fail) "save sentinel failure" else "" } });
    }
};
const Route = struct {
    srv: *Server,
    conn: *Conn,
    a: std.mem.Allocator,
    model: *Model,
    fn run(r: *@This()) void {
        slide.learn(r.srv, r.conn, r.a);
        r.model.finished.set(t.io);
    }
};
fn collect(fd: std.c.fd_t, out: *std.ArrayList(u8)) !void {
    var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
    while (try std.posix.poll(&fds, 100) > 0) {
        var buf: [4096]u8 = undefined;
        const n = try std.posix.read(fd, &buf);
        if (n == 0) return;
        try out.appendSlice(t.allocator, buf[0..n]);
    }
}
fn check(fail: bool) !void {
    var fds: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &fds) != 0) return error.SocketPair;
    defer for (fds) |fd| {
        _ = std.c.close(fd);
    };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m: Model = .{ .fail = fail };
    var text: ByteText = .{};
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
    defer srv.deinit();
    var conn = try Conn.init(t.allocator, fds[1], "test");
    defer conn.deinit();
    const body = "{\"text\":\"I like blue.\"}";
    @memcpy(conn.buf[0..body.len], body);
    conn.end = body.len;
    var digits: [20]u8 = undefined;
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.bufPrint(&digits, "{d}", .{body.len}) }};
    conn.headers = &headers;
    conn.version = "HTTP/1.1";
    // The byte template intentionally fails lesson construction; its accepted graph changes must still be saved.
    var r: Route = .{ .srv = srv, .conn = &conn, .a = a, .model = &m };
    const thread = try std.Thread.spawn(.{}, Route.run, .{&r});
    defer {
        m.release.set(t.io);
        thread.join();
    }
    try m.entered.waitTimeout(t.io, .{ .duration = .{ .raw = .fromMilliseconds(2000), .clock = .awake } });
    var wire: std.ArrayList(u8) = .empty;
    defer wire.deinit(t.allocator);
    try collect(fds[0], &wire);
    try t.expect(std.mem.indexOf(u8, wire.items, "[DONE]") == null);
    try t.expect(std.mem.indexOf(u8, wire.items, "\"saving\"") != null);
    m.release.set(t.io);
    try m.finished.waitTimeout(t.io, .{ .duration = .{ .raw = .fromMilliseconds(2000), .clock = .awake } });
    try collect(fds[0], &wire);
    try t.expectEqual(@as(usize, 1), m.saves.load(.monotonic));
    const done = std.mem.indexOf(u8, wire.items, "data: [DONE]\n\n") orelse return error.NoDone;
    const outcome = std.mem.indexOf(u8, wire.items, if (fail) "save sentinel failure" else "event: saved\n") orelse return error.NoSaveResult;
    try t.expect(outcome < done);
    try t.expect(std.mem.endsWith(u8, wire.items, "data: [DONE]\n\n"));
    const expected: @TypeOf(srv.slide.saved.state) = if (fail) .failed else .saved;
    try t.expectEqual(expected, srv.slide.saved.state);
}
test "lesson stream: a slow final save reports saving before DONE" {
    try check(false);
}
test "lesson stream: a failed final save reports failure before DONE" {
    try check(true);
}

test "lesson stream: a closed writer still saves on cleanup, including errors" {
    inline for (.{ false, true }) |fail| {
        var m: Model = .{ .fail = fail };
        m.release.set(t.io);
        var text: ByteText = .{};
        const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        defer srv.deinit();
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = t.allocator, .broken = true };
        var out: slide.Out = .{ .conn = &conn, .a = arena.allocator(), .open = false };
        slide.finishLearning(srv, arena.allocator(), &out);
        try t.expectEqual(@as(usize, 1), m.saves.load(.monotonic));
        const expected: @TypeOf(srv.slide.saved.state) = if (fail) .failed else .saved;
        try t.expectEqual(expected, srv.slide.saved.state);
        try t.expect(!out.open);
    }
}

//! The learn stream carries real progress and stays readable while a worker has not finished its next unit.
const std = @import("std");
const api = @import("engine_api");
const slide = @import("slide.zig");
const Server = @import("server.zig").Server;
const ByteText = @import("slide_forget_test.zig").ByteText;
const Conn = @import("http_conn.zig").Conn;
const t = std.testing;
const Model = struct {
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
    fn learn(_: *anyopaque, _: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        sink.event(sink.ctx, &.{ .progress = .{ .phase = .fitting, .at = 3, .of = 7, .loss = 0.25 } });
        sink.event(sink.ctx, &.{ .learned = .{ .recalled = true, .steps = 1, .loss = 0 } });
        sink.event(sink.ctx, &.{ .done = .{} });
    }
};
fn pair() ![2]std.c.fd_t {
    var fds: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &fds) != 0) return error.SocketPair;
    return fds;
}
fn read(fd: std.c.fd_t, buf: []u8) ![]u8 {
    var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
    if (try std.posix.poll(&fds, 300) == 0) return buf[0..0];
    return buf[0..try std.posix.read(fd, buf)];
}
test "lesson stream: learner progress reaches the SSE connection" {
    const fds = try pair();
    defer for (fds) |fd| {
        _ = std.c.close(fd);
    };
    var conn = try Conn.init(t.allocator, fds[1], "test");
    defer conn.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var out: slide.Out = .{ .conn = &conn, .a = a };
    var model: Model = .{};
    var text: ByteText = .{};
    const srv = try Server.init(t.allocator, t.io, model.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    const id = srv.slide.add("I like blue.", "chat", 1);
    const result = if (comptime @hasDecl(slide, "runReported")) slide.runReported(srv, a, &.{}, id, &out) else slide.run(srv, a, &.{}, id);
    try t.expect(result == .learned);
    var buf: [2048]u8 = undefined;
    const wire = try read(fds[0], &buf);
    try t.expect(std.mem.indexOf(u8, wire, "event: progress\n") != null);
    try t.expect(std.mem.indexOf(u8, wire, "\"phase\": \"fitting\"") != null);
    try t.expect(std.mem.indexOf(u8, wire, "\"at\": 3") != null);
}
test "lesson stream: heartbeat precedes a blocked unit and stops before DONE" {
    const fds = try pair();
    defer for (fds) |fd| {
        _ = std.c.close(fd);
    };
    var conn = try Conn.init(t.allocator, fds[1], "test");
    defer conn.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var out: slide.Out = .{ .conn = &conn, .a = arena.allocator() };
    if (comptime @hasDecl(slide.Out, "start")) try out.start(t.io, 10);
    defer if (comptime @hasDecl(slide.Out, "stop")) out.stop();
    var buf: [2048]u8 = undefined;
    const first = try read(fds[0], &buf);
    try t.expect(std.mem.indexOf(u8, first, "\"heartbeat\":true") != null);
    try t.expect(std.mem.indexOf(u8, first, "\"phase\":\"writing\"") != null);
    if (comptime @hasDecl(slide.Out, "done")) out.done();
    var rest: std.ArrayList(u8) = .empty;
    defer rest.deinit(t.allocator);
    while (true) {
        const bytes = try read(fds[0], &buf);
        if (bytes.len == 0) break;
        try rest.appendSlice(t.allocator, bytes);
    }
    try t.expect(std.mem.endsWith(u8, rest.items, "data: [DONE]\n\n"));
}

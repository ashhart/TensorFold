//! Real HTTP heads, bodies and protocol adapters over bounded local socket pairs, with a scripted research engine.
const std = @import("std");
const platform = @import("sys");
const json = @import("json");
const fixture = @import("cortheon_test.zig");
const live = @import("../stream_live.zig");
const GateEngine = @import("gate_engine.zig").Engine;
const Conn = @import("../http_conn.zig").Conn;
const Server = @import("../server.zig").Server;
const t = std.testing;
const io = t.io;
const posix = platform.posix;
const Route = enum { chat, messages, responses };

fn pair() ![2]posix.socket_t {
    var sockets: [2]posix.socket_t = undefined;
    if (platform.windows) try posix.socketpair(&sockets) else if (platform.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &sockets) != 0) return error.SocketPair;
    return sockets;
}
const Worker = struct {
    srv: *Server,
    fd: posix.socket_t,
    failed: bool = false,
    progress: ?*Progress = null,
    fn run(w: *Worker) void {
        defer _ = posix.system.close(w.fd);
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var conn = Conn.init(t.allocator, w.fd, "local request fixture") catch {
            w.failed = true;
            return;
        };
        defer conn.deinit();
        conn.quiet_log = true;
        conn.timeouts.read_ms = 1000;
        conn.timeouts.write_ms = 1000;
        const outcome = conn.readRequest(arena.allocator(), false) catch {
            w.failed = true;
            return;
        };
        if (outcome != .ready) {
            w.failed = true;
            return;
        }
        if (w.progress) |p| live.Testing.after_due_beat = .{ .ctx = p, .call = Progress.afterBeat };
        defer live.Testing.after_due_beat = null;
        w.srv.dispatch(&conn, arena.allocator());
    }
};
const Reply = struct { status: u16, raw: []const u8, head: []const u8, payload: []const u8 };

const Progress = struct {
    engine: *GateEngine,
    due: std.atomic.Value(bool) = .init(false),
    bad: std.atomic.Value(bool) = .init(false),
    released: bool = false,
    fn afterBeat(ctx: *anyopaque, opened: ?bool, sent: bool) void {
        const p: *Progress = @ptrCast(@alignCast(ctx));
        if (!p.engine.active.load(.acquire)) return;
        if (opened != false or sent) p.bad.store(true, .release);
        p.due.store(true, .release);
    }
    fn advance(p: *Progress) !void {
        if (!p.released and p.due.load(.acquire)) {
            try p.engine.releaseHeld();
            p.released = true;
        }
    }
};

fn request(srv: *Server, a: std.mem.Allocator, route_path: []const u8, request_body: []const u8, progress: ?*Progress) !Reply {
    const sockets = try pair();
    defer _ = posix.system.close(sockets[0]);
    var worker: Worker = .{ .srv = srv, .fd = sockets[1], .progress = progress };
    const thread = std.Thread.spawn(.{}, Worker.run, .{&worker}) catch |err| {
        _ = posix.system.close(sockets[1]);
        return err;
    };
    var joined = false;
    defer {
        _ = posix.system.shutdown(sockets[0], posix.SHUT.RDWR);
        if (!joined) thread.join();
    }
    const wire = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ route_path, request_body.len, request_body });
    var sender = try Conn.init(a, sockets[0], "local request sender");
    defer sender.deinit();
    sender.timeouts.write_ms = 1000;
    try sender.writeAll(wire);
    var raw: std.ArrayList(u8) = .empty;
    var progressed = std.Io.Clock.awake.now(io).toMilliseconds();
    while (true) {
        if (progress) |p| try p.advance();
        // An inactivity watchdog diagnoses a stuck fixture; elapsed time never releases the held engine.
        if (std.Io.Clock.awake.now(io).toMilliseconds() - progressed > 30_000) return error.RequestTimeout;
        var fds = [_]posix.pollfd{.{ .fd = sockets[0], .events = posix.POLL.IN, .revents = 0 }};
        if (try posix.poll(&fds, 100) == 0) continue;
        var buf: [8192]u8 = undefined;
        const n = try posix.read(sockets[0], &buf);
        if (n == 0) break;
        try raw.appendSlice(a, buf[0..n]);
        progressed = std.Io.Clock.awake.now(io).toMilliseconds();
    }
    thread.join();
    joined = true;
    try t.expect(!worker.failed);
    if (!std.mem.startsWith(u8, raw.items, "HTTP/")) {
        std.debug.print("context gate: bytes before HTTP head: {s}\n", .{raw.items[0..@min(256, raw.items.len)]});
        return error.BytesBeforeHttpHead;
    }
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, raw.items, "HTTP/1."));
    const end = std.mem.indexOf(u8, raw.items, "\r\n\r\n") orelse return error.MissingHttpHead;
    return .{ .status = try std.fmt.parseInt(u16, raw.items[9..12], 10), .raw = raw.items, .head = raw.items[0..end], .payload = raw.items[end + 4 ..] };
}

fn body(a: std.mem.Allocator, route: Route, streaming: bool, tools: bool) ![]const u8 {
    const tool: []const u8 = if (!tools) "" else switch (route) {
        .chat => ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]",
        .messages => ",\"tools\":[{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\"}}]",
        .responses => ",\"tools\":[{\"type\":\"function\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}]",
    };
    const fields = if (route == .responses) "\"max_output_tokens\":1,\"input\":\"What is the latest release of Zig?\"" else "\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"What is the latest release of Zig?\"}]";
    return std.fmt.allocPrint(a, "{{\"model\":\"m\",\"stream\":{s},{s}{s}}}", .{ if (streaming) "true" else "false", fields, tool });
}
fn path(route: Route) []const u8 {
    return switch (route) {
        .chat => "/v1/chat/completions",
        .messages => "/v1/messages",
        .responses => "/v1/responses",
    };
}
fn terminal(route: Route, payload: []const u8) !void {
    const end = switch (route) {
        .chat => "data: [DONE]",
        .messages => "event: error\n",
        .responses => "event: response.failed\n",
    };
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, payload, end));
    switch (route) {
        .chat => {
            try t.expect(std.mem.indexOf(u8, payload, "context_length_exceeded") != null);
            try t.expectEqual(@as(usize, 1), std.mem.count(u8, payload, "\"error\":"));
            const error_at = std.mem.indexOf(u8, payload, "\"error\":").?;
            try t.expect(std.mem.indexOf(u8, payload[error_at..], "\"choices\":") == null);
        },
        .messages => {
            try t.expect(std.mem.indexOf(u8, payload, "invalid_request_error") != null);
            try t.expectEqual(@as(usize, 1), std.mem.count(u8, payload, "event: message_start\n"));
            try t.expect(std.mem.indexOf(u8, payload, "event: message_stop") == null);
            try t.expect(std.mem.indexOf(u8, payload, "content_block_delta") == null);
        },
        .responses => {
            try t.expect(std.mem.indexOf(u8, payload, "invalid_prompt") != null);
            try t.expectEqual(@as(usize, 1), std.mem.count(u8, payload, "event: response.created\n"));
            try t.expect(std.mem.indexOf(u8, payload, "event: response.completed") == null);
            try t.expect(std.mem.indexOf(u8, payload, "output_text.delta") == null);
            try t.expect(std.mem.indexOf(u8, payload, "function_call_arguments.delta") == null);
        },
    }
    const last = std.mem.indexOf(u8, payload, end).?;
    const frame_end = std.mem.indexOfPos(u8, payload, last, "\n\n") orelse return error.UnterminatedEvent;
    try t.expectEqual(@as(usize, 0), std.mem.trim(u8, payload[frame_end + 2 ..], "\r\n \t").len);
}

test "context gate: real HTTP adapters keep late refusal ordering at both windows with and without tools" {
    for ([_]u32{ 2000, 2400 }) |window| for ([_]Route{ .chat, .messages, .responses }) |route| for ([_]bool{ false, true }) |streaming| for ([_]bool{ false, true }) |tools| {
        var r: fixture.Rig = .{ .eng = .{ .script = &.{"Zig latest release notes"} } };
        try r.start(true);
        defer r.stop();
        r.srv.info.context_window = window;
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const got = try request(r.srv, a, path(route), try body(a, route, streaming, tools), null);
        try t.expect(r.web.searches > 0);
        try t.expectEqual(@as(usize, 1), r.eng.prompts.items.len);
        try t.expectEqual(@as(i64, 0), r.srv.preparing.load(.acquire));
        try t.expectEqual(@as(i64, 0), r.minimum.load(.acquire));
        if (streaming) {
            try t.expectEqual(@as(u16, 200), got.status);
            try terminal(route, got.payload);
        } else {
            try t.expectEqual(@as(u16, 400), got.status);
            const value = (try json.parse(a, got.payload)).ok;
            try t.expect(value.has("error"));
        }
        const after = try request(r.srv, a, "/v1/chat/completions", "{\"model\":\"m\",\"cortheon\":false,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"ordinary\"}]}", null);
        try t.expectEqual(@as(u16, 200), after.status);
        try t.expectEqual(@as(i64, 0), r.srv.preparing.load(.acquire));
        try t.expectEqual(@as(i64, 0), r.minimum.load(.acquire));
    };
}

fn delayed(route: Route) !void {
    var r: fixture.Rig = .{ .eng = .{ .script = &.{"Zig latest release notes"} } };
    try r.start(true);
    defer r.stop();
    r.srv.info.context_window = 2000;
    r.srv.config.sse_keepalive_s = 0.000000001;
    var engine: GateEngine = .{ .original = &r.eng, .at = 0, .hold = true };
    defer engine.join();
    r.srv.engine = engine.engine();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var progress: Progress = .{ .engine = &engine };
    const got = try request(r.srv, a, path(route), try body(a, route, true, false), &progress);
    try t.expect(progress.due.load(.acquire) and progress.released);
    try t.expect(!progress.bad.load(.acquire));
    engine.join();
    try t.expect(r.web.searches > 0);
    try t.expectEqual(@as(usize, 1), engine.submissions);
    try t.expectEqual(@as(usize, 0), engine.cancellations);
    try terminal(route, got.payload);
    try t.expectEqual(@as(i64, 0), r.srv.preparing.load(.acquire));
    try t.expectEqual(@as(i64, 0), r.minimum.load(.acquire));
}

test "context gate: delayed research HTTP chat has its head before keepalive bytes" {
    try delayed(.chat);
}

test "context gate: delayed research HTTP messages has its head before keepalive bytes" {
    try delayed(.messages);
}

test "context gate: delayed research HTTP responses has its head before keepalive bytes" {
    try delayed(.responses);
}

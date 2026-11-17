//! Reply lines in TENSORFOLD_REQUEST_LOG: one per finished request, written off the decode path.
//! The CPU fake engine emits scripted tokens over a loopback socket, so these are real HTTP
//! request/reply pairs with a real request log, not mocks of either side.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const server = @import("server.zig");
const auth = @import("auth.zig");
const listener = @import("listener.zig");
const model_text = @import("model_text.zig");
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const posix = @import("sys").posix;
const t = std.testing;
const Allocator = std.mem.Allocator;

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
        const out = try a.alloc(u8, ids.len);
        for (out, ids) |*byte, id| byte.* = @intCast(id);
        return out;
    }
};

/// Closed until the test opens it: holds the engine between its first token chunk and its
/// terminal chunk so the log can be inspected while a stream is still open.
const Gate = struct {
    open: std.atomic.Value(bool) = .init(false),
};

const Engine = struct {
    gate: ?*Gate = null,
    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn cast(ptr: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ptr));
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096, .name = "fake-model" };
    }
    fn submit(ptr: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = cast(ptr);
        const worker = std.Thread.spawn(.{}, emit, .{ e, id, sink }) catch return error.Busy;
        worker.detach();
    }
    fn emit(e: *Engine, id: api.Id, sink: api.Sink) void {
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 2, 7, 1 } });
        if (e.gate) |g| while (!g.open.load(.acquire)) std.Thread.yield() catch {};
        sink.event(sink.ctx, id, &.{ .tokens = &.{3} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
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
    log_path: ?[]const u8 = null,

    fn start(h: *Harness, engine: *Engine) !void {
        h.srv = try server.Server.init(t.allocator, t.io, engine.engine(), Text.model(), .{
            .served_name = "fake-model",
            .model_ids = &.{"fake-model"},
            .request_log = h.log_path,
            .enable_thinking = false,
            .use_drafts = false,
            .keep_warm_s = 0,
        }, null);
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
            if (waited == 30000) @panic("request-log fixture did not quiesce");
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        h.listener.close();
        h.srv.deinit();
    }
    fn connect(h: *Harness, a: Allocator) !posix.socket_t {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.listener.port()) };
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&address, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.Socket;
        const fd: posix.socket_t = @intCast(rc);
        errdefer _ = posix.system.close(fd);
        if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.Connect;
        _ = a;
        return fd;
    }
    fn send(_: *Harness, fd: posix.socket_t, raw: []const u8) !void {
        var sent: usize = 0;
        while (sent < raw.len) {
            const n = posix.system.write(fd, raw[sent..].ptr, raw.len - sent);
            if (posix.errno(n) != .SUCCESS or n == 0) return error.Write;
            sent += @intCast(n);
        }
    }
    fn ask(h: *Harness, a: Allocator, path: []const u8, body: []const u8) ![]const u8 {
        const raw = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ path, body.len, body });
        const fd = try h.connect(a);
        defer _ = posix.system.close(fd);
        try h.send(fd, raw);
        var out: std.ArrayList(u8) = .empty;
        var buffer: [4096]u8 = undefined;
        while (true) {
            var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
            if (try posix.poll(&fds, 30000) == 0) return error.Timeout;
            const n = posix.system.read(fd, &buffer, buffer.len);
            if (posix.errno(n) != .SUCCESS) return error.Read;
            if (n == 0) return out.items;
            try out.appendSlice(a, buffer[0..@intCast(n)]);
        }
    }
};

const Stream = struct {
    fd: posix.socket_t,
    fn readMore(s: *Stream, a: Allocator) ![]const u8 {
        var fds = [_]posix.pollfd{.{ .fd = s.fd, .events = posix.POLL.IN, .revents = 0 }};
        if (try posix.poll(&fds, 5000) == 0) return error.Timeout;
        var buffer: [4096]u8 = undefined;
        const n = posix.system.read(s.fd, &buffer, buffer.len);
        if (posix.errno(n) != .SUCCESS) return error.Read;
        if (n == 0) return "";
        return a.dupe(u8, buffer[0..@intCast(n)]);
    }
    fn close(s: *Stream) void {
        _ = posix.system.close(s.fd);
    }
};

fn openStream(h: *Harness, a: Allocator, path: []const u8, body: []const u8) !Stream {
    const raw = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ path, body.len, body });
    const fd = try h.connect(a);
    errdefer _ = posix.system.close(fd);
    try h.send(fd, raw);
    return .{ .fd = fd };
}

fn readFrames(s: *Stream, a: Allocator, at_least: usize) ![]const u8 {
    var seen: std.ArrayList(u8) = .empty;
    while (std.mem.count(u8, seen.items, "\n\n") < at_least) {
        const chunk = try s.readMore(a);
        if (chunk.len == 0) return error.EarlyEof;
        try seen.appendSlice(a, chunk);
    }
    return seen.items;
}

fn drain(s: *Stream, a: Allocator) ![]const u8 {
    var seen: std.ArrayList(u8) = .empty;
    while (true) {
        const chunk = try s.readMore(a);
        if (chunk.len == 0) return seen.items;
        try seen.appendSlice(a, chunk);
    }
}

fn logPath(a: Allocator) ![]const u8 {
    const tmp = t.tmpDir(.{});
    // scratch dirs are intentionally left behind: the harness reads this file for the test's life
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
}

fn statusIs(reply: []const u8, code: u16) bool {
    var buf: [32]u8 = undefined;
    const want = std.fmt.bufPrint(&buf, "HTTP/1.1 {d} ", .{code}) catch return false;
    return std.mem.startsWith(u8, reply, want);
}

fn bodyOf(reply: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return "";
    return reply[at + 4 ..];
}

fn logValues(a: Allocator, h: *Harness) ![]json.Value {
    if (h.log_path == null) return &.{};
    const text = std.Io.Dir.cwd().readFileAlloc(t.io, h.log_path.?, a, .limited(1 << 20)) catch return &.{};
    var out: std.ArrayList(json.Value) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const parsed = json.parse(a, line) catch return error.BadLogLine;
        if (parsed != .ok) return error.BadLogLine;
        try out.append(a, parsed.ok);
    }
    return out.items;
}

fn isReply(v: json.Value) bool {
    const k = v.strField("kind") orelse return false;
    return std.mem.eql(u8, k, "reply");
}
fn findReply(vals: []json.Value) ?json.Value {
    for (vals) |v| if (isReply(v)) return v;
    return null;
}
fn findRequest(vals: []json.Value) ?json.Value {
    for (vals) |v| if (!isReply(v)) return v;
    return null;
}
fn waitReply(a: Allocator, h: *Harness) !json.Value {
    for (0..2000) |_| {
        if (findReply(try logValues(a, h))) |v| return v;
        std.Io.sleep(t.io, .fromMilliseconds(2), .awake) catch {};
    }
    return error.NoReplyLine;
}

/// Shared assertions for a completed 200 request/reply pair.
fn checkPair(a: Allocator, h: *Harness, route: []const u8, prefix: []const u8, want_text: ?[]const u8) !void {
    const vals = try logValues(a, h);
    const req = findRequest(vals) orelse return error.NoRequestLine;
    const rep = findReply(vals) orelse return error.NoReplyLine;
    const id = req.strField("id") orelse return error.NoRequestId; // fails on main: no id on request lines
    try t.expect(std.mem.startsWith(u8, id, prefix));
    try t.expectEqualStrings(id, rep.strField("id") orelse "");
    try t.expectEqualStrings(route, rep.strField("route") orelse "");
    try t.expectEqualStrings("fake-model", rep.strField("model") orelse "");
    try t.expectEqual(@as(i64, 200), rep.get("status").?.int64().?);
    const got_text = rep.strField("text") orelse return error.NoReplyText;
    try t.expect(got_text.len > 0);
    if (want_text) |w| try t.expectEqualStrings(w, got_text);
    try t.expectEqualStrings("stop", rep.strField("finish_reason") orelse "");
    try t.expect(rep.get("error") == null or rep.get("error").? == .null);
    const usage = rep.get("usage").?;
    try t.expect(usage.get("prompt_tokens") != null);
    try t.expect(usage.get("cached_tokens") != null);
    try t.expect(usage.get("completion_tokens") != null);
    const timings = rep.get("timings").?;
    for ([_][]const u8{ "queued", "ttft", "decode_tps", "total" }) |k| try t.expect(timings.get(k) != null);
}

fn chatContent(a: Allocator, reply: []const u8) ![]const u8 {
    const parsed = json.parse(a, bodyOf(reply)) catch return error.BadJson;
    if (parsed != .ok) return error.BadJson;
    const choices = parsed.ok.get("choices") orelse return error.BadJson;
    const msg = choices.array[0].get("message") orelse return error.BadJson;
    return msg.strField("content") orelse "";
}

const chat_body = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"stream\":false,\"options\":{\"temperature\":0,\"num_predict\":8}}";
const completion_body = "{\"model\":\"fake-model\",\"prompt\":\"H\",\"max_tokens\":8,\"temperature\":0}";
const messages_body = "{\"model\":\"fake-model\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"H\"}]}";
const responses_body = "{\"model\":\"fake-model\",\"input\":\"H\",\"max_output_tokens\":8}";
const ollama_body = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"stream\":false,\"options\":{\"temperature\":0,\"num_predict\":8}}";
const chat_stream = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"stream\":true,\"max_tokens\":8}";

test "request log: OpenAI chat and completions each write a request/reply pair" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]struct { path: []const u8, body: []const u8, route: []const u8, prefix: []const u8 }{
        .{ .path = "/v1/chat/completions", .body = chat_body, .route = "chat", .prefix = "chatcmpl-" },
        .{ .path = "/v1/completions", .body = completion_body, .route = "completions", .prefix = "cmpl-" },
    }) |c| {
        var engine: Engine = .{};
        var h: Harness = .{ .log_path = try logPath(a) };
        try h.start(&engine);
        defer h.deinit();
        const reply = try h.ask(a, c.path, c.body);
        try t.expect(statusIs(reply, 200));
        const want: ?[]const u8 = if (std.mem.eql(u8, c.route, "chat")) try chatContent(a, reply) else null;
        try checkPair(a, &h, c.route, c.prefix, want);
    }
}

test "request log: Anthropic, Responses and Ollama each write a request/reply pair" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]struct { path: []const u8, body: []const u8, route: []const u8, prefix: []const u8 }{
        .{ .path = "/v1/messages", .body = messages_body, .route = "messages", .prefix = "msg_" },
        .{ .path = "/v1/responses", .body = responses_body, .route = "responses", .prefix = "resp_" },
        .{ .path = "/api/chat", .body = ollama_body, .route = "ollama-chat", .prefix = "chatcmpl-" },
    }) |c| {
        var engine: Engine = .{};
        var h: Harness = .{ .log_path = try logPath(a) };
        try h.start(&engine);
        defer h.deinit();
        const reply = try h.ask(a, c.path, c.body);
        try t.expect(statusIs(reply, 200));
        try checkPair(a, &h, c.route, c.prefix, null);
    }
}

test "request log: a streamed reply line appears only after the terminal chunk" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var gate: Gate = .{};
    var engine: Engine = .{ .gate = &gate };
    var h: Harness = .{ .log_path = try logPath(a) };
    try h.start(&engine);
    defer h.deinit();
    defer gate.open.store(true, .release); // never leave the engine worker blocked

    var s = try openStream(&h, a, "/v1/chat/completions", chat_stream);
    defer s.close();
    const head = try readFrames(&s, a, 1); // one SSE frame: the stream is still open
    try t.expect(std.mem.indexOf(u8, head, "data:") != null);
    try t.expect(std.mem.indexOf(u8, head, "[DONE]") == null);

    // the engine is gated mid-stream: the request line exists, the reply line must not yet
    const mid = try logValues(a, &h);
    try t.expect(findRequest(mid) != null);
    try t.expect(findReply(mid) == null); // fails on main for the wrong reason; still proves no mid-stream write

    gate.open.store(true, .release);
    const tail = try drain(&s, a);
    try t.expect(std.mem.indexOf(u8, tail, "[DONE]") != null); // terminal chunk reached the client

    const rep = try waitReply(a, &h);
    try t.expectEqualStrings("chat", rep.strField("route") orelse "");
    try t.expectEqual(@as(i64, 200), rep.get("status").?.int64().?);
    try t.expectEqualStrings("stop", rep.strField("finish_reason") orelse "");
}

test "request log: a cancelled stream still writes a reply line" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var gate: Gate = .{};
    var engine: Engine = .{ .gate = &gate };
    var h: Harness = .{ .log_path = try logPath(a) };
    try h.start(&engine);
    defer h.deinit();
    defer gate.open.store(true, .release);

    var s = try openStream(&h, a, "/v1/chat/completions", chat_stream);
    _ = try readFrames(&s, a, 1);
    s.close(); // the client leaves mid-stream
    gate.open.store(true, .release);

    const rep = try waitReply(a, &h);
    try t.expectEqual(@as(i64, 0), rep.get("status").?.int64().?);
    try t.expectEqualStrings("cancelled", rep.strField("finish_reason") orelse "");
    try t.expect(rep.strField("id") != null);
}

test "request log: a refused request writes a request line and an error reply line" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{ .log_path = try logPath(a) };
    try h.start(&engine);
    defer h.deinit();

    // refused inside plan(), after the request line is written
    const bad = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"parallel_tool_calls\":\"yes\"}";
    const reply = try h.ask(a, "/v1/chat/completions", bad);
    try t.expect(statusIs(reply, 400));
    const rep = try waitReply(a, &h);
    try t.expectEqualStrings("error", rep.strField("finish_reason") orelse "");
    try t.expectEqual(@as(i64, 400), rep.get("status").?.int64().?);
    try t.expect((rep.strField("error") orelse @as([]const u8, "")).len > 0);
    try t.expect(findRequest(try logValues(a, &h)) != null);
}

fn normalizedBody(a: Allocator, reply: []const u8) ![]const u8 {
    const parsed = json.parse(a, bodyOf(reply)) catch return error.BadJson;
    if (parsed != .ok) return error.BadJson;
    var v = parsed.ok;
    if (v == .object) {
        _ = v.object.swapRemove("id"); // random per run
        _ = v.object.swapRemove("created"); // wall clock
        if (v.object.getPtr("tensorfold")) |tf| if (tf.* == .object) {
            for ([_][]const u8{ "seconds", "tokens_per_second", "time_to_first_token", "prefill_seconds" }) |key| _ = tf.object.swapRemove(key);
        };
    }
    return json.stringify(a, v, .{});
}

test "request log: reply bytes are identical with the log on and off" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var plain_engine: Engine = .{};
    var plain: Harness = .{};
    try plain.start(&plain_engine);
    defer plain.deinit();
    const off = try plain.ask(a, "/v1/chat/completions", chat_body);

    var logged_engine: Engine = .{};
    var logged: Harness = .{ .log_path = try logPath(a) };
    try logged.start(&logged_engine);
    defer logged.deinit();
    const on = try logged.ask(a, "/v1/chat/completions", chat_body);

    try t.expectEqualStrings(off[0 .. std.mem.indexOf(u8, off, "\r\n").?], on[0 .. std.mem.indexOf(u8, on, "\r\n").?]);
    try t.expectEqualStrings(try normalizedBody(a, off), try normalizedBody(a, on));
    // with the log off there is no log and no reply line to write
    try t.expectEqual(@as(usize, 0), (try logValues(a, &plain)).len);
}

test "request log: a reply line is present at all (fails against the unmodified server)" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{ .log_path = try logPath(a) };
    try h.start(&engine);
    defer h.deinit();
    try t.expect(statusIs(try h.ask(a, "/v1/chat/completions", chat_body), 200));
    const rep = findReply(try logValues(a, &h)) orelse return error.NoReplyLine;
    try t.expect(isReply(rep));
    try t.expect((rep.strField("kind") orelse @as([]const u8, "")).len > 0);
}

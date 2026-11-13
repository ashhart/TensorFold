//! ``return_progress`` on the wire: chat, completions and Responses streams, and the requests that never send it.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const http = @import("http_conn.zig");
const Server = @import("server.zig").Server;
const scan = @import("stream_scan_text.zig");
const A = std.mem.Allocator;
const V = json.Value;

const steps = [_]u32{ 8, 16, 32 };

/// A 32-row prompt with 4 rows restored; a paced progress request moves on a count each time the client reads one.
const Script = struct {
    paced: bool = true, // false: every count and the reply at once, so no read can hold the engine back
    asked: ?bool = null,
    sink: ?api.Sink = null,
    id: api.Id = 0,
    stage: usize = 0,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768, .reasoning = false };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.* = .{ .paced = s.paced, .asked = request.return_progress, .sink = sink, .id = id };
        if (!request.return_progress) s.stage = steps.len;
        s.next();
        if (!s.paced) while (s.stage <= steps.len) s.next();
    }
    /// The next count, or after the last the reply: the prefilled event, its tokens and its end.
    fn next(s: *Script) void {
        const sink = s.sink orelse return;
        defer s.stage += 1;
        if (s.stage < steps.len) return sink.event(sink.ctx, s.id, &.{ .progress = .{ .done = steps[s.stage], .total = 32, .cached = 4 } });
        if (s.stage > steps.len) return;
        sink.event(sink.ctx, s.id, &.{ .prefilled = 4 });
        sink.event(sink.ctx, s.id, &.{ .tokens = &.{ 'O', 'K' } });
        sink.event(sink.ctx, s.id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The client's bytes; each progress frame it reads lets the engine go on, as does a quiet stream's keepalive.
const Capture = struct {
    a: A,
    script: *Script,
    bytes: std.ArrayList(u8) = .empty,
    seen: usize = 0,
    fn write(ptr: *anyopaque, bytes: []const u8) error{Closed}!void {
        const c: *Capture = @ptrCast(@alignCast(ptr));
        c.bytes.appendSlice(c.a, bytes) catch return error.Closed;
        if (std.mem.eql(u8, bytes, ": keepalive\n\n")) return c.script.next(); // no count came: fail, not hang
        const counts = std.mem.count(u8, c.bytes.items, "\"done\":");
        while (c.seen < counts) : (c.seen += 1) c.script.next();
    }
};

const Route = enum { chat, completions, responses, messages };

const Frame = struct { event: ?[]const u8, data: V };

/// What one request's client read: its frames (null event: a plain ``data:`` frame), or the whole body.
fn post(a: A, script: *Script, route: Route, request: []const u8) ![]Frame {
    var plain: scan.Text = .{};
    const srv = try Server.init(a, std.testing.io, script.engine(), plain.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .sse_keepalive_s = 2 }, null);
    defer srv.deinit();
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{request.len}) }};
    var capture: Capture = .{ .a = a, .script = script };
    const path = if (route == .messages) "/v1/messages" else "";
    var conn: http.Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, request), .end = request.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = path, .headers = &headers, .write_hooks = .{ .ctx = &capture, .write = Capture.write } };
    switch (route) {
        .chat, .completions => @import("openai.zig").post(srv, &conn, a, route == .chat),
        .responses => @import("responses.zig").post(srv, &conn, a),
        .messages => @import("anthropic.zig").post(srv, &conn, a),
    }
    const wire = capture.bytes.items;
    try std.testing.expect(std.mem.startsWith(u8, wire, "HTTP/1.1 200"));
    const payload = wire[(std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.NoHeaders) + 4 ..];
    var out: std.ArrayList(Frame) = .empty;
    if (!std.mem.startsWith(u8, payload, "event: ") and !std.mem.startsWith(u8, payload, "data: ")) {
        try out.append(a, .{ .event = null, .data = (try json.parseText(a, payload)).ok });
        return out.items;
    }
    var name: ?[]const u8 = null;
    var lines = std.mem.splitScalar(u8, payload, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "event: ")) name = line[7..];
        if (!std.mem.startsWith(u8, line, "data: ") or std.mem.eql(u8, line[6..], "[DONE]")) continue;
        try out.append(a, .{ .event = name, .data = (try json.parseText(a, line[6..])).ok });
        name = null;
    }
    return out.items;
}

fn body(a: A, route: Route, stream: bool, asked: ?bool) ![]const u8 {
    const head = switch (route) {
        .chat => "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":8",
        .completions => "{\"model\":\"m\",\"prompt\":\"x\",\"max_tokens\":8",
        .responses => "{\"model\":\"m\",\"input\":\"x\",\"store\":false,\"max_output_tokens\":8",
        .messages => "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":8",
    };
    const flag = if (asked) |on| (if (on) ",\"return_progress\":true" else ",\"return_progress\":false") else "";
    return std.fmt.allocPrint(a, "{s},\"stream\":{s}{s}}}", .{ head, if (stream) "true" else "false", flag });
}

/// A frame's progress counts: a chunk's ``tensorfold.progress``, or a Responses ``tensorfold.progress`` event.
fn countsOf(f: Frame) ?V {
    if (f.event) |e| return if (std.mem.eql(u8, e, "tensorfold.progress")) f.data else null;
    const block = f.data.get("tensorfold") orelse return null;
    return block.get("progress");
}

/// The reply text a frame adds: a chat delta's content, a completion's text or a Responses text delta.
fn textOf(f: Frame) []const u8 {
    if (f.event) |e| return if (std.mem.eql(u8, e, "response.output_text.delta")) json.strOr(f.data.get("delta")) else "";
    const choices = f.data.get("choices") orelse return "";
    if (choices != .array or choices.array.len == 0) return "";
    const c = choices.array[0];
    return if (c.get("delta")) |d| json.strOr(d.get("content")) else json.strOr(c.get("text"));
}

test "prompt progress: each count streams once, in order, before the reply's first text, which it leaves unchanged" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Route{ .chat, .completions, .responses }) |route| {
        var texts: [2][]const u8 = undefined;
        for ([_]bool{ false, true }, &texts) |asked, *text| {
            var script: Script = .{};
            const frames = try post(a, &script, route, try body(a, route, true, if (asked) true else null));
            try std.testing.expectEqual(@as(?bool, asked), script.asked);
            var shown: std.ArrayList(i64) = .empty;
            var reply: std.ArrayList(u8) = .empty;
            for (frames) |f| {
                if (countsOf(f)) |c| {
                    try std.testing.expectEqual(@as(usize, 0), reply.items.len); // every count before any text
                    try std.testing.expectEqual(@as(?i64, 32), c.get("total").?.int64());
                    try std.testing.expectEqual(@as(?i64, 4), c.get("cached").?.int64());
                    try shown.append(a, c.get("done").?.int64().?);
                    if (f.event == null) {
                        try std.testing.expectEqual(@as(usize, 0), f.data.get("choices").?.array.len);
                        try std.testing.expectEqualStrings(json.strOr(frames[0].data.get("id")), json.strOr(f.data.get("id")));
                    }
                }
                try reply.appendSlice(a, textOf(f));
            }
            const want: []const i64 = if (asked) &.{ 8, 16, 32 } else &.{};
            try std.testing.expectEqualSlices(i64, want, shown.items);
            text.* = reply.items;
        }
        try std.testing.expectEqualStrings("OK", texts[0]);
        try std.testing.expectEqualStrings(texts[0], texts[1]);
    }
}

test "prompt progress: a whole reply, a false flag and the Messages route ask the engine for none" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { Route, bool, ?bool }{ .{ .chat, false, true }, .{ .responses, false, true }, .{ .chat, true, false }, .{ .messages, true, true } };
    for (cases) |c| {
        var script: Script = .{ .paced = false };
        const frames = try post(a, &script, c[0], try body(a, c[0], c[1], c[2]));
        try std.testing.expectEqual(@as(?bool, false), script.asked);
        for (frames) |f| try std.testing.expect(countsOf(f) == null);
    }
}

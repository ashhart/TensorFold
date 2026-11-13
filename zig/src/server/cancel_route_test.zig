//! POST /v1/requests/{id}/cancel over a one-lane engine: mid-decode, queued, Responses, unknown ids and who may ask.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const routes = @import("routes.zig");
const cancel_ids = @import("cancel_ids.zig");
const Server = @import("server.zig").Server;
const http = @import("http_conn.zig");
const Conn = http.Conn;
const Allocator = std.mem.Allocator;
const t = std.testing;

/// Bytes as tokens, as completion_prompt_test.zig's.
const ByteText = struct {
    fn text(b: *@This()) model_text.Text {
        return .{ .ctx = b, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try (a.dupe(u8, "prompt")) };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// One lane, a token a millisecond, others queued; a cancel ends the running one at its next token, a queued at once.
const Lane = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    jobs: std.ArrayList(Job) = .empty,
    asked: std.ArrayList(api.Id) = .empty,
    stop: std.atomic.Value(bool) = .init(false),
    cancels: std.atomic.Value(usize) = .init(0),
    worker: ?std.Thread = null,

    const Job = struct { id: api.Id, sink: api.Sink, max: u32, first: u32 };

    fn start(l: *Lane) !void {
        l.worker = try std.Thread.spawn(.{}, run, .{l});
    }

    fn deinit(l: *Lane) void {
        l.stop.store(true, .release);
        if (l.worker) |w| w.join();
        l.jobs.deinit(t.allocator);
        l.asked.deinit(t.allocator);
    }

    fn engine(l: *Lane) api.Engine {
        return .{ .ctx = l, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const l: *Lane = @ptrCast(@alignCast(ctx));
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        l.jobs.append(t.allocator, .{ .id = id, .sink = sink, .max = req.max_tokens, .first = req.prompt[0] }) catch return error.Busy;
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const l: *Lane = @ptrCast(@alignCast(ctx));
        _ = l.cancels.fetchAdd(1, .monotonic);
        l.mutex.lockUncancelable(l.io);
        // a queued request ends at once; the running one (the first) at its next token
        for (l.jobs.items[@min(1, l.jobs.items.len)..], 1..) |j, i| if (j.id == id) {
            const gone = l.jobs.orderedRemove(i);
            l.mutex.unlock(l.io);
            gone.sink.event(gone.sink.ctx, gone.id, &.{ .finished = .{ .reason = .cancelled } });
            return;
        };
        l.asked.append(t.allocator, id) catch {};
        l.mutex.unlock(l.io);
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn queued(l: *Lane, n: usize) !bool {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        return l.jobs.items.len >= n;
    }

    fn askedFor(l: *Lane, id: api.Id) bool {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        return std.mem.indexOfScalar(api.Id, l.asked.items, id) != null;
    }

    fn run(l: *Lane) void {
        while (!l.stop.load(.acquire)) {
            l.mutex.lockUncancelable(l.io);
            const job: ?Job = if (l.jobs.items.len > 0) l.jobs.items[0] else null;
            l.mutex.unlock(l.io);
            const j = job orelse {
                std.Io.sleep(l.io, .fromMilliseconds(1), .awake) catch {};
                continue;
            };
            j.sink.event(j.sink.ctx, j.id, &.{ .prefilled = 0 });
            var reason: api.Reason = .length;
            for (0..j.max) |i| {
                if (l.askedFor(j.id)) {
                    reason = .cancelled;
                    break;
                }
                const tok: u32 = if (i == 0) j.first else 'a' + @as(u32, @intCast(i % 26));
                j.sink.event(j.sink.ctx, j.id, &.{ .tokens = &.{tok} });
                std.Io.sleep(l.io, .fromMilliseconds(1), .awake) catch {};
            }
            j.sink.event(j.sink.ctx, j.id, &.{ .finished = .{ .reason = reason } });
            l.mutex.lockUncancelable(l.io);
            _ = l.jobs.orderedRemove(0);
            l.mutex.unlock(l.io);
        }
    }
};

/// A stream's events as text, read by the test while the request runs on its own thread.
const Stream = struct {
    mutex: std.Io.Mutex = .init,
    arena: std.heap.ArenaAllocator, // the events' text, written under `mutex`
    events: std.ArrayList([]const u8) = .empty,

    fn deinit(s: *Stream) void {
        s.arena.deinit();
    }

    fn out(s: *Stream) openai.Out {
        return .{ .ctx = s, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const s: *Stream = @ptrCast(@alignCast(ctx));
        s.mutex.lockUncancelable(t.io);
        defer s.mutex.unlock(t.io);
        const a = s.arena.allocator();
        const text = if (value) |v| json.stringify(a, v, .{ .ascii = false }) catch return error.Closed else "[DONE]";
        s.events.append(a, text) catch return error.Closed;
    }
    fn reply(_: *anyopaque, _: u16, _: json.Value) void {}

    /// The stream's id once its first chunk is out, else null.
    fn id(s: *Stream, a: Allocator) !?[]const u8 {
        s.mutex.lockUncancelable(t.io);
        defer s.mutex.unlock(t.io);
        if (s.events.items.len == 0) return null;
        return try a.dupe(u8, (try json.parse(a, s.events.items[0])).ok.strField("id").?);
    }

    /// The streamed text and the finish reason, once the stream ended.
    fn result(s: *Stream, a: Allocator) !struct { text: []const u8, finish: []const u8 } {
        s.mutex.lockUncancelable(t.io);
        defer s.mutex.unlock(t.io);
        var text: std.ArrayList(u8) = .empty;
        var finish: []const u8 = "";
        for (s.events.items) |e| {
            if (std.mem.eql(u8, e, "[DONE]")) continue;
            const v = (try json.parse(a, e)).ok;
            const l = v.get("choices").?.array;
            if (l.len == 0) continue;
            try text.appendSlice(a, l[0].strField("text") orelse "");
            if (l[0].strField("finish_reason")) |f| finish = f;
        }
        return .{ .text = text.items, .finish = finish };
    }
};

/// One streamed completion, on the caller's thread, in its own arena.
fn request(srv: *Server, s: *Stream, prompt: []const u8, max: u32) void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (json.parse(a, std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":{d},\"prompt\":\"{s}\",\"stream\":true}}", .{ max, prompt }) catch return) catch return).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    openai.run(srv, a, s.out(), .{ .conn = &conn }, false, raw);
}

fn waitFor(comptime f: anytype, args: anytype) !void {
    for (0..5000) |_| {
        if (try @call(.auto, f, args)) return;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

fn hasId(s: *Stream, a: Allocator) !bool {
    return (try s.id(a)) != null;
}

/// The held names that start with `head`, other than `skip`; the first, or null.
fn heldName(a: Allocator, head: []const u8, skip: []const u8) !?[]const u8 {
    for (try cancel_ids.registry.held(t.io, a)) |h| if (std.mem.startsWith(u8, h, head) and !std.mem.eql(u8, h, skip)) return h;
    return null;
}

fn holds(a: Allocator, head: []const u8) !bool {
    return (try heldName(a, head, "")) != null;
}

fn serve(lane: *Lane) !*Server {
    const S = struct {
        var text: ByteText = .{};
    };
    return Server.init(t.allocator, t.io, lane.engine(), S.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
}

/// The bytes a route writes, in place of a socket.
const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

const Answer = struct { status: u16, body: json.Value };

/// One request through the dispatcher from `peer`, as the connection thread hands it over after the key gate.
fn call(srv: *Server, a: Allocator, path: []const u8, peer: []const u8, extra: []const http.Header, body: []const u8) !Answer {
    var wire: Wire = .{ .a = a };
    const headers = try std.mem.concat(a, http.Header, &.{ &.{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }}, extra });
    var c: Conn = .{ .fd = -1, .peer = peer, .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = path, .headers = headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    routes.dispatch(srv, &c, a);
    const out = wire.bytes.items;
    const at = std.mem.indexOf(u8, out, "\r\n\r\n") orelse return error.MissingHttpHead;
    const status = try std.fmt.parseInt(u16, out["HTTP/1.1 ".len..][0..3], 10);
    return .{ .status = status, .body = (try json.parse(a, out[at + 4 ..])).ok };
}

fn cancelBy(srv: *Server, a: Allocator, id: []const u8) !Answer {
    return call(srv, a, try std.fmt.allocPrint(a, "/v1/requests/{s}/cancel", .{id}), "127.0.0.1", &.{}, "");
}

test "request cancel: mid-decode, the stream ends cancelled through the engine's cancel with its tokens so far" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    var s: Stream = .{ .arena = .init(t.allocator) };
    defer s.deinit();
    const th = try std.Thread.spawn(.{}, request, .{ srv, &s, "hi", 100000 });
    try waitFor(hasId, .{ &s, a });
    const id = (try s.id(a)).?;
    const got = try cancelBy(srv, a, id);
    try t.expectEqual(@as(u16, 200), got.status);
    try t.expectEqualStrings(id, got.body.strField("id").?);
    try t.expectEqualStrings("request.cancel", got.body.strField("object").?);
    th.join();
    const r = try s.result(a);
    try t.expectEqualStrings("cancelled", r.finish);
    try t.expect(r.text.len > 0 and r.text.len < 100000);
    try t.expectEqual(@as(usize, 1), lane.cancels.load(.monotonic));
    // ended, its id is no longer held
    try t.expectEqual(@as(u16, 404), (try cancelBy(srv, a, id)).status);
}

test "request cancel: a queued request ends at once with no tokens, and the running stream's tokens are unchanged" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    // the running stream alone, for its tokens
    var solo: Stream = .{ .arena = .init(t.allocator) };
    defer solo.deinit();
    request(srv, &solo, "go", 300);
    const want = try solo.result(a);
    try t.expectEqualStrings("length", want.finish);
    var first: Stream = .{ .arena = .init(t.allocator) };
    defer first.deinit();
    var second: Stream = .{ .arena = .init(t.allocator) };
    defer second.deinit();
    const t1 = try std.Thread.spawn(.{}, request, .{ srv, &first, "go", 300 });
    try waitFor(hasId, .{ &first, a });
    const t2 = try std.Thread.spawn(.{}, request, .{ srv, &second, "wait", 300 });
    try waitFor(Lane.queued, .{ &lane, 2 }); // the engine holds the second behind the first
    const first_id = (try first.id(a)).?;
    const second_id = (try heldName(a, "cmpl-", first_id)) orelse return error.TestUnexpectedResult;
    try t.expectEqual(@as(u16, 200), (try cancelBy(srv, a, second_id)).status);
    t2.join();
    const cancelled = try second.result(a);
    try t.expectEqualStrings("cancelled", cancelled.finish);
    try t.expectEqualStrings("", cancelled.text);
    t1.join();
    const kept = try first.result(a);
    try t.expectEqualStrings(want.finish, kept.finish);
    try t.expectEqualStrings(want.text, kept.text);
    try t.expectEqual(@as(usize, 1), lane.cancels.load(.monotonic));
}

/// One whole Responses turn through the dispatcher, its HTTP answer kept for the test.
fn respond(srv: *Server, a: Allocator, out: *?Answer) void {
    const body = "{\"model\":\"m\",\"input\":\"hi\",\"max_output_tokens\":100000,\"store\":false}";
    out.* = call(srv, a, "/v1/responses", "127.0.0.1", &.{}, body) catch null;
}

test "request cancel: a Responses turn cancelled by its resp_ id ends with status cancelled and its text so far" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var thread_arena = std.heap.ArenaAllocator.init(t.allocator);
    defer thread_arena.deinit();
    var lane: Lane = .{ .io = t.io };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    var answer: ?Answer = null;
    const th = try std.Thread.spawn(.{}, respond, .{ srv, thread_arena.allocator(), &answer });
    try waitFor(holds, .{ a, "resp_" });
    try waitFor(Lane.queued, .{ &lane, 1 }); // submitted, so the cancel reaches the engine
    const rid = (try heldName(a, "resp_", "")).?;
    try t.expectEqual(@as(u16, 200), (try cancelBy(srv, a, rid)).status);
    th.join();
    const got = answer orelse return error.TestUnexpectedResult;
    try t.expectEqual(@as(u16, 200), got.status);
    try t.expectEqualStrings(rid, got.body.strField("id").?);
    try t.expectEqualStrings("cancelled", got.body.strField("status").?);
    try t.expect(got.body.get("completed_at") == null);
    const item = got.body.get("output").?.array[0];
    try t.expectEqualStrings("incomplete", item.strField("status").?);
    try t.expect(item.get("content").?.array[0].strField("text").?.len > 0);
    try t.expectEqual(@as(u16, 404), (try cancelBy(srv, a, rid)).status);
}

test "request cancel: an unknown id answers 404, and without keys only a direct loopback caller may ask" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io };
    const srv = try serve(&lane);
    defer srv.deinit();
    const unknown = try cancelBy(srv, a, "cmpl-nobody");
    try t.expectEqual(@as(u16, 404), unknown.status);
    try t.expectEqualStrings("request_not_found", unknown.body.get("error").?.strField("code").?);
    var e: cancel_ids.Entry = .{};
    const names = [_][]const u8{"chatcmpl-route-held"};
    cancel_ids.registry.add(t.io, &names, &e);
    defer cancel_ids.registry.remove(t.io, &names, &e);
    const path = "/v1/requests/chatcmpl-route-held/cancel";
    const proxied = [_]http.Header{.{ .name = "X-Forwarded-For", .value = "127.0.0.1" }};
    for ([_]struct { []const u8, []const http.Header }{ .{ "10.1.2.3", &.{} }, .{ "2001:db8::1", &.{} }, .{ "127.0.0.1", &proxied } }) |c| {
        const refused = try call(srv, a, path, c[0], c[1], "{}");
        try t.expectEqual(@as(u16, 403), refused.status);
        try t.expectEqualStrings("loopback_only", refused.body.get("error").?.strField("code").?);
        try t.expect(!e.asked.load(.acquire));
    }
    try t.expectEqual(@as(u16, 200), (try call(srv, a, path, "::1", &.{}, "{}")).status);
    try t.expect(e.asked.load(.acquire));
}

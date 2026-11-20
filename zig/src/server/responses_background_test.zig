//! Background Responses through the dispatcher over a one-lane engine: queued, polled, completed, cancelled, refused.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const routes = @import("routes.zig");
const auth_scope = @import("auth_scope.zig");
const Server = @import("server.zig").Server;
const http = @import("http_conn.zig");
const TestText = @import("drain_test.zig").TestText;
const Allocator = std.mem.Allocator;
const t = std.testing;

/// One lane, a token a millisecond from the prompt and seed, others queued; up to 64 asked stop at half, more run out.
const Lane = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    output_ready: std.Io.Condition = .init,
    release_ready: std.Io.Condition = .init,
    first_tokens: usize = 0,
    hold_first: bool = false,
    release_first: bool = false,
    jobs: std.ArrayList(Job) = .empty,
    asked: std.ArrayList(api.Id) = .empty,
    taken: std.ArrayList([]const u8) = .empty, // each request's prompt, max_tokens and sampling, as text
    arena: std.heap.ArenaAllocator,
    stop: std.atomic.Value(bool) = .init(false),
    cancels: std.atomic.Value(usize) = .init(0),
    worker: ?std.Thread = null,

    const Job = struct { id: api.Id, sink: api.Sink, max: u32, salt: u64 };

    fn start(l: *Lane) !void {
        l.worker = try std.Thread.spawn(.{}, run, .{l});
    }

    fn deinit(l: *Lane) void {
        l.stop.store(true, .release);
        l.mutex.lockUncancelable(l.io);
        l.release_first = true;
        l.release_ready.broadcast(l.io);
        l.mutex.unlock(l.io);
        if (l.worker) |w| w.join();
        l.jobs.deinit(t.allocator);
        l.asked.deinit(t.allocator);
        l.taken.deinit(t.allocator);
        l.arena.deinit();
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
        var salt: u64 = if (req.sampling) |s| s.seed else 0;
        for (req.prompt) |p| salt +%= p;
        const shown = std.fmt.allocPrint(l.arena.allocator(), "{any} {d} {any} {}", .{ req.prompt, req.max_tokens, req.sampling, req.background }) catch return error.Busy;
        l.taken.append(t.allocator, shown) catch return error.Busy;
        l.jobs.append(t.allocator, .{ .id = id, .sink = sink, .max = req.max_tokens, .salt = salt }) catch return error.Busy;
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const l: *Lane = @ptrCast(@alignCast(ctx));
        _ = l.cancels.fetchAdd(1, .monotonic);
        l.mutex.lockUncancelable(l.io);
        for (l.jobs.items[@min(1, l.jobs.items.len)..], 1..) |j, i| if (j.id == id) {
            const gone = l.jobs.orderedRemove(i);
            l.mutex.unlock(l.io);
            gone.sink.event(gone.sink.ctx, gone.id, &.{ .finished = .{ .reason = .cancelled } });
            return;
        };
        l.asked.append(t.allocator, id) catch {};
        l.release_first = true;
        l.release_ready.broadcast(l.io);
        l.mutex.unlock(l.io);
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn submitted(l: *Lane, n: usize) !bool {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        return l.taken.items.len >= n;
    }

    fn releaseFirst(l: *Lane) void {
        l.mutex.lockUncancelable(l.io);
        l.release_first = true;
        l.release_ready.broadcast(l.io);
        l.mutex.unlock(l.io);
    }

    fn waitFirstTokens(l: *Lane, n: usize) !void {
        const timeout: std.Io.Timeout = .{ .duration = .{ .raw = .fromSeconds(5), .clock = .awake } };
        const deadline = timeout.toDeadline(l.io);
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        while (l.first_tokens < n) l.output_ready.waitTimeout(l.io, &l.mutex, deadline) catch return error.Timeout;
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
            if (l.hold_first) {
                l.mutex.lockUncancelable(l.io);
                while (!l.release_first and !l.stop.load(.acquire)) l.release_ready.waitUncancelable(l.io, &l.mutex);
                l.mutex.unlock(l.io);
            }
            var reason: api.Reason = if (j.max <= 64) .stop else .length;
            for (0..if (j.max <= 64) j.max / 2 else j.max) |i| {
                if (l.askedFor(j.id)) {
                    reason = .cancelled;
                    break;
                }
                const tok: u32 = 'a' + @as(u32, @intCast((j.salt +% i) % 26));
                j.sink.event(j.sink.ctx, j.id, &.{ .tokens = &.{tok} });
                if (i == 0) {
                    l.mutex.lockUncancelable(l.io);
                    l.first_tokens += 1;
                    l.output_ready.broadcast(l.io);
                    l.mutex.unlock(l.io);
                }
                std.Io.sleep(l.io, .fromMilliseconds(1), .awake) catch {};
            }
            j.sink.event(j.sink.ctx, j.id, &.{ .finished = .{ .reason = reason } });
            l.mutex.lockUncancelable(l.io);
            _ = l.jobs.orderedRemove(0);
            l.mutex.unlock(l.io);
        }
    }
};

var text: TestText = .{};

fn serve(lane: *Lane) !*Server {
    return Server.init(t.allocator, t.io, lane.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
}

const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

const Answer = struct { status: u16, body: json.Value };

/// One request through the dispatcher from a loopback peer, as the connection thread hands it over after the key gate.
fn send(srv: *Server, a: Allocator, method: []const u8, path: []const u8, body: []const u8, key: ?u8) !Answer {
    var wire: Wire = .{ .a = a };
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var c: http.Conn = .{ .fd = -1, .peer = "127.0.0.1", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .version = "HTTP/1.1", .method = method, .path = path, .headers = &headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    if (key) |k| c.key_digest = @splat(k);
    routes.dispatch(srv, &c, a);
    const out = wire.bytes.items;
    const at = std.mem.indexOf(u8, out, "\r\n\r\n") orelse return error.MissingHttpHead;
    return .{ .status = try std.fmt.parseInt(u16, out["HTTP/1.1 ".len..][0..3], 10), .body = (try json.parse(a, out[at + 4 ..])).ok };
}
const RawAnswer = struct { status: u16, body: []const u8 };

fn sendRaw(srv: *Server, a: Allocator, path: []const u8, key: ?u8) !RawAnswer {
    var wire: Wire = .{ .a = a };
    var c: http.Conn = .{ .fd = -1, .peer = "127.0.0.1", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "GET", .path = path, .headers = &.{}, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    if (key) |k| c.key_digest = @splat(k);
    routes.dispatch(srv, &c, a);
    const out = wire.bytes.items;
    const at = std.mem.indexOf(u8, out, "\r\n\r\n") orelse return error.MissingHttpHead;
    return .{ .status = try std.fmt.parseInt(u16, out["HTTP/1.1 ".len..][0..3], 10), .body = out[at + 4 ..] };
}

fn events(a: Allocator, body: []const u8) !std.ArrayList(json.Value) {
    var result: std.ArrayList(json.Value) = .empty;
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        try result.append(a, (try json.parse(a, line["data: ".len..])).ok);
    }
    return result;
}

fn eventSequences(a: Allocator, body: []const u8) !std.ArrayList(u64) {
    var result: std.ArrayList(u64) = .empty;
    for ((try events(a, body)).items) |one| {
        const number = one.get("sequence_number") orelse return error.MissingSequence;
        try result.append(a, try std.fmt.parseInt(u64, number.int, 10));
    }
    return result;
}

fn post(srv: *Server, a: Allocator, body: []const u8) !Answer {
    return send(srv, a, "POST", "/v1/responses", body, null);
}

fn urlOf(a: Allocator, id: []const u8, tail: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "/v1/responses/{s}{s}", .{ id, tail });
}

fn hasPrefill(srv: *Server, a: Allocator, id: []const u8) !bool {
    const got = try sendRaw(srv, a, try urlOf(a, id, "?stream=true"), null);
    return got.status == 200 and std.mem.indexOf(u8, got.body, "\"response.prefill\"") != null;
}

fn expectSettled(srv: *Server, a: Allocator, id: []const u8, terminal_kind: []const u8) !void {
    const path = try urlOf(a, id, "?stream=true");
    const before = try sendRaw(srv, a, path, null);
    const journal = try events(a, before.body);
    var terminals: usize = 0;
    var terminal = false;
    for (journal.items) |one| {
        const kind = one.strField("type").?;
        if (std.mem.eql(u8, kind, "response.prefill")) try t.expect(!terminal);
        if (std.mem.startsWith(u8, kind, "response.") and (std.mem.eql(u8, kind, "response.completed") or std.mem.eql(u8, kind, "response.failed") or std.mem.eql(u8, kind, "response.incomplete"))) {
            terminals += 1;
            terminal = true;
        }
    }
    try t.expectEqual(@as(usize, 1), terminals);
    try t.expectEqualStrings(terminal_kind, journal.items[journal.items.len - 1].strField("type").?);
    std.Io.sleep(t.io, .fromMilliseconds(5), .awake) catch {};
    try t.expectEqualStrings(before.body, (try sendRaw(srv, a, path, null)).body);
}

/// Polls until the response's status is `want`, which it returns.
fn waitStatus(srv: *Server, a: Allocator, id: []const u8, want: []const u8) !json.Value {
    for (0..5000) |_| {
        const got = try send(srv, a, "GET", try urlOf(a, id, ""), "", null);
        const state = got.body.strField("status") orelse "";
        if (std.mem.eql(u8, state, want)) return got.body;
        for ([_][]const u8{ "completed", "incomplete", "failed", "cancelled" }) |ended|
            if (std.mem.eql(u8, state, ended)) return error.UnexpectedStatus;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

fn waitFor(comptime f: anytype, args: anytype) !void {
    for (0..5000) |_| {
        if (try @call(.auto, f, args)) return;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

fn idle(srv: *Server) !bool {
    return srv.generating.load(.seq_cst) == 0;
}

fn textOf(v: json.Value) []const u8 {
    return v.get("output").?.array[0].get("content").?.array[0].strField("text").?;
}

const long = "{\"model\":\"m\",\"input\":\"hi\",\"background\":true,\"max_output_tokens\":100000}";

test "Responses owner: background jobs keep their creating key while running and after cancellation" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator) };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    const created = try send(srv, a, "POST", "/v1/responses", long, 1);
    try t.expectEqual(@as(u16, 200), created.status);
    const id = created.body.strField("id").?;
    const url = try urlOf(a, id, "");
    const cancel = try urlOf(a, id, "/cancel");
    const request_cancel = try std.fmt.allocPrint(a, "/v1/requests/{s}/cancel", .{id});
    const child = try std.fmt.allocPrint(a, "{{\"input\":\"x\",\"previous_response_id\":\"{s}\"}}", .{id});
    try lane.waitFirstTokens(1);
    for ([_]?u8{ 2, null }) |foreign| {
        try t.expectEqual(@as(u16, 404), (try send(srv, a, "GET", url, "", foreign)).status);
        try t.expectEqual(@as(u16, 404), (try send(srv, a, "DELETE", url, "", foreign)).status);
        try t.expectEqual(@as(u16, 404), (try send(srv, a, "POST", cancel, "", foreign)).status);
        try t.expectEqual(@as(u16, 404), (try send(srv, a, "POST", request_cancel, "", foreign)).status);
        const refused = try send(srv, a, "POST", "/v1/responses", child, foreign);
        try t.expectEqual(@as(u16, 400), refused.status);
        try t.expect(std.mem.indexOf(u8, refused.body.get("error").?.strField("message").?, "is not stored here") != null);
    }
    try t.expectEqual(@as(usize, 0), lane.cancels.load(.monotonic));
    try t.expectEqual(@as(u16, 200), (try send(srv, a, "GET", url, "", 1)).status);
    const cancelled = try send(srv, a, "POST", cancel, "", 1);
    try t.expectEqual(@as(u16, 200), cancelled.status);
    try t.expectEqualStrings("cancelled", cancelled.body.strField("status").?);
    try waitFor(idle, .{srv});
    try t.expectEqual(@as(usize, 1), lane.cancels.load(.monotonic));
    try t.expectEqual(@as(u16, 200), (try send(srv, a, "GET", url, "", 1)).status);
    for ([_]?u8{ 2, null }) |foreign| {
        try t.expectEqual(@as(u16, 404), (try send(srv, a, "GET", url, "", foreign)).status);
        try t.expectEqual(@as(u16, 404), (try send(srv, a, "DELETE", url, "", foreign)).status);
        try t.expectEqual(@as(u16, 404), (try send(srv, a, "POST", cancel, "", foreign)).status);
    }
    try t.expectEqual(@as(u16, 200), (try send(srv, a, "DELETE", url, "", 1)).status);
    try t.expectEqual(@as(u16, 404), (try send(srv, a, "GET", url, "", 1)).status);
}

test "background Responses: optional rows keep the queued id and reply, explicit unavailable rows refuse" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator) };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    const fields = "\"model\":\"m\",\"input\":\"hello\",\"max_output_tokens\":8,\"temperature\":1,\"seed\":7,\"include\":[\"message.output_text.logprobs\"]";
    const created = try post(srv, a, "{" ++ fields ++ ",\"background\":true}");
    try t.expectEqual(@as(u16, 200), created.status);
    try t.expectEqualStrings("queued", created.body.strField("status").?);
    const id = created.body.strField("id").?;
    const done = try waitStatus(srv, a, id, "completed");
    try waitFor(idle, .{srv});
    try t.expectEqualStrings(id, done.strField("id").?);
    const plain = try post(srv, a, "{" ++ fields ++ "}");
    try t.expectEqual(@as(u16, 200), plain.status);
    try t.expectEqualStrings(textOf(plain.body), textOf(done));
    try t.expectEqualStrings(plain.body.get("tensorfold").?.strField("token_sha256").?, done.get("tensorfold").?.strField("token_sha256").?);
    try t.expectEqual(@as(usize, 0), done.get("output").?.array[0].get("content").?.array[0].get("logprobs").?.array.len);
    const refused = try post(srv, a, "{" ++ fields ++ ",\"background\":true,\"top_logprobs\":2}");
    try t.expectEqual(@as(u16, 200), refused.status);
    const failed = try waitStatus(srv, a, refused.body.strField("id").?, "failed");
    try t.expectEqualStrings("invalid_prompt", failed.get("error").?.strField("code").?);
    try waitFor(idle, .{srv});
    try t.expectEqual(@as(usize, 2), lane.taken.items.len);
    try expectSettled(srv, a, failed.strField("id").?, "response.failed");
}

test "background Responses: queued at once, then completed with the reply the same request gets in the foreground" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator) };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    const fields = "\"model\":\"m\",\"input\":\"hello\",\"max_output_tokens\":40,\"temperature\":1,\"seed\":7";
    const created = try post(srv, a, "{" ++ fields ++ ",\"background\":true}");
    try t.expectEqual(@as(u16, 200), created.status);
    try t.expectEqualStrings("queued", created.body.strField("status").?);
    try t.expectEqual(true, created.body.get("background").?.bool);
    try t.expectEqual(@as(usize, 0), created.body.get("output").?.array.len);
    const id = created.body.strField("id").?;
    try t.expect(std.mem.startsWith(u8, id, "resp_") and id.len == "resp_".len + 32);
    const done = try waitStatus(srv, a, id, "completed");
    try waitFor(idle, .{srv}); // the counts the response took from its route go once it is stored
    const plain = try post(srv, a, "{" ++ fields ++ "}");
    try t.expectEqual(@as(u16, 200), plain.status);
    try t.expectEqual(false, plain.body.get("background").?.bool);
    try t.expectEqual(@as(usize, 2), lane.taken.items.len);
    try t.expectEqualStrings(lane.taken.items[1], lane.taken.items[0]); // the engine was asked the same thing
    try t.expectEqual(@as(usize, 20), textOf(done).len);
    try t.expectEqualStrings(textOf(plain.body), textOf(done));
    try t.expectEqualStrings(plain.body.get("tensorfold").?.strField("token_sha256").?, done.get("tensorfold").?.strField("token_sha256").?);
    try t.expect(json.equal(plain.body.get("usage").?, done.get("usage").?));
    try t.expect(done.get("completed_at") != null);
    try t.expectEqualStrings(id, (try send(srv, a, "GET", try urlOf(a, id, ""), "", null)).body.strField("id").?);
}

test "background Responses: polled in progress, cancelled with its text so far, and a second cancel answers the same" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator) };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    const id = (try post(srv, a, long)).body.strField("id").?;
    try lane.waitFirstTokens(1);
    _ = try waitStatus(srv, a, id, "in_progress");
    try t.expectEqual(@as(u32, 1), srv.generating.load(.seq_cst)); // the response holds the request gate's count
    const cancelled = try send(srv, a, "POST", try urlOf(a, id, "/cancel"), "", null);
    try t.expectEqual(@as(u16, 200), cancelled.status);
    try t.expectEqualStrings("cancelled", cancelled.body.strField("status").?);
    try t.expect(cancelled.body.get("completed_at") == null);
    const so_far = textOf(cancelled.body);
    try t.expect(so_far.len > 0 and so_far.len < 100000);
    try t.expectEqual(@as(usize, 1), lane.cancels.load(.monotonic));
    try t.expect(try idle(srv));
    const again = try send(srv, a, "POST", try std.fmt.allocPrint(a, "/responses/{s}/cancel", .{id}), "{}", null);
    try t.expectEqual(@as(u16, 200), again.status);
    try t.expectEqualStrings(so_far, textOf(again.body));
    try t.expectEqualStrings("cancelled", (try waitStatus(srv, a, id, "cancelled")).strField("status").?);
    try t.expectEqual(@as(usize, 1), lane.cancels.load(.monotonic));
}

test "background Responses: the request id cancel route, DELETE while running, and what cannot be cancelled" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator) };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    // POST /v1/requests/{id}/cancel takes the same resp_ id: the response ends "cancelled" with its text so far
    const first = (try post(srv, a, long)).body.strField("id").?;
    try lane.waitFirstTokens(1);
    try t.expectEqual(@as(u16, 200), (try send(srv, a, "POST", try std.fmt.allocPrint(a, "/v1/requests/{s}/cancel", .{first}), "", null)).status);
    try t.expect(textOf(try waitStatus(srv, a, first, "cancelled")).len > 0);
    // DELETE stops a running response and keeps nothing of it, even once its turn ends
    const second = (try post(srv, a, long)).body.strField("id").?;
    try lane.waitFirstTokens(2);
    const deleted = try send(srv, a, "DELETE", try urlOf(a, second, ""), "", null);
    try t.expectEqual(@as(u16, 200), deleted.status);
    try t.expectEqual(true, deleted.body.get("deleted").?.bool);
    try t.expectEqual(@as(u16, 404), (try send(srv, a, "GET", try urlOf(a, second, ""), "", null)).status);
    try waitFor(idle, .{srv});
    try t.expectEqual(@as(usize, 2), lane.cancels.load(.monotonic));
    try t.expectEqual(@as(u16, 404), (try send(srv, a, "GET", try urlOf(a, second, ""), "", null)).status);
    try t.expectEqual(@as(u16, 404), (try send(srv, a, "POST", try urlOf(a, second, "/cancel"), "", null)).status);
    // a completed background response, and a stored one that was never background, cannot be cancelled
    const short = (try post(srv, a, "{\"model\":\"m\",\"input\":\"x\",\"background\":true,\"max_output_tokens\":4}")).body.strField("id").?;
    _ = try waitStatus(srv, a, short, "completed");
    try t.expectEqual(@as(u16, 400), (try send(srv, a, "POST", try urlOf(a, short, "/cancel"), "", null)).status);
    const plain = (try post(srv, a, "{\"model\":\"m\",\"input\":\"x\",\"max_output_tokens\":3}")).body.strField("id").?;
    try t.expectEqual(@as(u16, 400), (try send(srv, a, "POST", try urlOf(a, plain, "/cancel"), "", null)).status);
    // a completed background response replays its typed events; any other stored response reads as before
    const streamed = try sendRaw(srv, a, try urlOf(a, short, "?stream=true&starting_after=0"), null);
    try t.expectEqual(@as(u16, 200), streamed.status);
    try t.expect(std.mem.indexOf(u8, streamed.body, "\"response.completed\"") != null);
    try t.expectEqual(@as(u16, 200), (try send(srv, a, "GET", try urlOf(a, plain, "?stream=true"), "", null)).status);
}

test "background Responses: refusals name the field, and the limits answer 429 before any work starts" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator) };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    const refused = [_][]const u8{
        "{\"input\":\"x\",\"background\":1}",
        "{\"input\":\"x\",\"background\":true,\"store\":false}",
        "{\"input\":\"x\",\"background\":true,\"stream\":true}",
    };
    for (refused) |body| try t.expectEqual(@as(u16, 400), (try post(srv, a, body)).status);
    // a restricted key is stateless: it may not create one, nor reach the cancel route
    var wire: Wire = .{ .a = a };
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{long.len}) }};
    var scoped: http.Conn = .{ .fd = -1, .peer = "127.0.0.1", .buf = try a.dupe(u8, long), .end = long.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = "/v1/responses", .headers = &headers, .key_scopes = .{ .inference = true }, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    routes.dispatch(srv, &scoped, a);
    try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 403 "));
    try t.expect(!auth_scope.allows(.{ .inference = true }, "POST", "/v1/responses/resp_x/cancel"));
    try t.expect(!auth_scope.allows(.{ .read = true }, "GET", "/v1/responses/resp_x"));
    try t.expectEqual(@as(usize, 0), lane.taken.items.len);
    srv.background_jobs.limits = .{ .jobs = 2, .per_key = 1 };
    const one = (try send(srv, a, "POST", "/v1/responses", long, 1)).body.strField("id").?;
    try waitFor(Lane.submitted, .{ &lane, 1 });
    const same_key = try send(srv, a, "POST", "/v1/responses", long, 1);
    try t.expectEqual(@as(u16, 429), same_key.status);
    try t.expectEqualStrings("background_limit_exceeded", same_key.body.get("error").?.strField("code").?);
    try t.expectEqual(@as(u16, 200), (try send(srv, a, "POST", "/v1/responses", long, 2)).status);
    try t.expectEqual(@as(u16, 429), (try send(srv, a, "POST", "/v1/responses", long, 3)).status); // the server's two
    try t.expectEqual(@as(u32, 2), srv.generating.load(.seq_cst)); // the refused ones released their counts
    const child = try std.fmt.allocPrint(a, "{{\"input\":\"x\",\"previous_response_id\":\"{s}\"}}", .{one});
    try t.expectEqual(@as(u16, 400), (try post(srv, a, child)).status);
    srv.background_jobs.limits = .{ .bytes = 1 };
    try t.expectEqual(@as(u16, 429), (try post(srv, a, long)).status);
    try waitFor(Lane.submitted, .{ &lane, 2 });
    try t.expectEqual(@as(usize, 2), lane.taken.items.len); // nothing refused reached the engine
}

test "background Responses: the server's stop cancels a running one and waits for it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator) };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    _ = (try post(srv, a, long)).body.strField("id").?;
    try waitFor(Lane.submitted, .{ &lane, 1 });
    srv.deinit();
    try t.expectEqual(@as(usize, 1), lane.cancels.load(.monotonic));
}

test "background Responses: prefill is contiguous before output and replay resumes exactly once" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator), .hold_first = true };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    srv.config.sse_keepalive_s = 0.001;
    const id = (try post(srv, a, "{\"model\":\"m\",\"input\":\"hi\",\"background\":true,\"max_output_tokens\":8}")).body.strField("id").?;
    try waitFor(hasPrefill, .{ srv, a, id });
    const early = try events(a, (try sendRaw(srv, a, try urlOf(a, id, "?stream=true"), null)).body);
    try t.expect(early.items.len >= 3);
    try t.expectEqualStrings("response.created", early.items[0].strField("type").?);
    try t.expectEqualStrings("response.in_progress", early.items[1].strField("type").?);
    try t.expectEqualStrings("response.prefill", early.items[2].strField("type").?);
    for (early.items[0..3]) |one| {
        try t.expectEqualStrings(id, one.get("response").?.strField("id").?);
        try t.expectEqualStrings("in_progress", one.get("response").?.strField("status").?);
    }
    lane.releaseFirst();
    _ = try waitStatus(srv, a, id, "completed");
    try waitFor(idle, .{srv});
    const live_path = try urlOf(a, id, "?stream=true");
    const whole = try sendRaw(srv, a, live_path, null);
    const journal = try events(a, whole.body);
    const sequences = try eventSequences(a, whole.body);
    for (sequences.items, 0..) |sequence, i| try t.expectEqual(@as(u64, @intCast(i)), sequence);
    var output_at: usize = 2;
    while (output_at < journal.items.len and std.mem.eql(u8, journal.items[output_at].strField("type").?, "response.prefill")) output_at += 1;
    try t.expect(output_at > 2 and output_at < journal.items.len);
    for (journal.items[output_at..]) |one| try t.expect(!std.mem.eql(u8, one.strField("type").?, "response.prefill"));
    try expectSettled(srv, a, id, "response.completed");
    try t.expectEqual(@as(usize, 1), lane.taken.items.len);
    const resumed = try sendRaw(srv, a, try urlOf(a, id, "?stream=true&starting_after=1"), null);
    const suffix = try events(a, resumed.body);
    try t.expectEqual(journal.items.len - 2, suffix.items.len);
    for (journal.items[2..], suffix.items) |want, got| try t.expect(json.equal(want, got));
    const future_path = try urlOf(a, id, "?stream=true&starting_after=999999");
    try t.expectEqual(@as(u16, 400), (try sendRaw(srv, a, future_path, null)).status);
    try t.expectEqual(@as(u16, 404), (try sendRaw(srv, a, live_path, 2)).status);
}

test "background Responses: cancellation stops delayed prefill without output or later progress" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator), .hold_first = true };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    srv.config.sse_keepalive_s = 0.001;
    const id = (try post(srv, a, long)).body.strField("id").?;
    try waitFor(hasPrefill, .{ srv, a, id });
    const cancelled = try send(srv, a, "POST", try urlOf(a, id, "/cancel"), "", null);
    try t.expectEqual(@as(u16, 200), cancelled.status);
    try t.expectEqualStrings("cancelled", cancelled.body.strField("status").?);
    try waitFor(idle, .{srv});
    try expectSettled(srv, a, id, "response.incomplete");
    const replay = try sendRaw(srv, a, try urlOf(a, id, "?stream=true"), null);
    try t.expect(std.mem.indexOf(u8, replay.body, "\"response.prefill\"") != null);
    try t.expect(std.mem.indexOf(u8, replay.body, "output_text.delta") == null);
    try t.expectEqual(@as(usize, 1), lane.taken.items.len);
}

test "background Responses: replay reports a queued turn before any visible generation" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lane: Lane = .{ .io = t.io, .arena = .init(t.allocator) };
    try lane.start();
    defer lane.deinit();
    const srv = try serve(&lane);
    defer srv.deinit();
    _ = (try post(srv, a, long)).body.strField("id").?;
    const id = (try post(srv, a, long)).body.strField("id").?;
    try waitFor(Lane.submitted, .{ &lane, 1 });
    const replay = try sendRaw(srv, a, try urlOf(a, id, "?stream=true"), null);
    try t.expectEqual(@as(u16, 200), replay.status);
    try t.expect(std.mem.indexOf(u8, replay.body, "\"response.created\"") != null);
    try t.expect(std.mem.indexOf(u8, replay.body, "output_text.delta") == null);
    try t.expect(std.mem.indexOf(u8, replay.body, "response.completed") == null);
}

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
    first_tokens: usize = 0,
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

fn post(srv: *Server, a: Allocator, body: []const u8) !Answer {
    return send(srv, a, "POST", "/v1/responses", body, null);
}

fn urlOf(a: Allocator, id: []const u8, tail: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "/v1/responses/{s}{s}", .{ id, tail });
}

/// Polls until the response's status is `want`, which it returns.
fn waitStatus(srv: *Server, a: Allocator, id: []const u8, want: []const u8) !json.Value {
    for (0..5000) |_| {
        const got = try send(srv, a, "GET", try urlOf(a, id, ""), "", null);
        if (std.mem.eql(u8, got.body.strField("status") orelse "", want)) return got.body;
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
    // a stream is refused by name for a background response; any other stored response reads as before
    const streamed = try send(srv, a, "GET", try urlOf(a, short, "?stream=true&starting_after=0"), "", null);
    try t.expectEqual(@as(u16, 400), streamed.status);
    try t.expectEqualStrings("stream", streamed.body.get("error").?.strField("param").?);
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

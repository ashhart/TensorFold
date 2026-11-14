//! ``background: true`` Responses: each runs on its own thread after its POST returns; GET polls it, cancel stops it.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const responses = @import("responses.zig");
const translate = @import("responses_translate.zig");
const gate = @import("request_gate.zig");
const cancel_ids = @import("cancel_ids.zig");
const routes = @import("routes.zig");
const log = @import("log.zig");
const Sink = @import("responses_sink.zig").Sink;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Background responses queued or running at once, on the server and per API key, and the request bytes they hold.
pub const Limits = struct { jobs: usize = 64, per_key: usize = 16, bytes: usize = 256 << 20 };

/// How long a cancel waits for its response to end before it answers with the response as it stands.
const cancel_wait_ms = 2000;

/// The background responses not yet ended; an ended one is an ordinary stored response.
pub const Manager = struct {
    mutex: std.Io.Mutex = .init,
    ended: std.Io.Condition = .init, // broadcast when a job leaves `jobs`
    jobs: std.ArrayList(*Job) = .empty,
    bytes: usize = 0, // the copied requests' memory, against limits.bytes
    closed: bool = false,
    limits: Limits = .{},

    fn find(m: *Manager, id: []const u8) ?*Job {
        for (m.jobs.items) |j| if (std.mem.eql(u8, j.id, id)) return j;
        return null;
    }

    /// Why another job cannot start now (429), else null.
    fn full(m: *Manager, job: *const Job) ?[]const u8 {
        if (m.jobs.items.len >= m.limits.jobs) return "too many background responses are queued or running on this server; retry when one ends";
        if (job.key) |k| {
            var n: usize = 0;
            for (m.jobs.items) |j| n += @intFromBool(j.key != null and std.crypto.timing_safe.eql([32]u8, j.key.?, k));
            if (n >= m.limits.per_key) return "too many background responses are queued or running for this API key; retry when one ends";
        }
        if (m.bytes + job.charge > m.limits.bytes) return "the background responses in progress hold too much request data; retry when one ends";
        return null;
    }

    /// The job's thread started with `held`'s counts and lease, which it keeps until its response is stored.
    fn start(m: *Manager, srv: *Server, job: *Job, held: *gate.Held) Started {
        m.mutex.lockUncancelable(srv.io);
        defer m.mutex.unlock(srv.io);
        if (m.closed) return .closed;
        if (m.full(job)) |why| return .{ .busy = why };
        m.jobs.append(srv.gpa, job) catch return .{ .unavailable = "out of memory" };
        job.held = held.take();
        const thread = std.Thread.spawn(.{ .stack_size = 8 << 20 }, Job.run, .{job}) catch {
            _ = m.jobs.pop();
            held.* = job.held.take(); // the route releases them as it would any refused request's
            return .{ .unavailable = "the server could not start a thread for it" };
        };
        thread.detach();
        m.bytes += job.charge;
        return .queued;
    }

    /// GET's answer for a queued or running response, copied into `a`; null when none has `id`.
    pub fn poll(m: *Manager, io: std.Io, a: Allocator, id: []const u8) ?Value {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        const j = m.find(id) orelse return null;
        return j.snapshot(a) catch null;
    }

    /// Whether `id` names a response still queued or running.
    pub fn running(m: *Manager, io: std.Io, id: []const u8) bool {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        return m.find(id) != null;
    }

    /// DELETE of a queued or running response: it stops, and its end is not stored; false when none has `id`.
    pub fn forget(m: *Manager, io: std.Io, id: []const u8) bool {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        const j = m.find(id) orelse return false;
        j.deleted = true;
        j.stop();
        return true;
    }

    /// Cancels `id` and waits briefly for its end; the response as it stands if still running, null if it ended.
    fn cancel(m: *Manager, io: std.Io, a: Allocator, id: []const u8) ?Value {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        (m.find(id) orelse return null).stop();
        const wait: std.Io.Timeout = .{ .duration = .{ .raw = .fromMilliseconds(cancel_wait_ms), .clock = .awake } };
        const deadline = wait.toDeadline(io);
        while (m.find(id) != null) m.ended.waitTimeout(io, &m.mutex, deadline) catch break;
        const j = m.find(id) orelse return null;
        return j.snapshot(a) catch null;
    }

    /// Server.deinit: every job stops and is waited for (a serve drain first gives them its window, like any request).
    pub fn shutdown(m: *Manager, srv: *Server) void {
        m.mutex.lockUncancelable(srv.io);
        defer m.mutex.unlock(srv.io);
        m.closed = true;
        for (m.jobs.items) |j| j.stop();
        while (m.jobs.items.len > 0) m.ended.waitUncancelable(srv.io, &m.mutex);
        m.jobs.deinit(srv.gpa);
        m.jobs = .empty;
    }

    /// The job's end kept (unless deleted), its counts released, and the job gone from `jobs`.
    fn end(m: *Manager, j: *Job, cx: *errors.Cx, got: ?Value) void {
        const srv = j.srv;
        m.mutex.lockUncancelable(srv.io);
        defer m.mutex.unlock(srv.io);
        // stored before the job leaves `jobs`, so a poll always finds one of the two
        if (!j.deleted) if (j.ending(cx.a, got)) |final| srv.store.put(srv.io, cx, final, j.t.added) catch |e|
            log.line("background response {s} was not stored: {s}", .{ j.id, @errorName(e) });
        j.held.release();
        for (m.jobs.items, 0..) |x, i| if (x == j) {
            _ = m.jobs.swapRemove(i);
            break;
        };
        m.bytes -= j.charge;
        j.free();
        m.ended.broadcast(srv.io);
    }
};

const Started = union(enum) { queued, closed, busy: []const u8, unavailable: []const u8 };

const State = enum { queued, in_progress };

/// One background response: its own copy of the request, run on its own thread through the ordinary turn.
const Job = struct {
    srv: *Server,
    arena: std.heap.ArenaAllocator,
    id: []const u8 = "",
    key: ?[32]u8,
    t: translate.Request,
    base: *json.Object = undefined, // the response as created; it does not change while the job runs
    held: gate.Held = undefined,
    charge: usize = 0,
    state: State = .queued, // state, cancelled and deleted change under the manager's mutex
    cancelled: bool = false, // a cancel was answered: the end is kept as "cancelled"
    deleted: bool = false,
    dropped: std.atomic.Value(bool) = .init(false), // the gone hook: the turn stops and keeps nothing

    fn make(srv: *Server, a: Allocator, body: Value, t: translate.Request, key: ?[32]u8) !*Job {
        const j = try srv.gpa.create(Job);
        j.* = .{ .srv = srv, .arena = .init(srv.gpa), .key = key, .t = t };
        errdefer j.free();
        const ja = j.arena.allocator();
        j.base = (try json.deepCopy(ja, .{ .object = try responses.newBase(srv, a, body, t) })).object;
        try j.base.put(ja, "status", .{ .string = "queued" });
        j.id = j.base.get("id").?.string;
        j.t.chat = try json.deepCopy(ja, t.chat);
        j.t.added = (try json.deepCopy(ja, .{ .array = t.added })).array;
        j.t.echo = j.base; // the turn starts from `base`, which holds the echoed fields
        j.t.store = false; // the job keeps its end itself, unless the response was deleted
        j.charge = j.arena.queryCapacity();
        return j;
    }

    fn free(j: *Job) void {
        const gpa = j.srv.gpa;
        j.arena.deinit();
        gpa.destroy(j);
    }

    /// The response as a poll shows it: as created, with the status it has now.
    fn snapshot(j: *const Job, a: Allocator) !Value {
        const v = try json.deepCopy(a, .{ .object = j.base });
        try v.object.put(a, "status", .{ .string = if (j.cancelled) "cancelled" else @tagName(j.state) });
        return v;
    }

    /// A cancel: a running turn ends "cancelled" with its text so far (cancel_ids.zig); before it, nothing runs.
    fn stop(j: *Job) void {
        if (j.cancelled) return;
        j.cancelled = true;
        if (!cancel_ids.registry.cancel(j.srv.io, j.id)) j.dropped.store(true, .release);
    }

    fn gone(ctx: *anyopaque) bool {
        const j: *Job = @ptrCast(@alignCast(ctx));
        return j.dropped.load(.acquire);
    }

    fn run(j: *Job) void {
        const m = &j.srv.background_jobs;
        var cx: errors.Cx = .{ .a = j.arena.allocator() };
        const begun = blk: {
            m.mutex.lockUncancelable(j.srv.io);
            defer m.mutex.unlock(j.srv.io);
            if (!j.cancelled) j.state = .in_progress;
            break :blk !j.cancelled;
        };
        m.end(j, &cx, if (begun) j.work(&cx) else null);
    }

    /// The turn as a whole (unstreamed) POST runs it, on a connection that is gone only when cancelled.
    fn work(j: *Job, cx: *errors.Cx) ?Value {
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = cx.a, .gone_hook = .{ .ctx = j, .call = gone } };
        const sink: Sink = .{ .ctx = j, .vt = &.{ .open = quiet, .event = unheard, .keepalive = quiet } };
        return switch (responses.turn(j.srv, cx.a, &conn, cx, .null, j.t, sink, .{ .base = j.base })) {
            .whole => |v| v,
            .streamed => |v| v,
            .refused => |r| failed(cx.a, j.base, r.payload) catch null,
        };
    }

    /// The end as kept: the turn's response, or a failure; once a cancel was answered, "cancelled".
    fn ending(j: *Job, a: Allocator, got: ?Value) ?Value {
        if (!j.cancelled) return got orelse failed(a, j.base, .null) catch null;
        const o = json.copyObject(a, if (got) |v| v.object else j.base) catch return null;
        _ = o.orderedRemove("completed_at");
        o.put(a, "status", .{ .string = "cancelled" }) catch return null;
        o.put(a, "incomplete_details", .null) catch return null;
        o.put(a, "error", .null) catch return null;
        return .{ .object = o };
    }

    fn quiet(_: *anyopaque) error{Closed}!void {}

    fn unheard(_: *anyopaque, _: Allocator, _: []const u8, _: *json.Object) error{Closed}!void {}
};

/// The response failed with the chat route's error (`payload`, or none): "invalid_prompt" for the request's own fault.
fn failed(a: Allocator, base: *json.Object, payload: Value) !Value {
    const why = payload.get("error") orelse Value.null;
    const own = std.mem.eql(u8, json.strOr(why.get("type")), "invalid_request_error");
    const e = try json.newObject(a);
    try e.put(a, "code", .{ .string = if (own) "invalid_prompt" else "server_error" });
    try e.put(a, "message", .{ .string = why.strField("message") orelse "the background response ended without a reply" });
    const o = try json.copyObject(a, base);
    try o.put(a, "status", .{ .string = "failed" });
    try o.put(a, "error", .{ .object = e });
    return .{ .object = o };
}

/// POST /v1/responses with ``background: true``: the queued response at once, its turn on its own thread.
pub fn create(srv: *Server, conn: *Conn, a: Allocator, body: Value, t: translate.Request, held: ?*gate.Held) void {
    const h = held orelse return responses.refuse(conn, a, "background: true is served by the HTTP route only", 400, "background");
    const job = Job.make(srv, a, body, t, conn.key_digest) catch return problem(conn, a, 503, "server_error", "out of memory");
    const queued = job.snapshot(a) catch {
        job.free();
        return problem(conn, a, 503, "server_error", "out of memory");
    };
    switch (srv.background_jobs.start(srv, job, h)) {
        .queued => routes.sendValue(conn, a, 200, queued),
        .closed => {
            job.free();
            routes.draining(conn, false);
        },
        .busy => |why| {
            job.free();
            problem(conn, a, 429, "rate_limit_error", why);
        },
        .unavailable => |why| {
            job.free();
            problem(conn, a, 503, "server_error", why);
        },
    }
}

fn problem(conn: *Conn, a: Allocator, status: u16, kind: []const u8, message: []const u8) void {
    const code = if (status == 429) "\"background_limit_exceeded\"" else "null";
    const body = std.fmt.allocPrint(a, "{{\"error\": {{\"message\": {s}, \"type\": \"{s}\", \"param\": \"background\", \"code\": {s}}}}}", .{ json.quote(a, message, .{}) catch return, kind, code }) catch return;
    conn.sendJson(status, body);
}

/// The id in a ``/v1/responses/{id}/cancel`` route (or without /v1), else null.
pub fn cancelPath(route: []const u8) ?[]const u8 {
    if (!std.mem.endsWith(u8, route, "/cancel")) return null;
    const id = responses.route(route[0 .. route.len - "/cancel".len]) orelse return null;
    return if (id.len > 0) id else null;
}

/// POST /v1/responses/{id}/cancel: the cancelled response; again for one already cancelled; 400 for any other end.
pub fn cancel(srv: *Server, conn: *Conn, a: Allocator, id: []const u8) void {
    if (srv.background_jobs.cancel(srv.io, a, id)) |still| return routes.sendValue(conn, a, 200, still);
    const kept = (srv.store.get(srv.io, a, id) catch null) orelse return responses.notStored(conn, a, id);
    if (!json.truthyField(kept, "background")) return responses.refuse(conn, a, "only a response created with background: true can be cancelled", 400, "response_id");
    const status = json.strOr(kept.get("status"));
    if (std.mem.eql(u8, status, "cancelled")) return routes.sendValue(conn, a, 200, kept);
    const message = std.fmt.allocPrint(a, "the response has already ended (status {s}); only a queued or in-progress one can be cancelled", .{status}) catch return;
    responses.refuse(conn, a, message, 400, "response_id");
}

/// ``stream=true`` (or 1) in a GET's query: an event stream asked of a stored response.
pub fn streamAsked(path: []const u8) bool {
    const q = std.mem.indexOfScalar(u8, path, '?') orelse return false;
    var pairs = std.mem.splitScalar(u8, path[q + 1 ..], '&');
    while (pairs.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
        if (!std.mem.eql(u8, pair[0..eq], "stream")) continue;
        return std.ascii.eqlIgnoreCase(pair[eq + 1 ..], "true") or std.mem.eql(u8, pair[eq + 1 ..], "1");
    }
    return false;
}

test "the cancel route names one response id, and only stream=true asks for a stream" {
    try std.testing.expectEqualStrings("resp_a", cancelPath("/v1/responses/resp_a/cancel").?);
    try std.testing.expectEqualStrings("resp_b", cancelPath("/responses/resp_b/cancel").?);
    for ([_][]const u8{ "/v1/responses/cancel", "/v1/responses//cancel", "/v1/responses/a/b/cancel", "/v1/responses/a" }) |r|
        try std.testing.expect(cancelPath(r) == null);
    try std.testing.expect(streamAsked("/v1/responses/r?stream=true&starting_after=3"));
    try std.testing.expect(streamAsked("/v1/responses/r?include=x&stream=1"));
    for ([_][]const u8{ "/v1/responses/r", "/v1/responses/r?stream=false", "/v1/responses/r?streams=true" }) |p|
        try std.testing.expect(!streamAsked(p));
}

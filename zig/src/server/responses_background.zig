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
const event = @import("responses_event_log.zig");
const sse = @import("sse.zig");
const Conn = @import("http_conn.zig").Conn;
const max_archives = 64;
const max_archive_bytes = 64 << 20;
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
    archives: std.ArrayList(*Archive) = .empty,
    archive_bytes: usize = 0,

    fn find(m: *Manager, id: []const u8, owner: ?[32]u8) ?*Job {
        for (m.jobs.items) |j| if (!j.deleted and std.mem.eql(u8, j.id, id) and std.meta.eql(j.key, owner)) return j;
        return null;
    }

    fn findArchive(m: *Manager, id: []const u8, owner: ?[32]u8) ?usize {
        for (m.archives.items, 0..) |archive, i|
            if (std.mem.eql(u8, archive.id, id) and std.meta.eql(archive.key, owner)) return i;
        return null;
    }

    fn dropArchive(m: *Manager, i: usize) void {
        const archive = m.archives.items[i];
        m.archive_bytes -= archive.log.charge();
        _ = m.archives.orderedRemove(i);
        archive.free(archive.log.gpa);
    }

    fn retain(m: *Manager, j: *Job) void {
        const archive = Archive.make(j) catch return;
        const charge = archive.log.charge();
        if (charge > max_archive_bytes) {
            archive.free(j.srv.gpa);
            return;
        }
        while (m.archives.items.len > 0 and (m.archives.items.len >= max_archives or m.archive_bytes + charge > max_archive_bytes))
            m.dropArchive(0);
        m.archives.append(j.srv.gpa, archive) catch {
            archive.free(j.srv.gpa);
            return;
        };
        m.archive_bytes += charge;
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
    pub fn poll(m: *Manager, io: std.Io, a: Allocator, id: []const u8, owner: ?[32]u8) ?Value {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        const j = m.find(id, owner) orelse return null;
        return j.snapshot(a) catch null;
    }

    pub fn copyReplay(m: *Manager, io: std.Io, a: Allocator, id: []const u8, owner: ?[32]u8, after: ?u64) !?[]*json.Object {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        if (m.find(id, owner)) |j|
            return try j.log.copyAfter(a, after, if (j.log.last == null and !j.cancelled) j.base else null);
        const i = m.findArchive(id, owner) orelse return null;
        return try m.archives.items[i].log.copyAfter(a, after, null);
    }

    /// Whether `id` names a response still queued or running.
    pub fn running(m: *Manager, io: std.Io, id: []const u8, owner: ?[32]u8) bool {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        return m.find(id, owner) != null;
    }

    /// DELETE of a queued or running response: it stops, and its end is not stored; false when none has `id`.
    pub fn forget(m: *Manager, io: std.Io, id: []const u8, owner: ?[32]u8) bool {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        const j = m.find(id, owner) orelse {
            if (m.findArchive(id, owner)) |i| m.dropArchive(i);
            return false;
        };
        j.deleted = true;
        j.stop();
        return true;
    }

    /// Cancels `id` and waits briefly for its end; the response as it stands if still running, null if it ended.
    fn cancel(m: *Manager, io: std.Io, a: Allocator, id: []const u8, owner: ?[32]u8) ?Value {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        (m.find(id, owner) orelse return null).stop();
        const wait: std.Io.Timeout = .{ .duration = .{ .raw = .fromMilliseconds(cancel_wait_ms), .clock = .awake } };
        const deadline = wait.toDeadline(io);
        while (m.find(id, owner) != null) m.ended.waitTimeout(io, &m.mutex, deadline) catch break;
        const j = m.find(id, owner) orelse return null;
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
        for (m.archives.items) |archive| archive.free(srv.gpa);
        m.archives.deinit(srv.gpa);
        m.archives = .empty;
        m.archive_bytes = 0;
    }

    /// The job's end kept (unless deleted), its counts released, and the job gone from `jobs`.
    fn end(m: *Manager, j: *Job, cx: *errors.Cx, got: ?Value) void {
        const srv = j.srv;
        m.mutex.lockUncancelable(srv.io);
        defer m.mutex.unlock(srv.io);
        // stored before the job leaves `jobs`, so a poll always finds one of the two
        const final: ?Value = if (j.deleted) null else j.ending(cx.a, got);
        if (final) |v| srv.store.put(srv.io, cx, v, j.t.added, j.key) catch |e|
            log.line("background response {s} was not stored: {s}", .{ j.id, @errorName(e) });
        j.held.release();
        for (m.jobs.items, 0..) |x, i| if (x == j) {
            _ = m.jobs.swapRemove(i);
            break;
        };
        m.bytes -= j.charge;
        if (!j.deleted) {
            if (final) |v| if (j.log.last == null) j.synthetic(cx.a, v);
            m.retain(j);
        }
        j.free();
        m.ended.broadcast(srv.io);
    }
};

const Started = union(enum) { queued, closed, busy: []const u8, unavailable: []const u8 };

const State = enum { queued, in_progress };

const Archive = struct {
    id: []const u8,
    key: ?[32]u8,
    log: event.Log,

    fn make(j: *Job) !*Archive {
        const gpa = j.srv.gpa;
        const archive = try gpa.create(Archive);
        archive.* = .{ .id = "", .key = j.key, .log = j.log };
        j.log = .init(gpa);
        errdefer archive.free(gpa);
        archive.id = try archive.log.arena.allocator().dupe(u8, j.id);
        return archive;
    }

    fn free(archive: *Archive, gpa: Allocator) void {
        archive.log.deinit();
        gpa.destroy(archive);
    }
};

/// One background response: its own copy of the request, run on its own thread through the ordinary turn.
const Job = struct {
    srv: *Server,
    arena: std.heap.ArenaAllocator,
    log: event.Log,
    id: []const u8 = "",
    key: ?[32]u8,
    t: translate.Request,
    base: *json.Object = undefined, // the response as created; it does not change while the job runs
    held: gate.Held = undefined,
    charge: usize = 0,
    state: State = .queued, // state, cancelled and deleted change under the manager's mutex
    cancelled: bool = false, // a cancel was answered: the end is kept as "cancelled"
    terminal: ?Value = null, // the first journaled terminal response; cancellation cannot rewrite it
    deleted: bool = false,
    emit_prefill: bool = true,
    dropped: std.atomic.Value(bool) = .init(false), // the gone hook: the turn stops and keeps nothing

    fn make(srv: *Server, a: Allocator, body: Value, t: translate.Request, key: ?[32]u8) !*Job {
        const j = try srv.gpa.create(Job);
        j.* = .{ .srv = srv, .arena = .init(srv.gpa), .log = .init(srv.gpa), .key = key, .t = t };
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
        j.log.deinit();
        j.arena.deinit();
        gpa.destroy(j);
    }

    /// The response as a poll shows it: its terminal snapshot, reconstructed cancel output, or current state.
    fn snapshot(j: *Job, a: Allocator) !Value {
        if (j.terminal) |v| return json.deepCopy(a, v);
        if (j.cancelled) {
            const partial = try j.log.response(a, j.base);
            return j.ending(a, partial) orelse error.OutOfMemory;
        }
        const v = try json.deepCopy(a, .{ .object = j.base });
        try v.object.put(a, "status", .{ .string = @tagName(j.state) });
        return v;
    }

    /// A cancel: a running turn ends "cancelled" with its text so far (cancel_ids.zig); before it, nothing runs.
    fn stop(j: *Job) void {
        if (j.cancelled or j.terminal != null) return;
        j.cancelled = true;
        if (!cancel_ids.registry.cancel(j.srv.io, j.id, j.key)) j.dropped.store(true, .release);
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

    /// The turn streams internally even though its POST is JSON: those typed events are its replay journal.
    fn work(j: *Job, cx: *errors.Cx) ?Value {
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = cx.a, .gone_hook = .{ .ctx = j, .call = gone }, .key_digest = j.key };
        j.t.stream = true;
        j.t.chat.object.put(cx.a, "stream", .{ .bool = true }) catch return null;
        j.base.put(cx.a, "status", .{ .string = "in_progress" }) catch return null;
        const sink: Sink = .{ .ctx = j, .vt = &.{ .open = quiet, .event = heard, .keepalive = prefilled } };
        return switch (responses.turn(j.srv, cx.a, &conn, cx, .null, j.t, sink, .{ .base = j.base })) {
            .whole => |v| v,
            .streamed => |v| v,
            .refused => |r| failed(cx.a, j.base, r.payload) catch null,
        };
    }

    /// The journaled terminal response wins; otherwise a cancellation keeps the partial turn as cancelled.
    fn ending(j: *Job, a: Allocator, got: ?Value) ?Value {
        if (j.terminal) |v| return json.deepCopy(a, v) catch null;
        if (!j.cancelled) return got orelse failed(a, j.base, .null) catch null;
        const o = json.copyObject(a, if (got) |v| v.object else j.base) catch return null;
        _ = o.orderedRemove("completed_at");
        o.put(a, "status", .{ .string = "cancelled" }) catch return null;
        o.put(a, "incomplete_details", .null) catch return null;
        o.put(a, "error", .null) catch return null;
        return .{ .object = o };
    }

    fn synthetic(j: *Job, a: Allocator, final: Value) void {
        const terminal_kind = if (std.mem.eql(u8, json.strOr(final.get("status")), "cancelled")) "response.incomplete" else "response.failed";
        var sequence: u64 = 0;
        for ([_]struct { kind: ?[]const u8, status: ?[]const u8, response: *json.Object }{
            .{ .kind = "response.created", .status = "queued", .response = j.base },
            .{ .kind = if (j.state == .in_progress) "response.in_progress" else null, .status = "in_progress", .response = j.base },
            .{ .kind = terminal_kind, .status = null, .response = final.object },
        }) |one| {
            const kind = one.kind orelse continue;
            const payload = json.newObject(a) catch {
                j.log.truncated = true;
                return;
            };
            payload.put(a, "type", .{ .string = kind }) catch {
                j.log.truncated = true;
                return;
            };
            payload.put(a, "sequence_number", json.intValue(a, sequence) catch {
                j.log.truncated = true;
                return;
            }) catch {
                j.log.truncated = true;
                return;
            };
            const response = json.copyObject(a, one.response) catch {
                j.log.truncated = true;
                return;
            };
            if (one.status) |status| response.put(a, "status", .{ .string = status }) catch {
                j.log.truncated = true;
                return;
            };
            payload.put(a, "response", .{ .object = response }) catch {
                j.log.truncated = true;
                return;
            };
            j.log.record(payload) catch {
                j.log.truncated = true;
                return;
            };
            sequence += 1;
        }
    }

    fn quiet(_: *anyopaque) error{Closed}!void {}

    fn prefilled(ctx: *anyopaque) error{Closed}!void {
        const j: *Job = @ptrCast(@alignCast(ctx));
        const m = &j.srv.background_jobs;
        m.mutex.lockUncancelable(j.srv.io);
        defer m.mutex.unlock(j.srv.io);
        if (!j.emit_prefill or j.state != .in_progress or j.cancelled or j.deleted or m.closed) return;
        const a = j.arena.allocator();
        const payload = json.newObject(a) catch return error.Closed;
        payload.put(a, "type", .{ .string = "response.prefill" }) catch return error.Closed;
        const next = (j.log.last orelse return) + 1;
        payload.put(a, "sequence_number", json.intValue(a, next) catch return error.Closed) catch return error.Closed;
        payload.put(a, "response", .{ .object = j.base }) catch return error.Closed;
        j.log.record(payload) catch {
            j.log.truncated = true;
            return error.Closed;
        };
    }

    fn heard(ctx: *anyopaque, a: Allocator, kind: []const u8, payload: *json.Object) error{Closed}!void {
        const j: *Job = @ptrCast(@alignCast(ctx));
        const m = &j.srv.background_jobs;
        m.mutex.lockUncancelable(j.srv.io);
        defer m.mutex.unlock(j.srv.io);
        if (!std.mem.eql(u8, kind, "response.created") and !std.mem.eql(u8, kind, "response.in_progress")) j.emit_prefill = false;
        if (j.log.last) |last| {
            const raw = payload.get("sequence_number") orelse return error.Closed;
            const sequence = std.fmt.parseInt(u64, raw.int, 10) catch return error.Closed;
            if (sequence <= last) payload.put(a, "sequence_number", json.intValue(a, last + 1) catch return error.Closed) catch return error.Closed;
        }
        j.log.record(payload) catch {
            j.log.truncated = true;
            return error.Closed;
        };
        if (std.mem.eql(u8, kind, "response.completed") or std.mem.eql(u8, kind, "response.failed") or std.mem.eql(u8, kind, "response.incomplete")) j.terminal = payload.get("response");
    }
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
    if (srv.background_jobs.cancel(srv.io, a, id, conn.key_digest)) |still| return routes.sendValue(conn, a, 200, still);
    const kept = (srv.store.get(srv.io, a, id, conn.key_digest) catch null) orelse return responses.notStored(conn, a, id);
    if (!json.truthyField(kept, "background")) return responses.refuse(conn, a, "only a response created with background: true can be cancelled", 400, "response_id");
    const status = json.strOr(kept.get("status"));
    if (std.mem.eql(u8, status, "cancelled")) return routes.sendValue(conn, a, 200, kept);
    const message = std.fmt.allocPrint(a, "the response has already ended (status {s}); only a queued or in-progress one can be cancelled", .{status}) catch return;
    responses.refuse(conn, a, message, 400, "response_id");
}

pub fn replay(srv: *Server, conn: *Conn, a: Allocator, id: []const u8) void {
    const after = startingAfter(conn.path) catch
        return responses.refuse(conn, a, "starting_after must be one nonnegative integer sequence number", 400, "starting_after");
    const found = srv.background_jobs.copyReplay(srv.io, a, id, conn.key_digest, after) catch |e| {
        switch (e) {
            error.InvalidCursor => responses.refuse(conn, a, "starting_after is newer than the response's latest event", 400, "starting_after"),
            error.StaleCursor => responses.refuse(conn, a, "the response's event journal was truncated; replay is no longer complete", 410, "starting_after"),
            else => responses.refuse(conn, a, "out of memory while copying the response's event journal", 503, null),
        }
        return;
    };
    const events = found orelse
        return responses.refuse(conn, a, "the response's event journal is no longer retained", 410, "starting_after");
    sse.open(conn) catch return;
    for (events) |payload| sse.event(conn, a, payload.get("type").?.string, .{ .object = payload }) catch return;
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

fn startingAfter(path: []const u8) error{InvalidCursor}!?u64 {
    const q = std.mem.indexOfScalar(u8, path, '?') orelse return null;
    var found: ?u64 = null;
    var pairs = std.mem.splitScalar(u8, path[q + 1 ..], '&');
    while (pairs.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
        if (!std.mem.eql(u8, pair[0..eq], "starting_after")) continue;
        if (found != null) return error.InvalidCursor;
        found = std.fmt.parseInt(u64, pair[eq + 1 ..], 10) catch return error.InvalidCursor;
    }
    return found;
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
    try std.testing.expectEqual(@as(?u64, null), try startingAfter("/v1/responses/r?stream=true"));
    try std.testing.expectEqual(@as(?u64, 0), try startingAfter("/v1/responses/r?starting_after=0"));
    try std.testing.expectEqual(@as(?u64, 3), try startingAfter("/v1/responses/r?stream=true&starting_after=3"));
    for ([_][]const u8{ "/v1/responses/r?starting_after=", "/v1/responses/r?starting_after=-1", "/v1/responses/r?starting_after=1x", "/v1/responses/r?starting_after=18446744073709551616", "/v1/responses/r?starting_after=1&starting_after=2" }) |p|
        try std.testing.expectError(error.InvalidCursor, startingAfter(p));
}

test "background Responses: cancellation cannot rewrite a journaled terminal response" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv: Server = undefined;
    srv.io = std.testing.io;
    srv.background_jobs = .{};
    for ([_][2][]const u8{ .{ "response.completed", "completed" }, .{ "response.failed", "failed" }, .{ "response.incomplete", "incomplete" } }) |one| {
        var j: Job = .{ .srv = &srv, .arena = .init(std.testing.allocator), .log = .init(std.testing.allocator), .key = null, .t = undefined, .base = try json.newObject(a) };
        defer j.log.deinit();
        defer j.arena.deinit();
        const got = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"id\":\"resp_terminal\",\"status\":\"{s}\",\"output\":[]}}", .{one[1]}))).ok.object;
        const payload = try json.newObject(a);
        try payload.put(a, "sequence_number", try json.intValue(a, 0));
        try payload.put(a, "response", .{ .object = got });
        try Job.heard(&j, a, one[0], payload);
        j.stop();
        try std.testing.expect(!j.cancelled);
        try std.testing.expect(json.equal(.{ .object = got }, (try j.snapshot(a))));
        try std.testing.expect(json.equal(.{ .object = got }, j.ending(a, .{ .object = got }).?));
    }
}

test "background Responses: synthetic replay preserves state and omits progress when work never began" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv: Server = undefined;
    var j: Job = .{ .srv = &srv, .arena = .init(std.testing.allocator), .log = .init(std.testing.allocator), .key = null, .t = undefined, .base = try json.newObject(a), .state = .in_progress };
    defer j.log.deinit();
    defer j.arena.deinit();
    const final = (try json.parse(a, "{\"id\":\"resp_failed\",\"status\":\"failed\",\"output\":[]}")).ok;
    j.synthetic(a, final);
    try std.testing.expectEqual(@as(usize, 3), j.log.events.items.len);
    for ([_][2][]const u8{ .{ "response.created", "queued" }, .{ "response.in_progress", "in_progress" }, .{ "response.failed", "failed" } }, j.log.events.items, 0..) |one, payload, i| {
        try std.testing.expectEqualStrings(one[0], (json.Value{ .object = payload }).strField("type").?);
        try std.testing.expectEqualStrings(one[1], payload.get("response").?.strField("status").?);
        try std.testing.expectEqual(@as(i64, @intCast(i)), payload.get("sequence_number").?.int64().?);
    }
    const suffix = try j.log.copyAfter(a, 1, null);
    try std.testing.expectEqual(@as(usize, 1), suffix.len);
    try std.testing.expectEqualStrings("response.failed", (json.Value{ .object = suffix[0] }).strField("type").?);

    var queued: Job = .{ .srv = &srv, .arena = .init(std.testing.allocator), .log = .init(std.testing.allocator), .key = null, .t = undefined, .base = try json.newObject(a), .cancelled = true };
    defer queued.log.deinit();
    defer queued.arena.deinit();
    queued.synthetic(a, (try json.parse(a, "{\"id\":\"resp_cancel\",\"status\":\"cancelled\",\"output\":[]}")).ok);
    try std.testing.expectEqual(@as(usize, 2), queued.log.events.items.len);
    for ([_][2][]const u8{ .{ "response.created", "queued" }, .{ "response.incomplete", "cancelled" } }, queued.log.events.items, 0..) |one, payload, i| {
        try std.testing.expectEqualStrings(one[0], (json.Value{ .object = payload }).strField("type").?);
        try std.testing.expectEqualStrings(one[1], payload.get("response").?.strField("status").?);
        try std.testing.expectEqual(@as(i64, @intCast(i)), payload.get("sequence_number").?.int64().?);
    }
}

test "background Responses: cancellation acknowledgement and storage expose the same partial output" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv: Server = undefined;
    srv.io = std.testing.io;
    srv.background_jobs = .{};
    var j: Job = .{ .srv = &srv, .arena = .init(std.testing.allocator), .log = .init(std.testing.allocator), .key = null, .t = undefined, .base = (try json.parse(a, "{\"id\":\"resp_cancel\",\"status\":\"in_progress\",\"output\":[]}")).ok.object, .cancelled = true };
    defer j.log.deinit();
    defer j.arena.deinit();
    const heard = (try json.parse(a,
        \\[{"type":"response.output_item.added","sequence_number":0,"output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","content":[]}},
        \\{"type":"response.content_part.added","sequence_number":1,"output_index":0,"part":{"type":"output_text","text":""}},
        \\{"type":"response.output_text.delta","sequence_number":2,"output_index":0,"delta":"partial"}]
    )).ok.array;
    for (heard) |payload| try j.log.record(payload.object);
    const acknowledged = try j.snapshot(a);
    const part = acknowledged.get("output").?.array[0].get("content").?.array[0];
    try std.testing.expectEqualStrings("partial", part.strField("text").?);
    try std.testing.expectEqualStrings("cancelled", acknowledged.strField("status").?);
    const stored = j.ending(a, acknowledged).?;
    try std.testing.expect(json.equal(acknowledged, stored));
}

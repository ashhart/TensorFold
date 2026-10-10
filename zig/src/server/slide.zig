//! Sliding Weights over HTTP: /v1/slide/learn streams learning, /v1/slide/graph lists or clears facts, /slide draws.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const sse = @import("sse.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const errors = @import("errors.zig");
const teach = @import("slide_lesson.zig");
const Gone = @import("openai.zig").Gone;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

const page = @embedFile("slide.html");

pub fn view(_: *Server, conn: *Conn, _: Allocator) void {
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/html; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{page.len}) catch return) catch return;
    conn.finish(page) catch {};
}

pub fn graph(srv: *Server, conn: *Conn, a: Allocator) void {
    const body = srv.slide.render(a) catch return refuse(conn, a, 500, "out of memory");
    conn.sendJson(200, body);
}

/// The graph goes; the weights keep what they learned.
pub fn forget(srv: *Server, conn: *Conn, _: Allocator) void {
    srv.slide.clear();
    conn.sendJson(200, "{\"cleared\": true}");
}

/// The body's facts, each written into a lesson by the model and learned in turn into the live weights.
pub fn learn(srv: *Server, conn: *Conn, a: Allocator) void {
    const raw = switch (http_body.read(conn, a, http_body.limit) catch return) {
        .ok => |b| b,
        .refused => |message| return refuse(conn, a, 400, message),
    };
    const body = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return) {
        .ok => |v| v,
        .err => |message| return refuse(conn, a, 400, message),
    };
    const text = body.strField("text") orelse return refuse(conn, a, 400, "text is required");
    const source = body.strField("source") orelse "chat";
    switch (run(srv, a, &.{})) {
        .refused => |e| return switch (e) {
            error.Unsupported => refuse(conn, a, 501, "this engine does not learn: serve its model with --slide"),
            error.Busy, error.Closed => refuse(conn, a, 503, "the engine cannot take a learn request now"),
        },
        else => {},
    }
    srv.teacher.learning.lockUncancelable(srv.io);
    defer srv.teacher.learning.unlock(srv.io);
    if (srv.teacher.stuck) return refuse(conn, a, 503, "a lesson that did not come back could not be taken out: restart the server");
    sse.open(conn) catch return;
    var out: Out = .{ .conn = conn, .a = a };
    var cx: errors.Cx = .{ .a = a };
    const gone: Gone = .{ .conn = conn };
    const facts = teach.facts(srv, &cx, text, source, gone) catch |e| return out.fail(&cx, e);
    if (facts.len == 0) {
        out.event("failed", .{ .message = "no facts found in the text: nothing was learned" });
        sse.done(conn) catch {};
        return;
    }
    const ids = a.alloc(u32, facts.len) catch return;
    for (facts, ids) |fact, *id| {
        id.* = srv.slide.add(fact, source, now(srv));
        out.event("fact", .{ .id = id.*, .text = fact });
    }
    defer {
        if (!srv.teacher.stuck) _ = run(srv, a, &.{ .save = true });
    }
    // each fact its own lesson and block, so each keeps a gate of its own
    for (facts, ids) |fact, id| {
        if (gone.check()) break;
        srv.slide.mark(id, .learning);
        out.event("learning", .{ .id = id });
        var kept = [1]bool{false};
        const plan = (teach.lesson(srv, &cx, &srv.teacher, &.{fact}, &kept, gone) catch |e| {
            missed(srv, &out, id, words(&cx, e));
            continue;
        }) orelse {
            missed(srv, &out, id, "too few clean answers to learn from");
            continue;
        };
        var result = rounds(srv, a, &cx, plan, gone);
        // clean rounds go on to the plain change and mining; the check after them decides keep or take out
        if (result.why == null) commit(srv, a, &cx, plan, gone, &result);
        const back = all(result.recalled);
        if (!back) _ = takeOut(srv, a, &result);
        srv.slide.mark(id, if (back) .learned else .missed);
        if (result.why) |why| {
            out.event("learned", .{ .id = id, .recalled = back, .steps = result.steps, .message = why });
        } else out.event("learned", .{ .id = id, .recalled = back, .steps = result.steps });
        if (srv.teacher.stuck) break;
    }
    sse.done(conn) catch {};
}

/// What a lesson's rounds left: which facts come back, the steps taken, and why they do not (null: not known).
pub const Rounds = struct { recalled: []bool, steps: u32 = 0, why: ?[]const u8 = null };

/// Whether every fact came back; a lesson whose facts could not be counted did not.
pub fn all(recalled: []const bool) bool {
    return recalled.len > 0 and std.mem.allEqual(bool, recalled, true);
}

/// A lesson in checked rounds until every fact comes back; a round that loops or leaks ends it.
fn rounds(srv: *Server, a: Allocator, cx: *errors.Cx, plan: teach.Plan, gone: Gone) Rounds {
    var out: Rounds = .{ .recalled = a.alloc(bool, plan.facts.len) catch &.{} };
    @memset(out.recalled, false);
    for (0..teach.rounds) |round| {
        var request = plan.request;
        request.more = round > 0;
        switch (run(srv, a, &request)) {
            .learned => |l| out.steps += l.steps,
            .failed => |message| {
                out.why = message;
                return out;
            },
            .refused => |e| {
                out.why = @errorName(e);
                return out;
            },
            .none => {
                out.why = "the learner ended the round without a result";
                return out;
            },
        }
        const verdict = teach.verify(srv, cx, plan, gone) catch |e| teach.Verdict{ .recalled = out.recalled, .damage = words(cx, e) };
        if (verdict.damage) |why| {
            out.why = std.fmt.allocPrint(a, "in round {d}, {s}", .{ round + 1, why }) catch why;
            @memset(out.recalled, false);
            return out;
        }
        @memcpy(out.recalled, verdict.recalled);
        if (all(out.recalled)) return out;
    }
    return out;
}

/// A lesson whose facts did not all come back, taken out whole before the save; false, and stuck, when its undo failed.
pub fn takeOut(srv: *Server, a: Allocator, out: *Rounds) bool {
    @memset(out.recalled, false);
    const why = out.why orelse "it did not come back on its held-out questions";
    out.why = why;
    // with no round finished the save keeps nothing of it, and an undo could reach back into the lesson before
    if (out.steps == 0) return true;
    const failed: ?[]const u8 = switch (run(srv, a, &.{ .undo = true })) {
        .none => null,
        .learned => "the learner reported a lesson instead of taking one out",
        .failed => |message| message,
        .refused => |e| @errorName(e),
    };
    if (failed) |no| {
        const said = "{s}; it could not be taken out ({s}), so nothing of this request is saved: restart the server";
        out.why = std.fmt.allocPrint(a, said, .{ why, no }) catch why;
        srv.teacher.stuck = true; // the lesson is still in the live weights, so no save may run until a restart
        return false;
    }
    out.why = std.fmt.allocPrint(a, "taken out: {s}", .{why}) catch why;
    return true;
}

/// The lesson made a plain weight change, its moved near misses trained back, then checked again.
fn commit(srv: *Server, a: Allocator, cx: *errors.Cx, plan: teach.Plan, gone: Gone, out: *Rounds) void {
    var request = plan.request;
    request.commit = true;
    if (!step(srv, a, &request, out)) return;
    for (0..teach.mining_rounds) |_| {
        if (gone.check()) break;
        const moved = teach.mine(srv, cx, plan, gone) catch break;
        const back = teach.recalledAll(srv, cx, plan, gone) catch break;
        if (moved.len == 0 and back) break;
        var near: std.ArrayList(api.Example) = .empty;
        near.appendSlice(a, plan.request.near) catch break;
        for (moved) |i| for (0..teach.mined_weight) |_| near.append(a, plan.request.near[i]) catch break;
        var more = plan.request;
        more.more = true;
        more.near = near.items;
        more.steps = teach.mining_steps;
        if (!step(srv, a, &more, out)) return;
    }
    const verdict = teach.verify(srv, cx, plan, gone) catch |e| teach.Verdict{ .recalled = out.recalled, .damage = words(cx, e) };
    if (verdict.damage) |why| {
        out.why = std.fmt.allocPrint(a, "after the mining rounds, {s}", .{why}) catch why;
        @memset(out.recalled, false);
        return;
    }
    @memcpy(out.recalled, verdict.recalled);
    if (!all(out.recalled)) out.why = "after the mining rounds, it did not come back on its held-out questions";
}

/// One round of the plain change through the learner; false (and why) when it did not run, which no fact survives.
pub fn step(srv: *Server, a: Allocator, request: *const api.LearnRequest, out: *Rounds) bool {
    out.why = switch (run(srv, a, request)) {
        .learned => |l| {
            out.steps += l.steps;
            return true;
        },
        .failed => |message| message,
        .refused => |e| @errorName(e),
        .none => "the learner ended the round without a result",
    };
    @memset(out.recalled, false);
    return false;
}

/// A fact the lesson could not learn, and why.
fn missed(srv: *Server, out: *Out, id: u32, why: []const u8) void {
    srv.slide.mark(id, .missed);
    out.event("learned", .{ .id = id, .recalled = false, .message = why });
}

const Outcome = union(enum) {
    none,
    learned: struct { recalled: bool, steps: u32 },
    failed: []const u8,
    refused: api.LearnError,
};

/// One lesson through the engine's learner, waited for to its end.
fn run(srv: *Server, a: Allocator, request: *const api.LearnRequest) Outcome {
    var box: Box = .{ .io = srv.io, .gpa = srv.gpa };
    defer box.deinit();
    srv.engine.learn(request, box.sink()) catch |e| return .{ .refused = e };
    var outcome: Outcome = .none;
    var batch: std.ArrayList(Box.Copy) = .empty;
    defer batch.deinit(srv.gpa);
    while (true) {
        const finished = box.take(&batch);
        for (batch.items) |c| {
            switch (c.kind) {
                .learned => outcome = .{ .learned = .{ .recalled = c.recalled, .steps = c.steps } },
                .done => if (c.text.len > 0) {
                    outcome = .{ .failed = a.dupe(u8, c.text) catch "the learner failed" };
                },
            }
            srv.gpa.free(c.text);
        }
        batch.clearRetainingCapacity();
        if (finished) return outcome;
    }
}

/// The learn stream: events while the client listens.
const Out = struct {
    conn: *Conn,
    a: Allocator,
    open: bool = true,

    fn event(o: *Out, kind: []const u8, fields: anytype) void {
        if (!o.open) return;
        const obj = json.newObject(o.a) catch return;
        inline for (@typeInfo(@TypeOf(fields)).@"struct".field_names) |name| {
            const v = @field(fields, name);
            const value: json.Value = switch (@TypeOf(v)) {
                bool => .{ .bool = v },
                u32 => json.intValue(o.a, v) catch return,
                else => .{ .string = v },
            };
            obj.put(o.a, name, value) catch return;
        }
        sse.event(o.conn, o.a, kind, .{ .object = obj }) catch {
            o.open = false;
        };
    }

    fn fail(o: *Out, cx: *errors.Cx, e: anyerror) void {
        o.event("failed", .{ .message = words(cx, e) });
        sse.done(o.conn) catch {};
    }
};

fn words(cx: *errors.Cx, e: anyerror) []const u8 {
    return if (e == error.Refused and cx.message.len > 0) cx.message else @errorName(e);
}

fn now(srv: *Server) i64 {
    return @intCast(@divTrunc(std.Io.Clock.real.now(srv.io).toNanoseconds(), std.time.ns_per_ms));
}

fn refuse(conn: *Conn, a: Allocator, code: u16, message: []const u8) void {
    const o = json.newObject(a) catch return;
    o.put(a, "message", .{ .string = message }) catch return;
    o.put(a, "type", .{ .string = "invalid_request_error" }) catch return;
    const wrapped = json.newObject(a) catch return;
    wrapped.put(a, "error", .{ .object = o }) catch return;
    routes.sendValue(conn, a, code, .{ .object = wrapped });
}

/// Learn events copied off the engine's thread, read by the request's own thread.
const Box = struct {
    io: std.Io,
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    events: std.ArrayList(Copy) = .empty,
    done: bool = false,

    const Copy = struct { kind: std.meta.Tag(api.LearnEvent), text: []const u8 = "", recalled: bool = false, steps: u32 = 0 };

    fn sink(b: *Box) api.LearnSink {
        return .{ .ctx = b, .event = onEvent };
    }

    fn onEvent(ctx: *anyopaque, e: *const api.LearnEvent) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        var c: Copy = .{ .kind = e.* };
        switch (e.*) {
            .learned => |l| {
                c.recalled = l.recalled;
                c.steps = l.steps;
            },
            .done => |d| c.text = b.gpa.dupe(u8, d.message) catch "",
        }
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        b.events.append(b.gpa, c) catch b.gpa.free(c.text);
        if (e.* == .done) b.done = true;
        b.cond.signal(b.io);
    }

    /// Waits for events and moves them to `out`; true once `done` has arrived.
    fn take(b: *Box, out: *std.ArrayList(Copy)) bool {
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        while (b.events.items.len == 0 and !b.done) b.cond.waitUncancelable(b.io, &b.mutex);
        out.appendSlice(b.gpa, b.events.items) catch {};
        b.events.clearRetainingCapacity();
        return b.done;
    }

    fn deinit(b: *Box) void {
        for (b.events.items) |c| b.gpa.free(c.text);
        b.events.deinit(b.gpa);
    }
};

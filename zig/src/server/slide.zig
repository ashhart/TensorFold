//! Sliding Weights over HTTP: /v1/slide/learn streams learning, /v1/slide/graph lists or clears facts, /slide draws them.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const sse = @import("sse.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const State = @import("slide_graph.zig").State;
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

/// Queue the body's text for the engine's learner and stream its progress; the graph follows every event.
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
    const request: api.LearnRequest = .{ .text = text, .source = body.strField("source") orelse "chat" };
    var box: Box = .{ .io = srv.io, .gpa = srv.gpa };
    defer box.deinit();
    srv.engine.learn(&request, box.sink()) catch |e| return switch (e) {
        error.Unsupported => refuse(conn, a, 501, "this engine has no learner for its model family"),
        error.Busy, error.Closed => refuse(conn, a, 503, "the engine cannot take a learn request now"),
    };
    var open = if (sse.open(conn)) |_| true else |_| false;
    var ids: std.ArrayList([2]u32) = .empty;
    var batch: std.ArrayList(Box.Copy) = .empty;
    defer batch.deinit(srv.gpa);
    while (true) {
        const finished = box.take(&batch);
        for (batch.items) |c| {
            open = relay(srv, conn, a, c, &ids, request.source, open);
            srv.gpa.free(c.text);
        }
        batch.clearRetainingCapacity();
        if (finished) break;
    }
    if (open) sse.done(conn) catch {};
}

/// One event into the graph and, when `write`, onto the stream; false once the stream has closed.
fn relay(srv: *Server, conn: *Conn, a: Allocator, c: Box.Copy, ids: *std.ArrayList([2]u32), source: []const u8, write: bool) bool {
    const o = json.newObject(a) catch return false;
    var kind: []const u8 = @tagName(c.kind);
    switch (c.kind) {
        .fact => {
            const at: i64 = @intCast(@divTrunc(std.Io.Clock.real.now(srv.io).toNanoseconds(), std.time.ns_per_ms));
            const id = srv.slide.add(c.text, source, at);
            ids.append(a, .{ c.id, id }) catch {};
            o.put(a, "id", json.intValue(a, id) catch return false) catch return false;
            o.put(a, "text", .{ .string = c.text }) catch return false;
        },
        .learning, .learned => {
            const id = for (ids.items) |p| {
                if (p[0] == c.id) break p[1];
            } else 0;
            const state: State = if (c.kind == .learning) .learning else if (c.recalled) .learned else .missed;
            srv.slide.mark(id, state);
            o.put(a, "id", json.intValue(a, id) catch return false) catch return false;
            if (c.kind == .learned) o.put(a, "recalled", .{ .bool = c.recalled }) catch return false;
        },
        .saved => o.put(a, "tensors", json.intValue(a, c.tensors) catch return false) catch return false,
        .done => {
            if (c.text.len == 0) return true;
            kind = "failed";
            o.put(a, "message", .{ .string = c.text }) catch return false;
        },
    }
    if (!write) return false;
    sse.event(conn, a, kind, .{ .object = o }) catch return false;
    return true;
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

    const Copy = struct { kind: std.meta.Tag(api.LearnEvent), id: u32 = 0, text: []const u8 = "", recalled: bool = false, tensors: u32 = 0 };

    fn sink(b: *Box) api.LearnSink {
        return .{ .ctx = b, .event = onEvent };
    }

    fn onEvent(ctx: *anyopaque, e: *const api.LearnEvent) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        var c: Copy = .{ .kind = e.* };
        switch (e.*) {
            .fact => |f| {
                c.id = f.id;
                c.text = b.gpa.dupe(u8, f.text) catch "";
            },
            .learning => |id| c.id = id,
            .learned => |l| {
                c.id = l.id;
                c.recalled = l.recalled;
            },
            .saved => |s| c.tensors = s.tensors,
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

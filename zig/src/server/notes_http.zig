//! Storage-only /v1/notes handler, dispatched after the server's ordinary /v1 authentication.
const std = @import("std");
const json = @import("json");
const request = @import("notes/http_request.zig");
const values = @import("notes/http_values.zig");
const ops = @import("notes/operations.zig");
const erase = @import("notes/erase_storage.zig");
const erase_reply = @import("notes_erase_reply.zig");
const line_mod = @import("notes/line.zig");
const http_body = @import("http_body.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
pub const route = request.route;

pub fn handle(srv: *Server, conn: *Conn, a: Allocator, r: request.Route) void {
    reply(srv, conn, a, r) catch fail(conn, a, 500, "out of memory");
}

fn reply(srv: *Server, conn: *Conn, a: Allocator, r: request.Route) Allocator.Error!void {
    const get = std.mem.eql(u8, conn.method, "GET");
    const payload = switch (try http_body.read(conn, a, request.body_limit)) {
        .ok => |b| b,
        .refused => |why| return fail(conn, a, if (std.mem.startsWith(u8, why, "request body exceeds")) 413 else 400, why),
    };
    const body = if (get) "" else payload;
    const s = srv.notes orelse return fail(conn, a, 404, "notes are off: this server was started without --notes");
    if (!validErase(conn.path)) return fail(conn, a, 400, "erase must occur once with true, 1, false or 0");
    if (r.bad_id) return fail(conn, a, 404, "a note id is a whole number: /v1/notes/ID");
    srv.notes_mutex.lockUncancelable(srv.io);
    defer srv.notes_mutex.unlock(srv.io);
    if (get) {
        const all = try ops.list(s, a, r.id != null or flag(conn.path, "history"));
        if (r.id) |id| {
            for (all) |v| if (v.id == id) return send(conn, a, 200, try values.viewValue(a, v));
            return fail(conn, a, 404, "no note has this id");
        }
        const o = try json.newObject(a);
        try o.put(a, "object", .{ .string = "list" });
        const data = try a.alloc(json.Value, all.len);
        for (all, data) |v, *slot| slot.* = try values.viewValue(a, v);
        try o.put(a, "data", .{ .array = data });
        try o.put(a, "history", .{ .bool = flag(conn.path, "history") });
        const work = if (s.journal) |j| j.erases.items else &.{};
        const pending = try a.alloc(json.Value, work.len);
        for (pending, work) |*value, intent| value.* = try json.intValue(a, intent.id);
        try o.put(a, "pending_erases", .{ .array = pending });
        return send(conn, a, 200, .{ .object = o });
    }
    if (std.mem.eql(u8, conn.method, "DELETE") and flag(conn.path, "erase")) {
        const id = r.id orelse return fail(conn, a, 405, "erase takes a note id");
        if (body.len != 0) return fail(conn, a, 400, "erase takes no body");
        _ = srv.generating.fetchAdd(1, .seq_cst);
        defer _ = srv.generating.fetchSub(1, .seq_cst);
        if (srv.draining.load(.seq_cst)) return @import("routes.zig").draining(conn, false);
        if (srv.idle_manager) |m| m.begin();
        defer if (srv.idle_manager) |m| m.end();
        if (srv.idle_manager) |m| {
            const problem = try @import("idle_unload.zig").wakeReload(m, srv, a);
            if (problem.len > 0) return fail(conn, a, 503, problem);
        }
        var p = erase.prepare(s, a, id) catch |e| return eraseError(conn, a, e);
        defer p.deinit();
        const response = try erase_reply.Reply.prepare(a, id, &p);
        p.commit(a) catch |e| return eraseError(conn, a, e);
        const encoded = response.finish(srv, a, &p) catch {
            conn.sendJson(500, "{\"erased\":true,\"complete\":false,\"state\":\"pending\",\"error\":{\"message\":\"erase applied; response capacity exceeded\"}}");
            return;
        };
        conn.sendJson(if (p.unsynced or p.retire_failed) 500 else 200, encoded);
        return;
    }
    var why: ops.Refusal = .{};
    const ask = request.parse(a, conn.method, r, body, &why) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Refused => fail(conn, a, why.status, why.message),
    };
    const told = @max(s.model.told_last + 100, localOrdinal(srv.io) * line_mod.tpd);
    var prepared = ops.prepare(s, a, ask, told, &why) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Refused => fail(conn, a, why.status, why.message),
    };
    defer prepared.deinit();
    const encoded = try json.stringify(a, try values.writtenValue(a, ask.op, prepared.reply), .{});
    prepared.commit(a, &why) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Refused => fail(conn, a, why.status, why.message),
    };
    conn.sendJson(if (prepared.reply.verdict.learn()) 201 else 200, encoded);
}

fn eraseError(conn: *Conn, a: Allocator, e: anyerror) Allocator.Error!void {
    switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.NoNote => fail(conn, a, 404, "no note has this id"),
        error.ReadOnly, error.SnapshotNote => fail(conn, a, 409, "only journal-backed notes can be erased"),
        error.JournalPoisoned => fail(conn, a, 503, "notes journal needs a restart"),
        error.SnapshotChanged => fail(conn, a, 409, "notes snapshot changed since load"),
        else => fail(conn, a, 500, "notes journal rewrite failed"),
    }
}

fn send(conn: *Conn, a: Allocator, code: u16, v: json.Value) Allocator.Error!void {
    conn.sendJson(code, try json.stringify(a, v, .{}));
}

fn fail(conn: *Conn, a: Allocator, code: u16, message: []const u8) void {
    const kind = switch (code) {
        404 => "not_found_error",
        409 => "conflict_error",
        413 => "request_too_large",
        503 => "unavailable_error",
        500 => "server_error",
        else => "invalid_request_error",
    };
    const body = std.fmt.allocPrint(a, "{{\"error\": {{\"message\": {s}, \"type\": \"{s}\", \"code\": {d}}}}}", .{ json.quote(a, message, .{}) catch "\"\"", kind, code }) catch return;
    conn.sendJson(code, body);
}

fn flag(path: []const u8, name: []const u8) bool {
    const q = std.mem.indexOfScalar(u8, path, '?') orelse return false;
    var it = std.mem.splitScalar(u8, path[q + 1 ..], '&');
    while (it.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse kv.len;
        if (!std.mem.eql(u8, kv[0..eq], name)) continue;
        const val = if (eq < kv.len) kv[eq + 1 ..] else "1";
        return std.mem.eql(u8, val, "1") or std.ascii.eqlIgnoreCase(val, "true");
    }
    return false;
}

fn localOrdinal(io: std.Io) i64 {
    const Tm = extern struct { sec: c_int, min: c_int, hour: c_int, mday: c_int, mon: c_int, year: c_int, wday: c_int, yday: c_int, isdst: c_int, gmtoff: c_long, zone: ?[*:0]const u8 };
    const ext = struct {
        extern "c" fn localtime_r(t: *const i64, out: *Tm) ?*Tm;
        extern "c" fn _localtime64_s(out: *Tm, t: *const i64) c_int;
    };
    const now: i64 = @intCast(@divFloor(std.Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_s));
    var tm: Tm = undefined;
    const ok = if (@import("sys").windows) ext._localtime64_s(&tm, &now) == 0 else ext.localtime_r(&now, &tm) != null;
    if (!ok) return @divFloor(now, 86400) + 719163;
    return line_mod.ordinalOf(@as(i64, tm.year) + 1900, @intCast(tm.mon + 1), @intCast(tm.mday));
}

fn validErase(path: []const u8) bool {
    const q = std.mem.indexOfScalar(u8, path, '?') orelse return true;
    var seen = false;
    var it = std.mem.splitScalar(u8, path[q + 1 ..], '&');
    while (it.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse kv.len;
        if (!std.mem.eql(u8, kv[0..eq], "erase")) continue;
        if (seen or eq == kv.len) return false;
        seen = true;
        const v = kv[eq + 1 ..];
        if (!std.mem.eql(u8, v, "1") and !std.mem.eql(u8, v, "0") and
            !std.ascii.eqlIgnoreCase(v, "true") and !std.ascii.eqlIgnoreCase(v, "false")) return false;
    }
    return true;
}

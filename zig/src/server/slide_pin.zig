//! POST /v1/slide/facts/{id}/pin, a durable pin or unpin under the same lock as learning and deletion.
const std = @import("std");
const json = @import("json");
const body = @import("http_body.zig");
const slide = @import("slide.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
pub const prefix = "/v1/slide/facts/";

pub fn id(path: []const u8) !u32 {
    if (!std.mem.startsWith(u8, path, prefix) or !std.mem.endsWith(u8, path, "/pin")) return error.BadId;
    if (path.len <= prefix.len + 4) return error.BadId;
    const raw = path[prefix.len .. path.len - 4];
    const n = std.fmt.parseInt(u32, raw, 10) catch return error.BadId;
    if (n == 0) return error.BadId;
    return n;
}

pub fn value(v: json.Value) !bool {
    const b = v.field("pinned") orelse return true;
    return if (b == .bool) b.bool else error.BadPin;
}

pub fn route(srv: *Server, conn: *Conn, a: std.mem.Allocator, path: []const u8) void {
    const raw = switch (body.read(conn, a, body.limit) catch return) {
        .ok => |v| v,
        .refused => |why| return slide.refuse(conn, a, 400, why),
    };
    const n = id(path) catch return slide.refuse(conn, a, 400, "pin needs a positive fact id");
    const v = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return) {
        .ok => |v| v,
        .err => |why| return slide.refuse(conn, a, 400, why),
    };
    const pinned = value(v) catch return slide.refuse(conn, a, 400, "pinned must be true or false");
    if (@import("slide_forget.zig").admit(srv, a)) |r| return slide.refuse(conn, a, r.code, r.message);
    defer srv.teacher.learning.unlock(srv.io);
    srv.slide.pin(n, pinned) catch |e| return slide.refuse(conn, a, switch (e) {
        error.NoSuchFact => 404,
        error.FactNotLearned => 409,
        else => 500,
    }, "the pin was not changed: the fact must be learned and its graph writable");
    const out = std.fmt.allocPrint(a, "{{\"id\":{d},\"pinned\":{s}}}", .{ n, if (pinned) "true" else "false" }) catch return;
    conn.sendJson(200, out);
}

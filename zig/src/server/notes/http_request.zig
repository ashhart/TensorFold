//! Notes HTTP routing and body validation, without inference or index dependencies.
const std = @import("std");
const json = @import("json");
const ops = @import("operations.zig");
const write = @import("write.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The most a notes request body may hold, and a note's text, and a subject.
pub const body_limit: usize = 1 << 20;
pub const text_limit: usize = 16 << 10;
pub const subject_limit: usize = 200;

/// A path this file answers: /v1/notes, or /v1/notes/ID (null id), or not ours (null).
pub const Route = struct { id: ?u32, bad_id: bool = false };

pub fn route(path: []const u8) ?Route {
    if (std.mem.eql(u8, path, "/v1/notes")) return .{ .id = null };
    if (!std.mem.startsWith(u8, path, "/v1/notes/")) return null;
    const tail = path["/v1/notes/".len..];
    for (tail) |c| if (!std.ascii.isDigit(c)) return .{ .id = null, .bad_id = true };
    const id = std.fmt.parseInt(u32, tail, 10) catch return .{ .id = null, .bad_id = true };
    return .{ .id = id };
}

fn refuse(why: *ops.Refusal, status: u16, message: []const u8) ops.Error {
    why.* = .{ .status = status, .message = message };
    return error.Refused;
}

/// Parse a write body after authentication; GET and erasure are handled by their own route paths.
pub fn parse(a: Allocator, m: []const u8, r: Route, body: []const u8, why: *ops.Refusal) ops.Error!ops.Ask {
    if (r.bad_id) return refuse(why, 404, "a note id is a whole number: /v1/notes/ID");
    if (body.len > body_limit) return refuse(why, 413, "request body exceeds the notes limit");
    const post = std.mem.eql(u8, m, "POST") or std.mem.eql(u8, m, "PUT");
    const delete = std.mem.eql(u8, m, "DELETE");
    if (!post and !delete or post and std.mem.eql(u8, m, "PUT") and r.id == null) return refuse(why, 405, "use GET or POST on /v1/notes, and GET, POST, PUT or DELETE on /v1/notes/ID");
    if (delete and r.id == null) return refuse(why, 405, "DELETE takes a note: /v1/notes/ID");
    const v = if (body.len == 0 and delete) Value.null else switch (try json.parse(a, body)) {
        .ok => |x| x,
        .err => |e| return refuse(why, 400, try std.fmt.allocPrint(a, "the request body is not JSON: {s}", .{e})),
    };
    if (v != .null and v != .object) return refuse(why, 400, "the request body must be a JSON object");
    const allowed: []const []const u8 = if (delete) &.{"text"} else if (r.id != null) &.{"text"} else &.{ "text", "subject", "replaces" };
    if (v == .object) for (v.object.keys()) |k| {
        for (allowed) |name| {
            if (std.mem.eql(u8, k, name)) break;
        } else return refuse(why, 400, try std.fmt.allocPrint(a, "unknown field {s}; the fields here are {s}", .{ k, if (allowed.len == 1) "text" else "text, subject and replaces" }));
    };
    var ask: ops.Ask = .{ .op = if (delete) .take_back else if (r.id != null) .change else .add, .text = write.default_takeback };
    if (r.id) |id| ask.target = id;
    if (v.field("text")) |t| {
        ask.text = t.str() orelse return refuse(why, 400, "text must be a string");
    } else if (!delete) return refuse(why, 400, "text is required: the note's sentence, as the user said it");
    if (std.mem.trim(u8, ask.text, " \t\r\n").len == 0) return refuse(why, 400, "text must not be empty");
    if (ask.text.len > text_limit) return refuse(why, 413, "text is over 16 KiB: a note is one thing the user said");
    if (!std.unicode.utf8ValidateSlice(ask.text)) return refuse(why, 400, "text must be valid Unicode (no lone surrogates)");
    if (v.field("subject")) |sub| {
        const name = sub.str() orelse return refuse(why, 400, "subject must be a string");
        if (!goodSubject(name)) return refuse(why, 400, "subject must be 1 to 200 bytes of text with no control characters and no space at either end");
        ask.subject = name;
    }
    if (v.field("replaces")) |x| {
        const n = if (x == .int) x.int64() else null;
        ask.target = std.math.cast(u32, n orelse -1) orelse return refuse(why, 400, "replaces must be a note id");
        ask.op = .change;
    }

    return ask;
}
fn goodSubject(s: []const u8) bool {
    if (s.len == 0 or s.len > subject_limit or !std.unicode.utf8ValidateSlice(s)) return false;
    for (s) |c| if (c < 0x20 or c == 0x7f) return false;
    return s[0] != ' ' and s[s.len - 1] != ' ';
}

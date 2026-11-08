//! POST /v1/ocr and POST /ocr: the body read once, the /ocr dialect chosen (docs/api.md OCR), then its handler.
const std = @import("std");
const Allocator = std.mem.Allocator;
const errors = @import("../errors.zig");
const http_body = @import("../http_body.zig");
const multipart = @import("../multipart.zig");
const query = @import("../query.zig");
const openai = @import("../openai.zig");
const routes = @import("../routes.zig");
const Conn = @import("../http_conn.zig").Conn;
const Server = @import("../server.zig").Server;
const mistral = @import("mistral.zig");
const mistral_reply = @import("mistral_reply.zig");
const liteparse = @import("liteparse.zig");

/// The client contracts POST /ocr answers in.
pub const Dialect = enum { liteparse };

/// A chosen dialect, with the multipart parts when choosing it parsed them.
pub const Selected = struct { dialect: Dialect, parts: ?[]const multipart.Part = null };

/// A request no dialect takes: its status and message, sent in the OpenAI error shape.
pub const Refusal = struct { status: u16, message: []const u8 };

pub const Selection = union(enum) { ok: Selected, refused: Refusal };

/// The handler for both OCR paths; ``route`` is the path as the dispatcher reads it (no query, no trailing slash).
pub fn post(srv: *Server, conn: *Conn, a: Allocator, route: []const u8) void {
    const is_mistral = std.mem.eql(u8, route, "/v1/ocr");
    const body = switch (http_body.read(conn, a, http_body.limit) catch {
        conn.close = true;
        if (is_mistral) return mistral_reply.sendFault(conn, a, .{ .class = .internal, .message = "out of memory" });
        return sendOpenAi(conn, a, 500, .server, "out of memory");
    }) {
        .ok => |b| b,
        .refused => |message| return if (is_mistral) mistral.refuseBody(conn, a, message) else sendOpenAi(conn, a, 400, .request, message),
    };
    const gone: openai.Gone = .{ .conn = conn };
    if (is_mistral) return mistral.post(srv, conn, a, body, gone);
    const sel = select(a, conn.path, conn.header("Content-Type"), body) catch return sendOpenAi(conn, a, 500, .server, "out of memory");
    switch (sel) {
        .refused => |r| sendOpenAi(conn, a, r.status, .request, r.message),
        .ok => |s| dialect(srv, conn, a, s, body, gone),
    }
}

/// Runs the chosen dialect, which gates the request and answers in its own error shape.
fn dialect(srv: *Server, conn: *Conn, a: Allocator, s: Selected, body: []const u8, gone: openai.Gone) void {
    switch (s.dialect) {
        .liteparse => liteparse.post(srv, conn, a, s.parts, conn.header("Content-Type"), body, gone),
    }
}

/// The /ocr dialect: ``?format=`` first, then the media type, then the multipart part names (docs/api.md OCR).
pub fn select(a: Allocator, raw_path: []const u8, content_type: ?[]const u8, body: []const u8) Allocator.Error!Selection {
    const format = query.get(a, raw_path, "format") catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Malformed => return refuse(400, "the format query parameter is malformed"),
    };
    if (format) |f| {
        const d = std.meta.stringToEnum(Dialect, f) orelse return refuse(400, "format must be liteparse");
        return .{ .ok = .{ .dialect = d } };
    }
    const ct = std.mem.trim(u8, content_type orelse "", " \t");
    if (ct.len == 0) return refuse(415, "the request has no Content-Type; send multipart/form-data (LiteParse)");
    const media = std.mem.trim(u8, ct[0 .. std.mem.indexOfScalar(u8, ct, ';') orelse ct.len], " \t");
    if (!std.ascii.eqlIgnoreCase(media, "multipart/form-data"))
        return refuse(415, try std.fmt.allocPrint(a, "unsupported Content-Type {s}; send multipart/form-data (LiteParse)", .{media}));
    const bound = multipart.boundary(ct) orelse return refuse(400, "multipart/form-data needs a boundary parameter");
    const parts = multipart.parse(a, body, bound) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Malformed => return refuse(400, "the multipart body is malformed"),
    };
    if (hasFiles(parts)) return refuse(400, files_parts);
    var file: usize = 0;
    for (parts) |p| file += @intFromBool(std.mem.eql(u8, p.name, "file"));
    if (file == 1) return .{ .ok = .{ .dialect = .liteparse, .parts = parts } };
    return refuse(400, "a multipart OCR request needs exactly one file part (LiteParse)");
}

/// The refusal for ``files`` parts: on /ocr this server reads only LiteParse's one ``file`` part, and drops nothing.
pub const files_parts = "a multipart OCR request has files parts, which this server does not read; LiteParse sends one file part";

/// Whether ``parts`` hold a ``files`` part; refused on every path, ``?format=`` included.
pub fn hasFiles(parts: []const multipart.Part) bool {
    for (parts) |p| if (std.mem.eql(u8, p.name, "files")) return true;
    return false;
}

fn refuse(status: u16, message: []const u8) Selection {
    return .{ .refused = .{ .status = status, .message = message } };
}

/// ``{"error": {"message", "type"}}`` with ``status``: the shape of refusals made before a dialect is known.
pub fn sendOpenAi(conn: *Conn, a: Allocator, status: u16, kind: errors.Kind, message: []const u8) void {
    const cx: errors.Cx = .{ .a = a, .kind = kind, .message = message };
    const body = openai.errorBody(a, &cx, null) catch return conn.sendJson(500, "{\"error\": {\"message\": \"out of memory\"}}");
    routes.sendValue(conn, a, status, openai.wrapError(a, body) catch return conn.sendJson(500, "{\"error\": {\"message\": \"out of memory\"}}"));
}

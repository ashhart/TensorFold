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
const doctr = @import("doctr.zig");
const liteparse = @import("liteparse.zig");

/// The client contracts POST /ocr answers in.
pub const Dialect = enum { doctr, liteparse };

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
        .doctr => doctr.post(srv, conn, a, s.parts, body, gone),
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
        const d = std.meta.stringToEnum(Dialect, f) orelse return refuse(400, "format must be doctr or liteparse");
        return .{ .ok = .{ .dialect = d } };
    }
    const ct = std.mem.trim(u8, content_type orelse "", " \t");
    if (ct.len == 0) return refuse(415, "the request has no Content-Type; send multipart/form-data (docTR, LiteParse)");
    const media = std.mem.trim(u8, ct[0 .. std.mem.indexOfScalar(u8, ct, ';') orelse ct.len], " \t");
    if (!std.ascii.eqlIgnoreCase(media, "multipart/form-data"))
        return refuse(415, try std.fmt.allocPrint(a, "unsupported Content-Type {s}; send multipart/form-data (docTR, LiteParse)", .{media}));
    const bound = multipart.boundary(ct) orelse return refuse(400, "multipart/form-data needs a boundary parameter");
    const parts = multipart.parse(a, body, bound) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Malformed => return refuse(400, "the multipart body is malformed"),
    };
    if (mixed(parts)) return refuse(400, mixed_parts);
    var files: usize = 0;
    var file: usize = 0;
    for (parts) |p| {
        files += @intFromBool(std.mem.eql(u8, p.name, "files"));
        file += @intFromBool(std.mem.eql(u8, p.name, "file"));
    }
    if (files > 0) return .{ .ok = .{ .dialect = .doctr, .parts = parts } };
    if (file == 1) return .{ .ok = .{ .dialect = .liteparse, .parts = parts } };
    return refuse(400, "a multipart OCR request needs one or more files parts (docTR) or exactly one file part (LiteParse)");
}

/// The refusal for parts of both dialects: docTR would drop the ``file`` part and LiteParse the ``files`` parts.
pub const mixed_parts = "a multipart OCR request has both files parts (docTR) and a file part (LiteParse); send one kind";

/// Whether ``parts`` hold both ``files`` and ``file`` parts; refused on every path, ``?format=`` included.
pub fn mixed(parts: []const multipart.Part) bool {
    var files = false;
    var file = false;
    for (parts) |p| {
        files = files or std.mem.eql(u8, p.name, "files");
        file = file or std.mem.eql(u8, p.name, "file");
    }
    return files and file;
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

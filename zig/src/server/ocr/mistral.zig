//! POST /v1/ocr in Mistral's OCR contract: fields and document, then the page loop (docs/api.md OCR: Mistral).
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const json = @import("json");
const errors = @import("../errors.zig");
const images = @import("../images.zig");
const http_body = @import("../http_body.zig");
const openai = @import("../openai.zig");
const Conn = @import("../http_conn.zig").Conn;
const Server = @import("../server.zig").Server;
const types = @import("types.zig");
const run = @import("run.zig");
const fetch = @import("fetch.zig");
const reply = @import("mistral_reply.zig");
const Value = json.Value;
const Fault = types.Fault;
const Step = run.Step;

pub const Options = reply.Options;

/// What decoding needs from the server: the reader's geometry and the URL fetcher.
pub const Facts = struct {
    geometry: api.ocr.Geometry,
    fetch: ?images.Fetch,

    pub fn of(srv: *const Server) Facts {
        return .{ .geometry = srv.info.ocr.?.geometry, .fetch = if (srv.config.vision_urls) images.Fetch.network(srv.io) else null };
    }
};

/// A decoded request: the reply's options, its one document and the document's decoded size.
pub const Request = struct { options: Options, source: types.Source, doc_size: usize };

/// Answers one POST /v1/ocr whose body has been read; a client that leaves gets nothing.
pub fn post(srv: *Server, conn: *Conn, a: Allocator, body: []const u8, gone: anytype) void {
    var fault: Fault = .{ .class = .internal, .message = "" };
    const out = handle(srv, a, body, gone, &fault) catch |e| switch (e) {
        error.Faulted => return reply.sendFault(conn, a, fault),
        error.OutOfMemory => return reply.sendFault(conn, a, .{ .class = .internal, .message = "out of memory" }),
        error.Cancelled => {
            conn.close = true;
            return;
        },
    };
    conn.sendJson(200, out);
}

/// Sends a body-read refusal (size, framing) in Mistral's error shape.
pub fn refuseBody(conn: *Conn, a: Allocator, message: []const u8) void {
    reply.sendFault(conn, a, .{ .class = .bad_input, .message = message });
}

fn handle(srv: *Server, a: Allocator, body: []const u8, gone: anytype, fault: *Fault) Step![]const u8 {
    try run.gate(srv, a, fault);
    const raw = switch (try json.parse(a, body)) {
        .ok => |v| v,
        .err => |m| return bad(a, fault, &.{"body"}, "{s}", .{m}),
    };
    const req = try decode(a, raw, Facts.of(srv), fault);
    var enc: reply.Encoder = .{ .o = req.options, .gpa = srv.gpa };
    try run.run(srv, a, &.{req.source}, enc.sink(), fault, gone);
    return enc.reply(a, openai.replyModel(srv, raw), req.doc_size);
}

/// Fills ``fault`` as bad input at ``loc`` (copied into ``a``); returns error.Faulted.
fn bad(a: Allocator, fault: *Fault, loc: []const []const u8, comptime fmt: []const u8, args: anytype) Step {
    fault.* = .{ .class = .bad_input, .message = try std.fmt.allocPrint(a, fmt, args), .loc = try a.dupe([]const u8, loc) };
    return error.Faulted;
}

/// The request's fields checked as docs/api.md OCR: Mistral states, and the document's bytes; ``null``: absent.
pub fn decode(a: Allocator, body: Value, facts: Facts, fault: *Fault) Step!Request {
    if (body != .object) return bad(a, fault, &.{"body"}, "the body must be a JSON object", .{});
    if (body.field("model")) |m| if (m != .string) return bad(a, fault, &.{ "body", "model" }, "model must be a string or null", .{});
    // annotations change content and are not supported: any non-null value is refused, never ignored
    for ([_][]const u8{ "bbox_annotation_format", "document_annotation_format", "document_annotation_prompt" }) |name|
        if (body.field(name) != null) return bad(a, fault, &.{ "body", name }, "annotations are not supported", .{});
    var o: Options = .{
        .header = try flag(a, fault, body, "extract_header") orelse false,
        .footer = try flag(a, fault, body, "extract_footer") orelse false,
        .blocks = try flag(a, fault, body, "include_blocks") orelse true,
        .base64 = try flag(a, fault, body, "include_image_base64") orelse false,
        .image_limit = try count(a, fault, body, "image_limit"),
        .image_min_size = try count(a, fault, body, "image_min_size"),
    };
    if (body.field("table_format")) |v| {
        const s = if (v == .string) v.string else "";
        o.tables = std.meta.stringToEnum(reply.TableFormat, s) orelse return bad(a, fault, &.{ "body", "table_format" }, "table_format must be \"markdown\", \"html\" or null", .{});
    }
    if (body.field("confidence_scores_granularity")) |v| {
        // accepted and left out of the reply: scores describe the content, they are not content
        const ok = v == .string and (std.mem.eql(u8, v.string, "word") or std.mem.eql(u8, v.string, "page") or std.mem.eql(u8, v.string, "block"));
        if (!ok) return bad(a, fault, &.{ "body", "confidence_scores_granularity" }, "confidence_scores_granularity must be \"word\", \"page\", \"block\" or null", .{});
    }
    if (facts.geometry == .none and (o.header or o.footer))
        return bad(a, fault, &.{ "body", if (o.header) "extract_header" else "extract_footer" }, "this model does not mark page headers or footers", .{});
    // this route reads one page image: pages selects PDF pages, and a PDF document is refused
    if (body.field("pages") != null) return bad(a, fault, &.{ "body", "pages" }, "pages selects pages of a PDF document; POST /v1/ocr reads one page image", .{});
    const bytes = try document(a, fault, body, facts);
    const image = switch (images.sniff(bytes)) {
        .png, .jpeg, .webp => true,
        else => false,
    };
    // the image signature first: a PNG text chunk or a JPEG comment may hold "%PDF-"
    const is_pdf = !image and std.mem.indexOf(u8, bytes[0..@min(bytes.len, 1024)], "%PDF-") != null;
    if (is_pdf) return bad(a, fault, &.{ "body", "document" }, "PDF documents are not read on POST /v1/ocr; send each page as a PNG, JPEG or WebP image", .{});
    if (!image) return bad(a, fault, &.{ "body", "document" }, "the document is not a PNG, JPEG or WebP image", .{});
    return .{ .options = o, .source = .{ .name = "document", .bytes = bytes }, .doc_size = bytes.len };
}

fn flag(a: Allocator, fault: *Fault, body: Value, name: []const u8) Step!?bool {
    const v = body.field(name) orelse return null;
    if (v != .bool) return bad(a, fault, &.{ "body", name }, "{s} must be a boolean or null", .{name});
    return v.bool;
}

fn count(a: Allocator, fault: *Fault, body: Value, name: []const u8) Step!?u64 {
    const v = body.field(name) orelse return null;
    if (v != .int or v.int[0] == '-') return bad(a, fault, &.{ "body", name }, "{s} must be a non-negative integer or null", .{name});
    return std.fmt.parseInt(u64, v.int, 10) catch std.math.maxInt(u64); // past u64: no bound, as an unbounded count
}

/// The document's bytes: a data URL, or an https URL fetched when --vision-urls is on.
fn document(a: Allocator, fault: *Fault, body: Value, facts: Facts) Step![]const u8 {
    const doc = body.field("document") orelse return bad(a, fault, &.{ "body", "document" }, "document is required", .{});
    const kind = doc.strField("type") orelse return bad(a, fault, &.{ "body", "document", "type" }, "document.type must be \"document_url\", \"image_url\" or \"file\"", .{});
    const url: []const u8 = if (std.mem.eql(u8, kind, "document_url"))
        doc.strField("document_url") orelse return bad(a, fault, &.{ "body", "document", "document_url" }, "document_url must be a string", .{})
    else if (std.mem.eql(u8, kind, "image_url")) blk: {
        const iu = doc.field("image_url") orelse .null;
        // a string, or {url, detail} with detail ignored
        break :blk (if (iu == .string) iu.string else iu.strField("url")) orelse return bad(a, fault, &.{ "body", "document", "image_url" }, "image_url must be a string or an object with a url string", .{});
    } else if (std.mem.eql(u8, kind, "file"))
        return bad(a, fault, &.{ "body", "document" }, "file documents are not supported; send document_url with a data URL", .{})
    else
        return bad(a, fault, &.{ "body", "document", "type" }, "document.type must be \"document_url\", \"image_url\" or \"file\"", .{});
    const loc: []const []const u8 = &.{ "body", "document", kind };
    var cx: errors.Cx = .{ .a = a };
    if (std.ascii.startsWithIgnoreCase(url, "data:")) return images.dataBytesOf(&cx, url, http_body.limit, "document") catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Refused => bad(a, fault, loc, "{s}", .{cx.message}),
    };
    return switch (try fetch.https(a, facts.fetch, url, false)) {
        .bytes => |b| b,
        .refused => |m| bad(a, fault, loc, "{s}", .{m}),
    };
}

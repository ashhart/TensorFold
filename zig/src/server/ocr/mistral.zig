//! POST /v1/ocr in Mistral's OCR contract: fields and document, then the page loop (docs/api.md OCR: Mistral).
const std = @import("std");
const Allocator = std.mem.Allocator;
const json = @import("json");
const errors = @import("../errors.zig");
const images = @import("../images.zig");
const http_body = @import("../http_body.zig");
const openai = @import("../openai.zig");
const pdf = @import("../pdf.zig");
const Conn = @import("../http_conn.zig").Conn;
const Server = @import("../server.zig").Server;
const types = @import("types.zig");
const budget = @import("budget.zig");
const run = @import("run.zig");
const fetch = @import("fetch.zig");
const reply = @import("mistral_reply.zig");
const Value = json.Value;
const Fault = types.Fault;
const Step = run.Step;

pub const Options = reply.Options;

/// The most page entries a ``pages`` value may name before expansion (a range counts its length).
pub const max_page_entries = 1024;

/// What decoding needs from the server: the PDF switch, the URL fetcher and the page budget.
pub const Facts = struct {
    pdf: bool,
    fetch: ?images.Fetch,
    pages_left: u32,

    pub fn of(srv: *const Server) Facts {
        return .{ .pdf = srv.config.vision_pdf, .fetch = if (srv.config.vision_urls) images.Fetch.network(srv.io) else null, .pages_left = budget.Pages.init().left };
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
    const pages: ?[]const u32 = if (body.field("pages")) |v| try pageList(a, fault, v, facts.pages_left) else null;
    const bytes = try document(a, fault, body, facts);
    const image = switch (images.sniff(bytes)) {
        .png, .jpeg, .webp => true,
        else => false,
    };
    // the image signature first: a PNG text chunk or a JPEG comment may hold "%PDF-"
    const is_pdf = !image and std.mem.indexOf(u8, bytes[0..@min(bytes.len, 1024)], "%PDF-") != null;
    if (is_pdf) {
        if (!facts.pdf) return bad(a, fault, &.{ "body", "document" }, "{s}", .{pdf.off_message});
        const sel: pdf.Selection = if (pages) |l| .{ .list = l } else .{ .all_within = std.math.maxInt(u32) };
        return .{ .options = o, .source = .{ .bytes = bytes, .kind = .pdf, .sel = sel }, .doc_size = bytes.len };
    }
    if (!image) return bad(a, fault, &.{ "body", "document" }, "the document is not a PDF, PNG, JPEG or WebP file", .{});
    if (pages) |l| if (l.len != 1 or l[0] != 0) return bad(a, fault, &.{ "body", "pages" }, "an image has one page; pages may select only page 0", .{});
    return .{ .options = o, .source = .{ .bytes = bytes, .kind = .image, .sel = .{ .all_within = 1 } }, .doc_size = bytes.len };
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

/// ``pages`` (indices, or a string of ``n`` and ``a-b`` ranges) as sorted distinct indices, bounded before expanding.
pub fn pageList(a: Allocator, fault: *Fault, v: Value, left: u32) Step![]const u32 {
    const loc: []const []const u8 = &.{ "body", "pages" };
    var ranges: std.ArrayList([2]u32) = .empty;
    var sum: u64 = 0;
    switch (v) {
        .array => |items| for (items) |item| {
            const n = (if (item == .int) std.fmt.parseInt(u32, item.int, 10) catch null else null) orelse
                return bad(a, fault, loc, "pages must hold page indices from 0 to 4294967295", .{});
            sum += 1;
            if (sum > max_page_entries) return bad(a, fault, loc, "pages selects too many pages", .{});
            try ranges.append(a, .{ n, n });
        },
        .string => |s| if (std.mem.trim(u8, s, " \t").len > 0) {
            var it = std.mem.splitScalar(u8, s, ',');
            while (it.next()) |raw| {
                const part = std.mem.trim(u8, raw, " \t");
                const dash = std.mem.indexOfScalar(u8, part, '-');
                const lo = try index(a, fault, if (dash) |d| part[0..d] else part);
                const hi = if (dash) |d| try index(a, fault, part[d + 1 ..]) else lo;
                if (lo > hi) return bad(a, fault, loc, "pages range {s} runs backwards", .{part});
                sum += @as(u64, hi) - lo + 1;
                if (sum > max_page_entries) return bad(a, fault, loc, "pages selects too many pages", .{});
                try ranges.append(a, .{ lo, hi });
            }
        },
        else => return bad(a, fault, loc, "pages must be a list of page indices or a string such as \"0,2-4\"", .{}),
    }
    if (ranges.items.len == 0) return bad(a, fault, loc, "pages selects no pages", .{});
    const all = try a.alloc(u32, @intCast(sum));
    var k: usize = 0;
    for (ranges.items) |r| {
        var i = r[0];
        while (true) : (i += 1) {
            all[k] = i;
            k += 1;
            if (i == r[1]) break;
        }
    }
    std.mem.sort(u32, all, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (all) |i| if (n == 0 or all[n - 1] != i) {
        all[n] = i;
        n += 1;
    };
    if (n > left) return bad(a, fault, loc, "pages selects {d} pages; a request may read at most {d} (TENSORFOLD_MAX_PDF_PAGES)", .{ n, left });
    return all[0..n];
}

fn index(a: Allocator, fault: *Fault, text: []const u8) Step!u32 {
    const t = std.mem.trim(u8, text, " \t");
    var digits = t.len > 0;
    for (t) |ch| digits = digits and std.ascii.isDigit(ch);
    if (digits) if (std.fmt.parseInt(u32, t, 10)) |n| return n else |_| {};
    return bad(a, fault, &.{ "body", "pages" }, "pages: \"{s}\" is not a page index from 0 to 4294967295", .{t});
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
    return switch (try fetch.https(a, facts.fetch, url, facts.pdf)) {
        .bytes => |b| b,
        .refused => |m| bad(a, fault, loc, "{s}", .{m}),
    };
}

//! POST /ocr in LiteParse's HTTP OCR contract: a page image in, a text item a line out (docs/api.md OCR: LiteParse).
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const json = @import("json");
const picture = @import("core").picture;
const images = @import("../images.zig");
const multipart = @import("../multipart.zig");
const Conn = @import("../http_conn.zig").Conn;
const Server = @import("../server.zig").Server;
const types = @import("types.zig");
const budget = @import("budget.zig");
const geometry = @import("geometry.zig");
const lines = @import("lines.zig");
const run = @import("run.zig");
const route = @import("route.zig");
const Value = json.Value;
const Quad = api.ocr.Quad;
const Fault = types.Fault;
const Step = run.Step;

/// The refusal for a reader without boxes: a whole-page box would place no text on the page.
pub const no_boxes = "this model gives no text boxes; LiteParse needs them";
/// The refusal for a PDF: LiteParse's OCR API spec sends an image file in the `file` field.
pub const no_pdf = "LiteParse sends page images; send the page as a PNG, JPEG or WebP file, not a PDF";

/// Answers one LiteParse request whose body has been read; ``parts`` is null when ``?format=`` chose the dialect.
pub fn post(srv: *Server, conn: *Conn, a: Allocator, parts: ?[]const multipart.Part, content_type: ?[]const u8, body: []const u8, gone: anytype) void {
    var fault: Fault = .{ .class = .internal, .message = "" };
    const out = handle(srv, a, parts, content_type, body, gone, &fault) catch |e| switch (e) {
        error.Faulted => return sendFault(conn, a, fault),
        error.OutOfMemory => return sendFault(conn, a, .{ .class = .internal, .message = "out of memory" }),
        error.Cancelled => {
            conn.close = true;
            return;
        },
    };
    conn.sendJson(200, out);
}

fn handle(srv: *Server, a: Allocator, parts: ?[]const multipart.Part, content_type: ?[]const u8, body: []const u8, gone: anytype, fault: *Fault) Step![]const u8 {
    try run.gate(srv, a, fault);
    const geo = srv.info.ocr.?.geometry;
    if (geo == .none) return fault.fail(a, .bad_input, "{s}", .{no_boxes});
    const source = try decode(a, parts orelse try parseParts(a, content_type, body, fault), fault);
    var enc: Encoder = .{ .geometry = geo };
    try run.run(srv, a, &.{source}, enc.sink(), fault, gone);
    return enc.reply(a);
}

/// The multipart parts of a body whose dialect the ``format`` query chose.
fn parseParts(a: Allocator, content_type: ?[]const u8, body: []const u8, fault: *Fault) Step![]const multipart.Part {
    const ct = content_type orelse "";
    const media = std.mem.trim(u8, ct[0 .. std.mem.indexOfScalar(u8, ct, ';') orelse ct.len], " \t");
    if (!std.ascii.eqlIgnoreCase(media, "multipart/form-data")) return fault.fail(a, .bad_input, "LiteParse sends multipart/form-data with one file part", .{});
    const bound = multipart.boundary(ct) orelse return fault.fail(a, .bad_input, "multipart/form-data needs a boundary parameter", .{});
    return multipart.parse(a, body, bound) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Malformed => fault.fail(a, .bad_input, "the multipart body is malformed", .{}),
    };
}

/// The one ``file`` part as an image source; ``language`` and other parts are ignored (no reader takes a hint).
pub fn decode(a: Allocator, parts: []const multipart.Part, fault: *Fault) Step!types.Source {
    if (route.mixed(parts)) return fault.fail(a, .bad_input, "{s}", .{route.mixed_parts});
    var file: ?multipart.Part = null;
    for (parts) |p| if (std.mem.eql(u8, p.name, "file")) {
        if (file != null) return fault.fail(a, .bad_input, "a LiteParse request has exactly one file part", .{});
        file = p;
    };
    const f = file orelse return fault.fail(a, .bad_input, "a LiteParse request has exactly one file part", .{});
    if (f.body.len == 0) return fault.fail(a, .bad_input, "the file part is empty", .{});
    // the image signature first: a PNG text chunk or a JPEG comment may hold "%PDF-"
    switch (images.sniff(f.body)) {
        .png, .jpeg, .webp => {},
        else => {
            if (std.mem.indexOf(u8, f.body[0..@min(f.body.len, 1024)], "%PDF-") != null) return fault.fail(a, .bad_input, "{s}", .{no_pdf});
            return fault.fail(a, .bad_input, "the file is not a PNG, JPEG or WebP image", .{});
        },
    }
    return .{ .name = f.filename orelse "file", .bytes = f.body, .kind = .image, .sel = .{ .all_within = 1 } };
}

/// The page sink: each line of the page as one LiteParse result.
pub const Encoder = struct {
    geometry: api.ocr.Geometry,
    results: std.ArrayList(Value) = .empty,

    pub fn sink(e: *Encoder) run.PageSink {
        return .{ .ctx = e, .page = onPage };
    }

    fn onPage(ctx: *anyopaque, a: Allocator, _: u32, page: *const types.OcrPage, _: picture.Rgb, _: *budget.Encodes, fault: *Fault) Step!void {
        const e: *Encoder = @ptrCast(@alignCast(ctx));
        const got = lines.fromBlocks(a, page, e.geometry) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Unreadable => return fault.fail(a, .unreadable, "page {d}: the model's output has a box that is not finite", .{page.index}),
        };
        for (got) |l| try e.results.append(a, try result(a, l.text, l.quad, page.width, page.height));
    }

    /// ``{"results": [...]}``, an empty list for a page without text.
    pub fn reply(e: *Encoder, a: Allocator) Allocator.Error![]const u8 {
        const o = try json.newObject(a);
        try o.put(a, "results", .{ .array = e.results.items });
        return json.stringify(a, .{ .object = o }, .{});
    }
};

/// One result: ``text``, ``bbox`` (min/max) and ``confidence`` 1.0; the spec's optional ``polygon`` is never sent.
pub fn result(a: Allocator, text: []const u8, quad: ?Quad, w: u32, h: u32) Allocator.Error!Value {
    // a line without a box is unreachable here (a .none reader is refused); it would cover the page
    const q = quad orelse Quad{ .{ 0, 0 }, .{ @floatFromInt(w), 0 }, .{ @floatFromInt(w), @floatFromInt(h) }, .{ 0, @floatFromInt(h) } };
    const b = geometry.minMax(q);
    const bbox = try a.alloc(Value, 4);
    for (b, bbox) |x, *v| v.* = number(x);
    const o = try json.newObject(a);
    try o.put(a, "text", .{ .string = text });
    try o.put(a, "bbox", .{ .array = bbox });
    try o.put(a, "confidence", .{ .float = 1.0 }); // the spec: "If your OCR engine doesn't provide confidence, use 1.0"
    return .{ .object = o };
}

/// A coordinate as its shortest f32 decimal (33.333332, not 33.33333206176758).
fn number(x: f32) Value {
    var buf: [32]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "{d}", .{x}) catch return .{ .float = x };
    return .{ .float = std.fmt.parseFloat(f64, text) catch x };
}

/// A fault's HTTP status: 400, 422, 503 or 500 (docs/api.md OCR: Failures; the spec names 400 and 500).
pub fn status(class: types.Class) u16 {
    return switch (class) {
        .bad_input => 400,
        .unreadable => 422,
        .capacity => 503,
        .internal => 500,
    };
}

/// ``{"error": msg}`` with the fault's status, the error format of LiteParse's OCR API spec.
pub fn sendFault(conn: *Conn, a: Allocator, fault: Fault) void {
    const body = errorBody(a, fault.message) catch return conn.sendJson(500, "{\"error\": \"out of memory\"}");
    if (fault.class != .capacity) return conn.sendJson(status(fault.class), body);
    var buf: [16]u8 = undefined;
    const retry = std.fmt.bufPrint(&buf, "{d}", .{fault.retry_after orelse 1}) catch unreachable;
    conn.sendJsonWith(status(fault.class), body, &.{.{ "Retry-After", retry }});
}

pub fn errorBody(a: Allocator, message: []const u8) Allocator.Error![]const u8 {
    const o = try json.newObject(a);
    try o.put(a, "error", .{ .string = message });
    return json.stringify(a, .{ .object = o }, .{});
}

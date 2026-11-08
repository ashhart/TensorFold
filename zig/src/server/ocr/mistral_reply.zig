//! Mistral OCR replies: each page's Markdown, images, tables and blocks, the reply around them, and its errors.
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const json = @import("json");
const picture = @import("core").picture;
const jpeg_out = @import("../jpeg_out.zig");
const pdf = @import("../pdf.zig");
const Conn = @import("../http_conn.zig").Conn;
const types = @import("types.zig");
const budget = @import("budget.zig");
const geometry = @import("geometry.zig");
const markdown = @import("markdown.zig");
const run = @import("run.zig");
const html = api.ocr.html;
const Value = json.Value;
const Fault = types.Fault;
const OcrPage = types.OcrPage;
const Step = run.Step;

/// Where tables go: inline in the page Markdown (``table_format`` null), or out to ``tables[]`` in one format.
pub const TableFormat = enum { markdown, html };

/// The request fields that shape the reply (docs/api.md OCR: Mistral).
pub const Options = struct {
    tables: ?TableFormat = null,
    header: bool = false,
    footer: bool = false,
    blocks: bool = true,
    base64: bool = false,
    image_limit: ?u64 = null,
    image_min_size: ?u64 = null,
};

/// JPEG quality of an image crop.
pub const crop_quality = 90;

/// The page sink: per-request image and table counters, and each page's JSON in the request arena.
pub const Encoder = struct {
    o: Options,
    /// Scratch for crop pixels and JPEG bytes, freed after each encode.
    gpa: Allocator,
    pages: std.ArrayList(Value) = .empty,
    images: u64 = 0, // kept images so far: the next image's N in img-N.jpeg
    tables: u64 = 0,

    pub fn sink(e: *Encoder) run.PageSink {
        return .{ .ctx = e, .page = onPage };
    }

    fn onPage(ctx: *anyopaque, a: Allocator, _: u32, page: *const OcrPage, rgb: picture.Rgb, enc: *budget.Encodes, fault: *Fault) Step!void {
        const e: *Encoder = @ptrCast(@alignCast(ctx));
        try e.pages.append(a, try e.encodePage(a, page, rgb, enc, fault));
    }

    /// The whole reply: pages, the model id it names, and the usage Mistral's SDKs require.
    pub fn reply(e: *Encoder, a: Allocator, model: []const u8, doc_size: usize) Allocator.Error![]const u8 {
        const usage = try json.newObject(a);
        try usage.put(a, "pages_processed", try json.intValue(a, e.pages.items.len));
        try usage.put(a, "doc_size_bytes", try json.intValue(a, doc_size));
        const o = try json.newObject(a);
        try o.put(a, "pages", .{ .array = e.pages.items });
        try o.put(a, "model", .{ .string = model });
        try o.put(a, "document_annotation", .null);
        try o.put(a, "usage_info", .{ .object = usage });
        return json.stringify(a, .{ .object = o }, .{});
    }

    fn encodePage(e: *Encoder, a: Allocator, page: *const OcrPage, rgb: picture.Rgb, enc: *budget.Encodes, fault: *Fault) Step!Value {
        var md: std.ArrayList([]const u8) = .empty;
        var header: std.ArrayList([]const u8) = .empty;
        var footer: std.ArrayList([]const u8) = .empty;
        var imgs: std.ArrayList(Value) = .empty;
        var tbls: std.ArrayList(Value) = .empty;
        var blocks: std.ArrayList(Value) = .empty;
        for (page.blocks) |b| {
            if (b.kind == .picture) {
                // a picture without a box cannot be cropped or placed: no reader with geometry writes one
                const q = b.quad orelse continue;
                const box = geometry.mistralBox(q);
                if (!e.keeps(box)) continue;
                const id = try std.fmt.allocPrint(a, "img-{d}.jpeg", .{e.images});
                e.images += 1;
                const img = try boxObject(a, "id", id, box);
                if (e.o.base64) try img.put(a, "image_base64", .{ .string = try e.crop(a, rgb, box, enc, fault) });
                try imgs.append(a, .{ .object = img });
                try md.append(a, try std.fmt.allocPrint(a, "![{s}]({s})", .{ id, id }));
                if (e.o.blocks) {
                    const blk = try boxObject(a, "type", "image", box);
                    try blk.put(a, "content", .{ .string = "" });
                    try blk.put(a, "image_id", .{ .string = id });
                    try blocks.append(a, .{ .object = blk });
                }
                continue;
            }
            const t = try e.placeTables(a, b.content, &tbls);
            if (b.kind == .page_header and e.o.header) {
                try header.append(a, t.markdown);
            } else if (b.kind == .page_footer and e.o.footer) {
                try footer.append(a, t.markdown);
            } else if (std.mem.trim(u8, t.markdown, " \t\r\n").len > 0) try md.append(a, t.markdown);
            if (!e.o.blocks) continue;
            const q = b.quad orelse continue; // a .page block has no geometry and so no block entry
            const blk = try boxObject(a, "type", blockType(b.kind), geometry.mistralBox(q));
            try blk.put(a, "content", .{ .string = t.content });
            if (b.kind == .table and e.o.tables != null) if (t.first) |tid| try blk.put(a, "table_id", .{ .string = tid });
            try blocks.append(a, .{ .object = blk });
        }
        const dims = try json.newObject(a);
        try dims.put(a, "dpi", try json.intValue(a, page.dpi));
        try dims.put(a, "height", try json.intValue(a, page.height));
        try dims.put(a, "width", try json.intValue(a, page.width));
        const o = try json.newObject(a);
        try o.put(a, "index", try json.intValue(a, page.index));
        try o.put(a, "markdown", .{ .string = try std.mem.join(a, "\n\n", md.items) });
        try o.put(a, "images", .{ .array = imgs.items });
        try o.put(a, "tables", .{ .array = tbls.items });
        try o.put(a, "header", if (e.o.header) .{ .string = try std.mem.join(a, "\n", header.items) } else .null);
        try o.put(a, "footer", if (e.o.footer) .{ .string = try std.mem.join(a, "\n", footer.items) } else .null);
        try o.put(a, "dimensions", .{ .object = dims });
        if (e.o.blocks) try o.put(a, "blocks", .{ .array = blocks.items });
        return .{ .object = o };
    }

    /// Whether the next picture is kept: ``image_min_size`` against its width and height, then ``image_limit``.
    fn keeps(e: *const Encoder, box: [4]i32) bool {
        if (e.o.image_min_size) |m| {
            const w: u64 = @intCast(@max(0, box[2] - box[0]));
            const h: u64 = @intCast(@max(0, box[3] - box[1]));
            if (w < m or h < m) return false;
        }
        if (e.o.image_limit) |l| if (e.images >= l) return false;
        return true;
    }

    /// The box's pixels as a JPEG data URL, under the request's image budget.
    fn crop(e: *Encoder, a: Allocator, rgb: picture.Rgb, box: [4]i32, enc: *budget.Encodes, fault: *Fault) Step![]const u8 {
        const x0: usize = @min(rgb.width - 1, @as(usize, @intCast(@max(0, box[0]))));
        const y0: usize = @min(rgb.height - 1, @as(usize, @intCast(@max(0, box[1]))));
        const x1: usize = @max(x0 + 1, @min(rgb.width, @as(usize, @intCast(@max(0, box[2])))));
        const y1: usize = @max(y0 + 1, @min(rgb.height, @as(usize, @intCast(@max(0, box[3])))));
        const w = x1 - x0;
        const h = y1 - y0;
        try enc.before(fault, @intCast(w), @intCast(h));
        const pixels = try e.gpa.alloc(u8, w * h * 3);
        defer e.gpa.free(pixels);
        for (0..h) |r| {
            const from = ((y0 + r) * rgb.width + x0) * 3;
            @memcpy(pixels[r * w * 3 ..][0 .. w * 3], rgb.pixels[from..][0 .. w * 3]);
        }
        const jpeg = jpeg_out.encodeRgb(e.gpa, pixels, w, h, crop_quality) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.NoLibrary => fault.fail(a, .bad_input, "include_image_base64 needs libjpeg-turbo, which this server cannot load", .{}),
            else => fault.fail(a, .internal, "an image crop could not be encoded", .{}),
        };
        defer e.gpa.free(jpeg);
        try enc.after(fault, jpeg.len);
        return pdf.makeDataUrl(a, "image/jpeg", jpeg);
    }

    /// One block's text placed two ways, and the id of its first table sent to ``tables[]``.
    const Placed = struct {
        markdown: []const u8, // for the page: each table inline, or a link to its ``tables[]`` entry
        content: []const u8, // for the block: each table as ``tables[]`` holds it (inline when table_format is null)
        first: ?[]const u8 = null,
    };

    /// The content's tables placed per ``table_format``, each sent ones also appended to ``tbls``.
    fn placeTables(e: *Encoder, a: Allocator, content: []const u8, tbls: *std.ArrayList(Value)) Allocator.Error!Placed {
        const found = try markdown.tables(a, content);
        if (found.len == 0) return .{ .markdown = content, .content = content };
        var out: std.ArrayList(u8) = .empty;
        var own: std.ArrayList(u8) = .empty;
        var first: ?[]const u8 = null;
        var from: usize = 0;
        for (found) |f| {
            try out.appendSlice(a, content[from..f.start]);
            try own.appendSlice(a, content[from..f.start]);
            from = f.end;
            const t = try tableText(a, e.o.tables, content[f.start..f.end], f.html);
            try own.appendSlice(a, t.text);
            if (e.o.tables == null) {
                try out.appendSlice(a, t.text);
                continue;
            }
            const id = try std.fmt.allocPrint(a, "tbl-{d}.{s}", .{ e.tables, if (t.format == .html) "html" else "md" });
            e.tables += 1;
            const entry = try json.newObject(a);
            try entry.put(a, "id", .{ .string = id });
            try entry.put(a, "content", .{ .string = t.text });
            try entry.put(a, "format", .{ .string = @tagName(t.format) });
            try tbls.append(a, .{ .object = entry });
            try out.print(a, "[{s}]({s})", .{ id, id });
            if (first == null) first = id;
        }
        try out.appendSlice(a, content[from..]);
        try own.appendSlice(a, content[from..]);
        return .{ .markdown = out.items, .content = own.items, .first = first };
    }
};

const TableText = struct { text: []const u8, format: TableFormat };

/// One table in the form ``mode`` asks for; a table no reader can parse keeps its source text and source format.
pub fn tableText(a: Allocator, mode: ?TableFormat, src: []const u8, is_html: bool) Allocator.Error!TableText {
    const own: TableFormat = if (is_html) .html else .markdown;
    if (!is_html and mode != .html) return .{ .text = src, .format = .markdown };
    const parsed = (if (is_html) html.parse(a, src) else markdown.pipeTable(a, src)) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Malformed => return .{ .text = src, .format = own },
    };
    const to: TableFormat = mode orelse .markdown;
    // markdown keeps a spanned HTML table as HTML, the only lossless form; inline (null) writes every table as pipes
    if (to == .html or (mode != null and parsed.spanned)) return .{ .text = try html.normalized(a, parsed), .format = .html };
    return .{ .text = try html.pipe(a, parsed), .format = .markdown };
}

/// Mistral's block type for a kind (docs/api.md OCR: Mistral); ``page`` never reaches here, having no geometry.
pub fn blockType(kind: api.ocr.Kind) []const u8 {
    return switch (kind) {
        .text, .footnote, .page => "text",
        .title, .section_header => "title",
        .list => "list",
        .table => "table",
        .formula => "equation",
        .code => "code",
        .caption => "caption",
        .page_header => "header",
        .page_footer => "footer",
        .picture => "image",
    };
}

/// ``{key: name, top_left_x, top_left_y, bottom_right_x, bottom_right_y}`` from an integer box.
fn boxObject(a: Allocator, key: []const u8, name: []const u8, box: [4]i32) Allocator.Error!*json.Object {
    const o = try json.newObject(a);
    try o.put(a, key, .{ .string = name });
    const keys = [_][]const u8{ "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y" };
    for (keys, box) |k, v| try o.put(a, k, try json.intValue(a, v));
    return o;
}

/// The HTTP status of each fault class (docs/api.md OCR: Failures).
pub fn status(class: types.Class) u16 {
    return switch (class) {
        .bad_input => 400,
        .unreadable => 422,
        .capacity => 503,
        .internal => 500,
    };
}

/// ``{"detail": [{"loc", "msg", "type"}]}``, the shape the SDKs parse as HTTPValidationError; ``loc`` defaults to body.
pub fn errorBody(a: Allocator, fault: Fault) Allocator.Error![]const u8 {
    const kind: []const u8 = switch (fault.class) {
        .bad_input => "bad_input",
        .unreadable => "unreadable_page",
        .capacity => "capacity",
        .internal => "internal",
    };
    const loc_in: []const []const u8 = if (fault.loc.len > 0) fault.loc else &.{"body"};
    const loc = try a.alloc(Value, loc_in.len);
    for (loc_in, loc) |s, *v| v.* = .{ .string = s };
    const item = try json.newObject(a);
    try item.put(a, "loc", .{ .array = loc });
    try item.put(a, "msg", .{ .string = fault.message });
    try item.put(a, "type", .{ .string = kind });
    const o = try json.newObject(a);
    try o.put(a, "detail", .{ .array = try a.dupe(Value, &.{.{ .object = item }}) });
    return json.stringify(a, .{ .object = o }, .{});
}

/// Sends the fault with its status; a capacity fault carries Retry-After.
pub fn sendFault(conn: *Conn, a: Allocator, fault: Fault) void {
    const body = errorBody(a, fault) catch {
        conn.sendJson(500, "{\"detail\": [{\"loc\": [\"body\"], \"msg\": \"out of memory\", \"type\": \"internal\"}]}");
        return;
    };
    if (fault.class != .capacity) return conn.sendJson(status(fault.class), body);
    var buf: [16]u8 = undefined;
    const retry = std.fmt.bufPrint(&buf, "{d}", .{fault.retry_after orelse 1}) catch unreachable;
    conn.sendJsonWith(status(fault.class), body, &.{.{ "Retry-After", retry }});
}

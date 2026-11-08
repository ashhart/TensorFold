//! POST /ocr in the docTR API's contract: ``files`` parts and query options in, one OCROut a page out (design 6.5).
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const json = @import("json");
const picture = @import("core").picture;
const multipart = @import("../multipart.zig");
const pdf = @import("../pdf.zig");
const query = @import("../query.zig");
const Conn = @import("../http_conn.zig").Conn;
const Server = @import("../server.zig").Server;
const types = @import("types.zig");
const budget = @import("budget.zig");
const lines = @import("lines.zig");
const run = @import("run.zig");
const Quad = api.ocr.Quad;
const Value = json.Value;
const Fault = types.Fault;
const OcrPage = types.OcrPage;
const Step = run.Step;

/// The query options that shape the reply; every other option is ignored or refused by ``options``.
pub const Options = struct {
    /// Boxes ``[xmin, ymin, xmax, ymax]``; false: 8 floats, TL, TR, BR, BL.
    straight: bool = true,
    /// One block a source block; false: one block a page.
    blocks: bool = false,
    /// One line a text line; false: one block holding one line of all the page's words.
    lines: bool = true,
};

/// Content options TensorFold cannot honour: accepted at docTR's default only.
const fixed_flags = [_]struct { []const u8, bool }{ .{ "detect_tables", false }, .{ "detect_layout", false }, .{ "straighten_pages", false } };
const fixed_numbers = [_]struct { []const u8, f64 }{ .{ "bin_thresh", 0.1 }, .{ "box_thresh", 0.1 }, .{ "paragraph_break", 0.0035 } };
/// Metadata options: any boolean is accepted, and ``orientation`` and ``language`` stay unknown (null).
const metadata = [_][]const u8{ "detect_orientation", "detect_language" };

/// Answers one POST /ocr chosen as docTR; ``parts`` is null when ``?format=`` chose it before any multipart parse.
pub fn post(srv: *Server, conn: *Conn, a: Allocator, parts: ?[]const multipart.Part, body: []const u8, gone: anytype) void {
    var fault: Fault = .{ .class = .internal, .message = "" };
    const out = handle(srv, a, conn.path, conn.header("Content-Type"), parts, body, gone, &fault) catch |e| switch (e) {
        error.Faulted => return sendFault(conn, a, fault),
        error.OutOfMemory => return sendFault(conn, a, .{ .class = .internal, .message = "out of memory" }),
        error.Cancelled => {
            conn.close = true;
            return;
        },
    };
    conn.sendJson(200, out);
}

fn handle(srv: *Server, a: Allocator, path: []const u8, content_type: ?[]const u8, parts_in: ?[]const multipart.Part, body: []const u8, gone: anytype, fault: *Fault) Step![]const u8 {
    try run.gate(srv, a, fault);
    const o = try options(a, path, fault);
    const parts = parts_in orelse try partsOf(a, content_type, body, fault);
    const sources = try files(a, parts, srv.config.vision_pdf, fault);
    var enc: Encoder = .{ .o = o, .geometry = srv.info.ocr.?.geometry, .sources = sources };
    try run.run(srv, a, sources, enc.sink(), fault, gone);
    return json.stringify(a, .{ .array = enc.pages.items }, .{});
}

fn bad(a: Allocator, fault: *Fault, comptime fmt: []const u8, args: anytype) Step {
    return fault.fail(a, .bad_input, fmt, args);
}

/// The query's options: pydantic booleans; content options at docTR's defaults; model and batch options unread.
pub fn options(a: Allocator, path: []const u8, fault: *Fault) Step!Options {
    for (metadata) |m| _ = try boolean(a, path, m, fault);
    for (fixed_flags) |f| if (try boolean(a, path, f[0], fault)) |v| if (v != f[1])
        return bad(a, fault, "{s}={} is not supported; TensorFold reads pages with {s}={}", .{ f[0], v, f[0], f[1] });
    for (fixed_numbers) |f| if (try param(a, path, f[0], fault)) |text| {
        const v = std.fmt.parseFloat(f64, std.mem.trim(u8, text, " ")) catch return bad(a, fault, "{s} must be a number", .{f[0]});
        if (v != f[1]) return bad(a, fault, "{s}={s} is not supported; TensorFold reads pages with {s}={d}", .{ f[0], text, f[0], f[1] });
    };
    return .{
        .straight = try boolean(a, path, "assume_straight_pages", fault) orelse true,
        .blocks = try boolean(a, path, "resolve_blocks", fault) orelse false,
        .lines = try boolean(a, path, "resolve_lines", fault) orelse true,
    };
}

fn boolean(a: Allocator, path: []const u8, name: []const u8, fault: *Fault) Step!?bool {
    const text = try param(a, path, name, fault) orelse return null;
    return query.boolean(text) orelse bad(a, fault, "{s} must be a boolean (true, false, 1, 0, yes, no, on or off)", .{name});
}

fn param(a: Allocator, path: []const u8, name: []const u8, fault: *Fault) Step!?[]const u8 {
    return query.get(a, path, name) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Malformed => bad(a, fault, "the {s} query parameter is malformed", .{name}),
    };
}

/// The body's multipart parts, read here when ``?format=doctr`` chose the dialect before any part was parsed.
fn partsOf(a: Allocator, content_type: ?[]const u8, body: []const u8, fault: *Fault) Step![]const multipart.Part {
    const ct = std.mem.trim(u8, content_type orelse "", " \t");
    const media = std.mem.trim(u8, ct[0 .. std.mem.indexOfScalar(u8, ct, ';') orelse ct.len], " \t");
    if (!std.ascii.eqlIgnoreCase(media, "multipart/form-data")) return bad(a, fault, "a docTR request is multipart/form-data with one or more files parts", .{});
    const bound = multipart.boundary(ct) orelse return bad(a, fault, "multipart/form-data needs a boundary parameter", .{});
    return multipart.parse(a, body, bound) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Malformed => bad(a, fault, "the multipart body is malformed", .{}),
    };
}

/// The ``files`` parts in order as sources: the part type exactly as docTR checks it, and the file's name.
pub fn files(a: Allocator, parts: []const multipart.Part, pdf_on: bool, fault: *Fault) Step![]const types.Source {
    var out: std.ArrayList(types.Source) = .empty;
    for (parts) |p| {
        if (!std.mem.eql(u8, p.name, "files")) continue;
        const mime = p.content_type orelse "";
        const name = p.filename orelse ""; // docTR refuses a part without a filename; TensorFold reads it unnamed
        const is_pdf = std.mem.eql(u8, mime, "application/pdf");
        if (!is_pdf and !std.mem.eql(u8, mime, "image/jpeg") and !std.mem.eql(u8, mime, "image/png"))
            return bad(a, fault, "Unsupported file format: {s} for file {s}", .{ p.content_type orelse "None", p.filename orelse "None" });
        if (is_pdf and !pdf_on) return bad(a, fault, "{s}", .{pdf.off_message});
        const sel: pdf.Selection = .{ .all_within = if (is_pdf) std.math.maxInt(u32) else 1 };
        try out.append(a, .{ .name = name, .bytes = p.body, .kind = if (is_pdf) .pdf else .image, .sel = sel });
    }
    if (out.items.len == 0) return bad(a, fault, "a docTR request needs one or more files parts", .{});
    return out.items;
}

/// The page sink: one OCROut a page, named by the page's file.
pub const Encoder = struct {
    o: Options,
    geometry: api.ocr.Geometry,
    sources: []const types.Source,
    pages: std.ArrayList(Value) = .empty,

    pub fn sink(e: *Encoder) run.PageSink {
        return .{ .ctx = e, .page = onPage };
    }

    fn onPage(ctx: *anyopaque, a: Allocator, source: u32, page: *const OcrPage, _: picture.Rgb, _: *budget.Encodes, fault: *Fault) Step!void {
        const e: *Encoder = @ptrCast(@alignCast(ctx));
        const out = e.encodePage(a, e.sources[source].name, page) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Unreadable => return fault.fail(a, .unreadable, "page {d}: the model's output has a box that is not finite", .{page.index}),
        };
        try e.pages.append(a, out);
    }

    /// One OCROut: name, unknown orientation and language, ``[height, width]``, no layout or tables, one item.
    pub fn encodePage(e: *const Encoder, a: Allocator, name: []const u8, page: *const OcrPage) (Allocator.Error || error{Unreadable})!Value {
        const size: Size = .{ .w = @floatFromInt(page.width), .h = @floatFromInt(page.height) };
        var built: std.ArrayList(Line) = .empty;
        for (try lines.fromBlocks(a, page, e.geometry)) |l| {
            const ws = try lines.words(a, l, page.width, page.height);
            // a line of only Unicode whitespace has no words and no box: left out, as is a block of only such lines
            if (ws.len > 0) try built.append(a, .{ .quad = l.quad, .words = ws, .block = l.block });
        }
        if (!e.o.lines and built.items.len > 0) {
            var all: std.ArrayList(lines.Word) = .empty;
            for (built.items) |l| try all.appendSlice(a, l.words);
            built.items[0] = .{ .quad = null, .words = all.items, .block = 0, .own = false };
            built.shrinkRetainingCapacity(1);
        }
        var blocks: std.ArrayList(Value) = .empty;
        var start: usize = 0;
        while (start < built.items.len) {
            var end = start + 1;
            if (e.o.blocks) {
                while (end < built.items.len and built.items[end].block == built.items[start].block) end += 1;
            } else end = built.items.len;
            try blocks.append(a, try e.block(a, built.items[start..end], size));
            start = end;
        }
        const item = try json.newObject(a);
        try item.put(a, "blocks", .{ .array = blocks.items });
        const o = try json.newObject(a);
        try o.put(a, "name", .{ .string = name });
        try o.put(a, "orientation", try unknown(a));
        try o.put(a, "language", try unknown(a));
        try o.put(a, "dimensions", .{ .array = try a.dupe(Value, &.{ try json.intValue(a, page.height), try json.intValue(a, page.width) }) });
        try o.put(a, "layout", .{ .array = &.{} });
        try o.put(a, "tables", .{ .array = &.{} });
        try o.put(a, "items", .{ .array = try a.dupe(Value, &.{.{ .object = item }}) });
        return .{ .object = o };
    }

    fn block(e: *const Encoder, a: Allocator, ls: []const Line, size: Size) Allocator.Error!Value {
        var box = empty_box;
        const out = try a.alloc(Value, ls.len);
        for (ls, out) |l, *v| {
            const r = try e.line(a, l, size);
            box = join(box, r.box);
            v.* = r.value;
        }
        const o = try json.newObject(a);
        try o.put(a, "geometry", try geometry(a, e.o.straight, box, null));
        try o.put(a, "objectness_score", .{ .float = 1.0 });
        try o.put(a, "lines", .{ .array = out });
        return .{ .object = o };
    }

    /// A line and the box that encloses its words (the block's union takes it).
    fn line(e: *const Encoder, a: Allocator, l: Line, size: Size) Allocator.Error!struct { value: Value, box: Box } {
        var box = empty_box;
        const ws = try a.alloc(Value, l.words.len);
        for (l.words, ws) |w, *v| {
            const wb = boxOf(w.quad, size);
            box = join(box, wb);
            const o = try json.newObject(a);
            try o.put(a, "value", .{ .string = w.text });
            try o.put(a, "geometry", try geometry(a, e.o.straight, wb, if (w.quad) |q| relative(q, size) else null));
            try o.put(a, "objectness_score", .{ .float = 1.0 });
            try o.put(a, "confidence", .{ .float = 1.0 });
            const crop = try json.newObject(a);
            try crop.put(a, "value", try json.intValue(a, 0));
            try crop.put(a, "confidence", .null);
            try o.put(a, "crop_orientation", .{ .object = crop });
            v.* = .{ .object = o };
        }
        // polygon mode writes the line's own quad when it encloses its words; else, and for a merged line, their union
        const own: ?[4][2]f64 = if (l.own) if (l.quad) |q| (if (encloses(q, l.words)) relative(q, size) else null) else null else null;
        const o = try json.newObject(a);
        try o.put(a, "geometry", try geometry(a, e.o.straight, box, own));
        try o.put(a, "objectness_score", .{ .float = 1.0 });
        try o.put(a, "words", .{ .array = ws });
        return .{ .value = .{ .object = o }, .box = box };
    }
};

const Size = struct { w: f64, h: f64 };
/// ``[xmin, ymin, xmax, ymax]`` relative to the page.
const Box = [4]f64;
const empty_box: Box = .{ std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64), -std.math.inf(f64) };
/// A page line, its words, its source block, and whether polygon mode writes its own quad.
const Line = struct { quad: ?Quad, words: []const lines.Word, block: u32, own: bool = true };

/// The quad's points divided by the page size, not rounded (docTR does not round).
fn relative(q: Quad, size: Size) [4][2]f64 {
    var out: [4][2]f64 = undefined;
    for (q, &out) |p, *o| o.* = .{ @as(f64, p[0]) / size.w, @as(f64, p[1]) / size.h };
    return out;
}

/// The quad's relative min/max box; the whole page for a reader without geometry.
fn boxOf(q: ?Quad, size: Size) Box {
    const r = relative(q orelse return .{ 0, 0, 1, 1 }, size);
    var box = empty_box;
    for (r) |p| box = join(box, .{ p[0], p[1], p[0], p[1] });
    return box;
}

/// Whether every corner of every word lies in the convex quad ``q``, to 1e-3 px (validation widens tiny words past it).
fn encloses(q: Quad, words: []const lines.Word) bool {
    var area: f64 = 0;
    for (0..4) |i| area += cross(q[i], q[(i + 1) % 4], .{ 0, 0 });
    const sign: f64 = if (area < 0) -1 else 1;
    for (words) |w| for (w.quad orelse return false) |p| for (0..4) |i| {
        const from = q[i];
        const to = q[(i + 1) % 4];
        const len = std.math.hypot(@as(f64, to[0]) - from[0], @as(f64, to[1]) - from[1]);
        if (sign * cross(from, to, p) < -1e-3 * len) return false;
    };
    return true;
}

/// The cross product of ``to - from`` and ``p - from``: its sign says which side of the edge ``p`` is on.
fn cross(from: [2]f32, to: [2]f32, p: [2]f32) f64 {
    const ax = @as(f64, to[0]) - from[0];
    const ay = @as(f64, to[1]) - from[1];
    return ax * (@as(f64, p[1]) - from[1]) - ay * (@as(f64, p[0]) - from[0]);
}

fn join(x: Box, y: Box) Box {
    return .{ @min(x[0], y[0]), @min(x[1], y[1]), @max(x[2], y[2]), @max(x[3], y[3]) };
}

/// Box mode: the 4 box floats; polygon mode: ``quad``'s 8, or the box's corners TL, TR, BR, BL.
fn geometry(a: Allocator, straight: bool, box: Box, quad: ?[4][2]f64) Allocator.Error!Value {
    if (straight) return floats(a, &box);
    const q = quad orelse [4][2]f64{ .{ box[0], box[1] }, .{ box[2], box[1] }, .{ box[2], box[3] }, .{ box[0], box[3] } };
    return floats(a, &@as([8]f64, @bitCast(q)));
}

fn floats(a: Allocator, xs: []const f64) Allocator.Error!Value {
    const out = try a.alloc(Value, xs.len);
    for (xs, out) |x, *v| v.* = .{ .float = x };
    return .{ .array = out };
}

fn unknown(a: Allocator) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "value", .null);
    try o.put(a, "confidence", .null);
    return .{ .object = o };
}

/// The HTTP status of each fault class (design 7.4).
pub fn status(class: types.Class) u16 {
    return switch (class) {
        .bad_input => 400,
        .unreadable => 422,
        .capacity => 503,
        .internal => 500,
    };
}

/// ``{"detail": message}``, docTR's HTTPException body, with the class's status; capacity carries Retry-After.
pub fn sendFault(conn: *Conn, a: Allocator, fault: Fault) void {
    const body = errorBody(a, fault.message) catch return conn.sendJson(500, "{\"detail\": \"out of memory\"}");
    if (fault.class != .capacity) return conn.sendJson(status(fault.class), body);
    var buf: [16]u8 = undefined;
    const retry = std.fmt.bufPrint(&buf, "{d}", .{fault.retry_after orelse 1}) catch unreachable;
    conn.sendJsonWith(status(fault.class), body, &.{.{ "Retry-After", retry }});
}

pub fn errorBody(a: Allocator, message: []const u8) Allocator.Error![]const u8 {
    const o = try json.newObject(a);
    try o.put(a, "detail", .{ .string = message });
    return json.stringify(a, .{ .object = o }, .{});
}

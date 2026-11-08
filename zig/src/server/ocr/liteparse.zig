//! POST /ocr in LiteParse's HTTP OCR contract: one page image in, one text item a line out (design 6.6).
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
const Value = json.Value;
const Quad = api.ocr.Quad;
const Fault = types.Fault;
const Step = run.Step;

/// The refusal for a reader without boxes: LiteParse's overlap rule would drop a whole-page box without a word.
pub const no_boxes = "this model gives no text boxes; LiteParse needs them";
/// The refusal for a PDF: LiteParse renders its pages itself and sends each as an image.
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

/// One result: ``text``, ``bbox`` (float min/max), ``confidence`` 1.0, and ``polygon`` when the rotations agree.
pub fn result(a: Allocator, text: []const u8, quad: ?Quad, w: u32, h: u32) Allocator.Error!Value {
    // a line without a box is unreachable here (a .none reader is refused); it would cover the page
    const q = quad orelse Quad{ .{ 0, 0 }, .{ @floatFromInt(w), 0 }, .{ @floatFromInt(w), @floatFromInt(h) }, .{ 0, @floatFromInt(h) } };
    const b = geometry.minMax(q);
    const bbox = try a.alloc(Value, 4);
    for (b, bbox) |x, *v| v.* = number(x);
    const o = try json.newObject(a);
    try o.put(a, "text", .{ .string = text });
    try o.put(a, "bbox", .{ .array = bbox });
    try o.put(a, "confidence", .{ .float = 1.0 });
    const lite = liteRotation(q);
    const reading = trueRotation(q);
    if (quad != null and lite != null and reading != null and lite.? == reading.?) {
        const poly = try a.alloc(Value, 4);
        for (q, poly) |p, *v| v.* = .{ .array = try a.dupe(Value, &.{ number(p[0]), number(p[1]) }) };
        try o.put(a, "polygon", .{ .array = poly });
    }
    return .{ .object = o };
}

/// A coordinate as its shortest f32 decimal (33.333332, not 33.33333206176758).
fn number(x: f32) Value {
    var buf: [32]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "{d}", .{x}) catch return .{ .float = x };
    return .{ .float = std.fmt.parseFloat(f64, text) catch x };
}

/// Degrees from a 45° boundary inside which a snap is left undecided: LiteParse's libm atan2f may round the other way.
pub const boundary_window = 0.01;

/// LiteParse's ``polygon_rotation_deg`` (ocr_merge.rs at 2514f8e); null when its snap sits on a 45° boundary.
pub fn liteRotation(q: Quad) ?i32 {
    const e0 = edge(q[0], q[1]);
    const e1 = edge(q[1], q[2]);
    const len0 = @sqrt(e0[0] * e0[0] + e0[1] * e0[1]);
    const len1 = @sqrt(e1[0] * e1[0] + e1[1] * e1[1]);
    if (@max(len0, len1) < 1.0) return 0;
    const longer = @max(len0, len1);
    const shorter = @min(len0, len1);
    if (shorter > 0 and longer / shorter < 1.3) return 0;
    return snapped(if (len0 >= len1) e0 else e1);
}

/// The quad's own reading rotation: its TL→TR edge snapped the same way; null on a 45° boundary.
pub fn trueRotation(q: Quad) ?i32 {
    return snapped(edge(q[0], q[1]));
}

fn edge(p: [2]f32, r: [2]f32) [2]f32 {
    return .{ r[0] - p[0], r[1] - p[1] };
}

/// ``-atan2(dy, dx)`` in degrees (y points down, so this is counter-clockwise), mod 360, rounded to a right angle.
fn snapped(d: [2]f32) ?i32 {
    const ccw = -std.math.radiansToDegrees(std.math.atan2(d[1], d[0]));
    const normalized = @mod(ccw, 360.0);
    // near 45, 135, 225 or 315 the snap rests on atan2's last bit, which may differ from LiteParse's libm
    if (@abs(@mod(normalized, 90.0) - 45.0) < boundary_window) return null;
    const quarter: i32 = @intFromFloat(@round(normalized / 90.0));
    return @mod(quarter * 90, 360);
}

/// The HTTP status of a fault: 400, 422, 503 or 500 (design 7.4; LiteParse retries 408, 425, 429, 500, 502-504).
pub fn status(class: types.Class) u16 {
    return switch (class) {
        .bad_input => 400,
        .unreadable => 422,
        .capacity => 503,
        .internal => 500,
    };
}

/// ``{"error": msg}`` with the fault's status; LiteParse reads only the status, the body is for curl and logs.
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

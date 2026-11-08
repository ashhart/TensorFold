//! PaddleX OCR replies: each page's prunedResult and images, the result around them, and the error envelope.
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const json = @import("json");
const picture = @import("core").picture;
const jpeg_out = @import("../jpeg_out.zig");
const Conn = @import("../http_conn.zig").Conn;
const types = @import("types.zig");
const budget = @import("budget.zig");
const geometry = @import("geometry.zig");
const lines = @import("lines.zig");
const draw = @import("draw.zig");
const run = @import("run.zig");
const Value = json.Value;
const Fault = types.Fault;
const Step = run.Step;

/// The request fields that shape the reply (docs/api.md OCR: PaddleX).
pub const Options = struct {
    thresh: f64 = 0, // textRecScoreThresh: rec_* keep lines with score >= thresh
    word_box: bool = false, // returnWordBox
    visualize: bool = true, // anything but false: ocrImage and inputImage
};

/// JPEG quality of ocrImage and inputImage.
pub const jpeg_quality = 85;

/// The page sink: each page's ocrResults entry and size, in the request arena.
pub const Encoder = struct {
    o: Options,
    geometry: api.ocr.Geometry,
    /// Scratch for JPEG bytes, freed after each encode.
    gpa: Allocator,
    results: std.ArrayList(Value) = .empty,
    sizes: std.ArrayList([2]u32) = .empty,

    pub fn sink(e: *Encoder) run.PageSink {
        return .{ .ctx = e, .page = onPage };
    }

    fn onPage(ctx: *anyopaque, a: Allocator, _: u32, page: *const types.OcrPage, rgb: picture.Rgb, enc: *budget.Encodes, fault: *Fault) Step!void {
        const e: *Encoder = @ptrCast(@alignCast(ctx));
        const w = page.width;
        const h = page.height;
        const ls = lines.fromBlocks(a, page, e.geometry) catch |err| return unreadable(a, fault, page.index, err);
        const polys = try a.alloc([4][2]i32, ls.len);
        for (ls, polys) |l, *p| p.* = if (l.quad) |q| geometry.paddlePoly(q, w, h) else whole(w, h);
        var rec: std.ArrayList(usize) = .empty;
        for (ls, 0..) |l, i| if (@as(f64, l.score) >= e.o.thresh) try rec.append(a, i);
        const settings = try json.newObject(a);
        try settings.put(a, "use_doc_preprocessor", .{ .bool = false });
        try settings.put(a, "use_textline_orientation", .{ .bool = false });
        const pruned = try json.newObject(a);
        try pruned.put(a, "model_settings", .{ .object = settings });
        try pruned.put(a, "dt_polys", try polyList(a, polys, null));
        try pruned.put(a, "text_type", .{ .string = "general" });
        if (ls.len > 0) {
            const angles = try a.alloc(Value, ls.len);
            for (angles) |*v| v.* = .{ .int = "-1" };
            try pruned.put(a, "textline_orientation_angles", .{ .array = angles });
        }
        try pruned.put(a, "text_rec_score_thresh", .{ .float = e.o.thresh });
        try pruned.put(a, "return_word_box", .{ .bool = e.o.word_box });
        const texts = try a.alloc(Value, rec.items.len);
        const scores = try a.alloc(Value, rec.items.len);
        const boxes = try a.alloc(Value, rec.items.len);
        for (rec.items, texts, scores, boxes) |i, *t, *s, *b| {
            t.* = .{ .string = ls[i].text };
            s.* = .{ .float = ls[i].score };
            b.* = try box(a, polys[i]);
        }
        try pruned.put(a, "rec_texts", .{ .array = texts });
        try pruned.put(a, "rec_scores", .{ .array = scores });
        try pruned.put(a, "rec_polys", try polyList(a, polys, rec.items));
        try pruned.put(a, "rec_boxes", .{ .array = boxes });
        if (e.o.word_box) try e.words(a, pruned, page, ls, rec.items, fault);
        const result = try json.newObject(a);
        try result.put(a, "prunedResult", .{ .object = pruned });
        if (e.o.visualize) {
            const input = try e.jpeg64(a, rgb, enc, fault); // before the outlines are drawn on the same pixels
            for (rec.items) |i| draw.outline(rgb.pixels, w, h, polys[i], draw.red, 2);
            try result.put(a, "ocrImage", .{ .string = try e.jpeg64(a, rgb, enc, fault) });
            try result.put(a, "inputImage", .{ .string = input });
        }
        try e.results.append(a, .{ .object = result });
        try e.sizes.append(a, .{ w, h });
    }

    /// ``text_word`` and ``text_word_boxes``: per kept line, its estimated words and their integer xyxy boxes.
    fn words(_: *Encoder, a: Allocator, pruned: *json.Object, page: *const types.OcrPage, ls: []const types.OcrLine, rec: []const usize, fault: *Fault) Step!void {
        const texts = try a.alloc(Value, rec.len);
        const boxes = try a.alloc(Value, rec.len);
        for (rec, texts, boxes) |i, *t, *b| {
            const ws = lines.words(a, ls[i], page.width, page.height) catch |err| return unreadable(a, fault, page.index, err);
            const wt = try a.alloc(Value, ws.len);
            const wb = try a.alloc(Value, ws.len);
            for (ws, wt, wb) |word, *x, *y| {
                x.* = .{ .string = word.text };
                y.* = try box(a, if (word.quad) |q| geometry.paddlePoly(q, page.width, page.height) else whole(page.width, page.height));
            }
            t.* = .{ .array = wt };
            b.* = .{ .array = wb };
        }
        try pruned.put(a, "text_word_boxes", .{ .array = boxes }); // PaddleX writes the boxes first
        try pruned.put(a, "text_word", .{ .array = texts });
    }

    /// The page pixels as base64 JPEG without a data-URL prefix, under the request's image budget.
    fn jpeg64(e: *Encoder, a: Allocator, rgb: picture.Rgb, enc: *budget.Encodes, fault: *Fault) Step![]const u8 {
        try enc.before(fault, @intCast(rgb.width), @intCast(rgb.height));
        const jpeg = jpeg_out.encodeRgb(e.gpa, rgb.pixels, rgb.width, rgb.height, jpeg_quality) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.NoLibrary => fault.fail(a, .internal, "visualize needs libjpeg-turbo, which this server cannot load; send visualize: false", .{}),
            else => fault.fail(a, .internal, "a visualisation image could not be encoded", .{}),
        };
        defer e.gpa.free(jpeg);
        try enc.after(fault, jpeg.len);
        const b64 = std.base64.standard.Encoder;
        const out = try a.alloc(u8, b64.calcSize(jpeg.len));
        return b64.encode(out, jpeg);
    }

    /// The whole reply: ``ocrResults`` in page order and ``dataInfo`` for an image or a PDF.
    pub fn reply(e: *Encoder, a: Allocator, log_id: []const u8, is_pdf: bool) Allocator.Error![]const u8 {
        const info = try json.newObject(a);
        if (is_pdf) {
            const pages = try a.alloc(Value, e.sizes.items.len);
            for (e.sizes.items, pages) |s, *p| p.* = .{ .object = try size(a, s) };
            try info.put(a, "numPages", try json.intValue(a, pages.len));
            try info.put(a, "pages", .{ .array = pages });
            try info.put(a, "type", .{ .string = "pdf" });
        } else {
            const s = if (e.sizes.items.len > 0) e.sizes.items[0] else [2]u32{ 0, 0 };
            try info.put(a, "width", try json.intValue(a, s[0]));
            try info.put(a, "height", try json.intValue(a, s[1]));
            try info.put(a, "type", .{ .string = "image" });
        }
        const result = try json.newObject(a);
        try result.put(a, "ocrResults", .{ .array = e.results.items });
        try result.put(a, "dataInfo", .{ .object = info });
        const o = try json.newObject(a);
        try o.put(a, "logId", .{ .string = log_id });
        try o.put(a, "errorCode", .{ .int = "0" });
        try o.put(a, "errorMsg", .{ .string = "Success" });
        try o.put(a, "result", .{ .object = result });
        return json.stringify(a, .{ .object = o }, .{});
    }
};

fn unreadable(a: Allocator, fault: *Fault, index: u32, err: (Allocator.Error || error{Unreadable})) Step {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.Unreadable => fault.fail(a, .unreadable, "page {d}: the model's output has a box that is not finite", .{index}),
    };
}

fn size(a: Allocator, s: [2]u32) Allocator.Error!*json.Object {
    const o = try json.newObject(a);
    try o.put(a, "width", try json.intValue(a, s[0]));
    try o.put(a, "height", try json.intValue(a, s[1]));
    return o;
}

/// The whole page as a polygon: a line of a reader without geometry.
fn whole(w: u32, h: u32) [4][2]i32 {
    const x: i32 = @intCast(@min(w, std.math.maxInt(i32)));
    const y: i32 = @intCast(@min(h, std.math.maxInt(i32)));
    return .{ .{ 0, 0 }, .{ x, 0 }, .{ x, y }, .{ 0, y } };
}

/// ``[[x, y] x 4]`` per polygon: every one, or those at ``only``.
fn polyList(a: Allocator, polys: []const [4][2]i32, only: ?[]const usize) Allocator.Error!Value {
    const n = if (only) |o| o.len else polys.len;
    const out = try a.alloc(Value, n);
    for (out, 0..) |*v, k| {
        const p = polys[if (only) |o| o[k] else k];
        const pts = try a.alloc(Value, 4);
        for (p, pts) |c, *pt| pt.* = .{ .array = try a.dupe(Value, &.{ try json.intValue(a, c[0]), try json.intValue(a, c[1]) }) };
        v.* = .{ .array = pts };
    }
    return .{ .array = out };
}

/// ``[xmin, ymin, xmax, ymax]`` of an integer polygon, as ``rec_boxes`` holds it.
fn box(a: Allocator, p: [4][2]i32) Allocator.Error!Value {
    var b: [4]i32 = .{ p[0][0], p[0][1], p[0][0], p[0][1] };
    for (p[1..]) |c| {
        b[0] = @min(b[0], c[0]);
        b[1] = @min(b[1], c[1]);
        b[2] = @max(b[2], c[0]);
        b[3] = @max(b[3], c[1]);
    }
    const out = try a.alloc(Value, 4);
    for (b, out) |x, *v| v.* = try json.intValue(a, x);
    return .{ .array = out };
}

/// The HTTP status and ``errorCode`` of each fault class: PaddleX answers 422 for every request and input failure.
pub fn status(class: types.Class) u16 {
    return switch (class) {
        .bad_input, .unreadable => 422,
        .capacity => 503,
        .internal => 500,
    };
}

/// ``{"logId", "errorCode", "errorMsg"}``, ``errorCode`` equal to the HTTP status.
pub fn errorBody(a: Allocator, log_id: []const u8, fault: Fault) Allocator.Error![]const u8 {
    const o = try json.newObject(a);
    try o.put(a, "logId", .{ .string = log_id });
    try o.put(a, "errorCode", try json.intValue(a, status(fault.class)));
    try o.put(a, "errorMsg", .{ .string = fault.message });
    return json.stringify(a, .{ .object = o }, .{});
}

/// Sends the fault with its status; a capacity fault carries Retry-After.
pub fn sendFault(conn: *Conn, a: Allocator, log_id: []const u8, fault: Fault) void {
    const body = errorBody(a, log_id, fault) catch return conn.sendJson(500, "{\"logId\": \"\", \"errorCode\": 500, \"errorMsg\": \"out of memory\"}");
    if (fault.class != .capacity) return conn.sendJson(status(fault.class), body);
    var buf: [16]u8 = undefined;
    const retry = std.fmt.bufPrint(&buf, "{d}", .{fault.retry_after orelse 1}) catch unreachable;
    conn.sendJsonWith(503, body, &.{.{ "Retry-After", retry }});
}

//! The OCR page loop: sources to page images, one greedy generation a page, parsed and validated blocks to a sink.
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const json = @import("json");
const picture = @import("core").picture;
const chat = @import("../chat.zig");
const errors = @import("../errors.zig");
const images = @import("../images.zig");
const messages = @import("../messages.zig");
const http_body = @import("../http_body.zig");
const ids = @import("../ids.zig");
const log = @import("../log.zig");
const pdf = @import("../pdf.zig");
const loop_stop = @import("../loop_stop.zig");
const Server = @import("../server.zig").Server;
const types = @import("types.zig");
const budget = @import("budget.zig");
const geometry = @import("geometry.zig");
const size = @import("size.zig");
const Fault = types.Fault;
const OcrPage = types.OcrPage;
const Source = types.Source;
const Cx = errors.Cx;

/// What every OCR step returns: a fault (details in the request's ``*Fault``), a client that left, or no memory.
pub const Step = types.Faulted || error{ Cancelled, OutOfMemory };

/// The dialect's per-page encoder, called in page order.
pub const PageSink = struct {
    ctx: *anyopaque,
    /// Called once per page while its pixels live; ``source`` indexes ``run``'s sources; encodes go into ``a``.
    page: *const fn (ctx: *anyopaque, a: Allocator, source: u32, page: *const OcrPage, rgb: picture.Rgb, enc: *budget.Encodes, fault: *Fault) Step!void,
};

/// The gate's refusal for a model without a reader (docs/api.md OCR).
pub const no_reader = "this model has no OCR reader; use POST /v1/chat/completions";
/// The gate's refusal for a server started without --vision.
pub const no_vision = "OCR needs image input; start the server with --vision";
/// The decoded-pixel limit of one page: dots.mocr's processor input limit, the one reader family's.
pub const max_page_pixels = picture.max_pixels;

/// Refuses as bad input unless the engine has an OCR reader and the server reads images.
pub fn gate(srv: *const Server, a: Allocator, fault: *Fault) Step!void {
    if (srv.info.ocr == null) return fault.fail(a, .bad_input, "{s}", .{no_reader});
    if (srv.info.vision == null) return fault.fail(a, .bad_input, "{s}", .{no_vision});
}

/// One page to read: its source, its index in that source, its encoded image, its DPI and its pixel count.
const Planned = struct { source: u32, index: u32, bytes: []const u8, dpi: u32, pixels: u64 };

/// Reads every page of ``sources`` in order into ``sink``; ``a`` is the request arena, ``gone`` says the client left.
pub fn run(srv: *Server, a: Allocator, sources: []const Source, sink: PageSink, fault: *Fault, gone: anytype) Step!void {
    try gate(srv, a, fault);
    const reader = srv.info.ocr.?;
    const id = try ids.make(a, "ocr-", 24);
    if (gone.check()) return error.Cancelled;
    const pages = try plan(srv, a, sources, fault);
    var total: u64 = 0;
    for (pages) |p| total +|= p.pixels;
    // one limit for the whole request, from every page's pixels, so no encode depends on the order of the pages
    var enc = budget.Encodes.init(total, pages.len);
    for (pages) |p| {
        if (gone.check()) return error.Cancelled;
        try onePage(srv, a, reader, id, p, sink, &enc, fault, gone);
    }
}

/// Every source's page within one page budget, its pixels counted before any inference.
fn plan(srv: *const Server, a: Allocator, sources: []const Source, fault: *Fault) Step![]const Planned {
    var left = budget.Pages.init();
    const cap = left.left;
    var out: std.ArrayList(Planned) = .empty;
    for (sources, 0..) |s, k| {
        if (!left.take(1)) return overBudget(a, fault, cap);
        const px = try imagePixels(srv, a, s.bytes, fault);
        try out.append(a, .{ .source = @intCast(k), .index = 0, .bytes = s.bytes, .dpi = 72, .pixels = px });
    }
    return out.items;
}

/// An image's pixel count from its header, else from one decode within the page limit (freed at once).
fn imagePixels(srv: *const Server, a: Allocator, bytes: []const u8, fault: *Fault) Step!u64 {
    // past the page limit the page's own decode refuses it, so no larger count can fund another page's encodes
    if (size.of(bytes)) |wh| return @min(@as(u64, wh[0]) * wh[1], max_page_pixels);
    var rgb = picture.decodeWithLimits(srv.io, srv.gpa, bytes, .{ .pixels = max_page_pixels }) catch |e| return decodeFault(a, fault, e);
    defer rgb.deinit(srv.gpa);
    return @as(u64, rgb.width) * rgb.height;
}

fn decodeFault(a: Allocator, fault: *Fault, e: picture.Error) Step {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => fault.fail(a, .bad_input, "{s}", .{picture.problem(e)}),
    };
}

fn overBudget(a: Allocator, fault: *Fault, cap: u32) Step {
    return fault.fail(a, .bad_input, "a request may read at most {d} pages (TENSORFOLD_MAX_PDF_PAGES); this one selects more", .{cap});
}

/// One page in its own arena: decode, generate, parse, validate, then the sink while the pixels live.
fn onePage(srv: *Server, a: Allocator, reader: api.ocr.Ocr, id: []const u8, p: Planned, sink: PageSink, enc: *budget.Encodes, fault: *Fault, gone: anytype) Step!void {
    var arena = std.heap.ArenaAllocator.init(srv.gpa);
    defer arena.deinit();
    const pa = arena.allocator();
    const bytes = p.bytes;
    const rgb = picture.decodeWithLimits(srv.io, pa, bytes, .{ .pixels = max_page_pixels }) catch |e| return decodeFault(a, fault, e);
    const w: u32 = @intCast(rgb.width);
    const h: u32 = @intCast(rgb.height);
    // the budget was planned from the header; pixels that disagree (an EXIF turn keeps the product) fail the request
    if (@min(@as(u64, w) * h, max_page_pixels) != p.pixels) return fault.fail(a, .bad_input, "page {d}: the image's header and its pixels disagree", .{p.index});
    var pcx: Cx = .{ .a = pa };
    const got = try generate(srv, &pcx, a, reader, id, p, bytes, fault, gone);
    const parsed = reader.parse(pa, got.content, .{ .width = w, .height = h }, got.frame) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        if (e != error.Unreadable) return fault.fail(a, .internal, "page {d}: the OCR reader failed ({s})", .{ p.index, @errorName(e) });
        log.line("{s}: page {d}: the model's output does not follow its format: {s}", .{ id, p.index, got.content });
        return fault.fail(a, .unreadable, "page {d}: the model's output does not follow its format", .{p.index});
    };
    const blocks = try a.alloc(api.ocr.OcrBlock, parsed.len);
    for (parsed, blocks) |b, *o| {
        const quad: ?api.ocr.Quad = if (b.quad) |q| geometry.validate(q, w, h) catch {
            log.line("{s}: page {d}: a box is not finite: {s}", .{ id, p.index, got.content });
            return fault.fail(a, .unreadable, "page {d}: the model's output has a box that is not finite", .{p.index});
        } else null;
        o.* = .{ .content = try a.dupe(u8, b.content), .quad = quad, .kind = b.kind };
    }
    const page: OcrPage = .{ .index = p.index, .width = w, .height = h, .dpi = p.dpi, .blocks = blocks };
    try sink.page(sink.ctx, a, p.source, &page, rgb, enc, fault);
}

const Generated = struct { content: []const u8, frame: api.Size };

/// The page as one user turn (image, then the reader's prompt) through chat.prepare and chat.generate, greedy.
fn generate(srv: *Server, pcx: *Cx, a: Allocator, reader: api.ocr.Ocr, id: []const u8, p: Planned, bytes: []const u8, fault: *Fault, gone: anytype) Step!Generated {
    const pa = pcx.a;
    const mime: []const u8 = switch (images.sniff(bytes)) {
        .png => "image/png",
        .jpeg => "image/jpeg",
        .webp => "image/webp",
        else => return fault.fail(a, .bad_input, "{s}", .{picture.problem(error.Invalid)}),
    };
    const turn = try userTurn(pa, try pdf.makeDataUrl(pa, mime, bytes), reader.prompt);
    const normalized = messages.normalizeWith(pcx, turn, "system", srv.needs_user_after_tool, srv.media()) catch |e| return chatFault(a, pcx, fault, e);
    const fields = try json.newObject(pa);
    try fields.put(pa, "enable_thinking", .{ .bool = false });
    try fields.put(pa, "temperature", .{ .float = 0 });
    const input: chat.Input = .{ .messages = normalized, .fields = .{ .object = fields }, .temperature = 0, .fill_room = true, .image_bytes = .{ .each = http_body.limit, .total = http_body.limit }, .id = id };
    const prepared = chat.prepare(srv, pcx, input, gone) catch |e| return chatFault(a, pcx, fault, e);
    const frame = (if (prepared.frames.len > 0) prepared.frames[0] else null) orelse {
        srv.release(prepared.preparing, prepared.request); // generate owns the request only once it is called
        return fault.fail(a, .internal, "the model's image step reported no frame", .{});
    };
    const reply = chat.generate(srv, pcx, prepared, null, gone) catch |e| return chatFault(a, pcx, fault, e);
    finished(a, fault, reply.finish_reason, p.index, gone) catch |e| {
        if (e == error.Faulted and fault.class == .unreadable) log.line("{s}: {s}: {s}", .{ id, fault.message, reply.content }); // the message names the page
        return e;
    };
    return .{ .content = reply.content, .frame = frame };
}

/// ``[{"role": "user", "content": [image part, text part]}]``, the text part left out when ``prompt`` is empty.
fn userTurn(a: Allocator, url: []const u8, prompt: []const u8) Allocator.Error!json.Value {
    const inner = try json.newObject(a);
    try inner.put(a, "url", .{ .string = url });
    const image = try json.newObject(a);
    try image.put(a, "type", .{ .string = "image_url" });
    try image.put(a, "image_url", .{ .object = inner });
    var parts: std.ArrayList(json.Value) = .empty;
    try parts.append(a, .{ .object = image });
    if (prompt.len > 0) {
        const text = try json.newObject(a);
        try text.put(a, "type", .{ .string = "text" });
        try text.put(a, "text", .{ .string = prompt });
        try parts.append(a, .{ .object = text });
    }
    const user = try json.newObject(a);
    try user.put(a, "role", .{ .string = "user" });
    try user.put(a, "content", .{ .array = parts.items });
    return .{ .array = try a.dupe(json.Value, &.{.{ .object = user }}) };
}

/// A chat failure as the request's fault; its message is copied into ``a`` before the page arena goes.
fn chatFault(a: Allocator, cx: *const Cx, fault: *Fault, e: chat.Failure) Step {
    return switch (e) {
        error.Refused => Fault.fromCx(a, cx, fault),
        error.Cancelled => error.Cancelled,
        error.OutOfMemory => error.OutOfMemory,
        error.Failed => fault.fail(a, .internal, "{s}", .{if (cx.message.len > 0) cx.message else "the engine failed the page"}),
    };
}

/// Only a reply that stopped on its end token is a finished page; every other finish fails the request.
pub fn finished(a: Allocator, fault: *Fault, reason: []const u8, index: u32, gone: anytype) Step!void {
    if (std.mem.eql(u8, reason, "stop")) return;
    if (std.mem.eql(u8, reason, "length")) return fault.fail(a, .unreadable, "page {d}: the model's output was cut at the context limit", .{index});
    if (std.mem.eql(u8, reason, loop_stop.finish_reason)) return fault.fail(a, .unreadable, "page {d}: the model's output repeated without end", .{index});
    if (std.mem.eql(u8, reason, "cancelled")) {
        if (gone.check()) return error.Cancelled;
        fault.* = .{ .class = .capacity, .message = "the engine stopped the page; retry", .retry_after = 1 }; // it closed or drained
        return error.Faulted;
    }
    return fault.fail(a, .internal, "page {d}: the engine ended the page with finish reason {s}", .{ index, reason });
}

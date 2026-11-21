//! A served OCR model for the route tests: a scripted engine, a one-image template, a fake reader and PNG pages.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("../model_text.zig");
const http = @import("../http_conn.zig");
const server = @import("../server.zig");
const TestText = @import("../stream_preflight_test.zig").TestText;
const Server = server.Server;
const Allocator = std.mem.Allocator;
const t = std.testing;

/// One scripted reply: its text, sent as one token a byte, and how it ends.
pub const Script = struct { text: []const u8, reason: api.Reason = .stop };

/// The engine, its vision step and its reader; ``srv`` serves them.
pub const Harness = struct {
    replies: []const Script = &.{},
    next: usize = 0,
    /// The tower input size the vision step reports; null: the missing-frame path.
    frame: ?api.Size = .{ .width = 100, .height = 50 },
    reader: ?api.ocr.Ocr = .{ .prompt = "Read the page.", .parse = parse },
    vision: bool = true,
    srv: *Server = undefined,

    pub fn start(h: *Harness, config: server.Config) !void {
        h.srv = try Server.init(t.allocator, t.io, h.engine(), text(), config, null);
    }

    pub fn stop(h: *Harness) void {
        h.srv.deinit();
    }

    fn engine(h: *Harness) api.Engine {
        return .{ .ctx = h, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        const v: api.Vision = .{ .ctx = h, .prepare = prepare, .image_token = '#', .image_tokens = 64, .max_images = 4 };
        return .{ .context_window = 4096, .vision = if (h.vision) v else null, .ocr = h.reader };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        const s = h.replies[h.next];
        h.next += 1;
        var ids: [8192]u32 = undefined;
        for (s.text, 0..) |b, i| ids[i] = b;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..s.text.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = s.reason } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn prepare(ctx: *anyopaque, a: Allocator, _: []const u8, _: u32) anyerror!api.PreparedImage {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        return .{ .pixels = try a.alloc(f32, 0), .gh = 1, .gw = 1, .tokens = 1, .hash = 7, .frame = h.frame };
    }
};

/// The fake reader: blocks split by "\n;;\n", each "kind x0 y0 x1 y1" in frame pixels, a newline, its content.
fn parse(a: Allocator, src: []const u8, original: api.Size, frame: api.Size) anyerror![]api.ocr.OcrBlock {
    if (std.mem.startsWith(u8, src, "!")) return error.Unreadable;
    var out: std.ArrayList(api.ocr.OcrBlock) = .empty;
    var it = std.mem.splitSequence(u8, src, "\n;;\n");
    while (it.next()) |block| {
        const nl = std.mem.indexOfScalar(u8, block, '\n') orelse block.len;
        var words = std.mem.tokenizeScalar(u8, block[0..nl], ' ');
        const kind = std.meta.stringToEnum(api.ocr.Kind, words.next() orelse return error.Unreadable) orelse return error.Unreadable;
        const content = if (nl < block.len) block[nl + 1 ..] else "";
        var v: [4]f32 = undefined;
        for (&v) |*x| x.* = try std.fmt.parseFloat(f32, words.next() orelse return error.Unreadable);
        const sx = @as(f32, @floatFromInt(original.width)) / @as(f32, @floatFromInt(frame.width));
        const sy = @as(f32, @floatFromInt(original.height)) / @as(f32, @floatFromInt(frame.height));
        const x0 = v[0] * sx;
        const y0 = v[1] * sy;
        const x1 = v[2] * sx;
        const y1 = v[3] * sy;
        try out.append(a, .{ .content = content, .quad = .{ .{ x0, y0 }, .{ x1, y0 }, .{ x1, y1 }, .{ x0, y1 } }, .kind = kind });
    }
    return out.items;
}

/// Bytes as tokens; every prompt renders "a#b", one image placeholder ('#', an added token) between two words.
fn text() model_text.Text {
    return .{ .ctx = undefined, .vtable = &.{ .encode = TestText.encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .added = added } };
}
fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
    const out = try a.alloc(u8, ids.len);
    for (ids, out) |id, *b| b.* = @intCast(id);
    return out;
}
fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
    return null;
}
fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
    return std.fmt.allocPrint(a, "{d}", .{id});
}
fn vocabSize(_: *anyopaque) u32 {
    return 256;
}
fn eosIds(_: *anyopaque) []const u32 {
    return &.{};
}
fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
    return .{ .text = try a.dupe(u8, "a#b") };
}
fn templateSource(_: *anyopaque) []const u8 {
    return "";
}
fn added(_: *anyopaque, id: u32) bool {
    return id == '#';
}

/// An 8-bit RGB PNG of ``w`` x ``h`` (stored deflate blocks), each pixel's colour from its position.
pub fn png(a: Allocator, w: u32, h: u32) ![]const u8 {
    var raw: std.ArrayList(u8) = .empty;
    for (0..h) |y| {
        try raw.append(a, 0); // filter: none
        for (0..w) |x| try raw.appendSlice(a, &.{ @truncate(x), @truncate(y), @truncate(x + y) });
    }
    var z: std.ArrayList(u8) = .empty;
    try z.appendSlice(a, &.{ 0x78, 0x01 });
    var rest = raw.items;
    while (true) {
        const n = @min(rest.len, 65535);
        const len: u16 = @intCast(n);
        try z.append(a, @intFromBool(n == rest.len));
        try z.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u16, len)));
        try z.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u16, ~len)));
        try z.appendSlice(a, rest[0..n]);
        rest = rest[n..];
        if (rest.len == 0) break;
    }
    try z.appendSlice(a, &std.mem.toBytes(std.mem.nativeToBig(u32, std.hash.Adler32.hash(raw.items))));
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "\x89PNG\r\n\x1a\n");
    var head: [13]u8 = undefined;
    std.mem.writeInt(u32, head[0..4], w, .big);
    std.mem.writeInt(u32, head[4..8], h, .big);
    @memcpy(head[8..], &[_]u8{ 8, 2, 0, 0, 0 });
    try chunk(a, &out, "IHDR", &head);
    try chunk(a, &out, "IDAT", z.items);
    try chunk(a, &out, "IEND", "");
    return out.items;
}

/// A valid PNG with a tEXt chunk saying "%PDF-1.7" right after its header, as an image editor may write.
pub fn pngWithPdfText(a: Allocator, w: u32, h: u32) ![]const u8 {
    const base = try png(a, w, h);
    const after_ihdr = 8 + 12 + 13; // signature, then IHDR's length, type, data and CRC
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, base[0..after_ihdr]);
    try chunk(a, &out, "tEXt", "Comment\x00%PDF-1.7");
    try out.appendSlice(a, base[after_ihdr..]);
    return out.items;
}

/// One PNG chunk: length, type, data and its CRC-32.
pub fn chunk(a: Allocator, out: *std.ArrayList(u8), kind: *const [4]u8, data: []const u8) !void {
    var len: [4]u8 = undefined;
    std.mem.writeInt(u32, &len, @intCast(data.len), .big);
    try out.appendSlice(a, &len);
    try out.appendSlice(a, kind);
    try out.appendSlice(a, data);
    var crc = std.hash.Crc32.init();
    crc.update(kind);
    crc.update(data);
    var sum: [4]u8 = undefined;
    std.mem.writeInt(u32, &sum, crc.final(), .big);
    try out.appendSlice(a, &sum);
}

/// What the connection wrote.
pub const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }

    /// The response's status code.
    pub fn status(w: *const Wire) u16 {
        return std.fmt.parseInt(u16, w.bytes.items[9..12], 10) catch 0;
    }

    /// The response's head, through the blank line.
    pub fn head(w: *const Wire) []const u8 {
        return w.bytes.items[0 .. (std.mem.indexOf(u8, w.bytes.items, "\r\n\r\n") orelse 0) + 4];
    }

    /// The response body parsed as JSON.
    pub fn parsed(w: *const Wire) !json.Value {
        return switch (try json.parse(w.a, w.bytes.items[w.head().len..])) {
            .ok => |v| v,
            .err => error.TestUnexpectedResult,
        };
    }
};

/// A POST of ``body`` to ``path`` with ``content_type``, its body already buffered; ``wire`` records the reply.
pub fn request(a: Allocator, wire: *Wire, path: []const u8, content_type: ?[]const u8, body: []const u8) !*http.Conn {
    const conn = try a.create(http.Conn);
    conn.* = try http.Conn.init(a, -1, "fixture");
    conn.write_hooks = .{ .ctx = wire, .write = Wire.write };
    conn.method = "POST";
    conn.version = "HTTP/1.1";
    conn.path = path;
    var headers: std.ArrayList(http.Header) = .empty;
    try headers.append(a, .{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) });
    if (content_type) |ct| try headers.append(a, .{ .name = "Content-Type", .value = ct });
    conn.headers = headers.items;
    if (body.len > conn.buf.len) return error.TestUnexpectedResult;
    @memcpy(conn.buf[0..body.len], body);
    conn.end = body.len;
    return conn;
}

//! The OCR page loop over a scripted engine: finish reasons, the missing frame, and whole /v1/ocr requests.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const routes = @import("../routes.zig");
const run = @import("run.zig");
const types = @import("types.zig");
const budget = @import("budget.zig");
const harness = @import("harness_test.zig");
const picture = @import("core").picture;
const t = std.testing;
const Allocator = std.mem.Allocator;
const Harness = harness.Harness;

const Gone = struct {
    left: bool = false,
    pub fn check(g: @This()) bool {
        return g.left;
    }
};

test "ocr run: only a stopped reply is a page; length, loops, a stop and other ends fail by class" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fault: types.Fault = undefined;
    try run.finished(a, &fault, "stop", 3, Gone{});
    try t.expectError(error.Faulted, run.finished(a, &fault, "length", 3, Gone{}));
    try t.expectEqual(types.Class.unreadable, fault.class);
    try t.expectEqualStrings("page 3: the model's output was cut at the context limit", fault.message);
    try t.expectError(error.Faulted, run.finished(a, &fault, "content_filter", 3, Gone{}));
    try t.expectEqualStrings("page 3: the model's output repeated without end", fault.message);
    try t.expectError(error.Cancelled, run.finished(a, &fault, "cancelled", 3, Gone{ .left = true }));
    try t.expectError(error.Faulted, run.finished(a, &fault, "cancelled", 3, Gone{}));
    try t.expectEqual(types.Class.capacity, fault.class);
    try t.expectEqual(@as(?u32, 1), fault.retry_after);
    try t.expectError(error.Faulted, run.finished(a, &fault, "tool_calls", 3, Gone{}));
    try t.expectEqual(types.Class.internal, fault.class);
}

/// Records each page the loop hands over.
const Seen = struct {
    pages: std.ArrayList(types.OcrPage) = .empty,
    sizes: std.ArrayList([2]usize) = .empty,
    limit: u64 = 0,
    fn sink(s: *Seen) run.PageSink {
        return .{ .ctx = s, .page = page };
    }
    fn page(ctx: *anyopaque, a: Allocator, p: *const types.OcrPage, rgb: picture.Rgb, enc: *budget.Encodes, _: *types.Fault) run.Step!void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        try s.pages.append(a, p.*);
        try s.sizes.append(a, .{ rgb.width, rgb.height });
        s.limit = enc.limit_pixels;
    }
};

fn imageSource(a: Allocator, w: u32, h: u32) !types.Source {
    return .{ .bytes = try harness.png(a, w, h) };
}

test "ocr run: blocks scale from the frame, quads are validated, and the sink sees the live pixels" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = .{ .replies = &.{ .{ .text = "title 10 5 90 10\n# T\n;;\ntext -5 40 120 60\nbody" }, .{ .text = "text 0 0 100 50\nall" } } };
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
    defer h.stop();
    var seen: Seen = .{};
    var fault: types.Fault = undefined;
    try run.run(h.srv, a, &.{ try imageSource(a, 200, 100), try imageSource(a, 30, 20) }, seen.sink(), &fault, Gone{});
    try t.expectEqual(@as(usize, 2), seen.pages.items.len);
    const p = seen.pages.items[0];
    try t.expectEqual(@as(u32, 200), p.width);
    try t.expectEqual(@as(u32, 72), p.dpi);
    try t.expectEqual([2]usize{ 200, 100 }, seen.sizes.items[0]);
    try t.expectEqual(api.ocr.Quad{ .{ 20, 10 }, .{ 180, 10 }, .{ 180, 20 }, .{ 20, 20 } }, p.blocks[0].quad);
    // a box past the page is clamped into it
    try t.expectEqual(api.ocr.Quad{ .{ 0, 80 }, .{ 200, 80 }, .{ 200, 100 }, .{ 0, 100 } }, p.blocks[1].quad);
    try t.expectEqualStrings("body", p.blocks[1].content);
    try t.expectEqual(api.ocr.Quad{ .{ 0, 0 }, .{ 30, 0 }, .{ 30, 20 }, .{ 0, 20 } }, seen.pages.items[1].blocks[0].quad);
    // the encode limit grew by four times each page's pixels
    try t.expectEqual(@as(u64, 4 * (200 * 100 + 30 * 20)), seen.limit);
    try t.expectEqual(@as(i64, 0), h.srv.preparing.load(.acquire));
}

test "ocr run: a missing frame releases the prepared request and fails as internal" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = .{ .frame = null };
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
    defer h.stop();
    var seen: Seen = .{};
    var fault: types.Fault = undefined;
    try t.expectError(error.Faulted, run.run(h.srv, a, &.{try imageSource(a, 8, 8)}, seen.sink(), &fault, Gone{}));
    try t.expectEqual(types.Class.internal, fault.class);
    try t.expectEqualStrings("the model's image step reported no frame", fault.message);
    try t.expectEqual(@as(i64, 0), h.srv.preparing.load(.acquire));
    try t.expectEqual(@as(usize, 0), h.next); // nothing was submitted
    // a client that left before a page is never generated
    try t.expectError(error.Cancelled, run.run(h.srv, a, &.{try imageSource(a, 8, 8)}, seen.sink(), &fault, Gone{ .left = true }));
}

test "ocr run: undecodable bytes refuse before any generation" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = .{};
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
    defer h.stop();
    var seen: Seen = .{};
    var fault: types.Fault = undefined;
    const junk: types.Source = .{ .bytes = "\x89PNG\r\n\x1a\nnope" };
    try t.expectError(error.Faulted, run.run(h.srv, a, &.{junk}, seen.sink(), &fault, Gone{}));
    try t.expectEqual(types.Class.bad_input, fault.class);
    try t.expectEqual(@as(usize, 0), h.next);
}

/// Reserves five whole-page encodes on the first page of a request, as five full-page pictures would.
const Greedy = struct {
    limit: u64 = 0,
    first: bool = true,
    fn sink(g: *Greedy) run.PageSink {
        g.first = true;
        return .{ .ctx = g, .page = page };
    }
    fn page(ctx: *anyopaque, _: Allocator, p: *const types.OcrPage, _: picture.Rgb, enc: *budget.Encodes, fault: *types.Fault) run.Step!void {
        const g: *Greedy = @ptrCast(@alignCast(ctx));
        if (!g.first) return;
        g.first = false;
        g.limit = enc.limit_pixels;
        for (0..5) |_| try enc.before(fault, p.width, p.height);
    }
};

test "ocr run: the encode budget is the whole request's, so a first page may use what a later page brings" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = .{ .replies = &.{ .{ .text = "text 0 0 1 1\na" }, .{ .text = "text 0 0 1 1\nb" }, .{ .text = "text 0 0 1 1\nc" } } };
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
    defer h.stop();
    var g: Greedy = .{};
    var fault: types.Fault = undefined;
    // two equal pages: 4 x 2 pages of pixels allow five whole-page encodes on page 0
    try run.run(h.srv, a, &.{ try imageSource(a, 40, 30), try imageSource(a, 40, 30) }, g.sink(), &fault, Gone{});
    try t.expectEqual(@as(u64, 4 * 2 * 40 * 30), g.limit);
    // one page alone allows four
    try t.expectError(error.Faulted, run.run(h.srv, a, &.{try imageSource(a, 40, 30)}, g.sink(), &fault, Gone{}));
    try t.expectEqualStrings(budget.over_message, fault.message);
    try t.expectEqual(types.Class.unreadable, fault.class);
}

/// An RGB IHDR chunk naming ``size``, appended to ``out``.
fn ihdr(a: Allocator, out: *std.ArrayList(u8), size: [2]u32) !void {
    var head: [13]u8 = undefined;
    std.mem.writeInt(u32, head[0..4], size[0], .big);
    std.mem.writeInt(u32, head[4..8], size[1], .big);
    @memcpy(head[8..], &[_]u8{ 8, 2, 0, 0, 0 });
    try harness.chunk(a, out, "IHDR", &head);
}

/// A PNG whose pixels are ``last`` but whose first IHDR names ``first``: the decoder reads the last header before IDAT.
fn twoHeaders(a: Allocator, first: [2]u32, last: [2]u32) ![]const u8 {
    const real = try harness.png(a, last[0], last[1]);
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, real[0..8]);
    try ihdr(a, &out, first);
    try out.appendSlice(a, real[8..]);
    return out.items;
}

/// A PNG whose one IHDR names ``size`` before an 8 x 8 image's pixel data: the header alone decides a size refusal.
fn bigHeader(a: Allocator, size: [2]u32) ![]const u8 {
    const real = try harness.png(a, 8, 8);
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, real[0..8]);
    try ihdr(a, &out, size);
    try out.appendSlice(a, real[8 + 25 ..]); // past the real IHDR: length, type, 13 bytes, CRC
    return out.items;
}

test "ocr run: an image whose first header and its pixels disagree is refused before generation" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = .{ .replies = &.{.{ .text = "text 0 0 1 1\na" }} };
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
    defer h.stop();
    const orders = [_][2][2]u32{ .{ .{ 80, 60 }, .{ 40, 30 } }, .{ .{ 40, 30 }, .{ 80, 60 } } };
    for (orders) |o| {
        var seen: Seen = .{};
        var fault: types.Fault = undefined;
        const src: types.Source = .{ .bytes = try twoHeaders(a, o[0], o[1]) };
        try t.expectError(error.Faulted, run.run(h.srv, a, &.{src}, seen.sink(), &fault, Gone{}));
        try t.expectEqual(types.Class.bad_input, fault.class);
        try t.expectEqualStrings("page 0: the image's header and its pixels disagree", fault.message);
        try t.expectEqual(@as(usize, 0), seen.pages.items.len);
    }
    try t.expectEqual(@as(usize, 0), h.next);
}

const page_text =
    "page_header 0 0 100 5\nRunning head\n;;\n" ++
    "title 10 5 90 10\n# Report\n;;\n" ++
    "picture 10 10 60 30\n\n;;\n" ++
    "picture 70 10 72 12\n\n;;\n" ++
    "table 10 30 90 40\n<table><tr><th>a</th><th>b</th></tr><tr><td colspan=\"2\">wide</td></tr></table>\n;;\n" ++
    "text 10 40 90 47\nSome **text**:\n\n| x | y |\n| --- | --- |\n| 1 | 2 |\n;;\n" ++
    "page_footer 0 47 100 50\nPage 1";

/// A whole POST /v1/ocr through the dispatcher on a 200 x 100 PNG; the reply as JSON, its status in ``status``.
fn post(a: Allocator, h: *Harness, extra: []const u8, status: *u16) !json.Value {
    return postPng(a, h, try harness.png(a, 200, 100), extra, status);
}

/// A whole POST /v1/ocr through the dispatcher on ``png``; the reply as JSON, its status in ``status``.
fn postPng(a: Allocator, h: *Harness, png: []const u8, extra: []const u8, status: *u16) !json.Value {
    const enc = std.base64.standard.Encoder;
    const b64 = try a.alloc(u8, enc.calcSize(png.len));
    const body = try std.fmt.allocPrint(a, "{{\"model\": \"mistral-ocr-latest\", \"document\": {{\"type\": \"image_url\", \"image_url\": \"data:image/png;base64,{s}\"}}{s}}}", .{ enc.encode(b64, png), extra });
    var wire: harness.Wire = .{ .a = a };
    const conn = try harness.request(a, &wire, "/v1/ocr", "application/json", body);
    routes.dispatch(h.srv, conn, a);
    status.* = wire.status();
    try t.expect(std.mem.indexOf(u8, wire.head(), "\r\nContent-Type: application/json\r\n") != null);
    return wire.parsed();
}

fn serve(h: *Harness, replies: []const harness.Script) !void {
    h.* = .{ .replies = replies };
    try h.start(.{ .served_name = "m", .model_ids = &.{ "m", "mistral-ocr-latest" }, .enable_thinking = false, .use_drafts = false });
}

test "ocr mistral: a whole request gives blocks, images and tables in the SDKs' shape" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    try serve(&h, &.{.{ .text = page_text }});
    defer h.stop();
    var status: u16 = 0;
    const r = try post(a, &h, ", \"table_format\": \"html\", \"extract_header\": true, \"image_min_size\": 10", &status);
    try t.expectEqual(@as(u16, 200), status);
    try t.expectEqualStrings("mistral-ocr-latest", r.strField("model").?);
    try t.expect(r.get("document_annotation").? == .null);
    try t.expectEqualStrings("1", r.get("usage_info").?.get("pages_processed").?.int);
    const page = r.get("pages").?.array[0];
    try t.expectEqualStrings("0", page.get("index").?.int);
    try t.expectEqualStrings("# Report\n\n![img-0.jpeg](img-0.jpeg)\n\n[tbl-0.html](tbl-0.html)\n\nSome **text**:\n\n[tbl-1.html](tbl-1.html)\n\nPage 1", page.strField("markdown").?);
    try t.expectEqualStrings("Running head", page.strField("header").?);
    try t.expect(page.get("footer").? == .null);
    try t.expect(page.get("hyperlinks") == null);
    const dims = page.get("dimensions").?;
    try t.expectEqualStrings("72", dims.get("dpi").?.int);
    try t.expectEqualStrings("100", dims.get("height").?.int);
    try t.expectEqualStrings("200", dims.get("width").?.int);
    // the 4 x 4 picture is under image_min_size: no image, no reference, no block
    const imgs = page.get("images").?.array;
    try t.expectEqual(@as(usize, 1), imgs.len);
    try t.expectEqualStrings("img-0.jpeg", imgs[0].strField("id").?);
    try t.expectEqualStrings("20", imgs[0].get("top_left_x").?.int);
    try t.expectEqualStrings("60", imgs[0].get("bottom_right_y").?.int);
    try t.expect(imgs[0].get("image_base64") == null);
    const tables = page.get("tables").?.array;
    try t.expectEqual(@as(usize, 2), tables.len);
    try t.expectEqualStrings("<table><tr><th>a</th><th>b</th></tr><tr><td colspan=\"2\">wide</td></tr></table>", tables[0].strField("content").?);
    try t.expectEqualStrings("html", tables[1].strField("format").?);
    try t.expectEqualStrings("<table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr></table>", tables[1].strField("content").?);
    const blocks = page.get("blocks").?.array;
    const want = [_][]const u8{ "header", "title", "image", "table", "text", "footer" };
    try t.expectEqual(want.len, blocks.len);
    for (blocks, want) |b, w| {
        try t.expectEqualStrings(w, b.strField("type").?);
        for ([_][]const u8{ "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y" }) |k| try t.expect(b.get(k).? == .int);
        try t.expect(b.get("content").? == .string);
    }
    try t.expectEqualStrings("img-0.jpeg", blocks[2].strField("image_id").?);
    try t.expectEqualStrings("", blocks[2].strField("content").?);
    try t.expectEqualStrings("tbl-0.html", blocks[3].strField("table_id").?);
    // a block holds its own text, its tables as tables[] holds them; the page Markdown holds the links
    try t.expectEqualStrings(tables[0].strField("content").?, blocks[3].strField("content").?);
    try t.expectEqualStrings("Some **text**:\n\n<table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr></table>", blocks[4].strField("content").?);
}

test "ocr mistral: with table_format markdown a block keeps its pipe table and a spanned table as HTML" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    try serve(&h, &.{.{ .text = page_text }});
    defer h.stop();
    var status: u16 = 0;
    const r = try post(a, &h, ", \"table_format\": \"markdown\"", &status);
    try t.expectEqual(@as(u16, 200), status);
    const page = r.get("pages").?.array[0];
    const tables = page.get("tables").?.array;
    try t.expectEqualStrings("tbl-0.html", tables[0].strField("id").?);
    try t.expectEqualStrings("html", tables[0].strField("format").?);
    try t.expectEqualStrings("tbl-1.md", tables[1].strField("id").?);
    try t.expectEqualStrings("| x | y |\n| --- | --- |\n| 1 | 2 |", tables[1].strField("content").?);
    try t.expect(std.mem.indexOf(u8, page.strField("markdown").?, "Some **text**:\n\n[tbl-1.md](tbl-1.md)") != null);
    const blocks = page.get("blocks").?.array; // both pictures kept: header, title, image, image, table, text, footer
    try t.expectEqualStrings("tbl-0.html", blocks[4].strField("table_id").?);
    try t.expectEqualStrings("<table><tr><th>a</th><th>b</th></tr><tr><td colspan=\"2\">wide</td></tr></table>", blocks[4].strField("content").?);
    try t.expectEqualStrings("Some **text**:\n\n| x | y |\n| --- | --- |\n| 1 | 2 |", blocks[5].strField("content").?);
    try t.expect(blocks[5].get("table_id") == null);
}

test "ocr mistral: inline tables, image_limit, no blocks, and the served name for an unknown model" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    try serve(&h, &.{.{ .text = page_text }});
    defer h.stop();
    var status: u16 = 0;
    const r = try post(a, &h, ", \"image_limit\": 1, \"include_blocks\": false, \"model\": \"other\"", &status);
    try t.expectEqual(@as(u16, 200), status);
    try t.expectEqualStrings("m", r.strField("model").?);
    const page = r.get("pages").?.array[0];
    try t.expect(page.get("blocks") == null);
    try t.expectEqual(@as(usize, 0), page.get("tables").?.array.len);
    try t.expectEqual(@as(usize, 1), page.get("images").?.array.len);
    const md = page.strField("markdown").?;
    try t.expect(std.mem.startsWith(u8, md, "Running head\n\n# Report\n\n![img-0.jpeg](img-0.jpeg)\n\n| a | b |\n| --- | --- |\n| wide |"));
    try t.expect(std.mem.indexOf(u8, md, "img-1") == null);
    try t.expect(std.mem.endsWith(u8, md, "| 1 | 2 |\n\nPage 1"));
}

test "ocr mistral: an image crop decodes to its box's size" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    try serve(&h, &.{.{ .text = "picture 10 10 60 30\n" }});
    defer h.stop();
    var status: u16 = 0;
    const r = try post(a, &h, ", \"include_image_base64\": true", &status);
    if (status == 400) { // only a host without the pinned libjpeg-turbo skips
        try t.expect(std.mem.indexOf(u8, r.get("detail").?.array[0].strField("msg").?, "libjpeg-turbo") != null);
        return error.SkipZigTest;
    }
    try t.expectEqual(@as(u16, 200), status);
    const url = r.get("pages").?.array[0].get("images").?.array[0].strField("image_base64").?;
    try t.expect(std.mem.startsWith(u8, url, "data:image/jpeg;base64,"));
    const b64 = url["data:image/jpeg;base64,".len..];
    const jpeg = try a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(b64));
    try std.base64.standard.Decoder.decode(jpeg, b64);
    const img = picture.decode(t.io, a, jpeg) catch |e| switch (e) {
        error.NoJpegLibrary, error.JpegLibraryVersion => return error.SkipZigTest,
        else => return e,
    };
    try t.expectEqual(@as(usize, 100), img.width);
    try t.expectEqual(@as(usize, 40), img.height);
}

test "ocr mistral: unreadable output, a cut page and a missing reader answer 422, 422 and 400" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    try serve(&h, &.{ .{ .text = "!not the grammar" }, .{ .text = "text 0 0 1 1\nhalf", .reason = .length } });
    defer h.stop();
    var status: u16 = 0;
    const cases = [_]struct { u16, []const u8 }{
        .{ 422, "page 0: the model's output does not follow its format" },
        .{ 422, "page 0: the model's output was cut at the context limit" },
    };
    for (cases) |c| {
        const r = try post(a, &h, "", &status);
        try t.expectEqual(c[0], status);
        const d = r.get("detail").?.array[0];
        try t.expectEqualStrings(c[1], d.strField("msg").?);
        try t.expectEqualStrings("unreadable_page", d.strField("type").?);
    }
    h.srv.info.ocr = null;
    const r = try post(a, &h, "", &status);
    try t.expectEqual(@as(u16, 400), status);
    try t.expectEqualStrings(run.no_reader, r.get("detail").?.array[0].strField("msg").?);
    h.srv.info.ocr = h.reader;
    h.srv.info.vision = null;
    const v = try post(a, &h, "", &status);
    try t.expectEqual(@as(u16, 400), status);
    try t.expectEqualStrings(run.no_vision, v.get("detail").?.array[0].strField("msg").?);
    try t.expectEqual(@as(usize, 2), h.next); // the two pages above only: the gate refused before generation
}

test "ocr mistral: a page past the decoded-pixel limit answers 400 with the decoder's message, before generation" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    try serve(&h, &.{.{ .text = "text 0 0 1 1\na" }});
    defer h.stop();
    // 4097 x 4096 is 16 Mi pixels and 4096 more, under 8192 a side: only the pixel limit refuses it
    const w = 4097;
    const ht = 4096;
    try t.expect(@as(u64, w) * ht > run.max_page_pixels and @max(w, ht) <= picture.max_side);
    var status: u16 = 0;
    const r = try postPng(a, &h, try bigHeader(a, .{ w, ht }), "", &status);
    try t.expectEqual(@as(u16, 400), status);
    try t.expectEqualStrings(picture.problem(error.Oversized), r.get("detail").?.array[0].strField("msg").?);
    try t.expectEqual(@as(usize, 0), h.next);
}

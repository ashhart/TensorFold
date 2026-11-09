//! dots.mocr's reader against Hugging Face oracle outputs, box scaling across frames, and content normalization.
const std = @import("std");
const api = @import("engine_api");
const ocr = @import("ocr.zig");
const image = @import("image.zig");
const cfg = @import("config.zig");
const cases = @import("config_test.zig");
const t = std.testing;

const Fixture = struct { text: []const u8, original: [2]u32, frame: [2]u32, source: []const u8, prompt_sha256: []const u8 };
const page_json = @embedFile("fixtures/ocr-demo-page.json");
const line_json = @embedFile("fixtures/ocr-one-line.json");
const cut_json = @embedFile("fixtures/ocr-demo-cut.json");

fn fixture(a: std.mem.Allocator, text: []const u8) !Fixture {
    return std.json.parseFromSliceLeaky(Fixture, a, text, .{});
}

fn read(a: std.mem.Allocator, f: Fixture) ![]api.ocr.OcrBlock {
    return ocr.parse(a, f.text, .{ .width = f.original[0], .height = f.original[1] }, .{ .width = f.frame[0], .height = f.frame[1] });
}

fn single(a: std.mem.Allocator, text: []const u8, kind: api.ocr.Kind) ![]const u8 {
    const blocks = try ocr.parse(a, text, .{ .width = 100, .height = 100 }, .{ .width = 100, .height = 100 });
    try t.expectEqual(@as(usize, 1), blocks.len);
    try t.expectEqual(kind, blocks[0].kind);
    return blocks[0].content;
}

test "dots: the reader's prompt is the one every oracle run recorded" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(ocr.prompt, &digest, .{});
    for ([_][]const u8{ page_json, line_json, cut_json }) |text|
        try t.expectEqualStrings((try fixture(arena.allocator(), text)).prompt_sha256, &std.fmt.bytesToHex(digest, .lower));
}

test "dots: the demo page reads as header, caption, spanned table and text blocks" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try fixture(a, page_json);
    const blocks = try read(a, f);
    try t.expect(blocks.len > 4);
    try t.expectEqual(api.ocr.Kind.page_header, blocks[0].kind);
    try t.expectEqualStrings("EXPOSURE TO MEAT AND RISK OF LYMPHOMA", blocks[0].content);
    // The oracle's first box [628, 172, 1077, 194] in a 1708x2240 frame, in the 1700x2250 original.
    const sx: f32 = 1700.0 / 1708.0;
    const sy: f32 = 2250.0 / 2240.0;
    const want: api.ocr.Quad = .{ .{ 628 * sx, 172 * sy }, .{ 1077 * sx, 172 * sy }, .{ 1077 * sx, 194 * sy }, .{ 628 * sx, 194 * sy } };
    for (want, blocks[0].quad) |w, g| for (w, g) |x, y| try t.expectApproxEqAbs(x, y, 1e-3);
    const kinds = [_]api.ocr.Kind{ .page_header, .page_header, .caption, .table, .text, .caption, .table, .table, .text, .text, .text, .text, .text };
    try t.expectEqual(kinds.len, blocks.len);
    for (kinds, blocks) |kind, b| {
        try t.expectEqual(kind, b.kind);
        try t.expect(b.content.len > 0);
        for (b.quad) |p| try t.expect(p[0] >= 0 and p[0] <= 1700 and p[1] >= 0 and p[1] <= 2250);
        // Every table on the page has spans, so each stays one clean HTML table.
        if (b.kind == .table) try t.expect(std.mem.startsWith(u8, b.content, "<table>") and std.mem.endsWith(u8, b.content, "</table>"));
    }
    try t.expect(std.mem.startsWith(u8, blocks[2].content, "TABLE II - ODDS RATIO OF HODGKIN LYMPHOMA"));
    try t.expect(std.mem.indexOf(u8, blocks[3].content, "<th colspan=\"3\">Hodgkin lymphoma</th>") != null);
    try t.expect(std.mem.indexOf(u8, blocks[6].content, "<th colspan=\"4\">Beef meat</th>") != null);
}

test "dots: one line reads as one text block" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const blocks = try read(a, try fixture(a, line_json));
    try t.expectEqual(@as(usize, 1), blocks.len);
    try t.expectEqual(api.ocr.Kind.text, blocks[0].kind);
    try t.expectEqualStrings("TensorFold reads one line of text.", blocks[0].content);
}

test "dots: output cut by the token limit is unreadable" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectError(error.Unreadable, read(a, try fixture(a, cut_json)));
}

test "dots: boxes scale to the original from the frames of two token budgets" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try image.Processor.parse(a, @embedFile("fixtures/preprocessor_config.json"), try cfg.Config.parse(a, cases.sample, cases.generation));
    const rgb = try a.alloc(u8, 1700 * 2250 * 3);
    @memset(rgb, 255);
    const original: api.Size = .{ .width = 1700, .height = 2250 };
    var frames: [2]api.Size = undefined;
    for ([_]u32{ 14400, 1200 }, &frames) |budget, *frame| {
        const prepared = ocr.prepared(try p.prepareRgb(a, .{ .width = 1700, .height = 2250, .pixels = rgb }, 0, budget), p.patches.patch);
        frame.* = prepared.frame.?;
        // One block covering the frame's middle half covers the original's middle half.
        const text = try std.fmt.allocPrint(a, "[{{\"bbox\": [{d}, {d}, {d}, {d}], \"category\": \"Text\", \"text\": \"x\"}}]", .{ frame.width / 4, frame.height / 4, frame.width / 4 * 3, frame.height / 4 * 3 });
        const blocks = try ocr.parse(a, text, original, frame.*);
        const q = blocks[0].quad;
        const sx = 1700.0 / @as(f32, @floatFromInt(frame.width));
        const sy = 2250.0 / @as(f32, @floatFromInt(frame.height));
        try t.expectApproxEqAbs(@as(f32, @floatFromInt(frame.width / 4)) * sx, q[0][0], 1e-3);
        try t.expectApproxEqAbs(@as(f32, @floatFromInt(frame.height / 4 * 3)) * sy, q[2][1], 1e-3);
        try t.expectApproxEqAbs(1700.0 / 4.0, q[0][0], 1700.0 / @as(f32, @floatFromInt(frame.width)) * 2);
        try t.expectApproxEqAbs(2250.0 / 4.0 * 3.0, q[2][1], 2250.0 / @as(f32, @floatFromInt(frame.height)) * 4);
    }
    try t.expect(frames[0].width > frames[1].width and frames[0].height > frames[1].height);
}

/// A golden case: the token budget, the frame smart_resize gives, one box in frame pixels and in client pixels.
const Golden = struct { budget: u32, frame: api.Size, tokens: u32, model: [4]u32, client: [4]f32 };

/// A 1000 x 333 page; each row's numbers are worked by hand from the processor's rules, not from the code under test.
const goldens = [_]Golden{
    // ample budget: rows = roundEven(333/28 = 11.89)*28 = 336, cols = roundEven(1000/28 = 35.71)*28 = 1008
    .{ .budget = 16384, .frame = .{ .width = 1008, .height = 336 }, .tokens = 24 * 72 / 4, .model = .{ 126, 112, 882, 224 }, .client = .{ 125, 111, 875, 222 } },
    // 100 tokens, 78400 px: d = sqrt(333000/78400) = 2.061; floor(333/d/28)*28 = 140, floor(1000/d/28)*28 = 476
    .{ .budget = 100, .frame = .{ .width = 476, .height = 140 }, .tokens = 10 * 34 / 4, .model = .{ 119, 35, 357, 105 }, .client = .{ 250, 83.25, 750, 249.75 } },
};

test "dots: golden boxes on a 1000 x 333 page come back in the client's pixels from two stretched frames" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try image.Processor.parse(a, @embedFile("fixtures/preprocessor_config.json"), try cfg.Config.parse(a, cases.sample, cases.generation));
    const rgb = try a.alloc(u8, 1000 * 333 * 3);
    @memset(rgb, 200);
    const original: api.Size = .{ .width = 1000, .height = 333 };
    for (goldens) |g| {
        // smart_resize stretches each side to a multiple of 28 (14 px patches, 2 x 2 merge); it never pads
        const prepared = ocr.prepared(try p.prepareRgb(a, .{ .width = 1000, .height = 333, .pixels = rgb }, 0, g.budget), p.patches.patch);
        // so the frame is the whole resized page (processor-reference.json: HF's 217 x 300 gives 224 x 308)
        try t.expectEqual(g.frame, prepared.frame.?);
        try t.expectEqual(g.tokens, prepared.tokens);
        // the frame's aspect differs from the page's (3.0 or 3.4 against 3.003), so x and y scale apart
        const text = try std.fmt.allocPrint(a, "[{{\"bbox\": [{d}, {d}, {d}, {d}], \"category\": \"Text\", \"text\": \"x\"}}]", .{ g.model[0], g.model[1], g.model[2], g.model[3] });
        const blocks = try ocr.parse(a, text, original, prepared.frame.?);
        try t.expectEqual(@as(usize, 1), blocks.len);
        const q = blocks[0].quad;
        const want: api.ocr.Quad = .{ .{ g.client[0], g.client[1] }, .{ g.client[2], g.client[1] }, .{ g.client[2], g.client[3] }, .{ g.client[0], g.client[3] } };
        for (want, q) |w, got| {
            try t.expectApproxEqAbs(w[0], got[0], 1e-3);
            try t.expectApproxEqAbs(w[1], got[1], 1e-3);
        }
    }
}

test "dots: prepare reports the processor's resized size as the frame" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try fixture(a, page_json);
    const p = try image.Processor.parse(a, @embedFile("fixtures/preprocessor_config.json"), try cfg.Config.parse(a, cases.sample, cases.generation));
    const rgb = try a.alloc(u8, @as(usize, f.original[0]) * f.original[1] * 3);
    @memset(rgb, 128);
    const prepared = ocr.prepared(try p.prepareRgb(a, .{ .width = f.original[0], .height = f.original[1], .pixels = rgb }, 0, 14400), p.patches.patch);
    try t.expectEqual(@as(u32, 160), prepared.gh);
    try t.expectEqual(@as(u32, 122), prepared.gw);
    try t.expectEqual(api.Size{ .width = f.frame[0], .height = f.frame[1] }, prepared.frame.?);
    try t.expectEqual(@as(u32, 1708 * 2240), prepared.frame.?.width * prepared.frame.?.height);
}

test "dots: categories map to kinds and content is normalized Markdown" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const block = "[{{\"bbox\": [1, 2, 3, 4], \"category\": \"{s}\", \"text\": \"{s}\"}}]";
    const Case = struct { []const u8, []const u8, api.ocr.Kind, []const u8 };
    for ([_]Case{
        .{ "Title", "Results", .title, "# Results" },
        .{ "Title", "# Results", .title, "# Results" },
        .{ "Section-header", "Methods", .section_header, "## Methods" },
        .{ "Section-header", "### 2.1 Data", .section_header, "### 2.1 Data" },
        .{ "Formula", "$$E = mc^2$$", .formula, "$$\nE = mc^2\n$$" },
        .{ "Formula", "\\\\[ a + b \\\\]", .formula, "$$\na + b\n$$" },
        .{ "Formula", "$x$", .formula, "$$\nx\n$$" },
        .{ "Formula", "x^2", .formula, "$$\nx^2\n$$" },
        .{ "Table", "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>", .table, "| a | b |\n| --- | --- |\n| 1 | 2 |" },
        .{ "List-item", "first", .list, "- first" },
        .{ "List-item", "- second", .list, "- second" },
        .{ "List-item", "3. third", .list, "3. third" },
        .{ "Caption", "TABLE I", .caption, "TABLE I" },
        .{ "Footnote", "a note", .footnote, "a note" },
        .{ "Page-header", "2763", .page_header, "2763" },
        .{ "Page-footer", "page 2", .page_footer, "page 2" },
        .{ "Text", "  body  ", .text, "body" },
        .{ "Diagram", "unknown", .text, "unknown" },
    }) |k| {
        const text = try std.fmt.allocPrint(a, block, .{ k[0], k[1] });
        try t.expectEqualStrings(k[3], try single(a, text, k[2]));
    }
}

test "dots: pictures keep empty content, empty blocks drop, reversed boxes swap" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // run011's case: a block whose text is empty is dropped; a picture needs no text.
    const text =
        \\[{"bbox": [10, 20, 30, 40], "category": "Picture"},
        \\ {"bbox": [1, 1, 5, 5], "category": "Text", "text": "  "},
        \\ {"bbox": [1, 1, 5, 5], "category": "Title", "text": ""},
        \\ {"bbox": [50, 60, 10, 20], "category": "Text", "text": "kept", "extra": 1},
        \\ {"bbox": [0.5, 1.5, 2.5, 3.5], "category": "Picture", "text": "ignored"}]
    ;
    const blocks = try ocr.parse(a, text, .{ .width = 200, .height = 400 }, .{ .width = 100, .height = 200 });
    try t.expectEqual(@as(usize, 3), blocks.len);
    try t.expectEqual(api.ocr.Kind.picture, blocks[0].kind);
    try t.expectEqualStrings("", blocks[0].content);
    try t.expectEqual(api.ocr.Quad{ .{ 20, 40 }, .{ 60, 40 }, .{ 60, 80 }, .{ 20, 80 } }, blocks[0].quad);
    try t.expectEqualStrings("kept", blocks[1].content);
    try t.expectEqual(api.ocr.Quad{ .{ 20, 40 }, .{ 100, 40 }, .{ 100, 120 }, .{ 20, 120 } }, blocks[1].quad);
    try t.expectEqual(api.ocr.Quad{ .{ 1, 3 }, .{ 5, 3 }, .{ 5, 7 }, .{ 1, 7 } }, blocks[2].quad);
    try t.expectEqual(@as(usize, 0), (try ocr.parse(a, " [] \n", .{ .width = 1, .height = 1 }, .{ .width = 1, .height = 1 })).len);
}

test "dots: text outside the block grammar is unreadable" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        "",
        "not json",
        "{\"bbox\": [1, 2, 3, 4], \"category\": \"Text\", \"text\": \"x\"}",
        "{\"layout\": [{\"bbox\": [1, 2, 3, 4], \"category\": \"Text\", \"text\": \"x\"}]}",
        "[1]",
        "[{\"category\": \"Text\", \"text\": \"x\"}]",
        "[{\"bbox\": [1, 2, 3], \"category\": \"Text\", \"text\": \"x\"}]",
        "[{\"bbox\": [1, 2, 3, \"4\"], \"category\": \"Text\", \"text\": \"x\"}]",
        "[{\"bbox\": [1, 2, 3, 1e999], \"category\": \"Text\", \"text\": \"x\"}]",
        "[{\"bbox\": [1, 2, 3, 4], \"category\": 7, \"text\": \"x\"}]",
        "[{\"bbox\": [1, 2, 3, 4], \"text\": \"x\"}]",
        "[{\"bbox\": [1, 2, 3, 4], \"category\": \"Text\"}]",
        "[{\"bbox\": [1, 2, 3, 4], \"category\": \"Text\", \"text\": null}]",
        "[{\"bbox\": [1, 2, 3, 4], \"category\": \"Table\", \"text\": \"<table><tr><td>a</td></tr></table> trailing\"}]",
        "[{\"bbox\": [1, 2, 3, 4], \"category\": \"Table\", \"text\": \"a | b\"}]",
    }) |text| try t.expectError(error.Unreadable, ocr.parse(a, text, .{ .width = 10, .height = 10 }, .{ .width = 10, .height = 10 }));
}

/// A PNG of a ``w`` x ``h`` RGB header, an empty IDAT and IEND: the size check runs before the image data is read.
fn pngHeader(a: std.mem.Allocator, w: u32, h: u32) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "\x89PNG\r\n\x1a\n");
    var head: [13]u8 = undefined;
    std.mem.writeInt(u32, head[0..4], w, .big);
    std.mem.writeInt(u32, head[4..8], h, .big);
    @memcpy(head[8..], &[_]u8{ 8, 2, 0, 0, 0 });
    for ([_]struct { []const u8, []const u8 }{ .{ "IHDR", &head }, .{ "IDAT", "" }, .{ "IEND", "" } }) |c| {
        try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToBig(u32, @intCast(c[1].len))));
        var crc = std.hash.Crc32.init();
        crc.update(c[0]);
        crc.update(c[1]);
        try out.appendSlice(a, c[0]);
        try out.appendSlice(a, c[1]);
        try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToBig(u32, crc.final())));
    }
    return out.items;
}

test "dots: the processor refuses a page past 16 megapixels and passes one at the limit to the decoder" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try image.Processor.parse(a, @embedFile("fixtures/preprocessor_config.json"), try cfg.Config.parse(a, cases.sample, cases.generation));
    // 4097 x 4096 is one row past 16 Mi pixels: refused from the header, the 400 docs/api.md names for dots.mocr
    try t.expectError(error.Oversized, p.prepare(t.io, a, try pngHeader(a, 4097, 4096), 16384));
    try t.expectError(error.Oversized, p.prepare(t.io, a, try pngHeader(a, 8193, 16), 16384));
    // exactly 4096 x 4096 passes the size check and fails only on the missing image data
    try t.expectError(error.Invalid, p.prepare(t.io, a, try pngHeader(a, 4096, 4096), 16384));
}

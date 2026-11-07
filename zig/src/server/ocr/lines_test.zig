//! Blocks to lines and words, with every derived quad validated (design 5.4).
const std = @import("std");
const api = @import("engine_api");
const types = @import("types.zig");
const lines = @import("lines.zig");
const g = @import("geometry.zig");
const t = std.testing;
const Quad = api.ocr.Quad;
const OcrBlock = api.ocr.OcrBlock;

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) Quad {
    return .{ .{ x0, y0 }, .{ x1, y0 }, .{ x1, y1 }, .{ x0, y1 } };
}

fn expectQuad(want: Quad, got: ?Quad) !void {
    const q = got orelse return error.TestExpectedEqual;
    for (want, q) |w, p| {
        try t.expectApproxEqAbs(w[0], p[0], 1e-4);
        try t.expectApproxEqAbs(w[1], p[1], 1e-4);
    }
}

/// Every quad passes validation unchanged against the page: simple, convex, 1 px² or more, inside the frame.
fn expectValid(q: ?Quad, w: u32, h: u32) !void {
    const v = q orelse return;
    try t.expect(g.isSimpleConvex(v));
    try t.expectEqual(v, try g.validate(v, w, h));
}

fn page(blocks: []OcrBlock, w: u32, h: u32) types.OcrPage {
    return .{ .index = 0, .width = w, .height = h, .dpi = 72, .blocks = blocks };
}

test "ocr lines: a block reader's lines split the block into equal bands, first line at the top edge" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var blocks = [_]OcrBlock{
        .{ .content = "# Title", .quad = rect(10, 0, 110, 12), .kind = .title },
        .{ .content = "", .quad = rect(0, 20, 50, 60), .kind = .picture },
        .{ .content = "one\n**two**\n\nthree", .quad = rect(0, 30, 100, 60), .kind = .text },
    };
    const p = page(&blocks, 200, 100);
    const got = try lines.fromBlocks(arena.allocator(), &p, .block);
    try t.expectEqual(@as(usize, 4), got.len);
    try t.expectEqualStrings("Title", got[0].text);
    try expectQuad(rect(10, 0, 110, 12), got[0].quad);
    try t.expectEqual(@as(u32, 0), got[0].block);
    const want = [_][]const u8{ "one", "two", "three" };
    for (got[1..], want, 0..) |l, text, k| {
        try t.expectEqualStrings(text, l.text);
        try t.expectEqual(@as(u32, 2), l.block);
        try t.expectEqual(@as(f32, 1.0), l.score);
        const y: f32 = 30 + 10 * @as(f32, @floatFromInt(k));
        try expectQuad(rect(0, y, 100, y + 10), l.quad);
    }
    for (got) |l| try expectValid(l.quad, 200, 100);
}

test "ocr lines: a rotated block splits along its TL→BL edge, in reading order" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    // text turned 90° clockwise: reading runs down, the first line is the rightmost column
    var blocks = [_]OcrBlock{.{ .content = "first\nsecond", .quad = .{ .{ 30, 0 }, .{ 30, 100 }, .{ 0, 100 }, .{ 0, 0 } }, .kind = .text }};
    const p = page(&blocks, 100, 100);
    const got = try lines.fromBlocks(arena.allocator(), &p, .block);
    try t.expectEqual(@as(usize, 2), got.len);
    try expectQuad(.{ .{ 30, 0 }, .{ 30, 100 }, .{ 15, 100 }, .{ 15, 0 } }, got[0].quad);
    try expectQuad(.{ .{ 15, 0 }, .{ 15, 100 }, .{ 0, 100 }, .{ 0, 0 } }, got[1].quad);
    for (got) |l| try expectValid(l.quad, 100, 100);
}

test "ocr lines: an f32 band that collapses is repaired to 1 px" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    // one f32 step tall at y 8191: exactly 1 px² of block, so either band can round to zero height
    var blocks = [_]OcrBlock{.{ .content = "a\nb", .quad = rect(0, 8191, 2048, 8191.00048828125), .kind = .text }};
    const p = page(&blocks, 2048, 8192);
    try t.expect(g.isSimpleConvex(blocks[0].quad.?));
    const got = try lines.fromBlocks(arena.allocator(), &p, .block);
    try t.expectEqual(@as(usize, 2), got.len);
    var repaired: usize = 0;
    for (got) |l| {
        try expectValid(l.quad, 2048, 8192);
        const b = g.minMax(l.quad.?);
        if (b[3] - b[1] == 1) repaired += 1;
        try t.expect(b[1] >= 8191 and b[3] <= 8192);
    }
    try t.expect(repaired >= 1); // the band that rounded to zero height
}

test "ocr lines: a geometry-less reader's lines keep a null quad" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var blocks = [_]OcrBlock{.{ .content = "# Page\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n$$\nx^2\n$$", .quad = null, .kind = .page }};
    const p = page(&blocks, 100, 100);
    const got = try lines.fromBlocks(arena.allocator(), &p, .none);
    const want = [_][]const u8{ "Page", "a\tb", "1\t2", "x^2" };
    try t.expectEqual(want.len, got.len);
    for (got, want) |l, text| {
        try t.expectEqualStrings(text, l.text);
        try t.expect(l.quad == null);
        try t.expectEqual(@as(u32, 0), l.block);
    }
}

test "ocr lines: a line reader's blocks pass through as one line each" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var blocks = [_]OcrBlock{
        .{ .content = "first *line*", .quad = rect(0, 0, 80, 10), .kind = .text },
        .{ .content = "  ", .quad = rect(0, 10, 80, 20), .kind = .text },
        .{ .content = "", .quad = rect(0, 20, 80, 60), .kind = .picture },
        .{ .content = "third", .quad = rect(0, 60, 300, 70), .kind = .text },
    };
    const p = page(&blocks, 100, 100);
    const got = try lines.fromBlocks(arena.allocator(), &p, .line);
    try t.expectEqual(@as(usize, 2), got.len);
    try t.expectEqualStrings("first line", got[0].text);
    try t.expectEqual(@as(u32, 0), got[0].block);
    try t.expectEqualStrings("third", got[1].text);
    try t.expectEqual(@as(u32, 3), got[1].block);
    try expectQuad(rect(0, 60, 100, 70), got[1].quad); // clamped to the page
}

test "ocr lines: words split the line by Unicode scalar count, not bytes" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const line: types.OcrLine = .{ .text = "héllo  wörld\tab", .quad = rect(0, 0, 120, 10), .block = 0 };
    const got = try lines.words(arena.allocator(), line, 200, 100);
    try t.expectEqual(@as(usize, 3), got.len);
    try t.expectEqualStrings("héllo", got[0].text);
    try t.expectEqualStrings("wörld", got[1].text);
    try t.expectEqualStrings("ab", got[2].text);
    try expectQuad(rect(0, 0, 50, 10), got[0].quad);
    try expectQuad(rect(50, 0, 100, 10), got[1].quad);
    try expectQuad(rect(100, 0, 120, 10), got[2].quad);
    for (got) |w| try expectValid(w.quad, 200, 100);
}

test "ocr lines: words follow the reading direction and keep a null quad" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    // upside-down text: TL is bottom right, reading runs right to left
    const turned: types.OcrLine = .{ .text = "ab cd", .quad = .{ .{ 100, 10 }, .{ 0, 10 }, .{ 0, 0 }, .{ 100, 0 } }, .block = 1 };
    const got = try lines.words(arena.allocator(), turned, 100, 100);
    try expectQuad(.{ .{ 100, 10 }, .{ 50, 10 }, .{ 50, 0 }, .{ 100, 0 } }, got[0].quad);
    try expectQuad(.{ .{ 50, 10 }, .{ 0, 10 }, .{ 0, 0 }, .{ 50, 0 } }, got[1].quad);
    const bare = try lines.words(arena.allocator(), .{ .text = "x y", .quad = null, .block = 0 }, 100, 100);
    try t.expectEqual(@as(usize, 2), bare.len);
    for (bare) |w| try t.expect(w.quad == null);
    // a one-pixel-wide line of many words: each word's sliver is widened to 1 px inside the page
    const thin = try lines.words(arena.allocator(), .{ .text = "a b c d", .quad = rect(99, 0, 100, 10), .block = 0 }, 100, 100);
    for (thin) |w| try expectValid(w.quad, 100, 100);
}

test "ocr lines: words split on Unicode whitespace as Python's str.split does" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const nbsp = try lines.words(a, .{ .text = "one\u{00A0}two", .quad = rect(0, 0, 60, 10), .block = 0 }, 100, 100);
    try t.expectEqual(@as(usize, 2), nbsp.len);
    try t.expectEqualStrings("one", nbsp[0].text);
    try t.expectEqualStrings("two", nbsp[1].text);
    try expectQuad(rect(0, 0, 30, 10), nbsp[0].quad);
    try expectQuad(rect(30, 0, 60, 10), nbsp[1].quad);
    const text = "a\u{3000}b\u{2009}c\x0bd\x0ce\u{85}f\u{202F}g\u{2028}h \xffi\x85j";
    const got = try lines.words(a, .{ .text = text, .quad = null, .block = 0 }, 100, 100);
    const want = [_][]const u8{ "a", "b", "c", "d", "e", "f", "g", "h", "\xffi\x85j" };
    try t.expectEqual(want.len, got.len);
    for (got, want) |w, s| try t.expectEqualStrings(s, w.text);
    // zero-width space (U+200B) is not whitespace to Python either
    try t.expectEqual(@as(usize, 1), (try lines.words(a, .{ .text = "a\u{200B}b", .quad = null, .block = 0 }, 100, 100)).len);
}

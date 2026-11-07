//! GLM-OCR's reader against Hugging Face oracle outputs, page Markdown normalization, and the prepared frame.
const std = @import("std");
const api = @import("engine_api");
const ocr = @import("ocr.zig");
const processor = @import("processor.zig");
const t = std.testing;

const Fixture = struct { text: []const u8, original: [2]u32, frame: [2]u32, source: []const u8, prompt_sha256: []const u8 };
const text_json = @embedFile("fixtures/ocr-text.json");
const table_json = @embedFile("fixtures/ocr-table.json");

fn fixture(a: std.mem.Allocator, text: []const u8) !Fixture {
    return std.json.parseFromSliceLeaky(Fixture, a, text, .{});
}

fn page(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    const blocks = try ocr.parse(a, text, .{ .width = 10, .height = 10 }, .{ .width = 28, .height = 28 });
    try t.expectEqual(@as(usize, 1), blocks.len);
    try t.expectEqual(api.ocr.Kind.page, blocks[0].kind);
    try t.expect(blocks[0].quad == null);
    return blocks[0].content;
}

test "GLM-OCR reader: the text oracle reads as one page block under the served prompt" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try fixture(a, text_json);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(ocr.prompt, &digest, .{});
    try t.expectEqualStrings(f.prompt_sha256, &std.fmt.bytesToHex(digest, .lower));
    try t.expectEqual(api.ocr.Geometry.none, ocr.reader.geometry);
    const content = try page(a, f.text);
    for ([_][]const u8{ "TensorFold OCR test", "Invoice 0042", "Total: 123.45 USD" }) |line|
        try t.expect(std.mem.indexOf(u8, content, line) != null);
}

test "GLM-OCR reader: the table oracle reads as a pipe table" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try fixture(a, table_json);
    try t.expectEqualStrings("| Item | Quantity |\n| --- | --- |\n| Apples | 3 |\n| Pears | 5 |", try page(a, f.text));
}

test "GLM-OCR reader: empty text is an empty page" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "", " \n\t " }) |text|
        try t.expectEqual(@as(usize, 0), (try ocr.parse(a, text, .{ .width = 1, .height = 1 }, .{ .width = 28, .height = 28 })).len);
}

test "GLM-OCR reader: tables inside the page and LaTeX delimiters become Markdown" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { []const u8, []const u8 };
    for ([_]Case{
        .{ "  plain text\n", "plain text" },
        .{ "Area \\(\\pi r^2\\) and\n\\[ E = mc^2 \\]", "Area $\\pi r^2$ and\n$$ E = mc^2 $$" },
        .{ "open \\( stays and \\[ too", "open \\( stays and \\[ too" },
        .{ "Before <table><tr><td>a</td><td>b</td></tr></table> after", "Before\n\n| a | b |\n| --- | --- |\n\nafter" },
        .{ "<table><tr><td colspan=\"2\">ab</td></tr><tr><td>1</td><td>2</td></tr></table>\n\n<table><tr><td>x</td></tr></table>", "<table><tr><td colspan=\"2\">ab</td></tr><tr><td>1</td><td>2</td></tr></table>\n\n| x |\n| --- |" },
        .{ "<table><tr><td>\\(x\\)</td></tr></table>", "| $x$ |\n| --- |" },
        .{ "<table><tr><td>\\[ y \\]</td><td>b</td></tr></table>", "| $$ y $$ | b |\n| --- | --- |" },
        .{ "kept <table><td>a</td></table> as written", "kept <table><td>a</td></table> as written" },
        .{ "cut <table><tr><td>a", "cut <table><tr><td>a" },
    }) |k| try t.expectEqualStrings(k[1], try page(a, k[0]));
}

fn repeat(a: std.mem.Allocator, part: []const u8, n: usize) ![]u8 {
    const text = try a.alloc(u8, part.len * n);
    for (0..n) |k| @memcpy(text[k * part.len ..][0..part.len], part);
    return text;
}

test "GLM-OCR reader: many unclosed math openers stay as written" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "\\(", "\\[" }) |opener| {
        const text = try repeat(a, opener, 32768);
        try t.expectEqualStrings(text, try page(a, text));
    }
    // A closer after many unclosed openers of the other kind still converts.
    const openers = try repeat(a, "\\[", 64);
    const mixed = try std.mem.concat(a, u8, &.{ openers, " \\(x\\)" });
    try t.expectEqualStrings(try std.mem.concat(a, u8, &.{ openers, " $x$" }), try page(a, mixed));
}

test "GLM-OCR reader: code blocks and code spans stay verbatim" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const table = "<table><tr><td>a</td></tr></table>";
    const Case = struct { []const u8, []const u8 };
    for ([_]Case{
        .{ "```python\nprint(r\"\\(x\\)\")\n" ++ table ++ "\n```\nafter \\(y\\)", "```python\nprint(r\"\\(x\\)\")\n" ++ table ++ "\n```\nafter $y$" },
        .{ "~~~~\n\\[ z \\]\n~~~\nstill code " ++ table, "~~~~\n\\[ z \\]\n~~~\nstill code " ++ table },
        .{ "use `\\(x\\)` and ``" ++ table ++ "`` here \\(y\\)", "use `\\(x\\)` and ``" ++ table ++ "`` here $y$" },
        .{ "`\\(a\\)` then ` lone \\(z\\)", "`\\(a\\)` then ` lone $z$" },
        .{ "a `` b ` c `` \\(x\\)", "a `` b ` c `` $x$" },
        .{ "x ``` inline ``` and ``` open \\(x\\)", "x ``` inline ``` and ``` open $x$" },
        .{ "text\n``` fence to the end \\(x\\)", "text\n``` fence to the end \\(x\\)" },
        .{ "intro\n  ~~~python\nprint(r\"\\(x\\)\")\n  ~~~", "intro\n  ~~~python\nprint(r\"\\(x\\)\")\n  ~~~" },
        .{ "```\n\\(x\\)\n   ```\nafter \\(y\\)", "```\n\\(x\\)\n   ```\nafter $y$" },
        .{ "x\n    ```\n\\(x\\)", "x\n    ```\n$x$" },
        .{ "lead `\n\n~~~text\n`\n~~~\nafter \\(y\\)", "lead `\n\n~~~text\n`\n~~~\nafter $y$" },
        .{ "a `b\n\nc` \\(x\\)", "a `b\n\nc` $x$" },
        .{ "<table><tr><td>`x`</td><td>\\(y\\)</td></tr></table>", "| `x` | $y$ |\n| --- | --- |" },
        .{ "<table><tr><td>`\\(x\\)`</td></tr></table>", "| `\\(x\\)` |\n| --- |" },
        .{ "bad <table><td>`\\(x\\)` \\(y\\)</td></table>", "bad <table><td>`\\(x\\)` $y$</td></table>" },
        .{ "<table> `\\(x\\)` and <table> again", "<table> `\\(x\\)` and <table> again" },
    }) |k| try t.expectEqualStrings(k[1], try page(a, k[0]));
}

test "GLM-OCR reader: prepare reports the processor's resized size as the frame" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ text_json, table_json }) |json| {
        const f = try fixture(a, json);
        const rgb = try a.alloc(u8, @as(usize, f.original[0]) * f.original[1] * 3);
        @memset(rgb, 255);
        const p = try processor.prepareRgb(a, .{ .width = f.original[0], .height = f.original[1], .pixels = rgb }, 6144);
        const prepared = ocr.prepared(p);
        try t.expectEqual(api.Size{ .width = f.frame[0], .height = f.frame[1] }, prepared.frame.?);
        try t.expectEqual(p.geometry.w * p.geometry.h, @as(usize, prepared.frame.?.width) * prepared.frame.?.height);
        try t.expectEqual(prepared.gw * 14, prepared.frame.?.width);
    }
}

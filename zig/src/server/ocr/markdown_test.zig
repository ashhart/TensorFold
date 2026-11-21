//! The table finder and the pipe-table reader over the reader Markdown subset (docs/api.md OCR: Reader Markdown).
const std = @import("std");
const md = @import("markdown.zig");
const t = std.testing;

/// ``cell`` as the one body cell of a pipe table, read back: its text after the inline clean-up.
fn cellBack(a: std.mem.Allocator, cell: []const u8) ![]const u8 {
    const table = try std.mem.concat(a, u8, &.{ "| h |\n| --- |\n| ", cell, " |" });
    const back = try md.pipeTable(a, table);
    try t.expectEqual(@as(u32, 2), back.rows);
    return back.cells[1].text;
}

test "ocr markdown: tables are found as byte ranges, none inside code" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const pipe_table = "| A | B |\n|---|---|\n| 1 | 2 |";
    const html_table = "<table><tr><td>x</td></tr></table>";
    const content = "Intro\n\n  " ++ pipe_table ++ "\n\nMiddle\n" ++ html_table ++ "\n```\n| c | d |\n|---|---|\n```\n$$\n|x|\n|-|\n$$\nEnd";
    const found = try md.tables(arena.allocator(), content);
    try t.expectEqual(@as(usize, 2), found.len);
    try t.expectEqualStrings(pipe_table, content[found[0].start..found[0].end]);
    try t.expect(!found[0].html);
    try t.expectEqualStrings(html_table, content[found[1].start..found[1].end]);
    try t.expect(found[1].html);
    try t.expectEqual(@as(usize, 0), (try md.tables(arena.allocator(), "no tables | here\njust text")).len);
}

test "ocr markdown: a pipe cell loses emphasis, code ticks and link syntax, keeping literal stars and underscores" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { []const u8, []const u8 };
    for ([_]Case{
        .{ "**bold** and *it* and _u_ and __s__ and ***both***", "bold and it and u and s and both" },
        .{ "snake_case_name stays", "snake_case_name stays" },
        .{ "5 * 3 = 15", "5 * 3 = 15" },
        .{ "an *unclosed star", "an *unclosed star" },
        .{ "a **nested *inner* run** ends", "a nested inner run ends" },
        .{ "escaped \\*not emphasis\\*", "escaped *not emphasis*" },
        .{ "use `a*b_c` here", "use a*b_c here" },
        .{ "double `` a ` b `` ticks", "double a ` b ticks" },
        .{ "an `unclosed tick", "an `unclosed tick" },
        .{ "see [the *docs*](http://x/y_(z)) now", "see the docs now" },
        .{ "![alt text](img.png)", "alt text" },
    }) |c| try t.expectEqualStrings(c[1], try cellBack(a, c[0]));
}

test "ocr markdown: many unclosed dollars and tables stay literal in one pass" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var line: std.ArrayList(u8) = .empty;
    for (0..40_000) |_| try line.appendSlice(a, "$x <b ");
    try t.expectEqualStrings(std.mem.trim(u8, line.items, " "), try cellBack(a, line.items));
    var tables: std.ArrayList(u8) = .empty;
    for (0..40_000) |_| try tables.appendSlice(a, "<table>\n");
    try t.expectEqual(@as(usize, 0), (try md.tables(a, tables.items)).len);
}

test "ocr markdown: many unmatched brackets and ampersands stay literal in one pass" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // 100k `[`, then `[a](` with a `(` that never closes, then `&` with no `;` near it
    const units = [_][]const u8{ "[", "[a](", "&", "&amp x" };
    for (units) |unit| {
        var cell: std.ArrayList(u8) = .empty;
        for (0..100_000) |_| try cell.appendSlice(a, unit);
        try t.expectEqualStrings(std.mem.trim(u8, cell.items, " "), try cellBack(a, cell.items));
    }
    // links still pair across nesting and escapes after the one-pass change
    try t.expectEqualStrings("a b d [x](y) p", try cellBack(a, "[a [b](c) d](e) \\[x](y) [p](q(r)s)"));
    try t.expectEqualStrings("\u{1F600} &< &abcdefghij;", try cellBack(a, "&#x1F600; &amp;&lt; &abcdefghij;"));
}

test "ocr markdown: table cell text survives the pipe form, read back as HTML" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const html = @import("engine_api").ocr.html;
    const texts = [_][]const u8{ "[x](y)", "a\\|b", "*not bold*", "`tick`", "back\\slash", "snake_case _x_", "$x_{1}|y$ and $$a*b$$", "costs $5", "a<b &lt;" };
    var cells: [texts.len]html.Cell = undefined;
    for (&cells, texts, 0..) |*c, text, k| c.* = .{ .text = text, .row = 0, .col = @intCast(k), .rows = 1, .cols = 1, .head = true };
    const table: html.Table = .{ .rows = 1, .cols = texts.len, .cells = &cells, .spanned = false };
    const md_table = try html.pipe(a, table);
    // Mistral's html form: the pipe table read back gives every cell's text, math included, and the same clean HTML
    const back = try md.pipeTable(a, md_table);
    for (back.cells, texts) |c, text| try t.expectEqualStrings(text, c.text);
    try t.expectEqualStrings(try html.normalized(a, table), try html.normalized(a, back));
}

test "ocr markdown: backslash runs before a pipe keep the columns and the cell text, in math and out" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const html = @import("engine_api").ocr.html;
    const Case = struct { text: []const u8 };
    const cases = [_]Case{
        .{ .text = "$a|b$" },
        .{ .text = "$a\\|b$" },
        .{ .text = "$a\\\\|b$" },
        .{ .text = "$a\\\\\\|b$" },
        .{ .text = "$$a\\|b\\\\|c$$" },
        .{ .text = "a\\|b" },
        .{ .text = "a\\\\|b" },
        .{ .text = "a\\\\\\|b" },
    };
    for (cases) |c| {
        var cells = [_]html.Cell{
            .{ .text = "h", .row = 0, .col = 0, .rows = 1, .cols = 1, .head = true },
            .{ .text = "k", .row = 0, .col = 1, .rows = 1, .cols = 1, .head = true },
            .{ .text = c.text, .row = 1, .col = 0, .rows = 1, .cols = 1, .head = false },
            .{ .text = "z", .row = 1, .col = 1, .rows = 1, .cols = 1, .head = false },
        };
        const table: html.Table = .{ .rows = 2, .cols = 2, .cells = &cells, .spanned = false };
        const md_table = try html.pipe(a, table);
        const back = try md.pipeTable(a, md_table);
        try t.expectEqual(@as(u32, 2), back.cols);
        try t.expectEqualStrings(c.text, back.cells[2].text);
        try t.expectEqualStrings("z", back.cells[3].text);
        try t.expectEqualStrings(try html.normalized(a, table), try html.normalized(a, back));
    }
}

test "ocr markdown: a raw pipe table reads as GFM renders it: a pipe right after a backslash is escaped" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { cell: []const u8, text: []const u8 };
    const cases = [_]Case{
        // GLM writes `$a\\\|b$`: GFM's table step removes the one backslash before `|`, leaving `$a\\|b$`
        .{ .cell = "$a\\\\\\|b$", .text = "$a\\\\|b$" },
        // `$\|x\|$` stays one cell; the table step leaves `$|x|$`
        .{ .cell = "$\\|x\\|$", .text = "$|x|$" },
    };
    for (cases) |c| {
        const table = try std.mem.concat(a, u8, &.{ "| h | k |\n| --- | --- |\n| ", c.cell, " | z |" });
        const back = try md.pipeTable(a, table);
        try t.expectEqual(@as(u32, 2), back.cols);
        try t.expectEqualStrings(c.text, back.cells[2].text);
        try t.expectEqualStrings("z", back.cells[3].text);
    }
}

//! Markdown to plain-text lines over the reader Markdown subset, and the table finder (docs/api.md OCR: Estimates).
const std = @import("std");
const md = @import("markdown.zig");
const t = std.testing;

fn expectLines(content: []const u8, want: []const []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const got = try md.plainLines(arena.allocator(), content);
    if (got.len != want.len) {
        std.debug.print("want {d} lines, got {d}:\n", .{ want.len, got.len });
        for (got) |l| std.debug.print("  [{s}]\n", .{l});
        return error.TestExpectedEqual;
    }
    for (want, got) |w, g| try t.expectEqualStrings(w, g);
}

test "ocr markdown: heading markers are removed" {
    try expectLines("# Title", &.{"Title"});
    try expectLines("## Section two ##", &.{"Section two"});
    try expectLines("###### Six", &.{"Six"});
    try expectLines("####### seven is text", &.{"####### seven is text"});
    try expectLines("#hashtag and C#", &.{"#hashtag and C#"});
    try expectLines("## **Bold** heading", &.{"Bold heading"});
}

test "ocr markdown: emphasis markers are removed, literal stars and underscores kept" {
    try expectLines("**bold** and *it* and _u_ and __s__ and ***both***", &.{"bold and it and u and s and both"});
    try expectLines("snake_case_name stays", &.{"snake_case_name stays"});
    try expectLines("5 * 3 = 15", &.{"5 * 3 = 15"});
    try expectLines("an *unclosed star", &.{"an *unclosed star"});
    try expectLines("a **nested *inner* run** ends", &.{"a nested inner run ends"});
    try expectLines("escaped \\*not emphasis\\*", &.{"escaped *not emphasis*"});
}

test "ocr markdown: inline code keeps its text without ticks" {
    try expectLines("use `a*b_c` here", &.{"use a*b_c here"});
    try expectLines("double `` a ` b `` ticks", &.{"double a ` b ticks"});
    try expectLines("an `unclosed tick", &.{"an `unclosed tick"});
}

test "ocr markdown: links and images keep their text" {
    try expectLines("see [the *docs*](http://x/y_(z)) now", &.{"see the docs now"});
    try expectLines("![alt text](img.png)", &.{"alt text"});
    try expectLines("a [1] reference and [x] (y) stay", &.{"a [1] reference and [x] (y) stay"});
    try expectLines("a &lt;b&gt; &amp; &#233;&#x20AC;", &.{"a <b> & é€"});
}

test "ocr markdown: list markers are kept" {
    try expectLines("- item **one**\n* star\n+ plus\n1. first\n12) twelfth", &.{ "- item one", "* star", "+ plus", "1. first", "12) twelfth" });
    try expectLines("  - indented", &.{"- indented"});
}

test "ocr markdown: fenced code keeps its lines verbatim" {
    try expectLines("before\n```zig\nconst x = *y_z*;\n\n  indented `tick`\n```\nafter", &.{ "before", "const x = *y_z*;", "  indented `tick`", "after" });
    try expectLines("~~~~\n| not | a |\n|---|---|\n~~~~", &.{ "| not | a |", "|---|---|" });
    try expectLines("```inline code``` stays text", &.{"inline code stays text"});
}

test "ocr markdown: math keeps its LaTeX without dollars" {
    try expectLines("$$\nE = mc^2\n\\frac{a}{b}\n$$", &.{ "E = mc^2", "\\frac{a}{b}" });
    try expectLines("$$x^2 + y_1$$", &.{"x^2 + y_1"});
    try expectLines("$$ a_1\nb_2 $$", &.{ "a_1", "b_2" });
    try expectLines("area $\\pi r^2$ and $$\\sum_i x_i$$ here", &.{"area \\pi r^2 and \\sum_i x_i here"});
    try expectLines("costs $5 and $10 today", &.{"costs $5 and $10 today"});
}

test "ocr markdown: a pipe table is one line a row, cells joined by tabs" {
    try expectLines("| A | B |\n| --- | :-: |\n| 1 | **2** |\n|  | x \\| y |", &.{ "A\tB", "1\t2", "\tx | y" });
    try expectLines("Name | Qty\n-|-\napple | 3\n\nafter", &.{ "Name\tQty", "apple\t3", "after" });
    try expectLines("| a &lt;b | c |\n| --- | --- |", &.{"a <b\tc"});
    // a delimiter row with the wrong cell count is not a table
    try expectLines("| a | b |\n| --- |", &.{ "| a | b |", "| --- |" });
}

test "ocr markdown: an HTML table is one line a row, a spanned cell once" {
    try expectLines(
        "<table><tr><th colspan=\"2\">Head</th></tr>\n<tr><td>a</td><td>b &amp; c</td></tr></table>",
        &.{ "Head\t", "a\tb & c" },
    );
    try expectLines("<TABLE>\n<tr><td rowspan=\"2\">x</td><td>1</td></tr><tr><td>2</td></tr>\n</TABLE> tail", &.{ "x\t1", "\t2", "tail" });
}

test "ocr markdown: lines empty after trimming are dropped" {
    try expectLines("a\n\n   \n\t\nb\r\n  \t  \n", &.{ "a", "b" });
    try expectLines("", &.{});
    try expectLines("# \n- ", &.{"-"});
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

test "ocr markdown: an HTML table that does not parse is read as text" {
    try expectLines("<table>\n</table>\nafter", &.{ "<table>", "</table>", "after" });
}

test "ocr markdown: many unclosed dollars and tables stay literal in one pass" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var line: std.ArrayList(u8) = .empty;
    for (0..40_000) |_| try line.appendSlice(a, "$x <b ");
    const got = try md.plainLines(a, line.items);
    try t.expectEqual(@as(usize, 1), got.len);
    try t.expectEqualStrings(std.mem.trim(u8, line.items, " "), got[0]);
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
    var content: std.ArrayList(u8) = .empty;
    for (units) |unit| {
        for (0..100_000) |_| try content.appendSlice(a, unit);
        try content.append(a, '\n');
    }
    const got = try md.plainLines(a, content.items);
    try t.expectEqual(units.len, got.len);
    var it = std.mem.splitScalar(u8, content.items, '\n');
    for (got) |l| try t.expectEqualStrings(std.mem.trim(u8, it.next().?, " "), l);
    // links still pair across nesting and escapes after the one-pass change
    try expectLines("[a [b](c) d](e) \\[x](y) [p](q(r)s)", &.{"a b d [x](y) p"});
    try expectLines("&#x1F600; &amp;&lt; &abcdefghij;", &.{"\u{1F600} &< &abcdefghij;"});
}

test "ocr markdown: table cell text survives the pipe form, as lines and read back as HTML" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const html = @import("engine_api").ocr.html;
    const texts = [_][]const u8{ "[x](y)", "a\\|b", "*not bold*", "`tick`", "back\\slash", "snake_case _x_", "$x_{1}|y$ and $$a*b$$", "costs $5", "a<b &lt;" };
    var cells: [texts.len]html.Cell = undefined;
    for (&cells, texts, 0..) |*c, text, k| c.* = .{ .text = text, .row = 0, .col = @intCast(k), .rows = 1, .cols = 1, .head = true };
    const table: html.Table = .{ .rows = 1, .cols = texts.len, .cells = &cells, .spanned = false };
    const md_table = try html.pipe(a, table);
    // the line dialects: each cell's literal text, math without its dollars (docs/api.md OCR: Estimates)
    const lines = try md.plainLines(a, md_table);
    try t.expectEqual(@as(usize, 1), lines.len);
    try t.expectEqualStrings("[x](y)\ta\\|b\t*not bold*\t`tick`\tback\\slash\tsnake_case _x_\tx_{1}|y and a*b\tcosts $5\ta<b &lt;", lines[0]);
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
    const Case = struct { text: []const u8, line: []const u8 };
    const cases = [_]Case{
        .{ .text = "$a|b$", .line = "a|b" },
        .{ .text = "$a\\|b$", .line = "a\\|b" },
        .{ .text = "$a\\\\|b$", .line = "a\\\\|b" },
        .{ .text = "$a\\\\\\|b$", .line = "a\\\\\\|b" },
        .{ .text = "$$a\\|b\\\\|c$$", .line = "a\\|b\\\\|c" },
        .{ .text = "a\\|b", .line = "a\\|b" },
        .{ .text = "a\\\\|b", .line = "a\\\\|b" },
        .{ .text = "a\\\\\\|b", .line = "a\\\\\\|b" },
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
        const lines = try md.plainLines(a, md_table);
        try t.expectEqual(@as(usize, 2), lines.len);
        // two columns, the first holding the cell's text (math without its dollars)
        try t.expectEqualStrings(try std.mem.concat(a, u8, &.{ c.line, "\tz" }), lines[1]);
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
    const Case = struct { cell: []const u8, line: []const u8, text: []const u8 };
    const cases = [_]Case{
        // GLM writes `$a\\\|b$`: GFM's table step removes the one backslash before `|`, leaving `$a\\|b$`
        .{ .cell = "$a\\\\\\|b$", .line = "a\\\\|b", .text = "$a\\\\|b$" },
        // `$\|x\|$` stays one cell; the table step leaves `$|x|$`
        .{ .cell = "$\\|x\\|$", .line = "|x|", .text = "$|x|$" },
    };
    for (cases) |c| {
        const table = try std.mem.concat(a, u8, &.{ "| h | k |\n| --- | --- |\n| ", c.cell, " | z |" });
        const lines = try md.plainLines(a, table);
        try t.expectEqual(@as(usize, 2), lines.len);
        try t.expectEqualStrings(try std.mem.concat(a, u8, &.{ c.line, "\tz" }), lines[1]);
        const back = try md.pipeTable(a, table);
        try t.expectEqual(@as(u32, 2), back.cols);
        try t.expectEqualStrings(c.text, back.cells[2].text);
        try t.expectEqualStrings("z", back.cells[3].text);
    }
}

test "ocr markdown: a spanned HTML table and a pipe table with the same cells give the same lines, math without dollars" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const html = @import("engine_api").ocr.html;
    const texts = [_][]const u8{ "$x_1$", "$$a$$", "costs $5", "a\\*b", "$p$ | $q" };
    var cells: [texts.len]html.Cell = undefined;
    for (&cells, texts, 0..) |*c, text, k| c.* = .{ .text = text, .row = 1, .col = @intCast(k), .rows = 1, .cols = 1, .head = false };
    var head = [_]html.Cell{.{ .text = "Head", .row = 0, .col = 0, .rows = 1, .cols = texts.len, .head = true }};
    const all = try std.mem.concat(a, html.Cell, &.{ &head, &cells });
    const spanned: html.Table = .{ .rows = 2, .cols = texts.len, .cells = all, .spanned = true };
    // the same cells with the header unspanned, as the pipe form writes them
    var plain_head = [_]html.Cell{.{ .text = "Head", .row = 0, .col = 0, .rows = 1, .cols = 1, .head = true }};
    const plain: html.Table = .{ .rows = 2, .cols = texts.len, .cells = try std.mem.concat(a, html.Cell, &.{ &plain_head, &cells }), .spanned = false };
    const from_html = try md.plainLines(a, try html.normalized(a, spanned));
    const from_pipe = try md.plainLines(a, try html.pipe(a, plain));
    try t.expectEqual(@as(usize, 2), from_html.len);
    try t.expectEqual(from_pipe.len, from_html.len);
    for (from_pipe, from_html) |p, h| try t.expectEqualStrings(p, h);
    try t.expectEqualStrings("x_1\ta\tcosts $5\ta\\*b\tp | $q", from_html[1]);
}

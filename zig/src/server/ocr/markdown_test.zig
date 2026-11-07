//! Markdown to plain-text lines over the 5.1 subset, and the table finder (design 5.3).
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

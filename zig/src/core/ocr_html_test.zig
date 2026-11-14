//! ocr_html.zig against tables dots.mocr and GLM-OCR wrote, and the subset's edges.
const std = @import("std");
const html = @import("ocr_html.zig");
const t = std.testing;

// dots.mocr, TABLE II of the layout oracle (run008), first body rows kept.
const dots_table = "<table><thead><tr><th rowspan=\"2\"></th><th rowspan=\"2\">Controls</th>" ++
    "<th colspan=\"3\">Hodgkin lymphoma</th><th colspan=\"3\">Non-Hodgkin lymphoma</th></tr>" ++
    "<tr><th>Cases</th><th>OR</th><th>95% CI</th><th>Cases</th><th>OR</th><th>95% CI</th></tr></thead><tbody>" ++
    "<tr><td>Never Exposed (reference group)</td><td>2,273</td><td>315</td><td>1.00</td><td>—</td>" ++
    "<td>1,823</td><td>1.00</td><td>—</td></tr>" ++
    "<tr><td>>1 year</td><td>90</td><td>7</td><td>0.84</td><td>0.37-1.91</td><td>86</td><td>1.17</td>" ++
    "<td>0.87-1.59</td></tr></tbody></table>";

// GLM-OCR, oracle case table-007: no header row.
const glm_table = "<table border=\"1\"><tr><td>Item</td><td>Quantity</td></tr><tr><td>Apples</td><td>3</td></tr>" ++
    "<tr><td>Pears</td><td>5</td></tr></table>";

fn expectCell(got: html.Cell, text: []const u8, row: u32, col: u32, rows: u32, cols: u32, head: bool) !void {
    const want: html.Cell = .{ .text = text, .row = row, .col = col, .rows = rows, .cols = cols, .head = head };
    try t.expectEqualDeep(want, got);
}

test "ocr html: the dots.mocr table keeps its spans" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const table = try html.parse(a, dots_table);
    try t.expectEqual(@as(u32, 4), table.rows);
    try t.expectEqual(@as(u32, 8), table.cols);
    try t.expect(table.spanned);
    try expectCell(table.cells[0], "", 0, 0, 2, 1, true);
    const nhl = html.Cell{ .text = "Non-Hodgkin lymphoma", .row = 0, .col = 5, .rows = 1, .cols = 3, .head = true };
    try t.expectEqualDeep(nhl, table.cells[3]);
    try expectCell(table.cells[4], "Cases", 1, 2, 1, 1, true);
    try t.expect(!table.cells[10].head);
}

test "ocr html: the GLM-OCR table has no header row and becomes a pipe table with its first row the header" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const table = try html.parse(a, glm_table);
    try t.expect(!table.spanned);
    for (table.cells) |c| try t.expect(!c.head);
    const want = "| Item | Quantity |\n| --- | --- |\n| Apples | 3 |\n| Pears | 5 |";
    try t.expectEqualStrings(want, try html.pipe(a, table));
    try t.expectEqualStrings("<table><tr><td>Item</td><td>Quantity</td></tr><tr><td>Apples</td><td>3</td></tr>" ++
        "<tr><td>Pears</td><td>5</td></tr></table>", try html.normalized(a, table));
}

test "ocr html: entities decode and unknown ones stay as written" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = "<table><tr><td>a &amp; b &lt;c&gt; &quot;q&quot; &#65;&#x42;&#X43; x&nbsp;y&#160;z" ++
        " &bogus; & &#0; &#xD800; &#x110000; &#12345678; &amp</td></tr></table>";
    const table = try html.parse(a, src);
    const want = "a & b <c> \"q\" ABC x y z &bogus; & \u{fffd} \u{fffd} \u{fffd} &#12345678; &amp";
    try t.expectEqualStrings(want, table.cells[0].text);
}

test "ocr html: other tags drop and keep their text; names and attributes in any case and quoting" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = "\n <!-- layout --><TABLE class=x><THEAD><TR><TH class='a>b' COLSPAN=2>" ++
        "<b>Bold</b> and <i>it<sup>2</sup></i><span class=\"s\">!</span></TH></TR></THEAD>" ++
        "<tr><td RowSpan = '1'>x</td><td colspan=2/>y</td></tr></TABLE>\n";
    const table = try html.parse(a, src);
    try t.expectEqual(@as(u32, 3), table.cols);
    try expectCell(table.cells[0], "Bold and it2!", 0, 0, 1, 2, true);
    try expectCell(table.cells[2], "y", 1, 1, 1, 2, false);
}

test "ocr html: <br> is a line break in the cell and a space in pipe cells" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = "<table><tr><td><br>line  one<br>line two <br/>\n three <BR /> </td><td>b</td></tr></table>";
    const table = try html.parse(a, src);
    try t.expectEqualStrings("line one\nline two\nthree", table.cells[0].text);
    try t.expectEqualStrings("| line one line two three | b |\n| --- | --- |", try html.pipe(a, table));
    const clean = "<table><tr><td>line one<br>line two<br>three</td><td>b</td></tr></table>";
    try t.expectEqualStrings(clean, try html.normalized(a, table));
}

test "ocr html: pipe cells escape |" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const table = try html.parse(a, "<table><tr><th>a|b</th><th></th></tr><tr><td>|</td><td>c</td></tr></table>");
    try t.expectEqualStrings("| a\\|b |  |\n| --- | --- |\n| \\| | c |", try html.pipe(a, table));
}

test "ocr html: pipe cells escape a literal < that would start a tag, and keep other < as written" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const table = try html.parse(a, "<table><tr><td>&lt;em&gt;x&lt;/em&gt;</td><td>a &lt; b</td></tr></table>");
    try t.expectEqualStrings("| &lt;em>x&lt;/em> | a < b |\n| --- | --- |", try html.pipe(a, table));
}

test "ocr html: pipe cells escape a literal & that would start an entity, and keep other & as written" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const table = try html.parse(a, "<table><tr><td>&amp;lt; &amp;#38; &amp;#x26;</td><td>AT&amp;T &amp; b; &amp;;</td></tr></table>");
    try t.expectEqualStrings("| &amp;lt; &amp;#38; &amp;#x26; | AT&T & b; &; |\n| --- | --- |", try html.pipe(a, table));
}

test "ocr html: an unknown tag whose name starts like a table tag is dropped" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const table = try html.parse(a, "<table><tr><td><table-note>x</table-note><tr-x/>y</td></tr></table>");
    try t.expectEqual(@as(usize, 1), table.cells.len);
    try t.expectEqualStrings("xy", table.cells[0].text);
}

test "ocr html: malformed tables and tables past the limits" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bad = [_][]const u8{
        "<table><tr><td>a</td></tr>", // unclosed table
        "<table><tr><td a=\"x></td></tr></table>", // unclosed quote
        "<table><tr><td>a</td></tr></table", // unclosed tag
        "<table><tr><td>a<!-- x</td></tr></table>", // unclosed comment
        "<table><tr><td><table><tr><td>a</td></tr></table></td></tr></table>", // nested
        "<table><td>a</td></table>", // a cell outside a row
        "<table>stray<tr><td>a</td></tr></table>", // text outside a cell
        "<p>x</p><table><tr><td>a</td></tr></table>", // markup before the table
        "<table><tr><td>a</td></tr></table>tail", // text after it
        "<table><tr><td>a</td></tr></table><table><tr><td>b</td></tr></table>", // two tables
        "<table></table>", // no cells
        "<table><tr></tr></table>",
        "",
    };
    for (bad) |src| try t.expectError(error.Malformed, html.parse(a, src));

    var wide: std.ArrayList(u8) = .empty;
    try wide.appendSlice(a, "<table><tr>");
    for (0..html.max_cols) |_| try wide.appendSlice(a, "<td>x</td>");
    const full_width = try std.mem.concat(a, u8, &.{ wide.items, "</tr></table>" });
    try t.expectEqual(@as(u32, html.max_cols), (try html.parse(a, full_width)).cols);
    try wide.appendSlice(a, "<td>x</td></tr></table>");
    try t.expectError(error.Malformed, html.parse(a, wide.items));

    var tall: std.ArrayList(u8) = .empty;
    try tall.appendSlice(a, "<table>");
    for (0..html.max_rows) |_| try tall.appendSlice(a, "<tr><td>x</td></tr>");
    const full_height = try std.mem.concat(a, u8, &.{ tall.items, "</table>" });
    try t.expectEqual(@as(u32, html.max_rows), (try html.parse(a, full_height)).rows);
    try tall.appendSlice(a, "<tr><td>x</td></tr></table>");
    try t.expectError(error.Malformed, html.parse(a, tall.items));

    const big = try a.alloc(u8, html.max_bytes + 1);
    @memset(big, ' ');
    @memcpy(big[0..7], "<table>");
    const tail = "<tr><td>a</td></tr></table>";
    @memcpy(big[html.max_bytes - tail.len ..][0..tail.len], tail);
    try t.expectEqual(@as(u32, 1), (try html.parse(a, big[0..html.max_bytes])).rows);
    try t.expectError(error.Malformed, html.parse(a, big));
}

test "ocr html: spans clip to the grid and to columns earlier rows cover" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = "<table><tr><td>A</td><td rowspan=\"2\">B</td></tr><tr><td colspan=\"3\">C</td></tr>" ++
        "<tr><td rowspan=\"9\" colspan=\"1000\">D</td></tr></table>";
    const table = try html.parse(a, src);
    try t.expectEqual(@as(u32, html.max_cols), table.cols);
    try expectCell(table.cells[2], "C", 1, 0, 1, 1, false);
    const d = html.Cell{ .text = "D", .row = 2, .col = 0, .rows = 1, .cols = html.max_cols, .head = false };
    try t.expectEqualDeep(d, table.cells[3]);
    const lone = try html.parse(a, "<table><tr><td rowspan=\"4\" colspan=0>x</td></tr></table>");
    try t.expect(!lone.spanned and lone.rows == 1 and lone.cols == 1);
}

test "ocr html: pipe rows place a spanned cell's text once and keep trailing cells" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = "<table><tr><td rowspan=\"2\">A</td><td colspan=\"2\">B</td></tr><tr><td>c</td><td>d</td></tr>" ++
        "<tr><td>e</td></tr></table>";
    const table = try html.parse(a, src);
    try t.expectEqual(@as(u32, 3), table.cols);
    try t.expectEqualStrings("| A | B |  |\n| --- | --- | --- |\n|  | c | d |\n| e |  |  |", try html.pipe(a, table));
}

test "ocr html: normalized HTML reads back as the same table" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const odd = "<table><tr><td rowspan=3>a&amp;b<br>c &lt;d&gt; &quot;</td><th colspan=2>h</th></tr><tr></tr>" ++
        "<tr><td>x</td></tr></table>";
    for ([_][]const u8{ dots_table, glm_table, odd }) |src| {
        const table = try html.parse(a, src);
        const clean = try html.normalized(a, table);
        try t.expectEqualDeep(table, try html.parse(a, clean));
        try t.expectEqualStrings(clean, try html.normalized(a, try html.parse(a, clean)));
    }
    const table = try html.parse(a, odd);
    const want = "<table><tr><td rowspan=\"3\">a&amp;b<br>c &lt;d&gt; &quot;</td>" ++
        "<th colspan=\"2\">h</th></tr><tr></tr><tr><td>x</td></tr></table>";
    try t.expectEqualStrings(want, try html.normalized(a, table));
}

fn everyWriter(a: std.mem.Allocator, src: []const u8) !void {
    const table = try html.parse(a, src);
    defer table.deinit(a);
    a.free(try html.pipe(a, table));
    a.free(try html.normalized(a, table));
}

test "ocr html: every allocation failure is returned and nothing leaks" {
    try t.checkAllAllocationFailures(t.allocator, everyWriter, .{dots_table});
    try t.expectError(error.Malformed, everyWriter(t.allocator, "<table><tr><td>a</td><td>b</td></tr><tr><td>c"));
}

test "ocr html: pipe cells escape Markdown markup to read back literally, and keep math as written" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { []const u8, []const u8 };
    const cases = [_]Case{
        .{ "[x](y)", "\\[x\\](y)" },
        .{ "a\\|b", "a\\\\\\|b" },
        .{ "*not bold*", "\\*not bold\\*" },
        .{ "`tick`", "\\`tick\\`" },
        .{ "back\\slash", "back\\slash" },
        .{ "snake_case _x_", "snake_case \\_x\\_" },
        .{ "$x_{1}|y$ and $$a*b$$", "$x_{1}\\|y$ and $$a*b$$" },
        // in math each `|` gets one backslash, whatever precedes it: GFM's table step removes exactly that one
        .{ "$a\\|b$", "$a\\\\|b$" },
        .{ "$a\\\\|b$", "$a\\\\\\|b$" },
        .{ "$$a\\|b$$ $\\alpha$", "$$a\\\\|b$$ $\\alpha$" },
        // outside math `\` before punctuation doubles: a literal `a\` then `|` reads back through `a\\|` to `a\|`
        .{ "a\\|b", "a\\\\\\|b" },
        .{ "costs $5 only", "costs \\$5 only" },
        .{ "$$ alone", "\\$\\$ alone" },
        .{ "\\*", "\\\\\\*" },
        .{ "a<b & c", "a&lt;b & c" },
    };
    for (cases) |c| {
        var cell = [_]html.Cell{.{ .text = c[0], .row = 0, .col = 0, .rows = 1, .cols = 1, .head = true }};
        const got = try html.pipe(a, .{ .rows = 1, .cols = 1, .cells = &cell, .spanned = false });
        try t.expectEqualStrings(try std.mem.concat(a, u8, &.{ "| ", c[1], " |\n| --- |" }), got);
    }
}

//! The HTML table subset OCR models write: cells and spans, and the table as GFM pipes, clean HTML or text rows.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Most HTML bytes, grid rows and grid columns ``parse`` reads; a table past them is ``error.Malformed``.
pub const max_bytes = 64 * 1024;
pub const max_rows = 512;
pub const max_cols = 64;

/// One cell at its top-left grid position, covering ``rows`` x ``cols``; ``text`` holds '\n' where the HTML had <br>.
pub const Cell = struct { text: []const u8, row: u32, col: u32, rows: u32, cols: u32, head: bool };

/// A table's grid and its cells in row-major order; ``spanned``: some cell covers more than one position.
pub const Table = struct {
    rows: u32,
    cols: u32,
    cells: []Cell,
    spanned: bool,

    /// Frees what ``parse`` allocated in ``a``.
    pub fn deinit(t: Table, a: Allocator) void {
        for (t.cells) |c| a.free(c.text);
        a.free(t.cells);
    }
};

pub const ParseError = error{ Malformed, OutOfMemory };

/// The one <table> in ``html``, with only whitespace and comments around it; cells and texts allocated in ``a``.
pub fn parse(a: Allocator, html: []const u8) ParseError!Table {
    if (html.len > max_bytes) return error.Malformed;
    var p: Parser = .{ .a = a, .src = html };
    defer p.text.deinit(a);
    errdefer {
        for (p.cells.items) |c| a.free(c.text);
        p.cells.deinit(a);
    }
    try p.run();
    var cols: u32 = 0;
    var spanned = false;
    for (p.cells.items) |*c| {
        c.rows = @min(c.rows, p.rows - c.row); // a rowspan past the last row ends there, as in HTML
        cols = @max(cols, c.col + c.cols);
        spanned = spanned or c.rows > 1 or c.cols > 1;
    }
    if (cols == 0) return error.Malformed;
    return .{ .rows = p.rows, .cols = cols, .cells = try p.cells.toOwnedSlice(a), .spanned = spanned };
}

/// The table as a GFM pipe table: first row the header, cell text escaped to read literally (``cellText``); spans lost.
pub fn pipe(a: Allocator, t: Table) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var k: usize = 0;
    for (0..t.rows) |r| {
        for (0..t.cols) |c| {
            try out.appendSlice(a, if (c == 0) "| " else " | ");
            try cellText(a, &out, if (topLeft(t, &k, r, c)) |cell| cell.text else "");
        }
        try out.appendSlice(a, " |");
        if (r == 0) {
            try out.append(a, '\n');
            for (0..t.cols) |_| try out.appendSlice(a, "| --- ");
            try out.append(a, '|');
        }
        if (r + 1 < t.rows) try out.append(a, '\n');
    }
    return out.toOwnedSlice(a);
}

/// The table as one clean <table>: rows of <th>/<td>, rowspan and colspan only above 1, text entity-escaped.
pub fn normalized(a: Allocator, t: Table) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "<table>");
    var k: usize = 0;
    for (0..t.rows) |r| {
        try out.appendSlice(a, "<tr>");
        while (k < t.cells.len and t.cells[k].row == r) : (k += 1) {
            const cell = t.cells[k];
            const name = if (cell.head) "th" else "td";
            try out.print(a, "<{s}", .{name});
            if (cell.rows > 1) try out.print(a, " rowspan=\"{d}\"", .{cell.rows});
            if (cell.cols > 1) try out.print(a, " colspan=\"{d}\"", .{cell.cols});
            try out.append(a, '>');
            for (cell.text) |ch| switch (ch) {
                '&' => try out.appendSlice(a, "&amp;"),
                '<' => try out.appendSlice(a, "&lt;"),
                '>' => try out.appendSlice(a, "&gt;"),
                '"' => try out.appendSlice(a, "&quot;"),
                '\n' => try out.appendSlice(a, "<br>"),
                else => try out.append(a, ch),
            };
            try out.print(a, "</{s}>", .{name});
        }
        try out.appendSlice(a, "</tr>");
    }
    try out.appendSlice(a, "</table>");
    return out.toOwnedSlice(a);
}

/// One line a grid row, its cells joined by '\t'; a spanned cell's text at its top-left only, trailing cells kept.
pub fn rowsText(a: Allocator, t: Table) Allocator.Error![][]const u8 {
    const lines = try a.alloc([]const u8, t.rows);
    var done: usize = 0;
    errdefer {
        for (lines[0..done]) |l| a.free(l);
        a.free(lines);
    }
    var line: std.ArrayList(u8) = .empty;
    defer line.deinit(a);
    var k: usize = 0;
    for (lines, 0..) |*l, r| {
        line.clearRetainingCapacity();
        for (0..t.cols) |c| {
            if (c > 0) try line.append(a, '\t');
            const text = if (topLeft(t, &k, r, c)) |cell| cell.text else "";
            for (text) |ch| try line.append(a, if (ch == '\n') ' ' else ch);
        }
        l.* = try a.dupe(u8, line.items);
        done += 1;
    }
    return lines;
}

/// A cell's text in a pipe row: math as written (only `|` escaped), markup escaped elsewhere, to read back literally.
fn cellText(a: Allocator, out: *std.ArrayList(u8), text: []const u8) Allocator.Error!void {
    var no_close = false; // a `$` found no closer, so no later `$` in the cell can (markdown.zig's rule)
    var i: usize = 0;
    while (i < text.len) {
        const ch = text[i];
        if (ch == '$') {
            if (mathEnd(text, i, &no_close)) |end| {
                try mathText(a, out, text[i..end]);
                i = end;
                continue;
            }
            // a `$` that opens no math is escaped, so a reader never pairs it with another
            const lone: usize = if (i + 1 < text.len and text[i + 1] == '$') 2 else 1;
            for (0..lone) |_| try out.appendSlice(a, "\\$");
            i += lone;
            continue;
        }
        switch (ch) {
            '\n' => try out.append(a, ' '),
            '|', '`', '*', '[', ']' => try out.appendSlice(a, &[_]u8{ '\\', ch }),
            // CommonMark reads a backslash as an escape only before ASCII punctuation; elsewhere it is literal
            '\\' => try out.appendSlice(a, if (i + 1 < text.len and isPunct(text[i + 1])) "\\\\" else "\\"),
            // `_` between two word characters can neither open nor close emphasis, so snake_case stays readable
            '_' => try out.appendSlice(a, if (i > 0 and i + 1 < text.len and isWord(text[i - 1]) and isWord(text[i + 1])) "_" else "\\_"),
            '<' => try out.appendSlice(a, if (opensTag(text[i + 1 ..])) "&lt;" else "<"),
            '&' => try out.appendSlice(a, if (opensEntity(text[i + 1 ..])) "&amp;" else "&"),
            else => try out.append(a, ch),
        }
        i += 1;
    }
}

/// Math as written, each `|` as `\|`: GFM's table step removes exactly that backslash, whatever precedes it.
fn mathText(a: Allocator, out: *std.ArrayList(u8), m: []const u8) Allocator.Error!void {
    for (m) |ch| switch (ch) {
        '|' => try out.appendSlice(a, "\\|"),
        '\n' => try out.append(a, ' '),
        else => try out.append(a, ch),
    };
}

/// The end of the `$$…$$` or `$…$` math opening at ``s[i]``, by markdown.zig's ``mathSpan`` rule; null when none.
fn mathEnd(s: []const u8, i: usize, no_close: *bool) ?usize {
    if (i + 1 < s.len and s[i + 1] == '$') return if (std.mem.indexOfPos(u8, s, i + 2, "$$")) |j| j + 2 else null;
    if (i + 1 >= s.len or isSpace(s[i + 1]) or no_close.*) return null;
    var j = i + 2;
    while (j < s.len) : (j += 1) {
        if (s[j] != '$' or isSpace(s[j - 1]) or escaped(s, j)) continue;
        if (j + 1 < s.len and std.ascii.isDigit(s[j + 1])) continue;
        return j + 1;
    }
    no_close.* = true;
    return null;
}

/// Whether ``s[i]`` follows an odd run of backslashes.
fn escaped(s: []const u8, i: usize) bool {
    var n: usize = 0;
    while (n < i and s[i - 1 - n] == '\\') n += 1;
    return n % 2 == 1;
}

fn isSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r';
}

fn isPunct(c: u8) bool {
    return c < 0x80 and std.ascii.isPrint(c) and !std.ascii.isAlphanumeric(c) and c != ' ';
}

/// A byte of a word for emphasis rules: neither whitespace nor ASCII punctuation (UTF-8 letters count).
fn isWord(c: u8) bool {
    return !isSpace(c) and !isPunct(c);
}

/// Whether text after a literal `<` would start raw HTML in Markdown: a tag, an end tag, a comment or a declaration.
fn opensTag(rest: []const u8) bool {
    return rest.len > 0 and (std.ascii.isAlphabetic(rest[0]) or std.mem.indexOfScalar(u8, "/!?", rest[0]) != null);
}

/// Whether text after a literal `&` reads as an entity reference in Markdown (`name;`, `#digits;` or `#xhex;`).
fn opensEntity(rest: []const u8) bool {
    var n: usize = 0;
    while (n < rest.len and n < 33 and (std.ascii.isAlphanumeric(rest[n]) or (n == 0 and rest[n] == '#'))) n += 1;
    return n > 0 and n < rest.len and rest[n] == ';';
}

/// The cell whose top-left is (r, c), walking ``k`` through the row-major cells; null: covered or empty.
fn topLeft(t: Table, k: *usize, r: usize, c: usize) ?Cell {
    if (k.* >= t.cells.len or t.cells[k.*].row != r or t.cells[k.*].col != c) return null;
    k.* += 1;
    return t.cells[k.* - 1];
}

const Tag = enum { table, tr, td, th, br, other };

fn tagKind(name: []const u8) Tag {
    inline for (.{ Tag.table, Tag.tr, Tag.td, Tag.th, Tag.br }) |t| {
        if (std.ascii.eqlIgnoreCase(name, @tagName(t))) return t;
    }
    return .other;
}

/// A span attribute's value as HTML reads it: leading digits, and 1 when there are none or they read 0.
fn spanValue(v: []const u8) u32 {
    var i: usize = 0;
    while (i < v.len and std.ascii.isWhitespace(v[i])) i += 1;
    var n: u32 = 0;
    while (i < v.len and std.ascii.isDigit(v[i])) : (i += 1) n = @min(n * 10 + (v[i] - '0'), 1_000_000);
    return @max(n, 1);
}

/// The code point a named or numeric entity body (between '&' and ';') stands for; null: not one this reader knows.
fn entityPoint(name: []const u8) ?u21 {
    const named = .{ .{ "amp", '&' }, .{ "lt", '<' }, .{ "gt", '>' }, .{ "quot", '"' }, .{ "apos", '\'' } };
    if (std.mem.eql(u8, name, "nbsp")) return 0xa0;
    inline for (named) |e| if (std.mem.eql(u8, name, e[0])) return e[1];
    if (name.len < 2 or name[0] != '#') return null;
    const hex = name[1] == 'x' or name[1] == 'X';
    const base: u8 = if (hex) 16 else 10;
    const digits = name[if (hex) 2 else 1..];
    if (digits.len == 0 or digits.len > 7) return null;
    var v: u32 = 0;
    for (digits) |d| v = v * base + (std.fmt.charToDigit(d, base) catch return null);
    return if (v == 0 or v > 0x10ffff or (v >= 0xd800 and v <= 0xdfff)) 0xfffd else @intCast(v);
}

/// Whitespace or one of ``ends``: where an attribute's name or unquoted value stops.
fn stops(ch: u8, ends: []const u8) bool {
    return std.ascii.isWhitespace(ch) or std.mem.indexOfScalar(u8, ends, ch) != null;
}

const Parser = struct {
    a: Allocator,
    src: []const u8,
    i: usize = 0,
    where: enum { before, inside, after } = .before,
    cells: std.ArrayList(Cell) = .empty,
    cell: ?Cell = null, // the open cell; its text gathers in ``text`` until it closes
    text: std.ArrayList(u8) = .empty,
    space: bool = false, // whitespace met in the open cell since its last character
    row_open: bool = false,
    rows: u32 = 0, // rows opened so far; the open row is the last
    col: u32 = 0, // the open row's next column to try
    until: [max_cols]u32 = @splat(0), // each column is covered by earlier cells through row until - 1

    fn run(p: *Parser) ParseError!void {
        while (p.i < p.src.len) {
            const ch = p.src[p.i];
            const next = if (p.i + 1 < p.src.len) p.src[p.i + 1] else 0;
            const after = if (p.i + 2 < p.src.len) p.src[p.i + 2] else 0;
            if (ch == '<' and (next == '!' or next == '?')) {
                try p.skipMarkup();
            } else if (ch == '<' and std.ascii.isAlphabetic(if (next == '/') after else next)) {
                try p.tag();
            } else if (ch == '&') {
                try p.entity();
            } else {
                try p.char(ch);
                p.i += 1;
            }
        }
        if (p.where != .after) return error.Malformed;
    }

    /// A comment, a doctype or a processing instruction: skipped whole.
    fn skipMarkup(p: *Parser) ParseError!void {
        if (std.mem.startsWith(u8, p.src[p.i..], "<!--")) {
            const end = std.mem.indexOfPos(u8, p.src, p.i + 4, "-->") orelse return error.Malformed;
            p.i = end + 3;
        } else {
            const end = std.mem.indexOfScalarPos(u8, p.src, p.i, '>') orelse return error.Malformed;
            p.i = end + 1;
        }
    }

    fn entity(p: *Parser) ParseError!void {
        const rest = p.src[p.i..];
        const semi = std.mem.indexOfScalar(u8, rest[0..@min(rest.len, 12)], ';');
        const point = if (semi) |s| entityPoint(rest[1..s]) else null;
        const cp = point orelse {
            p.i += 1;
            return p.char('&');
        };
        p.i += semi.? + 1;
        if (cp == 0xa0) return p.char(' ');
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &buf) catch unreachable;
        for (buf[0..n]) |b| try p.char(b);
    }

    /// One text byte: whitespace runs fold to one space between words; outside a cell only whitespace may stand.
    fn char(p: *Parser, ch: u8) ParseError!void {
        const blank = std.ascii.isWhitespace(ch);
        if (p.cell == null) {
            if (!blank) return error.Malformed;
            return;
        }
        if (blank) {
            p.space = true;
            return;
        }
        if (p.space and p.text.items.len > 0 and p.text.getLast() != '\n') try p.text.append(p.a, ' ');
        p.space = false;
        try p.text.append(p.a, ch);
    }

    /// A start or end tag at ``i``: its name, its span attributes in any quoting, then its effect.
    fn tag(p: *Parser) ParseError!void {
        const src = p.src;
        var i = p.i + 1;
        const close = src[i] == '/';
        if (close) i += 1;
        const start = i;
        while (i < src.len and (std.ascii.isAlphanumeric(src[i]) or std.mem.indexOfScalar(u8, "-_:.", src[i]) != null)) i += 1;
        const kind = tagKind(src[start..i]);
        var span = [2]u32{ 1, 1 }; // rowspan, colspan
        while (true) {
            while (i < src.len and (std.ascii.isWhitespace(src[i]) or src[i] == '/')) i += 1;
            if (i >= src.len) return error.Malformed;
            if (src[i] == '>') break;
            const name_start = i;
            while (i < src.len and !stops(src[i], "=>/")) i += 1;
            const name = src[name_start..i];
            while (i < src.len and std.ascii.isWhitespace(src[i])) i += 1;
            if (i >= src.len or src[i] != '=') continue;
            i += 1;
            while (i < src.len and std.ascii.isWhitespace(src[i])) i += 1;
            if (i >= src.len) return error.Malformed;
            var value: []const u8 = undefined;
            if (src[i] == '"' or src[i] == '\'') {
                const end = std.mem.indexOfScalarPos(u8, src, i + 1, src[i]) orelse return error.Malformed;
                value = src[i + 1 .. end];
                i = end + 1;
            } else {
                const value_start = i;
                while (i < src.len and !stops(src[i], ">")) i += 1;
                value = src[value_start..i];
            }
            if (std.ascii.eqlIgnoreCase(name, "rowspan")) span[0] = spanValue(value);
            if (std.ascii.eqlIgnoreCase(name, "colspan")) span[1] = spanValue(value);
        }
        p.i = i + 1;
        try p.apply(kind, close, span);
    }

    fn apply(p: *Parser, kind: Tag, close: bool, span: [2]u32) ParseError!void {
        switch (p.where) {
            .before => {
                if (kind != .table or close) return error.Malformed;
                p.where = .inside;
                return;
            },
            .after => return error.Malformed,
            .inside => {},
        }
        switch (kind) {
            .table => {
                if (!close) return error.Malformed; // a table inside a cell
                try p.endCell();
                p.row_open = false;
                p.where = .after;
            },
            .tr => {
                try p.endCell();
                p.row_open = false;
                if (!close) try p.openRow();
            },
            .td, .th => {
                try p.endCell();
                if (!close) try p.openCell(kind == .th, span);
            },
            .br => if (p.cell != null) {
                p.space = false;
                try p.text.append(p.a, '\n');
            },
            .other => {},
        }
    }

    fn openRow(p: *Parser) ParseError!void {
        if (p.rows == max_rows) return error.Malformed;
        p.rows += 1;
        p.row_open = true;
        p.col = 0;
    }

    /// A cell at the open row's first free column; its spans clipped to the grid and to columns earlier rows cover.
    fn openCell(p: *Parser, head: bool, span: [2]u32) ParseError!void {
        if (!p.row_open) return error.Malformed;
        const row = p.rows - 1;
        while (p.col < max_cols and p.until[p.col] > row) p.col += 1;
        if (p.col == max_cols) return error.Malformed;
        var cols: u32 = 0;
        while (cols < span[1] and p.col + cols < max_cols and p.until[p.col + cols] <= row) cols += 1;
        const rows = @min(span[0], max_rows - row);
        for (p.until[p.col..][0..cols]) |*u| u.* = row + rows;
        p.cell = .{ .text = "", .row = row, .col = p.col, .rows = rows, .cols = cols, .head = head };
        p.col += cols;
    }

    fn endCell(p: *Parser) ParseError!void {
        var cell = p.cell orelse return;
        p.cell = null;
        p.space = false;
        defer p.text.clearRetainingCapacity();
        try p.cells.ensureUnusedCapacity(p.a, 1);
        cell.text = try p.a.dupe(u8, std.mem.trim(u8, p.text.items, " \n"));
        p.cells.appendAssumeCapacity(cell);
    }
};

//! A block's tables as byte ranges, and a pipe table read back as cells (docs/api.md OCR: Reader Markdown).
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const html = api.ocr.html;

/// The byte range of one table in the content; ``html``: an HTML <table>, else a GFM pipe table.
pub const Found = struct { start: usize, end: usize, html: bool };

/// Every table in the content in order, found by a line scanner (none inside code or math).
pub fn tables(a: Allocator, content: []const u8) Allocator.Error![]Found {
    var out: std.ArrayList(Found) = .empty;
    var sc: Scanner = .{ .src = content };
    while (sc.next()) |item| switch (item) {
        .table => |f| try out.append(a, f),
        else => {},
    };
    return out.toOwnedSlice(a);
}

/// A GFM pipe table (a ``Found`` range that is not HTML) as an ``ocr_html`` table: the header row first, cells cleaned.
pub fn pipeTable(a: Allocator, table: []const u8) html.ParseError!html.Table {
    var rows: std.ArrayList([]const u8) = .empty;
    defer rows.deinit(a);
    var it = std.mem.splitScalar(u8, table, '\n');
    var n: usize = 0;
    while (it.next()) |raw| : (n += 1) if (n != 1) try rows.append(a, trim(raw)); // row 1 is the delimiter row
    const cols = if (rows.items.len > 0) Cells.count(rows.items[0]) else 0;
    if (cols == 0 or cols > html.max_cols or rows.items.len > html.max_rows) return error.Malformed;
    const cells = try a.alloc(html.Cell, rows.items.len * cols);
    var done: usize = 0;
    errdefer {
        for (cells[0..done]) |c| a.free(c.text);
        a.free(cells);
    }
    for (rows.items, 0..) |row, r| {
        var cs = Cells.init(row); // GFM: a short row is padded, cells past the header's count are dropped
        for (0..cols) |c| {
            // the cell's own text: math keeps its dollars, as the HTML cell it came from held them
            const text = if (cs.next()) |cell| try cellText(a, cell) else try a.alloc(u8, 0);
            cells[done] = .{ .text = text, .row = @intCast(r), .col = @intCast(c), .rows = 1, .cols = 1, .head = r == 0 };
            done += 1;
        }
    }
    return .{ .rows = @intCast(rows.items.len), .cols = @intCast(cols), .cells = cells, .spanned = false };
}

/// The cells of one pipe-table row, its outer pipes dropped; as cmark-gfm, a `|` right after a `\` is escaped.
const Cells = struct {
    rest: ?[]const u8,

    fn init(row: []const u8) Cells {
        var r = row;
        if (r.len > 0 and r[0] == '|') r = r[1..];
        if (r.len > 0 and r[r.len - 1] == '|' and !pipeEscaped(r, r.len - 1)) r = r[0 .. r.len - 1];
        return .{ .rest = r };
    }

    fn next(c: *Cells) ?[]const u8 {
        const r = c.rest orelse return null;
        var i: usize = 0;
        while (i < r.len) : (i += 1) {
            if (r[i] == '|' and !pipeEscaped(r, i)) {
                c.rest = r[i + 1 ..];
                return r[0..i];
            }
        }
        c.rest = null;
        return r;
    }

    fn count(row: []const u8) usize {
        var c = init(row);
        var n: usize = 0;
        while (c.next()) |_| n += 1;
        return n;
    }
};

/// One pipe-table cell's text: GFM drops the backslash before each `|` (in code and math too), then inline rules.
fn cellText(a: Allocator, cell: []const u8) Allocator.Error![]u8 {
    const t = trim(cell);
    const bare = try a.alloc(u8, t.len);
    defer a.free(bare);
    var n: usize = 0;
    for (t, 0..) |ch, i| {
        if (ch == '\\' and i + 1 < t.len and t[i + 1] == '|') continue;
        bare[n] = ch;
        n += 1;
    }
    return clean(a, bare[0..n], 0);
}

/// GFM's table rule (cmark-gfm `escaped_char = [\\][|]`): the `|` at ``s[i]`` is escaped when `\` is right before it.
fn pipeEscaped(s: []const u8, i: usize) bool {
    return i > 0 and s[i - 1] == '\\';
}

/// Whether ``s[i]`` follows an odd run of backslashes.
fn escaped(s: []const u8, i: usize) bool {
    var n: usize = 0;
    while (n < i and s[i - 1 - n] == '\\') n += 1;
    return n % 2 == 1;
}

fn trim(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r");
}

/// What the scanner yields: an ordinary line, a code or math line kept as written, or a table's byte range.
const Item = union(enum) { text: []const u8, verbatim: []const u8, table: Found };

/// A line scanner over docs/api.md OCR: Reader Markdown: fenced code, `$$` blocks, HTML and pipe tables, other lines.
const Scanner = struct {
    src: []const u8,
    pos: usize = 0,
    fence: ?struct { ch: u8, len: usize } = null,
    math: bool = false,
    no_table_close: bool = false, // a search for `</table>` failed: none further on either

    const Line = struct { text: []const u8, next: usize };

    fn lineAt(s: *const Scanner, pos: usize) Line {
        const end = std.mem.indexOfScalarPos(u8, s.src, pos, '\n') orelse s.src.len;
        const text = std.mem.trimEnd(u8, s.src[pos..end], "\r");
        return .{ .text = text, .next = @min(end + 1, s.src.len) };
    }

    /// Where a slice of the source starts in it.
    fn offset(s: *const Scanner, part: []const u8) usize {
        return @intFromPtr(part.ptr) - @intFromPtr(s.src.ptr);
    }

    fn next(s: *Scanner) ?Item {
        while (s.pos < s.src.len) {
            const l = s.lineAt(s.pos);
            const t = trim(l.text);
            s.pos = l.next;
            if (s.fence) |f| {
                if (t.len >= f.len and std.mem.allEqual(u8, t, f.ch)) {
                    s.fence = null;
                    continue;
                }
                return .{ .verbatim = l.text };
            }
            if (s.math) {
                if (std.mem.endsWith(u8, t, "$$")) {
                    s.math = false;
                    const inner = trim(t[0 .. t.len - 2]);
                    if (inner.len > 0) return .{ .verbatim = inner };
                    continue;
                }
                return .{ .verbatim = t };
            }
            if (fenceOpen(t)) |f| {
                s.fence = .{ .ch = t[0], .len = f };
                continue;
            }
            if (std.mem.startsWith(u8, t, "$$")) {
                const rest = t[2..];
                if (std.mem.endsWith(u8, rest, "$$")) return .{ .verbatim = trim(rest[0 .. rest.len - 2]) };
                s.math = true;
                return .{ .verbatim = trim(rest) };
            }
            if (startsHtmlTable(t) and !s.no_table_close) {
                const start = s.offset(t);
                if (indexOfIgnoreCase(s.src, start, "</table>")) |close| {
                    s.pos = close + "</table>".len;
                    return .{ .table = .{ .start = start, .end = s.pos, .html = true } };
                }
                s.no_table_close = true;
            }
            if (std.mem.indexOfScalar(u8, t, '|') != null and l.next < s.src.len) {
                const d = s.lineAt(l.next);
                if (delimiterRow(trim(d.text), Cells.count(t))) {
                    var end = d.next;
                    var last = s.offset(trim(d.text)) + trim(d.text).len;
                    while (end < s.src.len) {
                        const r = s.lineAt(end);
                        const rt = trim(r.text);
                        if (rt.len == 0 or std.mem.indexOfScalar(u8, rt, '|') == null) break;
                        last = s.offset(rt) + rt.len;
                        end = r.next;
                    }
                    s.pos = end;
                    return .{ .table = .{ .start = s.offset(t), .end = last, .html = false } };
                }
            }
            return .{ .text = l.text };
        }
        return null;
    }
};

/// The fence length when ``t`` opens a fenced code block (3+ backticks or tildes; a backtick fence's info has none).
fn fenceOpen(t: []const u8) ?usize {
    if (t.len < 3 or (t[0] != '`' and t[0] != '~')) return null;
    var n: usize = 0;
    while (n < t.len and t[n] == t[0]) n += 1;
    if (n < 3 or (t[0] == '`' and std.mem.indexOfScalar(u8, t[n..], '`') != null)) return null;
    return n;
}

fn startsHtmlTable(t: []const u8) bool {
    if (t.len < 6 or !std.ascii.eqlIgnoreCase(t[0..6], "<table")) return false;
    return t.len == 6 or t[6] == '>' or t[6] == '/' or std.ascii.isWhitespace(t[6]);
}

fn indexOfIgnoreCase(s: []const u8, start: usize, needle: []const u8) ?usize {
    var i = start;
    while (i + needle.len <= s.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(s[i .. i + needle.len], needle)) return i;
    }
    return null;
}

/// Whether ``t`` is a GFM delimiter row (`:?-+:?` cells, a pipe somewhere) with ``cells`` cells.
fn delimiterRow(t: []const u8, cells: usize) bool {
    if (std.mem.indexOfScalar(u8, t, '|') == null) return false;
    var it = Cells.init(t);
    var n: usize = 0;
    while (it.next()) |raw| : (n += 1) {
        var c = trim(raw);
        if (c.len > 0 and c[0] == ':') c = c[1..];
        if (c.len > 0 and c[c.len - 1] == ':') c = c[0 .. c.len - 1];
        if (c.len == 0 or !std.mem.allEqual(u8, c, '-')) return false;
    }
    return n == cells;
}

/// The deepest link-in-link the cleaner unwraps; a deeper `[` stays literal.
const max_depth = 4;

/// One emphasis delimiter run in the output: where it sits, its length, what it may do and how much a match removed.
const Run = struct { at: usize, len: usize, ch: u8, open: bool, close: bool, left: usize, remove: usize = 0 };

/// Inline markup removed: emphasis, code ticks, link syntax, escapes, entities; math kept as written, dollars too.
fn clean(a: Allocator, s: []const u8, depth: u8) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var runs: std.ArrayList(Run) = .empty;
    defer runs.deinit(a);
    var no_dollar_close = false; // a `$` found no closer: no later `$` on the line can, so it skips the scan
    const pairs = try Pairs.init(a, s);
    defer pairs.deinit(a);
    var i: usize = 0;
    while (i < s.len) {
        const ch = s[i];
        if (ch == '\\' and i + 1 < s.len and isPunct(s[i + 1])) {
            try out.append(a, s[i + 1]);
            i += 2;
        } else if (ch == '`') {
            const span = codeSpan(s, i);
            try out.appendSlice(a, span.body);
            i = span.end;
        } else if (ch == '$') {
            const span = mathSpan(s, i, &no_dollar_close);
            try out.appendSlice(a, s[i..span.end]);
            i = span.end;
        } else if ((ch == '[' or (ch == '!' and i + 1 < s.len and s[i + 1] == '[')) and depth < max_depth) {
            const open = if (ch == '!') i + 1 else i;
            if (link(s, pairs, open)) |l| {
                const text = try clean(a, l.text, depth + 1);
                defer a.free(text);
                try out.appendSlice(a, text);
                i = l.end;
            } else {
                try out.append(a, ch);
                i += 1;
            }
        } else if (ch == '*' or ch == '_') {
            var n: usize = 1;
            while (i + n < s.len and s[i + n] == ch) n += 1;
            const prev: u8 = if (i > 0) s[i - 1] else ' ';
            const next: u8 = if (i + n < s.len) s[i + n] else ' ';
            const left = !isSpace(next) and (!isPunct(next) or isSpace(prev) or isPunct(prev));
            const right = !isSpace(prev) and (!isPunct(prev) or isSpace(next) or isPunct(next));
            const open = if (ch == '*') left else left and (!right or isPunct(prev));
            const close = if (ch == '*') right else right and (!left or isPunct(next));
            try runs.append(a, .{ .at = out.items.len, .len = n, .ch = ch, .open = open, .close = close, .left = n });
            try out.appendSlice(a, s[i .. i + n]);
            i += n;
        } else if (ch == '&') {
            var buf: [4]u8 = undefined;
            if (entity(s, i, &buf)) |e| {
                try out.appendSlice(a, buf[0..e.len]);
                i = e.end;
            } else {
                try out.append(a, ch);
                i += 1;
            }
        } else {
            try out.append(a, ch);
            i += 1;
        }
    }
    try match(a, runs.items);
    var res: std.ArrayList(u8) = .empty;
    errdefer res.deinit(a);
    var from: usize = 0;
    for (runs.items) |r| {
        try res.appendSlice(a, out.items[from..r.at]);
        try res.appendSlice(a, out.items[r.at + r.remove .. r.at + r.len]);
        from = r.at + r.len;
    }
    try res.appendSlice(a, out.items[from..]);
    return res.toOwnedSlice(a);
}

/// Pairs closers with the nearest open run of the same character, as CommonMark does without its rule of three.
fn match(a: Allocator, runs: []Run) Allocator.Error!void {
    var stack: std.ArrayList(usize) = .empty;
    defer stack.deinit(a);
    var bottom: [2]usize = .{ 0, 0 }; // per character: no usable opener below this stack index
    for (runs, 0..) |*c, i| {
        const b = &bottom[@intFromBool(c.ch == '_')];
        while (c.close and c.left > 0) {
            var k = stack.items.len;
            while (k > b.* and runs[stack.items[k - 1]].ch != c.ch) k -= 1;
            if (k == b.*) {
                b.* = stack.items.len;
                break;
            }
            const o = &runs[stack.items[k - 1]];
            const n = @min(o.left, c.left);
            o.left -= n;
            c.left -= n;
            o.remove += n;
            c.remove += n;
            stack.shrinkRetainingCapacity(if (o.left > 0) k else k - 1);
            for (&bottom) |*x| x.* = @min(x.*, stack.items.len);
        }
        if (c.open and c.left > 0) try stack.append(a, i);
    }
}

const Span = struct { body: []const u8, end: usize };

/// A code span's body (one padding space trimmed each side), or the backtick run itself when it never closes.
fn codeSpan(s: []const u8, i: usize) Span {
    var n: usize = 0;
    while (i + n < s.len and s[i + n] == '`') n += 1;
    var j = i + n;
    while (j < s.len) {
        if (s[j] != '`') {
            j += 1;
            continue;
        }
        var m: usize = 0;
        while (j + m < s.len and s[j + m] == '`') m += 1;
        if (m == n) {
            var body = s[i + n .. j];
            if (body.len >= 2 and body[0] == ' ' and body[body.len - 1] == ' ' and !std.mem.allEqual(u8, body, ' ')) body = body[1 .. body.len - 1];
            return .{ .body = body, .end = j + m };
        }
        j += m;
    }
    return .{ .body = s[i .. i + n], .end = i + n };
}

/// Inline `$$…$$` or `$…$` math's LaTeX, or a literal `$` when it opens nothing (`$` before a space, `$5`-style text).
fn mathSpan(s: []const u8, i: usize, no_close: *bool) Span {
    if (i + 1 < s.len and s[i + 1] == '$') {
        if (std.mem.indexOfPos(u8, s, i + 2, "$$")) |j| return .{ .body = trim(s[i + 2 .. j]), .end = j + 2 };
        return .{ .body = s[i .. i + 2], .end = i + 2 };
    }
    if (i + 1 < s.len and !isSpace(s[i + 1]) and !no_close.*) {
        var j = i + 2;
        while (j < s.len) : (j += 1) {
            if (s[j] != '$' or isSpace(s[j - 1]) or escaped(s, j)) continue;
            if (j + 1 < s.len and std.ascii.isDigit(s[j + 1])) continue;
            return .{ .body = s[i + 1 .. j], .end = j + 1 };
        }
        no_close.* = true;
    }
    return .{ .body = s[i .. i + 1], .end = i + 1 };
}

const Link = struct { text: []const u8, end: usize };

/// ``[text](target)`` at ``s[i]`` (the `[`): its text and the index after the `)`.
fn link(s: []const u8, pairs: Pairs, i: usize) ?Link {
    const close = pairs.of(i) orelse return null;
    if (close + 1 >= s.len or s[close + 1] != '(') return null;
    const end = pairs.of(close + 1) orelse return null;
    return .{ .text = s[i + 1 .. close], .end = end + 1 };
}

/// Each `[` and `(` with the bracket that closes it, nesting counted and escaped brackets skipped, in one pass.
const Pairs = struct {
    close: []usize, // empty when the line has no `[`; else per byte, `none` unless an opener with a partner
    const none = std.math.maxInt(usize);

    fn init(a: Allocator, s: []const u8) Allocator.Error!Pairs {
        if (std.mem.indexOfScalar(u8, s, '[') == null) return .{ .close = &.{} };
        const close = try a.alloc(usize, s.len);
        errdefer a.free(close);
        @memset(close, none);
        var brackets: std.ArrayList(usize) = .empty;
        defer brackets.deinit(a);
        var parens: std.ArrayList(usize) = .empty;
        defer parens.deinit(a);
        var j: usize = 0;
        while (j < s.len) : (j += 1) switch (s[j]) {
            '\\' => j += 1,
            '[' => try brackets.append(a, j),
            '(' => try parens.append(a, j),
            ']' => if (brackets.pop()) |o| {
                close[o] = j;
            },
            ')' => if (parens.pop()) |o| {
                close[o] = j;
            },
            else => {},
        };
        return .{ .close = close };
    }

    fn deinit(p: Pairs, a: Allocator) void {
        if (p.close.len > 0) a.free(p.close);
    }

    fn of(p: Pairs, i: usize) ?usize {
        if (i >= p.close.len or p.close[i] == none) return null;
        return p.close[i];
    }
};

const Entity = struct { len: usize, end: usize };

/// A character reference at ``s[i]``: the five XML names or a numeric one, as UTF-8 in ``buf``.
fn entity(s: []const u8, i: usize, buf: *[4]u8) ?Entity {
    const window = s[i + 1 .. @min(s.len, i + 11)]; // a name past 9 bytes is no reference: never scan further
    const semi = i + 1 + (std.mem.indexOfScalar(u8, window, ';') orelse return null);
    const name = s[i + 1 .. semi];
    const names = [_]struct { []const u8, u8 }{ .{ "lt", '<' }, .{ "gt", '>' }, .{ "amp", '&' }, .{ "quot", '"' }, .{ "apos", '\'' } };
    for (names) |n| if (std.mem.eql(u8, name, n[0])) {
        buf[0] = n[1];
        return .{ .len = 1, .end = semi + 1 };
    };
    if (name.len < 2 or name[0] != '#') return null;
    const hex = name[1] == 'x' or name[1] == 'X';
    const digits = if (hex) name[2..] else name[1..];
    const cp = std.fmt.parseInt(u21, digits, if (hex) 16 else 10) catch return null;
    if (cp == 0) return null;
    const len = std.unicode.utf8Encode(cp, buf) catch return null;
    return .{ .len = len, .end = semi + 1 };
}

fn isSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r';
}

fn isPunct(c: u8) bool {
    return c < 0x80 and std.ascii.isPrint(c) and !std.ascii.isAlphanumeric(c) and c != ' ';
}

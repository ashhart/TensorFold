//! Readable text from an HTML page: its title, then its main content with block breaks kept and entities decoded.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Text = struct { title: []const u8, text: []const u8, truncated: bool };

/// Elements whose content is never prose (scripts, styles, page chrome, controls).
const skipped = [_][]const u8{ "script", "style", "noscript", "svg", "template", "iframe", "head", "nav", "footer", "aside", "form", "button", "select", "textarea", "canvas", "object", "video", "audio", "map" };
const blocks = [_][]const u8{ "p", "div", "br", "ul", "ol", "tr", "table", "section", "article", "header", "blockquote", "hr", "dl", "dt", "dd", "figure", "figcaption", "details", "summary", "main", "tbody", "thead" };

const entities = [_]struct { []const u8, []const u8 }{
    .{ "amp", "&" },        .{ "lt", "<" },          .{ "gt", ">" },        .{ "quot", "\"" },      .{ "apos", "'" },
    .{ "nbsp", " " },       .{ "ndash", "\u{2013}" }, .{ "mdash", "\u{2014}" }, .{ "hellip", "\u{2026}" }, .{ "rsquo", "\u{2019}" },
    .{ "lsquo", "\u{2018}" }, .{ "rdquo", "\u{201d}" }, .{ "ldquo", "\u{201c}" }, .{ "laquo", "\u{ab}" },   .{ "raquo", "\u{bb}" },
    .{ "middot", "\u{b7}" }, .{ "bull", "\u{2022}" }, .{ "times", "\u{d7}" }, .{ "copy", "\u{a9}" },     .{ "reg", "\u{ae}" },
    .{ "trade", "\u{2122}" }, .{ "deg", "\u{b0}" },   .{ "rarr", "\u{2192}" }, .{ "larr", "\u{2190}" },   .{ "zwj", "" },
    .{ "zwnj", "" },        .{ "shy", "" },
};

fn isName(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '-' or ch == ':';
}

fn inList(name: []const u8, list: []const []const u8) bool {
    for (list) |x| if (std.ascii.eqlIgnoreCase(name, x)) return true;
    return false;
}

/// Where the tag starting at `lt` ends (past its `>`), skipping `>` inside quoted attribute values.
fn tagEnd(html: []const u8, lt: usize) usize {
    var quote: u8 = 0;
    var i = lt + 1;
    while (i < html.len) : (i += 1) {
        const ch = html[i];
        if (quote != 0) {
            if (ch == quote) quote = 0;
        } else if (ch == '"' or ch == '\'') quote = ch else if (ch == '>') return i + 1;
    }
    return html.len;
}

/// The decoded text of `&...;` at `at`, and its length; null when it is not an entity.
fn entity(html: []const u8, at: usize, buf: *[4]u8) ?struct { text: []const u8, len: usize } {
    const semi = std.mem.indexOfScalarPos(u8, html, at, ';') orelse return null;
    if (semi - at > 10 or semi == at + 1) return null;
    const body = html[at + 1 .. semi];
    if (body[0] == '#') {
        const code = if (body.len > 1 and (body[1] == 'x' or body[1] == 'X'))
            std.fmt.parseInt(u21, body[2..], 16) catch return null
        else
            std.fmt.parseInt(u21, body[1..], 10) catch return null;
        const n = std.unicode.utf8Encode(if (code == 0 or code > 0x10ffff or (code >= 0xd800 and code <= 0xdfff)) 0xfffd else code, buf) catch return null;
        return .{ .text = buf[0..n], .len = semi + 1 - at };
    }
    for (entities) |e| if (std.mem.eql(u8, body, e[0])) return .{ .text = e[1], .len = semi + 1 - at };
    return null;
}

const Out = struct {
    a: Allocator,
    buf: std.ArrayList(u8) = .empty,
    space: bool = false, // a collapsed space waits before the next text

    fn text(o: *Out, s: []const u8, pre: bool) Allocator.Error!void {
        for (s) |ch| {
            if (!pre and (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r' or ch == 0x0c)) {
                o.space = o.buf.items.len > 0 and o.buf.items[o.buf.items.len - 1] != '\n';
                continue;
            }
            if (o.space) try o.buf.append(o.a, ' ');
            o.space = false;
            try o.buf.append(o.a, ch);
        }
    }

    fn newline(o: *Out) Allocator.Error!void {
        o.space = false;
        const items = o.buf.items;
        if (items.len >= 2 and items[items.len - 1] == '\n' and items[items.len - 2] == '\n') return;
        if (items.len > 0) try o.buf.append(o.a, '\n');
    }
};

/// The span of the first `<name ...>` element's content in `html`, or null.
fn element(html: []const u8, name: []const u8) ?[]const u8 {
    var from: usize = 0;
    while (std.ascii.findIgnoreCasePos(html, from, name)) |at| : (from = at + 1) {
        if (at == 0 or html[at - 1] != '<') continue;
        const after = at + name.len;
        if (after < html.len and isName(html[after])) continue;
        const open_end = tagEnd(html, at - 1);
        var close_name: [16]u8 = undefined;
        const closer = std.fmt.bufPrint(&close_name, "</{s}", .{name}) catch return null;
        const close = std.ascii.findIgnoreCasePos(html, open_end, closer) orelse html.len;
        return html[open_end..close];
    }
    return null;
}

/// Readable text of `html`: the title and the main, article or body content, at most `max_chars` bytes of it.
pub fn toText(a: Allocator, html: []const u8, max_chars: usize) Allocator.Error!Text {
    var title: Out = .{ .a = a };
    if (element(html, "title")) |t| try decodeInto(&title, t);
    const region = element(html, "main") orelse element(html, "article") orelse element(html, "body") orelse html;
    var o: Out = .{ .a = a };
    var pre: usize = 0;
    var i: usize = 0;
    var ent: [4]u8 = undefined;
    while (i < region.len) {
        const ch = region[i];
        if (ch == '&') {
            if (entity(region, i, &ent)) |e| {
                try o.text(e.text, pre > 0);
                i += e.len;
                continue;
            }
        }
        if (ch != '<') {
            const next = std.mem.indexOfAnyPos(u8, region, i, "<&") orelse region.len;
            try o.text(region[i..next], pre > 0);
            i = next;
            continue;
        }
        if (std.mem.startsWith(u8, region[i..], "<!--")) {
            i = if (std.mem.indexOfPos(u8, region, i + 4, "-->")) |e| e + 3 else region.len;
            continue;
        }
        const closing = i + 1 < region.len and region[i + 1] == '/';
        const name_start = i + 1 + @intFromBool(closing);
        var name_end = name_start;
        while (name_end < region.len and isName(region[name_end])) name_end += 1;
        const name = region[name_start..name_end];
        const end = tagEnd(region, i);
        if (name.len == 0) {
            try o.text(region[i .. i + 1], pre > 0);
            i += 1;
            continue;
        }
        i = end;
        if (!closing and inList(name, &skipped)) {
            var closer_buf: [16]u8 = undefined;
            const closer = std.fmt.bufPrint(&closer_buf, "</{s}", .{name}) catch continue;
            const close = std.ascii.findIgnoreCasePos(region, i, closer) orelse region.len;
            i = if (close < region.len) tagEnd(region, close) else region.len;
            continue;
        }
        if (std.ascii.eqlIgnoreCase(name, "pre")) {
            try o.newline();
            pre = if (closing) pre -| 1 else pre + 1;
        } else if (name.len == 2 and (name[0] == 'h' or name[0] == 'H') and name[1] >= '1' and name[1] <= '6') {
            try o.newline();
            if (!closing) {
                for (0..name[1] - '0') |_| try o.buf.append(a, '#');
                try o.buf.append(a, ' ');
            }
        } else if (std.ascii.eqlIgnoreCase(name, "li")) {
            try o.newline();
            if (!closing) try o.buf.appendSlice(a, "- ");
        } else if (std.ascii.eqlIgnoreCase(name, "td") or std.ascii.eqlIgnoreCase(name, "th")) {
            if (!closing) try o.text(" | ", true);
        } else if (inList(name, &blocks)) try o.newline();
    }
    const tidy = try tidyLines(a, o.buf.items);
    const cut = cutAt(tidy, max_chars);
    return .{ .title = std.mem.trim(u8, title.buf.items, " "), .text = tidy[0..cut], .truncated = cut < tidy.len };
}

fn decodeInto(o: *Out, s: []const u8) Allocator.Error!void {
    var i: usize = 0;
    var ent: [4]u8 = undefined;
    while (i < s.len) {
        if (s[i] == '&') if (entity(s, i, &ent)) |e| {
            try o.text(e.text, false);
            i += e.len;
            continue;
        };
        try o.text(s[i .. i + 1], false);
        i += 1;
    }
}

/// Lines with trailing blanks cut, and no more than one empty line in a row.
fn tidyLines(a: Allocator, text: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var blank = false;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const t = std.mem.trimEnd(u8, line, " \t\r");
        if (std.mem.trim(u8, t, " \t").len == 0) {
            if (!blank and out.items.len > 0) try out.append(a, '\n');
            blank = true;
            continue;
        }
        blank = false;
        try out.appendSlice(a, t);
        try out.append(a, '\n');
    }
    return std.mem.trim(u8, out.items, "\n");
}

/// The largest length up to `max` that ends on a UTF-8 character boundary.
pub fn cutAt(text: []const u8, max: usize) usize {
    if (text.len <= max) return text.len;
    var n = max;
    while (n > 0 and (text[n] & 0xc0) == 0x80) n -= 1;
    return n;
}

test "title, main content, blocks, lists and entities" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const page =
        \\<html><head><title>Zig &amp; Friends</title><style>p{color:red}</style></head>
        \\<body><nav>Home | Docs</nav><main><h1>Release notes</h1><p>Version  0.17&nbsp;adds
        \\ <b>std.Io</b>.</p><script>alert(1)</script><ul><li>One &lt;thing&gt;</li><li>Two&#33; &#x263A;</li></ul>
        \\<pre>fn main() void {
        \\    run();
        \\}</pre><!-- hidden --><table><tr><td>a</td><td>b</td></tr></table></main><footer>legal</footer></body></html>
    ;
    const t = try toText(a, page, 10_000);
    try std.testing.expectEqualStrings("Zig & Friends", t.title);
    try std.testing.expect(std.mem.indexOf(u8, t.text, "# Release notes") != null);
    try std.testing.expect(std.mem.indexOf(u8, t.text, "Version 0.17 adds std.Io.") != null);
    try std.testing.expect(std.mem.indexOf(u8, t.text, "- One <thing>") != null);
    try std.testing.expect(std.mem.indexOf(u8, t.text, "- Two! \u{263a}") != null);
    try std.testing.expect(std.mem.indexOf(u8, t.text, "    run();") != null);
    try std.testing.expect(std.mem.indexOf(u8, t.text, "| a | b") != null);
    for ([_][]const u8{ "alert", "Home", "legal", "hidden", "color" }) |gone| try std.testing.expect(std.mem.indexOf(u8, t.text, gone) == null);
    try std.testing.expect(!t.truncated);
}

test "a cut never splits a character" {
    const s = "ab\u{263a}cd";
    try std.testing.expectEqual(@as(usize, 2), cutAt(s, 3));
    try std.testing.expectEqual(@as(usize, 5), cutAt(s, 5));
    try std.testing.expectEqual(s.len, cutAt(s, 100));
}

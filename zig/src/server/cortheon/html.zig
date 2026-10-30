//! Readable text from an HTML page: its title, then its main content with block breaks kept and entities decoded.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Budget = @import("budget.zig").Budget;

pub const Text = struct { title: []const u8, text: []const u8, truncated: bool };

/// Elements whose content is never prose (scripts, styles, page chrome, controls).
const skipped = [_][]const u8{ "script", "style", "noscript", "svg", "template", "iframe", "head", "nav", "footer", "aside", "form", "button", "select", "textarea", "canvas", "object", "video", "audio", "map" };
const blocks = [_][]const u8{ "p", "div", "br", "ul", "ol", "tr", "table", "section", "article", "header", "blockquote", "hr", "dl", "dt", "dd", "figure", "figcaption", "details", "summary", "main", "tbody", "thead" };

const entities = [_]struct { []const u8, []const u8 }{
    .{ "amp", "&" },          .{ "lt", "<" },           .{ "gt", ">" },           .{ "quot", "\"" },         .{ "apos", "'" },
    .{ "nbsp", " " },         .{ "ndash", "\u{2013}" }, .{ "mdash", "\u{2014}" }, .{ "hellip", "\u{2026}" }, .{ "rsquo", "\u{2019}" },
    .{ "lsquo", "\u{2018}" }, .{ "rdquo", "\u{201d}" }, .{ "ldquo", "\u{201c}" }, .{ "laquo", "\u{ab}" },    .{ "raquo", "\u{bb}" },
    .{ "middot", "\u{b7}" },  .{ "bull", "\u{2022}" },  .{ "times", "\u{d7}" },   .{ "copy", "\u{a9}" },     .{ "reg", "\u{ae}" },
    .{ "trade", "\u{2122}" }, .{ "deg", "\u{b0}" },     .{ "rarr", "\u{2192}" },  .{ "larr", "\u{2190}" },   .{ "zwj", "" },
    .{ "zwnj", "" },          .{ "shy", "" },
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
    const semi = std.mem.indexOfScalarPos(u8, html[0..@min(html.len, at + 12)], at, ';') orelse return null; // bounded: a bare & costs O(1)
    if (semi == at + 1) return null;
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

/// Readable text of `html`: title and main, article or body content, at most `max_chars`, less if `budget` runs out.
pub fn toText(a: Allocator, html: []const u8, max_chars: usize, budget: ?*Budget) Allocator.Error!Text {
    var title: Out = .{ .a = a };
    if (element(html, "title")) |t| try decodeInto(&title, t);
    const region = element(html, "main") orelse element(html, "article") orelse element(html, "body") orelse html;
    var o: Out = .{ .a = a };
    var pre: usize = 0;
    var i: usize = 0;
    var ent: [4]u8 = undefined;
    var stopped = false;
    while (i < region.len) {
        if (budget) |b| if (b.out()) {
            stopped = true;
            break;
        };
        const ch = region[i];
        if (ch == '&') {
            if (entity(region, i, &ent)) |e| {
                try o.text(e.text, pre > 0);
                i += e.len;
                continue;
            }
        }
        if (ch != '<') {
            const next = std.mem.indexOfAnyPos(u8, region, i + 1, "<&") orelse region.len; // past i: a bare & is text
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
        if (name.len == 0) {
            try o.text(region[i .. i + 1], pre > 0);
            i += 1;
            continue;
        }
        i = tagEnd(region, i); // only a named tag is read to its end, so a run of '<' costs one pass
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
    const named = std.mem.trim(u8, title.buf.items, " ");
    return .{ .title = named, .text = tidy[0..cut], .truncated = cut < tidy.len or stopped };
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

const stop_words = [_][]const u8{ "the", "and", "for", "with", "what", "are", "how", "this", "that", "from", "into", "use", "using", "can", "you", "your", "its", "was", "were", "has", "have", "does", "which", "about", "all", "any", "not", "but", "out", "via", "per", "new", "name", "main", "latest", "current", "please", "show", "tell", "write", "make", "get" };

/// The lowercased words of `focus` worth matching: three letters or more, no stop words, no repeats.
fn focusTerms(a: Allocator, focus: []const u8) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, focus, " \t\r\n.,;:!?()[]{}<>\"'`/=");
    while (it.next()) |w| {
        if (w.len < 3 or out.items.len >= 24) continue;
        const lower = try std.ascii.allocLowerString(a, w);
        var skip = false;
        for (stop_words) |s| skip = skip or std.mem.eql(u8, lower, s);
        for (out.items) |seen| skip = skip or std.mem.eql(u8, lower, seen);
        if (!skip) try out.append(a, lower);
    }
    return out.items;
}

/// At most `max` bytes of `text`: its opening, then the lines sharing the rarest words with `focus`, in page order.
pub fn excerpt(
    a: Allocator,
    text: []const u8,
    focus: []const u8,
    max: usize,
    budget: ?*Budget,
) Allocator.Error![]const u8 {
    if (text.len <= max) return text;
    const head_len = blk: {
        const cut = cutAt(text, max / 3);
        break :blk if (std.mem.lastIndexOfScalar(u8, text[0..cut], '\n')) |nl| nl else cut;
    };
    const terms = try focusTerms(a, focus);
    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text[head_len..], '\n');
    while (it.next()) |line| {
        if (budget) |b| if (b.out()) break; // lines past a spent budget are left out
        if (std.mem.trim(u8, line, " ").len > 0) try lines.append(a, line);
    }
    const seen_in = try a.alloc(usize, terms.len);
    @memset(seen_in, 0);
    for (lines.items) |line| for (terms, seen_in) |t, *n| {
        if (std.ascii.findIgnoreCase(line, t) != null) n.* += 1;
    };
    const scores = try a.alloc(f32, lines.items.len);
    for (lines.items, scores) |line, *score| {
        score.* = 0;
        if (budget) |b| if (b.out()) continue;
        for (terms, seen_in) |t, n| if (std.ascii.findIgnoreCase(line, t) != null) {
            score.* += 1.0 / @as(f32, @floatFromInt(n)); // a word on every line says little about any one of them
        };
        if (std.mem.startsWith(u8, line, "#")) score.* *= 1.5;
    }
    const order = try a.alloc(usize, lines.items.len);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, scores, struct {
        fn more(s: []f32, x: usize, y: usize) bool {
            return s[x] > s[y] or (s[x] == s[y] and x < y);
        }
    }.more);
    const keep = try a.alloc(bool, lines.items.len);
    @memset(keep, false);
    var used = head_len;
    var kept: std.StringHashMapUnmanaged(void) = .empty;
    for (order) |i| {
        if (scores[i] <= 0) break;
        const line = std.mem.trim(u8, lines.items[i], " ");
        if (line.len < 16 and !std.mem.startsWith(u8, line, "#")) continue; // a fragment carries little on its own
        if (kept.contains(line)) continue; // boilerplate repeats: one copy says it
        const cost = lines.items[i].len + 7; // the line and, at most, the "\n[...]\n" before it
        if (used + cost > max) continue;
        try kept.put(a, line, {});
        keep[i] = true;
        used += cost;
    }
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, text[0..head_len]);
    var gap = true;
    for (lines.items, keep) |line, k| {
        if (!k) {
            gap = true;
            continue;
        }
        try out.appendSlice(a, if (gap) "\n[...]\n" else "\n");
        try out.appendSlice(a, line);
        gap = false;
    }
    return out.items;
}

/// The page's link targets as written (`<a href>`, entities decoded), at most `max`; fragments and scripts left out.
pub fn hrefs(a: Allocator, page: []const u8, max: usize, budget: ?*Budget) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var from: usize = 0;
    var ent: [4]u8 = undefined;
    while (out.items.len < max) {
        if (budget) |b| if (b.out()) break;
        const lt = std.ascii.findIgnoreCasePos(page, from, "<a") orelse break;
        from = lt + 2;
        if (from >= page.len or !(page[from] == ' ' or page[from] == '\t' or page[from] == '\n')) continue;
        const end = tagEnd(page, lt);
        from = end; // the next link starts after this tag, so an unclosed one is read once
        const tag = page[lt..end];
        const at = std.ascii.findIgnoreCase(tag, "href=") orelse continue;
        var v = tag[at + 5 ..];
        const q: u8 = if (v.len > 0 and (v[0] == '"' or v[0] == '\'')) v[0] else ' ';
        if (q != ' ') v = v[1..];
        const stop = std.mem.indexOfAny(u8, v, if (q == ' ') " >" else &.{q}) orelse v.len;
        const raw = std.mem.trim(u8, v[0..stop], " \t\r\n");
        if (raw.len == 0 or raw[0] == '#' or std.ascii.startsWithIgnoreCase(raw, "javascript:") or std.ascii.startsWithIgnoreCase(raw, "mailto:")) continue;
        var text: std.ArrayList(u8) = .empty;
        var i: usize = 0;
        while (i < raw.len) {
            if (raw[i] == '&') if (entity(raw, i, &ent)) |e| {
                try text.appendSlice(a, e.text);
                i += e.len;
                continue;
            };
            try text.append(a, raw[i]);
            i += 1;
        }
        try out.append(a, text.items);
    }
    return out.items;
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
    const t = try toText(a, page, 10_000, null);
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

test "bare ampersands are text, and a page full of them reads in linear time" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try toText(a, "<p>Tom & Jerry && x &y; z &amp; w &#33 &</p>", 1000, null);
    try std.testing.expectEqualStrings("Tom & Jerry && x &y; z & w &#33 &", t.text);
    var big: std.ArrayList(u8) = .empty;
    try big.appendSlice(a, "<body>");
    for (0..200_000) |_| try big.appendSlice(a, "&a ");
    const many = try toText(a, big.items, 100, null);
    try std.testing.expect(many.truncated and many.text.len <= 100);
}

/// Two pages of one unit repeated, 256 KiB and 2 MiB, read for their links or as text.
const Unclosed = struct {
    pages: [2][]u8,
    links: bool,
    fn run(u: *const Unclosed, i: usize) anyerror!void {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        if (u.links) {
            _ = try hrefs(arena.allocator(), u.pages[i], 400, null);
        } else {
            _ = try toText(arena.allocator(), u.pages[i], 100, null);
        }
    }
};

test "a page of unclosed tags reads in linear time" {
    for ([_]struct { []const u8, bool }{ .{ "<", false }, .{ "<a ", true } }) |case| {
        var u: Unclosed = .{ .pages = undefined, .links = case[1] };
        for (&u.pages, [_]usize{ 256 << 10, 2 << 20 }) |*page, size| {
            page.* = try std.testing.allocator.alloc(u8, size);
            for (page.*, 0..) |*c, i| c.* = case[0][i % case[0].len];
        }
        defer for (u.pages) |page| std.testing.allocator.free(page);
        const ns = try @import("../linear_time.zig").fastest(3, &u, Unclosed.run);
        try std.testing.expect(ns[1] < 5 * std.time.ns_per_s); // 2 MiB took minutes while each '<' rescanned the rest
        try std.testing.expect(ns[1] < 24 * @max(ns[0], 1)); // 8x the bytes: near 8x the time, not a quadratic's 64x
    }
}

test "an excerpt keeps the opening and the lines about the focus, in order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(a, "# Zig release notes\nIntro about Zig.\n");
    for (0..40) |i| try text.print(a, "Zig line {d} about the build system and other work.\n", .{i});
    try text.appendSlice(a, "## std.Io\nThe new std.Io interface replaces std.io writers.\n");
    for (0..40) |i| try text.print(a, "Zig line {d} about packaging.\n", .{i + 40});
    const out = try excerpt(a, text.items, "How do I use the std.Io writer in zig?", 300, null);
    try std.testing.expect(out.len <= 300);
    try std.testing.expect(std.mem.startsWith(u8, out, "# Zig release notes\nIntro about Zig."));
    try std.testing.expect(std.mem.indexOf(u8, out, "The new std.Io interface replaces std.io writers.") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "[...]") != null);
    try std.testing.expectEqualStrings("short", try excerpt(a, "short", "x", 300, null));
}

test "link targets are read from anchors, decoded, without fragments or scripts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try hrefs(a, "<p><a href=\"/docs?a=1&amp;b=2\">x</a> <A HREF='https://o.org/'>y</A> <a href=#top>t</a> <abbr href=\"/no\">z</abbr> <a class=k href=rel/page>r</a> <a href=\"javascript:go()\">j</a></p>", 10, null);
    try std.testing.expectEqual(@as(usize, 3), got.len);
    try std.testing.expectEqualStrings("/docs?a=1&b=2", got[0]);
    try std.testing.expectEqualStrings("https://o.org/", got[1]);
    try std.testing.expectEqualStrings("rel/page", got[2]);
}

test "a spent budget stops the text and the links, keeping what was read" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var page: std.ArrayList(u8) = .empty;
    for (0..200) |i| try page.print(a, "<p>Line {d}.</p><a href=\"/p{d}\">p</a>", .{ i, i });
    const full = try toText(a, page.items, 1 << 20, null);
    var ten: Budget = .{ .steps = 10 };
    const t = try toText(a, page.items, 1 << 20, &ten);
    try std.testing.expect(t.truncated and t.text.len < full.text.len and std.mem.startsWith(u8, full.text, t.text));
    var two: Budget = .{ .steps = 2 };
    try std.testing.expectEqual(@as(usize, 2), (try hrefs(a, page.items, 400, &two)).len);
}

test "a cut never splits a character" {
    const s = "ab\u{263a}cd";
    try std.testing.expectEqual(@as(usize, 2), cutAt(s, 3));
    try std.testing.expectEqual(@as(usize, 5), cutAt(s, 5));
    try std.testing.expectEqual(s.len, cutAt(s, 100));
}

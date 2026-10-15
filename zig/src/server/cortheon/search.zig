//! Web search for Cortheon: DuckDuckGo's keyless HTML results, or Brave's search API when a key is set.
const std = @import("std");
const json = @import("../json.zig");
const html = @import("html.zig");
const Allocator = std.mem.Allocator;

pub const Result = struct { title: []const u8, url: []const u8, snippet: []const u8 };

fn unreserved(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '-' or ch == '_' or ch == '.' or ch == '~';
}

/// `text` percent-encoded for a query string value.
pub fn encode(a: Allocator, text: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (text) |ch| {
        if (unreserved(ch)) try out.append(a, ch) else if (ch == ' ') try out.append(a, '+') else try out.print(a, "%{X:0>2}", .{ch});
    }
    return out.items;
}

/// `text` with %XX escapes (and + as space) decoded.
pub fn decode(a: Allocator, text: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] == '+') {
            try out.append(a, ' ');
        } else if (text[i] == '%' and i + 2 < text.len) {
            const v = std.fmt.parseInt(u8, text[i + 1 .. i + 3], 16) catch {
                try out.append(a, '%');
                continue;
            };
            try out.append(a, v);
            i += 2;
        } else try out.append(a, text[i]);
    }
    return out.items;
}

pub fn duckduckgoUrl(a: Allocator, query: []const u8) Allocator.Error![]const u8 {
    return std.fmt.allocPrint(a, "https://html.duckduckgo.com/html/?q={s}", .{try encode(a, query)});
}

/// DuckDuckGo's lite page, asked when its HTML page refuses or comes back empty.
pub fn duckduckgoLiteUrl(a: Allocator, query: []const u8) Allocator.Error![]const u8 {
    return std.fmt.allocPrint(a, "https://lite.duckduckgo.com/lite/?q={s}", .{try encode(a, query)});
}

pub fn braveUrl(a: Allocator, query: []const u8, count: usize) Allocator.Error![]const u8 {
    return std.fmt.allocPrint(a, "https://api.search.brave.com/res/v1/web/search?q={s}&count={d}", .{ try encode(a, query), @min(count, 20) });
}

/// Text of an HTML fragment: tags dropped, entities decoded, spaces collapsed.
fn fragmentText(a: Allocator, fragment: []const u8) Allocator.Error![]const u8 {
    const t = try html.toText(a, fragment, 2000, null);
    return std.mem.trim(u8, t.text, " \n");
}

/// The target of a DuckDuckGo result link: its `uddg` parameter decoded, or the link itself.
fn resultTarget(a: Allocator, href_raw: []const u8) Allocator.Error!?[]const u8 {
    const href = try std.mem.replaceOwned(u8, a, href_raw, "&amp;", "&");
    if (std.mem.indexOf(u8, href, "duckduckgo.com/y.js") != null) return null; // an ad
    if (std.mem.indexOf(u8, href, "uddg=")) |at| {
        const start = at + 5;
        const end = std.mem.indexOfScalarPos(u8, href, start, '&') orelse href.len;
        return try decode(a, href[start..end]);
    }
    if (std.mem.startsWith(u8, href, "http://") or std.mem.startsWith(u8, href, "https://")) return href;
    return null;
}

/// How a DuckDuckGo results page marks a result's link and its snippet.
const Marks = struct { link: []const u8, snippet: []const u8, snippet_end: []const u8 };
const html_marks: Marks = .{ .link = "class=\"result__a\"", .snippet = "class=\"result__snippet\"", .snippet_end = "</a>" };
const lite_marks: Marks = .{ .link = "class='result-link'", .snippet = "class='result-snippet'", .snippet_end = "</td>" };

/// Results from DuckDuckGo's HTML page: each `result__a` link's target and title, with its `result__snippet`.
pub fn parseDuckDuckGo(a: Allocator, page: []const u8, limit: usize) Allocator.Error![]Result {
    return parseMarked(a, page, limit, html_marks);
}

/// Results from DuckDuckGo's lite page: each `result-link` and its `result-snippet` cell.
pub fn parseDuckDuckGoLite(a: Allocator, page: []const u8, limit: usize) Allocator.Error![]Result {
    return parseMarked(a, page, limit, lite_marks);
}

fn parseMarked(a: Allocator, page: []const u8, limit: usize, m: Marks) Allocator.Error![]Result {
    var out: std.ArrayList(Result) = .empty;
    const marker = m.link;
    var from: usize = 0;
    var last: usize = 0; // where the previous link ended: every search back stops there, so each byte is read once
    while (out.items.len < limit) {
        const at = std.mem.indexOfPos(u8, page, from, marker) orelse break;
        const lt = std.mem.lastIndexOfScalar(u8, page[last..at], '<') orelse { // a mark outside any tag: skip it
            from = at + marker.len;
            last = at;
            continue;
        };
        const tag_start = last + lt;
        const tag_end = std.mem.indexOfScalarPos(u8, page, at, '>') orelse break;
        const next = std.mem.indexOfPos(u8, page, tag_end, marker) orelse page.len;
        from = tag_end;
        const lo = last;
        last = tag_end;
        const block = std.mem.lastIndexOf(u8, page[lo..tag_start], "class=\"result ");
        const block_start = if (block) |b| lo + b else tag_start;
        if (std.mem.indexOf(u8, page[block_start..tag_start], "result--ad") != null) continue;
        const tag = page[tag_start..tag_end];
        const href_at = std.mem.indexOf(u8, tag, "href=\"") orelse continue;
        const href_end = std.mem.indexOfScalarPos(u8, tag, href_at + 6, '"') orelse continue;
        const url = (try resultTarget(a, tag[href_at + 6 .. href_end])) orelse continue;
        const close = std.mem.indexOfPos(u8, page[0..next], tag_end, "</a>") orelse continue; // within its own result
        const title = try fragmentText(a, page[tag_end + 1 .. close]);
        var snippet: []const u8 = "";
        if (std.mem.indexOfPos(u8, page[0..next], close, m.snippet)) |s| {
            const s_open = (std.mem.indexOfScalarPos(u8, page[0..next], s, '>') orelse s) + 1;
            // a snippet with no end (a page cut at its size cap) is left out rather than sliced backwards
            if (std.mem.indexOfPos(u8, page[0..next], s_open, m.snippet_end)) |s_close|
                snippet = try fragmentText(a, page[s_open..s_close]);
        }
        var seen = false;
        for (out.items) |r| seen = seen or std.mem.eql(u8, r.url, url);
        if (!seen and title.len > 0) try out.append(a, .{ .title = title, .url = url, .snippet = snippet });
    }
    return out.items;
}

/// Results from Brave's JSON reply: `web.results[]` titles, urls and descriptions.
pub fn parseBrave(a: Allocator, text: []const u8, limit: usize) Allocator.Error![]Result {
    const parsed = switch (try json.parseText(a, text)) {
        .ok => |v| v,
        .err => return &.{},
    };
    const web = parsed.get("web") orelse return &.{};
    const items = (web.get("results") orelse return &.{}).items() orelse return &.{};
    var out: std.ArrayList(Result) = .empty;
    for (items) |item| {
        if (out.items.len >= limit) break;
        const title = item.strField("title") orelse continue;
        const url = item.strField("url") orelse continue;
        try out.append(a, .{ .title = try fragmentText(a, title), .url = url, .snippet = try fragmentText(a, item.strField("description") orelse "") });
    }
    return out.items;
}

test "DuckDuckGo results: targets decoded, ads skipped, snippets read" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const page =
        \\<div class="result results_links result--ad"><h2><a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad=1">Buy now</a></h2></div>
        \\<div class="result results_links web-result"><h2 class="result__title">
        \\<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fziglang.org%2Fdownload%2F0.17.0%2Frelease%2Dnotes.html&amp;rut=abc">0.17.0 Release Notes &amp; More</a></h2>
        \\<a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">The <b>std.Io</b> interface &mdash; new.</a></div>
        \\<div class="result results_links web-result"><h2><a class="result__a" href="https://example.org/page">Example</a></h2></div>
    ;
    const results = try parseDuckDuckGo(a, page, 5);
    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("https://ziglang.org/download/0.17.0/release-notes.html", results[0].url);
    try std.testing.expectEqualStrings("0.17.0 Release Notes & More", results[0].title);
    try std.testing.expectEqualStrings("The std.Io interface \u{2014} new.", results[0].snippet);
    try std.testing.expectEqualStrings("https://example.org/page", results[1].url);
    try std.testing.expectEqualStrings("", results[1].snippet);
}

test "DuckDuckGo's lite page reads the same way" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const page =
        \\<tr><td>1.&nbsp;</td><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fziglang.org%2Fnews%2F&amp;rut=x" class='result-link'>Zig <b>News</b></a></td></tr>
        \\<tr><td>&nbsp;</td><td class='result-snippet'>
        \\  Release <b>notes</b> &amp; more
        \\</td></tr>
        \\<tr><td>2.&nbsp;</td><td><a rel="nofollow" href="https://example.org/two" class='result-link'>Two</a></td></tr>
    ;
    const results = try parseDuckDuckGoLite(a, page, 5);
    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("https://ziglang.org/news/", results[0].url);
    try std.testing.expectEqualStrings("Zig News", results[0].title);
    try std.testing.expectEqualStrings("Release notes & more", results[0].snippet);
    try std.testing.expectEqualStrings("Two", results[1].title);
}

test "a page cut inside a snippet keeps its result without the snippet" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const full = "<a class=\"result__a\" href=\"https://example.org/a\">A</a><a class=\"result__snippet\" href=\"x\">";
    const short = "<a href=\"https://example.org/b\" class='result-link'>B</a></td></tr><tr><td class='result-snippet'>";
    for ([_][]const u8{ full, short }, [_]bool{ false, true }) |page, is_lite| {
        const results = if (is_lite) try parseDuckDuckGoLite(a, page, 5) else try parseDuckDuckGo(a, page, 5);
        try std.testing.expectEqual(@as(usize, 1), results.len);
        try std.testing.expectEqualStrings("", results[0].snippet);
    }
}

test "Brave results and query encoding" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const results = try parseBrave(a, "{\"web\": {\"results\": [{\"title\": \"Zig <strong>0.17</strong>\", \"url\": \"https://ziglang.org/\", \"description\": \"A language\"}, {\"url\": \"https://x.org\"}]}}", 5);
    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("Zig 0.17", results[0].title);
    try std.testing.expectEqualStrings("q=zig+0.17+%22std.Io%22", (try duckduckgoUrl(a, "zig 0.17 \"std.Io\""))[std.mem.indexOf(u8, try duckduckgoUrl(a, "zig 0.17 \"std.Io\""), "q=").?..]);
    try std.testing.expectEqualStrings("a b/c", try decode(a, "a+b%2Fc"));
}

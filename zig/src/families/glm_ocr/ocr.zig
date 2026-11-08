//! GLM-OCR's OCR reader: the model card's text prompt, and the page as one block with tables and math normalized.
const std = @import("std");
const api = @import("engine_api");
const c = @import("config.zig");
const processor = @import("processor.zig");
const ocr = api.ocr;
const Allocator = std.mem.Allocator;

/// The model card's text recognition prompt.
pub const prompt = "Text Recognition:";

/// The reader the family offers (Offer.ocr).
pub const reader: ocr.Ocr = .{ .prompt = prompt, .geometry = .none, .parse = parse };

/// The prepared image the tower reads, with its frame: the resized size, one patch a grid cell.
pub fn prepared(p: processor.Prepared) api.PreparedImage {
    const gh: u32 = @intCast(p.geometry.h / c.patch);
    const gw: u32 = @intCast(p.geometry.w / c.patch);
    return .{ .pixels = p.pixels, .gh = gh, .gw = gw, .tokens = @intCast(p.geometry.tokens()), .hash = p.hash, .frame = .{ .width = gw * c.patch, .height = gh * c.patch } };
}

/// The page as one `.page` block (no geometry); empty text is an empty page. ``a`` is a request arena.
pub fn parse(a: Allocator, text: []const u8, original: api.Size, frame: api.Size) anyerror![]ocr.OcrBlock {
    _ = original;
    _ = frame;
    const page = std.mem.trim(u8, text, &std.ascii.whitespace);
    if (page.len == 0) return &.{};
    const blocks = try a.alloc(ocr.OcrBlock, 1);
    blocks[0] = .{ .content = try markdown(a, page), .quad = null, .kind = .page };
    return blocks;
}

/// Backtick runs this long or longer are literal text: the failed-closer set stays small and the scan linear.
const max_tick = 64;

/// The page with code (fenced blocks, backtick spans) verbatim; tables and math normalized in the prose between.
fn markdown(a: Allocator, page: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try split(a, &out, page, true, prose);
    return out.toOwnedSlice(a);
}

/// ``text`` to ``out``: code verbatim, the prose between through ``each``; ``tables``: a table is one prose range.
fn split(a: Allocator, out: *std.ArrayList(u8), text: []const u8, tables: bool, comptime each: anytype) !void {
    var scanner: Scanner = .{ .page = text, .unclosed_table = !tables };
    var start: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        const s = scanner.at(i);
        if (s.code) {
            try each(a, out, text[start..i]);
            try out.appendSlice(a, text[i..s.end]);
            start = s.end;
        }
        i = s.end;
    }
    try each(a, out, text[start..]);
}

const Span = struct { end: usize, code: bool };

/// A forward scan for code; every search it remembers failing covers a range no later search can reach past.
const Scanner = struct {
    page: []const u8,
    /// Backtick run lengths with no closer before ``paragraph`` (the current paragraph's end).
    failed: std.StaticBitSet(max_tick) = .empty,
    paragraph: usize = 0,
    /// No `</table>` follows a `<table` already seen.
    unclosed_table: bool = false,

    /// What starts at ``i``: code up to ``end``, or prose up to ``end`` (a byte, an unclosed run or a whole table).
    fn at(s: *Scanner, i: usize) Span {
        const page = s.page;
        if (i == 0 or page[i - 1] == '\n') if (fence(page, i)) |end| return .{ .end = end, .code = true };
        if (page[i] == '<' and !s.unclosed_table and std.mem.startsWith(u8, page[i..], "<table")) {
            // A table reaches prose whole, so its cells' backticks cannot split it.
            if (std.mem.indexOfPos(u8, page, i, "</table>")) |end| return .{ .end = end + "</table>".len, .code = false };
            s.unclosed_table = true;
        }
        if (page[i] != '`') return .{ .end = i + 1, .code = false };
        const k = run(page, i, '`');
        if (i >= s.paragraph) {
            s.paragraph = paragraphEnd(page, i);
            s.failed = .empty;
        }
        // No closer of length k before the paragraph's end after one position means none after any later one.
        if (k < max_tick and !s.failed.isSet(k)) {
            const within = page[0..s.paragraph];
            var j = i + k;
            while (std.mem.indexOfScalarPos(u8, within, j, '`')) |closer| {
                const r = run(within, closer, '`');
                if (r == k) return .{ .end = closer + r, .code = true };
                j = closer + r;
            }
            s.failed.set(k);
        }
        return .{ .end = i + k, .code = false };
    }
};

/// The start of the first line after ``i``'s line that is blank or opens a fence, or the text's end.
fn paragraphEnd(page: []const u8, i: usize) usize {
    var at = i;
    while (std.mem.indexOfScalarPos(u8, page, at, '\n')) |nl| {
        const next = nl + 1;
        const line_end = std.mem.indexOfScalarPos(u8, page, next, '\n') orelse page.len;
        if (std.mem.trim(u8, page[next..line_end], " \t\r").len == 0 or opens(page, next) != null) return next;
        at = line_end;
    }
    return page.len;
}

const Fence = struct { ch: u8, n: usize, at: usize };

/// The fence a line starting at ``i`` opens: up to 3 spaces, then 3 or more ` or ~.
fn opens(page: []const u8, i: usize) ?Fence {
    var j = i;
    while (j < page.len and j - i < 3 and page[j] == ' ') j += 1;
    if (j >= page.len or (page[j] != '`' and page[j] != '~')) return null;
    const n = run(page, j, page[j]);
    return if (n >= 3) .{ .ch = page[j], .n = n, .at = j } else null;
}

/// The end of a fenced block opening at line start ``i``: its closing fence line (3 spaces at most), or the text's end.
fn fence(page: []const u8, i: usize) ?usize {
    const f = opens(page, i) orelse return null;
    var at = std.mem.indexOfScalarPos(u8, page, f.at, '\n') orelse return page.len;
    while (true) {
        const line_end = std.mem.indexOfScalarPos(u8, page, at + 1, '\n') orelse page.len;
        const line = page[at + 1 .. line_end];
        const indent = run(line[0..@min(line.len, 3)], 0, ' ');
        const r = run(line, indent, f.ch);
        if (r >= f.n and std.mem.trim(u8, line[indent + r ..], " \t\r").len == 0) return line_end;
        if (line_end == page.len) return page.len;
        at = line_end;
    }
}

/// How many ``ch`` bytes start at ``i``.
fn run(text: []const u8, i: usize, ch: u8) usize {
    var j = i;
    while (j < text.len and text[j] == ch) j += 1;
    return j - i;
}

/// Prose with HTML tables as pipe tables (clean HTML when spanned), and math outside code as `$…$`, cells included.
fn prose(a: Allocator, out: *std.ArrayList(u8), text: []const u8) !void {
    var rest = text;
    while (std.mem.indexOf(u8, rest, "<table")) |open| {
        const end = (std.mem.indexOfPos(u8, rest, open, "</table>") orelse break) + "</table>".len;
        var html: std.ArrayList(u8) = .empty;
        try split(a, &html, rest[open..end], false, math);
        const t = ocr.html.parse(a, html.items) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Malformed => {
                // A table the subset cannot read stays as the model wrote it: raw HTML is still Markdown.
                try math(a, out, rest[0..open]);
                try out.appendSlice(a, html.items);
                rest = rest[end..];
                continue;
            },
        };
        try math(a, out, rest[0..open]);
        if (out.items.len > 0) try separate(a, out);
        try out.appendSlice(a, if (t.spanned) try ocr.html.normalized(a, t) else try ocr.html.pipe(a, t));
        rest = rest[end..];
        if (std.mem.trim(u8, rest, &std.ascii.whitespace).len > 0) {
            try separate(a, out);
            rest = std.mem.trimStart(u8, rest, &std.ascii.whitespace);
        }
    }
    try math(a, out, rest);
}

/// Ends ``out`` with one blank line, so a pipe table starts and ends a Markdown block.
fn separate(a: Allocator, out: *std.ArrayList(u8)) !void {
    out.shrinkRetainingCapacity(std.mem.trimEnd(u8, out.items, &std.ascii.whitespace).len);
    if (out.items.len > 0) try out.appendSlice(a, "\n\n");
}

/// ``text`` with each closed `\[…\]` as `$$…$$` and `\(…\)` as `$…$`; an unclosed opener stays.
fn math(a: Allocator, out: *std.ArrayList(u8), text: []const u8) !void {
    // A closer missing after one opener is missing after every later one: each closer is searched until it fails.
    var open_paren = true;
    var open_bracket = true;
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '\\' and i + 1 < text.len and (text[i + 1] == '[' or text[i + 1] == '(')) {
            const display = text[i + 1] == '[';
            const searching = if (display) &open_bracket else &open_paren;
            if (searching.*) {
                if (std.mem.indexOfPos(u8, text, i + 2, if (display) "\\]" else "\\)")) |end| {
                    const delimiter = if (display) "$$" else "$";
                    try out.appendSlice(a, delimiter);
                    try out.appendSlice(a, text[i + 2 .. end]);
                    try out.appendSlice(a, delimiter);
                    i = end + 2;
                    continue;
                }
                searching.* = false;
            }
        }
        try out.append(a, text[i]);
        i += 1;
    }
}

//! Single-call tool policy and streamed prose tag filter (parallel_tool_calls: false).
const std = @import("std");
const reply_code = @import("reply_code.zig");
const reply_text = @import("reply_text.zig");
const Allocator = std.mem.Allocator;

fn nameLike(s: []const u8) bool {
    if (s.len == 0 or !(std.ascii.isAlphabetic(s[0]) or s[0] == '_')) return false;
    for (s[1..]) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '-' or ch >= 0x80)) return false;
    return true;
}

/// ``(?:NAME:)?tool_call``, any case.
pub fn callTag(name: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(name, "tool_call")) return true;
    const colon = std.mem.indexOfScalar(u8, name, ':') orelse return false;
    return nameLike(name[0..colon]) and std.ascii.eqlIgnoreCase(name[colon + 1 ..], "tool_call");
}

/// Hides tool envelopes from streamed prose, holding only a partial tag or open code span between deltas.
pub const TextFilter = struct {
    buf: std.ArrayList(u8) = .empty,
    sent: usize = 0,
    scan: usize = 0,
    hidden: ?[2]usize = null,
    hidden_from: usize = 0,
    deferred: bool = false, // a closed envelope was hidden: later prose waits for the reply's final content
    code: reply_code.Scanner = .{},

    pub fn feed(f: *TextFilter, a: Allocator, text: []const u8) Allocator.Error![]const u8 {
        if (f.deferred) return "";
        try f.buf.appendSlice(a, text);
        return f.step(false);
    }

    pub fn finish(f: *TextFilter) []const u8 {
        const tail = if (f.hidden != null or f.deferred) "" else f.step(true);
        f.buf = .empty;
        return tail;
    }

    fn step(f: *TextFilter, finished: bool) []const u8 {
        const b = f.buf.items;
        const emit_start = f.sent;
        if (f.hidden) |tag| {
            f.matchCloser(b, tag);
            return b[emit_start..f.sent];
        }
        while (std.mem.indexOfScalarPos(u8, b, f.scan, '<')) |start| {
            switch (f.code.check(b, start, finished)) {
                .skip => |to| {
                    f.sent = @min(to, b.len);
                    f.scan = f.sent;
                    continue;
                },
                .pending => {
                    f.sent = start;
                    f.scan = start;
                    return b[emit_start..f.sent];
                },
                .outside => {},
            }
            const rest = b[start..];
            const end = std.mem.indexOfScalar(u8, rest, '>') orelse {
                if (!finished) {
                    const name = rest[1..];
                    const colon = std.mem.indexOfScalar(u8, name, ':');
                    const possible = name.len == 0 or nameLike(name) or (colon != null and nameLike(name[0..colon.?]) and
                        std.ascii.startsWithIgnoreCase("tool_call", name[colon.? + 1 ..]));
                    if (possible) {
                        f.sent = start;
                        f.scan = start;
                        return b[emit_start..f.sent];
                    }
                }
                f.sent = start + 1;
                f.scan = start + 1;
                continue;
            };
            if (callTag(rest[1..end])) {
                f.sent = start;
                f.hidden = .{ start + 1, start + end };
                f.hidden_from = start + end + 1;
                f.matchCloser(b, f.hidden.?);
                return b[emit_start..f.sent];
            }
            f.sent = start + 1;
            f.scan = start + 1;
        }
        f.sent = b.len;
        f.scan = b.len;
        return b[emit_start..f.sent];
    }

    fn matchCloser(f: *TextFilter, b: []const u8, tag: [2]usize) void {
        const name = b[tag[0]..tag[1]];
        var at = f.hidden_from;
        while (std.mem.indexOfScalarPos(u8, b, at, '<')) |i| {
            if (i + 3 + name.len > b.len) {
                f.hidden_from = i;
                return;
            }
            if (b[i + 1] == '/' and std.ascii.eqlIgnoreCase(b[i + 2 .. i + 2 + name.len], name) and b[i + 2 + name.len] == '>') {
                f.hidden = null;
                f.deferred = true;
                return;
            }
            at = i + 1;
        }
        f.hidden_from = b.len;
    }
};

/// ``parallel_tool_calls: false`` keeps one call and hides call markup from streamed prose.
pub const Policy = struct {
    single: bool = false,
    filter: TextFilter = .{},

    pub fn maxCalls(p: *const Policy) ?usize {
        return if (p.single) 1 else null;
    }

    /// Filters a str delta's text (dict deltas lose tool_calls and filter content), or passes it through.
    pub fn text(p: *Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        return if (p.single) p.filter.feed(a, s) else s;
    }

    pub fn flush(p: *Policy) []const u8 {
        return if (p.single) p.filter.finish() else "";
    }

    pub fn content(p: *const Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        if (!p.single) return s;
        var f: TextFilter = .{};
        const head = try f.feed(a, s);
        return std.mem.concat(a, u8, &.{ head, f.finish() });
    }

    /// parsed_content: parsed reply content; whole envelopes left as text kept, rest filtered; unclosed stays hidden.
    pub fn parsedContent(p: *const Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        if (!p.single) return s;
        var out: std.ArrayList(u8) = .empty;
        var cursor: usize = 0;
        var code: reply_code.Scanner = .{};
        while (try wholeEnvelope(a, s, cursor, &code)) |e| : (cursor = e.end) {
            try out.appendSlice(a, try p.content(a, s[cursor..e.start]));
            try out.appendSlice(a, s[e.start..e.end]);
        }
        try out.appendSlice(a, try p.content(a, s[cursor..]));
        return out.items;
    }

    /// Held-back malformed or unoffered call envelopes kept as reply text: ``final`` past the prose ``sent``.
    pub fn kept(p: *const Policy, final: []const u8, sent: []const u8) ?[]const u8 {
        // Replies without parsed calls retain their leading whitespace; parsed content may be stripped.
        const shown = if (std.mem.startsWith(u8, final, sent)) sent else reply_text.pyLstrip(sent);
        if (!std.mem.startsWith(u8, final, shown)) return null;
        const rest = final[shown.len..];
        if (p.single) return if (rest.len > 0) rest else null;
        return if (reply_text.pyStrip(rest).len > 0) rest else null;
    }
};

/// The first whole envelope from ``from`` (``<((?:NAME:)?tool_call)>.*?</\1>``, any case).
fn wholeEnvelope(a: Allocator, s: []const u8, from: usize, code: *reply_code.Scanner) Allocator.Error!?struct { start: usize, end: usize } {
    var at = from;
    while (std.mem.indexOfScalarPos(u8, s, at, '<')) |start| {
        if (code.check(s, start, true) == .skip) {
            at = code.skip_end;
            continue;
        }
        at = start + 1;
        const close = std.mem.indexOfScalarPos(u8, s, start, '>') orelse return null;
        if (!callTag(s[start + 1 .. close])) continue;
        const closer = try std.mem.concat(a, u8, &.{ "</", s[start + 1 .. close], ">" });
        const end = std.ascii.findIgnoreCasePos(s, close + 1, closer) orelse continue;
        code.jumpTo(end + closer.len);
        return .{ .start = start, .end = end + closer.len };
    }
    return null;
}

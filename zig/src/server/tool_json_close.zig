//! A legacy envelope closer outside a JSON-led payload's strings, retained between stream feeds.
const std = @import("std");

const tag = "</tool_call>";

pub const Closer = struct {
    at: usize,
    json: ?bool = null, // decided by the payload's first non-blank byte, once one has arrived
    in_string: bool = false,
    escaped: bool = false,

    /// The closer's start in `text` (the reply so far), or null until it has arrived whole.
    pub fn next(c: *Closer, text: []const u8) ?usize {
        if (c.json == null) {
            while (c.at < text.len and (text[c.at] == ' ' or text[c.at] == '\t' or text[c.at] == '\r' or text[c.at] == '\n')) c.at += 1;
            if (c.at >= text.len) return null; // nothing to decide on yet
            c.json = text[c.at] == '{' or text[c.at] == '[';
        }
        if (!c.json.?) {
            // not JSON: the first closer anywhere; a feed resumes where a partial closer may begin
            if (std.ascii.findIgnoreCasePos(text, c.at, tag)) |at| return at;
            c.at = @max(c.at, text.len -| (tag.len - 1));
            return null;
        }
        while (c.at < text.len) : (c.at += 1) {
            const ch = text[c.at];
            if (c.in_string) {
                if (c.escaped) {
                    c.escaped = false;
                } else if (ch == '\\') {
                    c.escaped = true;
                } else if (ch == '"') {
                    c.in_string = false;
                }
            } else if (ch == '"') {
                c.in_string = true;
            } else if (ch == '<') {
                const rest = text[c.at..];
                // a closer cut by the feed's end is looked at again with the next feed, not stepped over
                if (rest.len < tag.len) {
                    if (std.ascii.startsWithIgnoreCase(tag, rest)) return null;
                } else if (std.ascii.startsWithIgnoreCase(rest, tag)) return c.at;
            }
        }
        return null;
    }
};

pub fn find(text: []const u8, at: usize) ?usize {
    var closer: Closer = .{ .at = at };
    return closer.next(text);
}

test "the closer: found whole or fed a byte at a time, JSON-led or not, never inside a JSON string" {
    const t = std.testing;
    const cases = [_]struct { text: []const u8, want: ?usize }{
        .{ .text = "{\"a\":\"</tool_call>\"}</tool_call>", .want = 20 },
        .{ .text = "  {\"q\":1}\n</TOOL_CALL>", .want = 10 },
        .{ .text = "<function=other>\n\"x\n</function>\n</tool_call>", .want = 32 },
        .{ .text = "lookup<arg_key>q</arg_key><arg_value>\"</arg_value></tool_call>", .want = 50 },
        .{ .text = "{\"q\":1} 1 < 2 </tool_call>", .want = 14 },
        .{ .text = "{\"q\":\"open", .want = null },
        .{ .text = "   ", .want = null },
    };
    for (cases) |case| {
        try t.expectEqual(case.want, find(case.text, 0));
        var c: Closer = .{ .at = 0 };
        var got: ?usize = null;
        for (1..case.text.len + 1) |n| {
            got = c.next(case.text[0..n]);
            if (got != null) break;
        }
        try t.expectEqual(case.want, got);
    }
}

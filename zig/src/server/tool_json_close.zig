//! A legacy envelope closer outside a JSON-led payload's strings, retained between stream feeds.
const std = @import("std");

pub const Closer = struct {
    at: usize,
    decided: bool = false,
    in_string: bool = false,
    escaped: bool = false,

    pub fn next(c: *Closer, text: []const u8) ?usize {
        if (!c.decided) {
            while (c.at < text.len and (text[c.at] == ' ' or text[c.at] == '\t' or text[c.at] == '\r' or text[c.at] == '\n')) c.at += 1;
            const json = c.at < text.len and (text[c.at] == '{' or text[c.at] == '[');
            c.decided = true;
            if (!json) return std.ascii.findIgnoreCasePos(text, c.at, "</tool_call>");
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
            } else if (ch == '<' and std.ascii.startsWithIgnoreCase(text[c.at..], "</tool_call>")) {
                return c.at;
            }
        }
        return null;
    }
};

pub fn find(text: []const u8, at: usize) ?usize {
    var closer: Closer = .{ .at = at };
    return closer.next(text);
}

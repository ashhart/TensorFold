//! A tool call's arguments that max_tokens cut, closed as far as written: a whole Messages reply needs an input object.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;
const Value = json.Value;

const max_depth = 64;
const kept = 8; // the last points a cut may close at, newest first when tried

/// A place the text can be closed: its length, the closers it needs, a cut string to end first.
const Point = struct { len: usize, closers: [max_depth]u8 = undefined, n: usize = 0, string: bool = false };

/// `text`, a cut JSON object, closed where it parses: in its cut string value, else after its last whole member.
pub fn close(a: Allocator, text: []const u8) Allocator.Error!?Value {
    var stack: [max_depth]u8 = undefined;
    var keys: [max_depth]bool = undefined; // an object's next string is a key
    var depth: usize = 0;
    var in_string = false;
    var escape = false;
    var key_string = false;
    var ring: [kept]Point = undefined;
    var count: usize = 0;
    const start = std.mem.indexOfNone(u8, text, " \t\r\n") orelse return null;
    if (text[start] != '{') return null;
    for (text[start..], start..) |ch, i| {
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
                continue;
            } else if (ch == '"') {
                in_string = false;
                if (!key_string) mark(&ring, &count, i + 1, stack[0..depth], false);
                continue;
            }
            if (!key_string) mark(&ring, &count, i + 1, stack[0..depth], true); // a cut escape or UTF-8 fails to parse
            continue;
        }
        switch (ch) {
            '{', '[' => {
                if (depth == max_depth) return null;
                stack[depth] = if (ch == '{') '}' else ']';
                keys[depth] = ch == '{';
                depth += 1;
                mark(&ring, &count, i + 1, stack[0..depth], false);
            },
            '}', ']' => {
                if (depth == 0) return null;
                depth -= 1;
                mark(&ring, &count, i + 1, stack[0..depth], false);
            },
            '"' => {
                in_string = true;
                key_string = depth > 0 and keys[depth - 1];
                if (!key_string) mark(&ring, &count, i + 1, stack[0..depth], true);
            },
            ':' => if (depth > 0) {
                keys[depth - 1] = false;
            },
            ',' => if (depth > 0) {
                mark(&ring, &count, i, stack[0..depth], false); // a whole member or element before it
                keys[depth - 1] = stack[depth - 1] == '}';
            },
            else => {},
        }
    }
    var tried: usize = 0;
    while (tried < @min(count, kept)) : (tried += 1) {
        const p = &ring[(count - 1 - tried) % kept];
        var buf: std.ArrayList(u8) = .empty;
        try buf.appendSlice(a, text[0..p.len]);
        if (p.string) try buf.append(a, '"');
        var j = p.n;
        while (j > 0) : (j -= 1) try buf.append(a, p.closers[j - 1]);
        switch (try json.parseText(a, buf.items)) {
            .ok => |v| if (v == .object) return v,
            .err => {},
        }
    }
    return null;
}

fn mark(ring: *[kept]Point, count: *usize, len: usize, open: []const u8, string: bool) void {
    const p = &ring[count.* % kept];
    p.* = .{ .len = len, .n = open.len, .string = string };
    @memcpy(p.closers[0..open.len], open);
    count.* += 1;
}

fn expectClosed(want: ?[]const u8, text: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try close(a, text);
    if (want == null) return std.testing.expect(got == null);
    try std.testing.expectEqualStrings(want.?, try json.stringify(a, got orelse return error.TestExpectedClosed, .{ .ascii = false, .compact = true }));
}

test "cut arguments close inside the cut string value, or after the last whole member" {
    try expectClosed("{\"path\":\"notes.txt\",\"content\":\"The history of\"}", "{\"path\": \"notes.txt\", \"content\": \"The history of");
    try expectClosed("{\"path\":\"notes.txt\"}", "{\"path\": \"notes.txt\", \"cont");
    try expectClosed("{\"path\":\"notes.txt\"}", "{\"path\": \"notes.txt\", \"content\":");
    try expectClosed("{\"path\":\"notes.txt\"}", "{\"path\": \"notes.txt\", \"n\": 12");
    try expectClosed("{\"a\":[1,{\"b\":\"x\"}]}", "{\"a\": [1, {\"b\": \"x\"}");
    try expectClosed("{\"a\":[1,2]}", "{\"a\": [1, 2, tr");
    try expectClosed("{\"s\":\"line\\n\"}", "{\"s\": \"line\\n\\");
    try expectClosed("{\"s\":\"x\"}", "{\"s\": \"x\\u00");
    try expectClosed("{}", "{");
    try expectClosed(null, "");
    try expectClosed(null, "[1, 2");
}

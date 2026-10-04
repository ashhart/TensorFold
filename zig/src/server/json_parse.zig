//! ``json.loads`` on request bytes, with CPython's decoder and scanner error messages.
const std = @import("std");
const json = @import("json.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const Result = union(enum) {
    ok: Value,
    err: []const u8, // ``str(exc)`` as Python words it
};

const max_depth = 900; // nesting past this answers as Python's RecursionError does

const Failure = error{ Syntax, OutOfMemory };

/// Decode ``bytes`` as UTF-8 (a BOM dropped, surrogates passed) and parse one JSON document.
pub fn parse(a: Allocator, bytes: []const u8) Allocator.Error!Result {
    var text = bytes;
    if (std.mem.startsWith(u8, text, "\xef\xbb\xbf")) text = text[3..];
    if (try utf8Error(a, text, if (text.ptr == bytes.ptr) 0 else 3)) |message| return .{ .err = message };
    var p: Parser = .{ .a = a, .s = text };
    const start = p.skip(0);
    const value = p.value(start, 0) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Syntax => return .{ .err = p.message },
    };
    const end = p.skip(p.pos);
    if (end != text.len) return .{ .err = try p.fail("Extra data", end) };
    return .{ .ok = value };
}

/// ``json.loads`` on a Python str: a leading BOM is refused, as for text that was already decoded.
pub fn parseText(a: Allocator, text: []const u8) Allocator.Error!Result {
    if (std.mem.startsWith(u8, text, "\xef\xbb\xbf")) return .{ .err = "Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)" };
    return parse(a, text);
}

/// CPython's UnicodeDecodeError text for the first bad sequence, or null when ``s`` decodes.
fn utf8Error(a: Allocator, s: []const u8, offset: usize) Allocator.Error!?[]const u8 {
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c < 0x80) {
            i += 1;
            continue;
        }
        const need: usize = if (c >= 0xc2 and c <= 0xdf) 1 else if (c >= 0xe0 and c <= 0xef) 2 else if (c >= 0xf0 and c <= 0xf4) 3 else 0;
        if (need == 0) return try decodeError(a, c, offset + i, offset + i + 1, "invalid start byte");
        var k: usize = 1;
        while (k <= need) : (k += 1) {
            if (i + k >= s.len) return try decodeError(a, c, offset + i, offset + s.len, "unexpected end of data");
            const b = s[i + k];
            const lo: u8, const hi: u8 = if (k == 1) switch (c) {
                0xe0 => .{ 0xa0, 0xbf },
                0xf0 => .{ 0x90, 0xbf },
                0xf4 => .{ 0x80, 0x8f },
                else => .{ 0x80, 0xbf },
            } else .{ 0x80, 0xbf };
            if (b < lo or b > hi) return try decodeError(a, c, offset + i, offset + i + k, "invalid continuation byte");
        }
        i += need + 1;
    }
    return null;
}

fn decodeError(a: Allocator, byte: u8, start: usize, end: usize, reason: []const u8) Allocator.Error![]const u8 {
    if (end - start == 1) return try std.fmt.allocPrint(a, "'utf-8' codec can't decode byte 0x{x:0>2} in position {d}: {s}", .{ byte, start, reason });
    return try std.fmt.allocPrint(a, "'utf-8' codec can't decode bytes in position {d}-{d}: {s}", .{ start, end - 1, reason });
}

const Parser = struct {
    a: Allocator,
    s: []const u8,
    pos: usize = 0,
    message: []const u8 = "",

    fn skip(p: *const Parser, at: usize) usize {
        var i = at;
        while (i < p.s.len and (p.s[i] == ' ' or p.s[i] == '\t' or p.s[i] == '\n' or p.s[i] == '\r')) i += 1;
        return i;
    }

    /// Records Python's ``msg: line L column C (char N)`` for a failure at byte ``at``.
    fn fail(p: *Parser, msg: []const u8, at: usize) Allocator.Error![]const u8 {
        var chars: usize = 0;
        var line: usize = 1;
        var last_newline: ?usize = null;
        for (p.s[0..at]) |b| {
            if (b & 0xc0 == 0x80) continue;
            if (b == '\n') {
                line += 1;
                last_newline = chars;
            }
            chars += 1;
        }
        const column = if (last_newline) |n| chars - n else chars + 1;
        p.message = try std.fmt.allocPrint(p.a, "{s}: line {d} column {d} (char {d})", .{ msg, line, column, chars });
        return p.message;
    }

    fn syntax(p: *Parser, msg: []const u8, at: usize) Failure {
        _ = try p.fail(msg, at);
        return error.Syntax;
    }

    fn value(p: *Parser, at: usize, depth: usize) Failure!Value {
        if (at >= p.s.len) return p.syntax("Expecting value", at);
        switch (p.s[at]) {
            '"' => return .{ .string = try p.string(at + 1) },
            '{' => {
                if (depth >= max_depth) return p.recursion("object");
                return p.object(at + 1, depth + 1);
            },
            '[' => {
                if (depth >= max_depth) return p.recursion("array");
                return p.array(at + 1, depth + 1);
            },
            'n' => return p.word(at, "null", .null),
            't' => return p.word(at, "true", .{ .bool = true }),
            'f' => return p.word(at, "false", .{ .bool = false }),
            'N' => return p.word(at, "NaN", .{ .float = std.math.nan(f64) }),
            'I' => return p.word(at, "Infinity", .{ .float = std.math.inf(f64) }),
            '-' => {
                if (std.mem.startsWith(u8, p.s[at..], "-Infinity")) {
                    p.pos = at + 9;
                    return .{ .float = -std.math.inf(f64) };
                }
                return p.numberAt(at);
            },
            else => return p.numberAt(at),
        }
    }

    fn recursion(p: *Parser, kind: []const u8) Failure {
        p.message = try std.fmt.allocPrint(p.a, "maximum recursion depth exceeded while decoding a JSON {s} from a unicode string", .{kind});
        return error.Syntax;
    }

    fn word(p: *Parser, at: usize, text: []const u8, v: Value) Failure!Value {
        if (!std.mem.startsWith(u8, p.s[at..], text)) return p.syntax("Expecting value", at);
        p.pos = at + text.len;
        return v;
    }

    fn numberAt(p: *Parser, start: usize) Failure!Value {
        const s = p.s;
        var i = start;
        if (i < s.len and s[i] == '-') i += 1;
        if (i < s.len and s[i] >= '1' and s[i] <= '9') {
            i += 1;
            while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        } else if (i < s.len and s[i] == '0') {
            i += 1;
        } else return p.syntax("Expecting value", start);
        var is_float = false;
        if (i + 1 < s.len and s[i] == '.' and std.ascii.isDigit(s[i + 1])) {
            is_float = true;
            i += 2;
            while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        }
        if (i + 1 < s.len and (s[i] == 'e' or s[i] == 'E')) {
            const e_start = i;
            i += 1;
            if (i + 1 < s.len and (s[i] == '-' or s[i] == '+')) i += 1;
            while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
            if (std.ascii.isDigit(s[i - 1])) is_float = true else i = e_start;
        }
        p.pos = i;
        const text = s[start..i];
        if (is_float) return .{ .float = std.fmt.parseFloat(f64, text) catch unreachable };
        return .{ .int = try canonicalInt(p, text) };
    }

    fn canonicalInt(p: *Parser, text: []const u8) Failure![]const u8 {
        const digits = if (text[0] == '-') text.len - 1 else text.len;
        if (digits > 4300) {
            p.message = try std.fmt.allocPrint(p.a, "Exceeds the limit (4300 digits) for integer string conversion: value has {d} digits; use sys.set_int_max_str_digits() to increase the limit", .{digits});
            return error.Syntax;
        }
        return if (std.mem.eql(u8, text, "-0")) "0" else text;
    }

    fn string(p: *Parser, begin: usize) Failure![]const u8 {
        const s = p.s;
        var out: std.ArrayList(u8) = .empty;
        var i = begin;
        while (true) {
            const chunk_start = i;
            while (i < s.len and s[i] != '"' and s[i] != '\\' and s[i] >= 0x20) i += 1;
            if (i >= s.len) return p.syntax("Unterminated string starting at", begin - 1);
            if (s[i] < 0x20) return p.syntax("Invalid control character at", i);
            if (s[i] == '"') {
                if (out.items.len == 0) {
                    p.pos = i + 1;
                    return s[chunk_start..i]; // no escapes: the input slice itself
                }
                try out.appendSlice(p.a, s[chunk_start..i]);
                p.pos = i + 1;
                return out.items;
            }
            try out.appendSlice(p.a, s[chunk_start..i]);
            i += 1; // the backslash
            if (i >= s.len) return p.syntax("Unterminated string starting at", begin - 1);
            const c = s[i];
            if (c != 'u') {
                const mapped: u8 = switch (c) {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'b' => 8,
                    'f' => 12,
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    else => return p.syntax("Invalid \\escape", i - 1),
                };
                try out.append(p.a, mapped);
                i += 1;
                continue;
            }
            const u_at = i;
            i += 1;
            if (i + 4 >= s.len) return p.syntax("Invalid \\uXXXX escape", u_at);
            var code: u21 = hex4(s[i..][0..4]) orelse return p.syntax("Invalid \\uXXXX escape", u_at);
            i += 4;
            if (code >= 0xd800 and code <= 0xdbff and i + 6 < s.len and s[i] == '\\' and s[i + 1] == 'u') {
                const low = hex4(s[i + 2 ..][0..4]) orelse return p.syntax("Invalid \\uXXXX escape", i + 1);
                if (low >= 0xdc00 and low <= 0xdfff) {
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                    i += 6;
                }
            }
            var buf: [4]u8 = undefined;
            const n = wtf8Encode(code, &buf);
            try out.appendSlice(p.a, buf[0..n]);
        }
    }

    fn object(p: *Parser, after_brace: usize, depth: usize) Failure!Value {
        const o = try json.newObject(p.a);
        var i = p.skip(after_brace);
        if (i < p.s.len and p.s[i] == '}') {
            p.pos = i + 1;
            return .{ .object = o };
        }
        while (true) {
            if (i >= p.s.len or p.s[i] != '"') return p.syntax("Expecting property name enclosed in double quotes", i);
            const key = try p.string(i + 1);
            i = p.skip(p.pos);
            if (i >= p.s.len or p.s[i] != ':') return p.syntax("Expecting ':' delimiter", i);
            i = p.skip(i + 1);
            const v = try p.value(i, depth);
            try o.put(p.a, key, v); // a repeated key keeps its first place and its last value, as dict() does
            i = p.skip(p.pos);
            if (i < p.s.len and p.s[i] == '}') {
                p.pos = i + 1;
                return .{ .object = o };
            }
            if (i >= p.s.len or p.s[i] != ',') return p.syntax("Expecting ',' delimiter", i);
            const comma = i;
            i = p.skip(i + 1);
            if (i < p.s.len and p.s[i] == '}') return p.syntax("Illegal trailing comma before end of object", comma);
        }
    }

    fn array(p: *Parser, after_bracket: usize, depth: usize) Failure!Value {
        var list: std.ArrayList(Value) = .empty;
        var i = p.skip(after_bracket);
        if (i < p.s.len and p.s[i] == ']') {
            p.pos = i + 1;
            return .{ .array = &.{} };
        }
        while (true) {
            try list.append(p.a, try p.value(i, depth));
            i = p.skip(p.pos);
            if (i < p.s.len and p.s[i] == ']') {
                p.pos = i + 1;
                return .{ .array = list.items };
            }
            if (i >= p.s.len or p.s[i] != ',') return p.syntax("Expecting ',' delimiter", i);
            const comma = i;
            i = p.skip(i + 1);
            if (i < p.s.len and p.s[i] == ']') return p.syntax("Illegal trailing comma before end of array", comma);
        }
    }
};

fn hex4(h: *const [4]u8) ?u21 {
    var v: u21 = 0;
    for (h) |c| v = v * 16 + (std.fmt.charToDigit(c, 16) catch return null);
    return v;
}

/// UTF-8, with a lone surrogate written as its three bytes (WTF-8) as Python's str keeps it.
pub fn wtf8Encode(code: u21, buf: *[4]u8) usize {
    if (code < 0x80) {
        buf[0] = @intCast(code);
        return 1;
    }
    if (code < 0x800) {
        buf[0] = @intCast(0xc0 | (code >> 6));
        buf[1] = @intCast(0x80 | (code & 0x3f));
        return 2;
    }
    if (code < 0x10000) {
        buf[0] = @intCast(0xe0 | (code >> 12));
        buf[1] = @intCast(0x80 | ((code >> 6) & 0x3f));
        buf[2] = @intCast(0x80 | (code & 0x3f));
        return 3;
    }
    buf[0] = @intCast(0xf0 | (code >> 18));
    buf[1] = @intCast(0x80 | ((code >> 12) & 0x3f));
    buf[2] = @intCast(0x80 | ((code >> 6) & 0x3f));
    buf[3] = @intCast(0x80 | (code & 0x3f));
    return 4;
}

test "python messages" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "", "Expecting value: line 1 column 1 (char 0)" },
        .{ "{\"a\":1,}", "Illegal trailing comma before end of object: line 1 column 7 (char 6)" },
        .{ "[1,\n 2,\n x]", "Expecting value: line 3 column 2 (char 9)" },
        .{ "\"\\u12\"", "Invalid \\uXXXX escape: line 1 column 3 (char 2)" },
        .{ "\"\xe2\x82\"", "'utf-8' codec can't decode bytes in position 1-2: invalid continuation byte" },
        .{ "\xff", "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte" },
        .{ "1.", "Extra data: line 1 column 2 (char 1)" },
        .{ "\"\xc3\xa9\\x\"", "Invalid \\escape: line 1 column 3 (char 2)" },
    };
    for (cases) |c| {
        const r = try parse(a, c[0]);
        try std.testing.expectEqualStrings(c[1], r.err);
    }
}

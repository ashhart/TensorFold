//! A strict subset of TOML 1.0 for flat settings files: top-level ``key = value`` lines, the rest refused by line.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// A number keeps its text, ready to be a flag's value: no ``_``, and no leading ``+`` on an integer.
pub const Scalar = union(enum) { string: []const u8, integer: []const u8, float: []const u8, boolean: bool };
pub const Value = union(enum) { scalar: Scalar, array: []const []const u8 };
pub const Entry = struct { key: []const u8, line: u32, value: Value };

/// Why a text was refused and on which line (1-based).
pub const Problem = struct { line: u32 = 0, message: []const u8 = "" };

pub const Error = error{ Invalid, OutOfMemory };

/// Top-level ``key = value`` lines in file order; every text this accepts is valid TOML 1.0 with the same meaning.
pub fn parse(a: Allocator, text: []const u8, problem: *Problem) Error![]const Entry {
    var r: Reader = .{ .a = a, .text = text, .problem = problem };
    if (std.mem.startsWith(u8, text, "\xef\xbb\xbf")) return r.fail("a byte order mark is not TOML 1.0: save the file without one");
    if (!std.unicode.utf8ValidateSlice(text)) {
        var lines = std.mem.splitScalar(u8, text, '\n'); // a line feed is never inside a multi-byte sequence
        while (lines.next()) |l| : (r.line += 1) if (!std.unicode.utf8ValidateSlice(l)) break;
        return r.fail("invalid UTF-8");
    }
    var out: std.ArrayList(Entry) = .empty;
    while (true) {
        r.blanks();
        if (r.done()) break;
        if (try r.lineEnd()) continue;
        if (r.peek() == '#') {
            try r.comment();
            continue;
        }
        if (r.peek() == '[') return r.fail("tables are not read: every key is a top-level setting");
        const line = r.line;
        const key = try r.key();
        for (out.items) |e| if (std.mem.eql(u8, e.key, key)) return r.fail("key defined twice");
        r.blanks();
        if (r.peek() == '.') return r.fail("dotted keys are not read: every key is a top-level setting");
        if (r.peek() != '=') return r.fail("expected '=' after the key");
        r.at += 1;
        r.blanks();
        const value = try r.value();
        r.blanks();
        if (r.peek() == '#') try r.comment();
        if (!r.done() and !try r.lineEnd()) return r.fail("unexpected text after the value");
        try out.append(a, .{ .key = key, .line = line, .value = value });
    }
    return out.items;
}

const Reader = struct {
    a: Allocator,
    text: []const u8,
    problem: *Problem,
    at: usize = 0,
    line: u32 = 1,

    fn fail(r: *Reader, message: []const u8) Error {
        r.problem.* = .{ .line = r.line, .message = message };
        return error.Invalid;
    }

    fn done(r: *const Reader) bool {
        return r.at >= r.text.len;
    }

    /// The byte at the cursor, or 0 at the end of the text (a NUL in the text is refused wherever it can occur).
    fn peek(r: *const Reader) u8 {
        return if (r.at < r.text.len) r.text[r.at] else 0;
    }

    fn blanks(r: *Reader) void {
        while (r.peek() == ' ' or r.peek() == '\t') r.at += 1;
    }

    /// Consumes a line end (LF or CRLF) at the cursor; a CR alone is refused.
    fn lineEnd(r: *Reader) Error!bool {
        if (r.peek() == '\r') {
            if (r.at + 1 >= r.text.len or r.text[r.at + 1] != '\n') return r.fail("a carriage return without a line feed");
            r.at += 1;
        }
        if (r.peek() != '\n' or r.done()) return false;
        r.at += 1;
        r.line += 1;
        return true;
    }

    fn control(c: u8) bool {
        return (c < 0x20 and c != '\t') or c == 0x7f;
    }

    /// From ``#`` to the line end, which stays unread.
    fn comment(r: *Reader) Error!void {
        while (!r.done() and r.peek() != '\n' and !(r.peek() == '\r' and r.at + 1 < r.text.len and r.text[r.at + 1] == '\n')) : (r.at += 1)
            if (control(r.peek())) return r.fail("a control character in a comment");
    }

    fn key(r: *Reader) Error![]const u8 {
        if (r.peek() == '"' or r.peek() == '\'') return r.string();
        const start = r.at;
        while (std.ascii.isAlphanumeric(r.peek()) or r.peek() == '_' or r.peek() == '-') r.at += 1;
        if (r.at == start) return r.fail("expected a key");
        return r.text[start..r.at];
    }

    fn value(r: *Reader) Error!Value {
        if (r.peek() == '{') return r.fail("inline tables are not read");
        if (r.peek() != '[') return .{ .scalar = try r.scalar() };
        r.at += 1;
        var items: std.ArrayList([]const u8) = .empty;
        while (true) {
            try r.gap();
            if (r.done()) return r.fail("unterminated array");
            if (r.peek() == ']') break;
            if (r.peek() == '[' or r.peek() == '{') return r.fail("nested arrays and inline tables are not read");
            if (r.peek() != '"' and r.peek() != '\'') return r.fail("an array holds strings only: quote each item");
            try items.append(r.a, try r.string());
            try r.gap();
            if (r.peek() == ',') r.at += 1 else if (r.peek() != ']') return r.fail(if (r.done()) "unterminated array" else "expected ',' or ']' in the array");
        }
        r.at += 1;
        return .{ .array = items.items };
    }

    /// Blanks, line ends and comments between an array's items.
    fn gap(r: *Reader) Error!void {
        while (true) {
            r.blanks();
            if (r.peek() == '#') try r.comment() else if (!try r.lineEnd()) return;
        }
    }

    fn scalar(r: *Reader) Error!Scalar {
        if (r.peek() == '"' or r.peek() == '\'') return .{ .string = try r.string() };
        const start = r.at;
        while (!r.done() and std.mem.indexOfScalar(u8, " \t\r\n,]#", r.peek()) == null) r.at += 1;
        const token = r.text[start..r.at];
        if (token.len == 0) return r.fail("expected a value");
        if (std.mem.eql(u8, token, "true")) return .{ .boolean = true };
        if (std.mem.eql(u8, token, "false")) return .{ .boolean = false };
        const kind = number(token) orelse
            return r.fail("not a string, a decimal number, true or false (dates, times and 0x, 0o and 0b integers are not read)");
        const body = if (kind == .integer and token[0] == '+') token[1..] else token;
        const clean = try r.a.alloc(u8, body.len - std.mem.count(u8, body, "_"));
        var n: usize = 0;
        for (body) |c| if (c != '_') {
            clean[n] = c;
            n += 1;
        };
        return if (kind == .integer) .{ .integer = clean } else .{ .float = clean };
    }

    /// A basic (``"``) or literal (``'``) string on one line, decoded.
    fn string(r: *Reader) Error![]const u8 {
        const quote = r.peek();
        if (std.mem.startsWith(u8, r.text[r.at..], if (quote == '"') "\"\"\"" else "'''")) return r.fail("multi-line strings are not read");
        r.at += 1;
        var out: std.ArrayList(u8) = .empty;
        while (true) : (r.at += 1) {
            if (r.done() or r.peek() == '\n' or r.peek() == '\r') return r.fail("unterminated string");
            const c = r.peek();
            if (c == quote) break;
            if (control(c)) return r.fail("a control character in a string: write it as an escape");
            if (c != '\\' or quote == '\'') {
                try out.append(r.a, c);
                continue;
            }
            r.at += 1;
            const simple: ?u8 = switch (r.peek()) {
                'b' => 0x08,
                't' => '\t',
                'n' => '\n',
                'f' => 0x0c,
                'r' => '\r',
                '"' => '"',
                '\\' => '\\',
                else => null,
            };
            if (simple) |s| {
                try out.append(r.a, s);
                continue;
            }
            const width: usize = switch (r.peek()) {
                'u' => 4,
                'U' => 8,
                else => return r.fail("unknown escape in a string (TOML has \\b \\t \\n \\f \\r \\\" \\\\ \\uXXXX \\UXXXXXXXX)"),
            };
            if (r.at + width >= r.text.len) return r.fail("unterminated string");
            const hex = r.text[r.at + 1 .. r.at + 1 + width];
            for (hex) |h| if (!std.ascii.isHex(h)) return r.fail(bad_escape);
            const code = std.fmt.parseInt(u21, hex, 16) catch return r.fail(bad_escape);
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(code, &buf) catch return r.fail(bad_escape);
            try out.appendSlice(r.a, buf[0..len]);
            r.at += width;
        }
        r.at += 1;
        return out.items;
    }
};

const bad_escape = "a \\u or \\U escape needs 4 or 8 hexadecimal digits of a Unicode scalar value";

const Number = enum { integer, float };

/// TOML 1.0's decimal integer or float, or null: a sign, digits with ``_`` only between two, fraction, exponent.
fn number(token: []const u8) ?Number {
    var s = token;
    if (s.len > 0 and (s[0] == '+' or s[0] == '-')) s = s[1..];
    if (std.mem.eql(u8, s, "inf") or std.mem.eql(u8, s, "nan")) return .float;
    const whole_end = std.mem.indexOfAny(u8, s, ".eE") orelse s.len;
    const whole = s[0..whole_end];
    if (!digits(whole) or (whole.len > 1 and whole[0] == '0')) return null;
    var rest = s[whole_end..];
    if (rest.len == 0) return .integer;
    if (rest[0] == '.') {
        const frac_end = std.mem.indexOfAny(u8, rest, "eE") orelse rest.len;
        if (!digits(rest[1..frac_end])) return null;
        rest = rest[frac_end..];
        if (rest.len == 0) return .float;
    }
    rest = rest[1..]; // past the e
    if (rest.len > 0 and (rest[0] == '+' or rest[0] == '-')) rest = rest[1..];
    return if (digits(rest)) .float else null;
}

/// One or more digits, with single underscores only between two digits.
fn digits(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s, 0..) |c, i| {
        if (std.ascii.isDigit(c)) continue;
        if (c != '_' or i == 0 or i + 1 == s.len or !std.ascii.isDigit(s[i - 1]) or !std.ascii.isDigit(s[i + 1])) return false;
    }
    return true;
}

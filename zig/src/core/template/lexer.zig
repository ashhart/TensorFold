//! Jinja2 tokenizer with Hugging Face's settings: trim_blocks, lstrip_blocks and no trailing newline.
const std = @import("std");
const uni = @import("unicode.zig");
const Allocator = std.mem.Allocator;

pub const Op = enum { add, sub, div, floordiv, mul, mod, pow, tilde, lbracket, rbracket, lparen, rparen, lbrace, rbrace, eq, ne, gt, gteq, lt, lteq, assign, dot, colon, pipe, comma, semicolon };

pub const Tag = enum { data, var_begin, var_end, block_begin, block_end, name, string, integer, float, op, eof };

pub const Token = struct {
    tag: Tag,
    text: []const u8 = "",
    op: Op = .add,
    int: i64 = 0,
    float: f64 = 0,
    line: u32 = 1,
};

pub const Error = error{ SyntaxError, OutOfMemory };

const ops = [_]struct { []const u8, Op }{
    .{ "//", .floordiv }, .{ "**", .pow },      .{ "==", .eq },    .{ "!=", .ne },    .{ ">=", .gteq },  .{ "<=", .lteq },
    .{ "+", .add },       .{ "-", .sub },       .{ "/", .div },    .{ "*", .mul },    .{ "%", .mod },    .{ "~", .tilde },
    .{ "[", .lbracket },  .{ "]", .rbracket },  .{ "(", .lparen }, .{ ")", .rparen }, .{ "{", .lbrace }, .{ "}", .rbrace },
    .{ ">", .gt },        .{ "<", .lt },        .{ "=", .assign }, .{ ".", .dot },    .{ ":", .colon },  .{ "|", .pipe },
    .{ ",", .comma },     .{ ";", .semicolon },
};

pub const Lexer = struct {
    a: Allocator,
    s: []const u8,
    pos: usize = 0,
    line: u32 = 1,
    line_starting: bool = true,
    tokens: std.ArrayList(Token) = .empty,
    msg: []const u8 = "",

    fn fail(l: *Lexer, comptime fmt: []const u8, args: anytype) Error {
        l.msg = std.fmt.allocPrint(l.a, "line {d}: " ++ fmt, .{l.line} ++ args) catch return error.OutOfMemory;
        return error.SyntaxError;
    }

    fn emit(l: *Lexer, t: Token) !void {
        var tok = t;
        tok.line = l.line;
        try l.tokens.append(l.a, tok);
    }

    /// Consumes `n` bytes; like Jinja2, a match ending in a newline starts a line for lstrip_blocks.
    fn skip(l: *Lexer, n: usize) void {
        if (n == 0) return;
        l.line += @intCast(std.mem.count(u8, l.s[l.pos..][0..n], "\n"));
        l.line_starting = l.s[l.pos + n - 1] == '\n';
        l.pos += n;
    }

    /// Length of the Unicode whitespace run (Python's \s) at `from`.
    fn spaces(l: *const Lexer, from: usize) usize {
        var i = from;
        while (i < l.s.len) {
            var j = i;
            if (!uni.isSpace(uni.next(l.s, &j))) break;
            i = j;
        }
        return i - from;
    }

    fn rstripped(text: []const u8) []const u8 {
        var end = text.len;
        while (end > 0) {
            var start = end - 1;
            while (start > 0 and text[start] & 0xC0 == 0x80) start -= 1;
            var j = start;
            if (!uni.isSpace(uni.next(text, &j))) break;
            end = start;
        }
        return text[0..end];
    }

    fn allSpace(text: []const u8) bool {
        var i: usize = 0;
        while (i < text.len) if (!uni.isSpace(uni.next(text, &i))) return false;
        return text.len > 0;
    }

    /// Text before a tag: `-` strips all trailing whitespace; lstrip_blocks drops a line's indent before `{%` and `{#`.
    fn data(l: *const Lexer, text: []const u8, sign: u8, variable: bool) []const u8 {
        if (sign == '-') return rstripped(text);
        if (sign != '+' and !variable) {
            const from = if (std.mem.lastIndexOfScalar(u8, text, '\n')) |nl| nl + 1 else 0;
            if ((from > 0 or l.line_starting) and allSpace(text[from..])) return text[0..from];
        }
        return text;
    }

    fn signAt(l: *const Lexer, at: usize) u8 {
        return if (at < l.s.len and (l.s[at] == '-' or l.s[at] == '+')) l.s[at] else 0;
    }

    fn root(l: *Lexer) !void {
        while (l.pos < l.s.len) {
            var at = l.pos;
            const kind: u8 = while (std.mem.indexOfScalarPos(u8, l.s, at, '{')) |brace| : (at = brace + 1) {
                if (brace + 1 < l.s.len and std.mem.indexOfScalar(u8, "{%#", l.s[brace + 1]) != null) {
                    at = brace;
                    break l.s[brace + 1];
                }
            } else 0;
            if (kind == 0) {
                try l.emit(.{ .tag = .data, .text = l.s[l.pos..] });
                l.skip(l.s.len - l.pos);
                return;
            }
            const sign = l.signAt(at + 2);
            const text = l.data(l.s[l.pos..at], sign, kind == '{');
            if (text.len > 0) try l.emit(.{ .tag = .data, .text = text });
            const begin = at + 2 + @intFromBool(sign != 0);
            if (kind == '%' and try l.raw(begin)) continue;
            l.skip(begin - l.pos);
            switch (kind) {
                '#' => try l.comment(),
                '%' => {
                    try l.emit(.{ .tag = .block_begin });
                    try l.inside(false);
                },
                else => {
                    try l.emit(.{ .tag = .var_begin });
                    try l.inside(true);
                },
            }
        }
    }

    /// `{% raw %}...{% endraw %}` emits its body as data; the raw tag itself never trims a newline.
    fn raw(l: *Lexer, begin: usize) !bool {
        var i = begin + l.spaces(begin);
        if (!std.mem.startsWith(u8, l.s[i..], "raw")) return false;
        i += 3;
        i += l.spaces(i);
        if (std.mem.startsWith(u8, l.s[i..], "-%}")) {
            i += 3 + l.spaces(i + 3);
        } else if (std.mem.startsWith(u8, l.s[i..], "%}")) i += 2 else return false;
        l.skip(i - l.pos);
        var k = l.pos;
        while (std.mem.indexOfPos(u8, l.s, k, "{%")) |at| : (k = at + 1) {
            const sign = l.signAt(at + 2);
            var m = at + 2 + @intFromBool(sign != 0);
            m += l.spaces(m);
            if (!std.mem.startsWith(u8, l.s[m..], "endraw")) continue;
            m += 6;
            m += l.spaces(m);
            const close = l.blockEnd(m) orelse continue;
            const body = l.data(l.s[l.pos..at], sign, false);
            if (body.len > 0) try l.emit(.{ .tag = .data, .text = body });
            l.skip(close - l.pos);
            return true;
        }
        return l.fail("Missing end of raw directive", .{});
    }

    /// End of a block tag at `at` (`+%}`, `-%}` with following whitespace, or `%}` and one trimmed newline).
    fn blockEnd(l: *const Lexer, at: usize) ?usize {
        const rest = l.s[at..];
        if (std.mem.startsWith(u8, rest, "+%}")) return at + 3;
        if (std.mem.startsWith(u8, rest, "-%}")) return at + 3 + l.spaces(at + 3);
        if (std.mem.startsWith(u8, rest, "%}")) return at + 2 + @intFromBool(rest.len > 2 and rest[2] == '\n');
        return null;
    }

    fn comment(l: *Lexer) !void {
        const at = std.mem.indexOfPos(u8, l.s, l.pos, "#}") orelse return l.fail("Missing end of comment tag", .{});
        const sign: u8 = if (at > l.pos) l.signAt(at - 1) else 0;
        const end = at + 2;
        const stop = switch (sign) {
            '+' => end,
            '-' => end + l.spaces(end),
            else => end + @intFromBool(end < l.s.len and l.s[end] == '\n'),
        };
        l.skip(stop - l.pos);
    }

    fn inside(l: *Lexer, variable: bool) !void {
        var depth: usize = 0;
        while (true) {
            if (l.pos >= l.s.len) return l.fail("unexpected end of template", .{});
            const rest = l.s[l.pos..];
            if (depth == 0) {
                const close: ?usize = if (!variable) l.blockEnd(l.pos) else if (std.mem.startsWith(u8, rest, "-}}")) l.pos + 3 + l.spaces(l.pos + 3) else if (std.mem.startsWith(u8, rest, "}}")) l.pos + 2 else null;
                if (close) |end| {
                    try l.emit(.{ .tag = if (variable) .var_end else .block_end });
                    l.skip(end - l.pos);
                    return;
                }
            }
            const ws = l.spaces(l.pos);
            if (ws > 0) {
                l.skip(ws);
                continue;
            }
            if (try l.number()) continue;
            const c = rest[0];
            if (std.ascii.isAlphabetic(c) or c == '_') {
                var n: usize = 1;
                while (n < rest.len and (std.ascii.isAlphanumeric(rest[n]) or rest[n] == '_')) n += 1;
                try l.emit(.{ .tag = .name, .text = rest[0..n] });
                l.skip(n);
                continue;
            }
            if (c == '\'' or c == '"') {
                var n: usize = 1;
                while (n < rest.len and rest[n] != c) n += if (rest[n] == '\\' and n + 1 < rest.len) 2 else 1;
                if (n >= rest.len) return l.fail("unexpected char '{c}'", .{c});
                try l.emit(.{ .tag = .string, .text = try l.unescape(rest[1..n]) });
                l.skip(n + 1);
                continue;
            }
            const op = for (ops) |o| {
                if (std.mem.startsWith(u8, rest, o[0])) break o;
            } else return l.fail("unexpected char '{c}'", .{c});
            switch (op[1]) {
                .lbrace, .lparen, .lbracket => depth += 1,
                .rbrace, .rparen, .rbracket => {
                    if (depth == 0) return l.fail("unexpected '{c}'", .{c});
                    depth -= 1;
                },
                else => {},
            }
            try l.emit(.{ .tag = .op, .op = op[1] });
            l.skip(op[0].len);
        }
    }

    /// End of a run of `base` digits from `from`, with single underscores between digits.
    fn digits(s: []const u8, from: usize, base: u8) usize {
        var i = from;
        while (i < s.len) {
            const under = s[i] == '_';
            const d = if (under) (if (i + 1 < s.len) s[i + 1] else break) else s[i];
            _ = std.fmt.charToDigit(d, base) catch break;
            i += if (under) 2 else 1;
        }
        return i;
    }

    /// float_re is tried before integer_re, and a float may not follow a dot.
    fn number(l: *Lexer) !bool {
        const s = l.s;
        const p = l.pos;
        if (!std.ascii.isDigit(s[p])) return false;
        if (p == 0 or s[p - 1] != '.') {
            var i = digits(s, p, 10);
            var is_float = false;
            if (i + 1 < s.len and s[i] == '.' and std.ascii.isDigit(s[i + 1])) {
                i = digits(s, i + 1, 10);
                is_float = true;
            }
            if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
                var k = i + 1;
                if (k < s.len and (s[k] == '+' or s[k] == '-')) k += 1;
                if (k < s.len and std.ascii.isDigit(s[k])) {
                    i = digits(s, k, 10);
                    is_float = true;
                }
            }
            if (is_float) {
                const clean = try std.mem.replaceOwned(u8, l.a, s[p..i], "_", "");
                try l.emit(.{ .tag = .float, .float = std.fmt.parseFloat(f64, clean) catch return l.fail("invalid float", .{}) });
                l.skip(i - p);
                return true;
            }
        }
        var base: u8 = 10;
        var start = p;
        var i = p + 1;
        if (s[p] == '0' and p + 1 < s.len) {
            const prefixed: u8 = switch (std.ascii.toLower(s[p + 1])) {
                'b' => 2,
                'o' => 8,
                'x' => 16,
                else => 0,
            };
            if (prefixed != 0 and digits(s, p + 2, prefixed) > p + 2) {
                base = prefixed;
                start = p + 2;
                i = digits(s, start, base);
            } else {
                while (i < s.len and (s[i] == '0' or (s[i] == '_' and i + 1 < s.len and s[i + 1] == '0'))) i += if (s[i] == '_') 2 else 1;
            }
        } else i = digits(s, p, 10);
        const clean = try std.mem.replaceOwned(u8, l.a, s[start..i], "_", "");
        if (std.fmt.parseInt(i64, clean, base)) |n| {
            try l.emit(.{ .tag = .integer, .int = n });
        } else |_| {
            if (base != 10) return l.fail("non-decimal integer literal out of range", .{});
            try l.emit(.{ .tag = .integer, .text = clean });
        }
        l.skip(i - p);
        return true;
    }

    /// Python's unicode-escape decoding of a literal after backslashreplace, as Jinja2 does.
    fn unescape(l: *Lexer, text: []const u8) ![]const u8 {
        if (std.mem.indexOfScalar(u8, text, '\\') == null) return text;
        var out: std.ArrayList(u8) = .empty;
        var i: usize = 0;
        while (i < text.len) {
            if (text[i] != '\\') {
                try out.append(l.a, text[i]);
                i += 1;
                continue;
            }
            if (i + 1 >= text.len) return l.fail("\\ at end of string", .{});
            const e = text[i + 1];
            i += 2;
            switch (e) {
                '\n' => {},
                '\\', '\'', '"' => try out.append(l.a, e),
                'a' => try out.append(l.a, 7),
                'b' => try out.append(l.a, 8),
                'f' => try out.append(l.a, 12),
                'n' => try out.append(l.a, '\n'),
                'r' => try out.append(l.a, '\r'),
                't' => try out.append(l.a, '\t'),
                'v' => try out.append(l.a, 11),
                '0'...'7' => {
                    var v: u21 = e - '0';
                    var n: usize = 1;
                    while (n < 3 and i < text.len and text[i] >= '0' and text[i] <= '7') : (n += 1) {
                        v = v * 8 + (text[i] - '0');
                        i += 1;
                    }
                    try uni.encode(&out, l.a, v);
                },
                'x', 'u', 'U' => {
                    const n: usize = switch (e) {
                        'x' => 2,
                        'u' => 4,
                        else => 8,
                    };
                    if (i + n > text.len) return l.fail("truncated \\{c} escape", .{e});
                    const v = std.fmt.parseInt(u32, text[i..][0..n], 16) catch return l.fail("truncated \\{c} escape", .{e});
                    if (v > 0x10FFFF or (v >= 0xD800 and v <= 0xDFFF)) return l.fail("illegal Unicode character", .{});
                    try uni.encode(&out, l.a, @intCast(v));
                    i += n;
                },
                'N' => return l.fail("\\N{{...}} escapes are not supported", .{}),
                else => if (e < 0x80) {
                    try out.appendSlice(l.a, &.{ '\\', e });
                } else {
                    var k = i - 1;
                    const cp = uni.next(text, &k);
                    i = k;
                    if (cp < 0x100) try out.print(l.a, "\\x{x:0>2}", .{cp}) else if (cp < 0x10000) try out.print(l.a, "\\u{x:0>4}", .{cp}) else try out.print(l.a, "\\U{x:0>8}", .{cp});
                },
            }
        }
        return out.items;
    }
};

/// Jinja2 turns \r\n and \r into \n and drops one trailing newline before lexing.
pub fn normalize(a: Allocator, source: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(a, source.len);
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        if (source[i] == '\r') {
            out.appendAssumeCapacity('\n');
            if (i + 1 < source.len and source[i + 1] == '\n') i += 1;
        } else out.appendAssumeCapacity(source[i]);
    }
    if (out.items.len > 0 and out.items[out.items.len - 1] == '\n') out.items.len -= 1;
    return out.items;
}

pub fn tokenize(l: *Lexer) Error![]Token {
    try l.root();
    try l.emit(.{ .tag = .eof });
    return l.tokens.items;
}

//! RFC 7578 multipart/form-data over a body already read: parts as slices of the body, no copies of part bodies.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The most parts one body may hold; one more is malformed.
pub const max_parts = 256;
/// The most header bytes one part may carry, its blank line included.
pub const max_header_bytes = 8 * 1024;

pub const Part = struct {
    /// The `name` parameter of `Content-Disposition: form-data`.
    name: []const u8,
    /// The `filename` parameter, or null when the part sends none (`filename*=` is ignored).
    filename: ?[]const u8,
    /// The `Content-Type` header value exactly as sent, trimmed; docTR checks and quotes it.
    content_type: ?[]const u8,
    /// The lower-cased type/subtype without parameters; null when the header is absent or empty.
    media: ?[]const u8,
    body: []const u8,
};

/// The `boundary` parameter of a Content-Type header (quoted or token, 1..70 characters), or null.
pub fn boundary(content_type: []const u8) ?[]const u8 {
    const semi = std.mem.indexOfScalar(u8, content_type, ';') orelse return null;
    var it: Params = .{ .text = content_type[semi..] };
    while (it.next() catch return null) |p| {
        if (!std.ascii.eqlIgnoreCase(p.key, "boundary")) continue;
        if (p.value.len == 0 or p.value.len > 70 or p.value[p.value.len - 1] == ' ') return null;
        for (p.value) |ch| if (!boundaryChar(ch) or (ch == ' ' and !p.quoted)) return null;
        return p.value;
    }
    return null;
}

/// RFC 2046 bchars: a quoted boundary may also hold spaces, but not as its last character.
fn boundaryChar(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or std.mem.indexOfScalar(u8, "'()+_,-./:=? ", ch) != null;
}

/// The parts of `body` in order; the slice and unescaped names come from `a`, which should be a request arena.
pub fn parse(a: Allocator, body: []const u8, bound: []const u8) error{ Malformed, OutOfMemory }![]Part {
    if (bound.len == 0) return error.Malformed;
    var parts: std.ArrayList(Part) = .empty;
    errdefer parts.deinit(a);
    var d = delimiter(body, 0, bound) orelse return error.Malformed;
    while (!d.close) {
        if (parts.items.len == max_parts) return error.Malformed;
        var head = try headers(a, body, d.next);
        const next = delimiter(body, head.start, bound) orelse return error.Malformed;
        head.part.body = body[head.start..bodyEnd(body, head.start, next.line)];
        try parts.append(a, head.part);
        d = next;
    }
    return parts.toOwnedSlice(a);
}

const Delimiter = struct {
    /// Where the `--boundary` line starts.
    line: usize,
    /// Where the next part's headers start (after the line break).
    next: usize,
    /// Whether this is the close delimiter `--boundary--`.
    close: bool,
};

/// The first `--boundary` at a line start that `--`, or blanks and a CRLF or LF, follows; other such text is body.
fn delimiter(body: []const u8, from: usize, bound: []const u8) ?Delimiter {
    var pos = from + 2;
    while (std.mem.indexOfPos(u8, body, pos, bound)) |at| : (pos = at + 1) {
        const line = at - 2;
        if (!std.mem.eql(u8, body[line..at], "--")) continue;
        if (line != 0 and body[line - 1] != '\n') continue;
        const end = at + bound.len;
        if (std.mem.startsWith(u8, body[end..], "--")) return .{ .line = line, .next = end + 2, .close = true };
        var i = end;
        while (i < body.len and (body[i] == ' ' or body[i] == '\t')) i += 1;
        if (std.mem.startsWith(u8, body[i..], "\r\n")) return .{ .line = line, .next = i + 2, .close = false };
        if (std.mem.startsWith(u8, body[i..], "\n")) return .{ .line = line, .next = i + 1, .close = false };
    }
    return null;
}

/// Where a part body that starts at `start` ends, given the next delimiter line: before its CRLF or LF.
fn bodyEnd(body: []const u8, start: usize, line: usize) usize {
    if (line <= start) return start; // the blank line after the headers was the delimiter's own line break
    var end = line - 1;
    if (end > start and body[end - 1] == '\r') end -= 1;
    return end;
}

const Head = struct { part: Part, start: usize };

/// One part's headers from `at`: the part without its body, and where its body starts.
fn headers(a: Allocator, body: []const u8, at: usize) error{ Malformed, OutOfMemory }!Head {
    var part: Part = .{ .name = "", .filename = null, .content_type = null, .media = null, .body = "" };
    var disposition = false;
    var typed = false;
    var pos = at;
    while (true) {
        const window = body[0..@min(body.len, at + max_header_bytes)];
        const nl = std.mem.indexOfScalarPos(u8, window, pos, '\n') orelse return error.Malformed;
        const line = std.mem.trimEnd(u8, body[pos..nl], "\r");
        pos = nl + 1;
        if (line.len == 0) break;
        if (line[0] == ' ' or line[0] == '\t') return error.Malformed; // no obsolete header folding
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.Malformed;
        const name = line[0..colon];
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(name, "Content-Disposition")) {
            if (disposition) return error.Malformed;
            disposition = true;
            try formData(a, value, &part);
        } else if (std.ascii.eqlIgnoreCase(name, "Content-Type")) {
            if (typed) return error.Malformed;
            typed = true;
            part.content_type = value;
            part.media = try media(a, value);
        }
    }
    if (!disposition) return error.Malformed;
    return .{ .part = part, .start = pos };
}

/// The `name` and `filename` of a `form-data` disposition into `part`.
fn formData(a: Allocator, value: []const u8, part: *Part) error{ Malformed, OutOfMemory }!void {
    const semi = std.mem.indexOfScalar(u8, value, ';') orelse value.len;
    if (!std.ascii.eqlIgnoreCase(std.mem.trim(u8, value[0..semi], " \t"), "form-data")) return error.Malformed;
    var named = false;
    var it: Params = .{ .text = value[semi..] };
    while (try it.next()) |p| {
        if (std.ascii.eqlIgnoreCase(p.key, "name")) {
            if (named) return error.Malformed;
            named = true;
            part.name = try unescape(a, p);
        } else if (std.ascii.eqlIgnoreCase(p.key, "filename")) {
            if (part.filename != null) return error.Malformed;
            part.filename = try unescape(a, p);
        }
    }
    if (!named) return error.Malformed;
}

/// The lower-cased type/subtype of a Content-Type value, or null when it is empty.
fn media(a: Allocator, value: []const u8) Allocator.Error!?[]const u8 {
    const semi = std.mem.indexOfScalar(u8, value, ';') orelse value.len;
    const kind = std.mem.trim(u8, value[0..semi], " \t");
    if (kind.len == 0) return null;
    for (kind) |ch| {
        if (std.ascii.isUpper(ch)) return std.ascii.lowerString(try a.alloc(u8, kind.len), kind);
    }
    return kind;
}

/// A quoted value with `\"` and `\\` decoded; other backslashes stay, as Windows paths arrive unescaped.
fn unescape(a: Allocator, p: Param) Allocator.Error![]const u8 {
    if (!p.quoted or std.mem.indexOfScalar(u8, p.value, '\\') == null) return p.value;
    var out = try std.ArrayList(u8).initCapacity(a, p.value.len);
    var i: usize = 0;
    while (i < p.value.len) : (i += 1) {
        const ch = p.value[i];
        if (ch == '\\' and i + 1 < p.value.len and (p.value[i + 1] == '"' or p.value[i + 1] == '\\')) i += 1;
        out.appendAssumeCapacity(p.value[i]);
    }
    return out.toOwnedSlice(a);
}

const Param = struct { key: []const u8, value: []const u8, quoted: bool };

/// `; key=value` parameters, values as tokens or quoted strings (still escaped).
const Params = struct {
    text: []const u8,
    pos: usize = 0,

    fn skip(it: *Params) void {
        while (it.pos < it.text.len and (it.text[it.pos] == ' ' or it.text[it.pos] == '\t')) it.pos += 1;
    }

    fn next(it: *Params) error{Malformed}!?Param {
        it.skip();
        while (it.pos < it.text.len and it.text[it.pos] == ';') : (it.skip()) it.pos += 1;
        if (it.pos == it.text.len) return null;
        const key_start = it.pos;
        while (it.pos < it.text.len and std.mem.indexOfScalar(u8, "=; \t", it.text[it.pos]) == null) it.pos += 1;
        const key = it.text[key_start..it.pos];
        it.skip();
        if (key.len == 0 or it.pos == it.text.len or it.text[it.pos] != '=') return error.Malformed;
        it.pos += 1;
        it.skip();
        if (it.pos < it.text.len and it.text[it.pos] == '"') {
            const start = it.pos + 1;
            var i = start;
            while (i < it.text.len and it.text[i] != '"') : (i += 1) {
                if (it.text[i] == '\\' and i + 1 < it.text.len) i += 1;
            }
            if (i == it.text.len) return error.Malformed;
            it.pos = i + 1;
            it.skip();
            if (it.pos < it.text.len and it.text[it.pos] != ';') return error.Malformed;
            return .{ .key = key, .value = it.text[start..i], .quoted = true };
        }
        const start = it.pos;
        while (it.pos < it.text.len and it.text[it.pos] != ';') it.pos += 1;
        const value = std.mem.trimEnd(u8, it.text[start..it.pos], " \t");
        if (value.len == 0) return error.Malformed;
        return .{ .key = key, .value = value, .quoted = false };
    }
};

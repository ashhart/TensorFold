//! Bounded request bodies, fixed-length or chunked, refused as the Python server refuses them.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

pub const limit = 32 * 1024 * 1024;
const metadata_limit = 65536;

/// A body, or the refusal a RequestError carries (the connection then closes).
pub const Body = union(enum) { ok: []const u8, refused: []const u8 };

fn refuse(c: *Conn, message: []const u8) Body {
    c.close = true; // once framing is uncertain, unread bytes must not become another request
    return .{ .refused = message };
}

fn isSpace(ch: u8) bool {
    return switch (ch) {
        ' ', '\t', '\n', '\r', 0x0b, 0x0c, 0x1c...0x1f, 0x85, 0xa0 => true,
        else => false,
    };
}

fn strip(s: []const u8) []const u8 {
    var start: usize = 0;
    var end = s.len;
    while (start < end and isSpace(s[start])) start += 1;
    while (end > start and isSpace(s[end - 1])) end -= 1;
    return s[start..end];
}

/// Reads one body without consuming the next request on the connection.
pub fn read(c: *Conn, a: Allocator, max: usize) Allocator.Error!Body {
    const too_big = try std.fmt.allocPrint(a, "request body exceeds the {d} MiB limit", .{max / (1024 * 1024)});
    var transfers = false;
    var lengths = false;
    for (c.headers) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "Transfer-Encoding")) transfers = true;
        if (std.ascii.eqlIgnoreCase(h.name, "Content-Length")) lengths = true;
    }
    if (transfers) {
        if (lengths) return refuse(c, "Content-Length and Transfer-Encoding cannot be combined");
        var codings: usize = 0;
        var chunked = true;
        for (c.headers) |h| {
            if (!std.ascii.eqlIgnoreCase(h.name, "Transfer-Encoding")) continue;
            var it = std.mem.splitScalar(u8, h.value, ',');
            while (it.next()) |coding| {
                codings += 1;
                if (!std.ascii.eqlIgnoreCase(strip(coding), "chunked")) chunked = false;
            }
        }
        if (!chunked or codings != 1 or !std.mem.eql(u8, c.version, "HTTP/1.1"))
            return refuse(c, "unsupported request Transfer-Encoding; expected chunked over HTTP/1.1");
        return chunks(c, a, max, too_big);
    }
    if (!lengths) return .{ .ok = "" };
    var size: ?u64 = null;
    for (c.headers) |h| {
        if (!std.ascii.eqlIgnoreCase(h.name, "Content-Length")) continue;
        var it = std.mem.splitScalar(u8, h.value, ',');
        while (it.next()) |raw| {
            const value = strip(raw);
            if (value.len == 0 or value.len > 4300) return refuse(c, "invalid Content-Length");
            for (value) |ch| if (!std.ascii.isDigit(ch)) return refuse(c, "invalid Content-Length");
            const n = std.fmt.parseInt(u64, value, 10) catch std.math.maxInt(u64);
            if (size != null and size.? != n) return refuse(c, "conflicting Content-Length values");
            size = n;
        }
    }
    if (size.? > max) return refuse(c, too_big);
    var body: std.ArrayList(u8) = .empty;
    if ((c.readExact(a, &body, size.?) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => 0,
    }) != size.?) return refuse(c, "incomplete request body");
    return .{ .ok = body.items };
}

fn metaLine(c: *Conn, a: Allocator) Allocator.Error!?[]const u8 {
    const data = c.readLine(a, metadata_limit + 1, c.timeouts.read_ms) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    if (data.len > metadata_limit or !std.mem.endsWith(u8, data, "\r\n")) return null;
    return data[0 .. data.len - 2];
}

fn chunks(c: *Conn, a: Allocator, max: usize, too_big: []const u8) Allocator.Error!Body {
    var body: std.ArrayList(u8) = .empty;
    var extensions: usize = 0;
    while (true) {
        const head = try metaLine(c, a) orelse return refuse(c, "invalid or oversized chunked request framing");
        const semi = std.mem.indexOfScalar(u8, head, ';');
        const size_text = if (semi) |i| std.mem.trimEnd(u8, head[0..i], " \t") else head;
        if (size_text.len == 0) return refuse(c, "invalid chunk size");
        for (size_text) |ch| if (!std.ascii.isHex(ch)) return refuse(c, "invalid chunk size");
        if (semi) |i| extensions += head.len - i - 1;
        if (extensions > metadata_limit) return refuse(c, "chunk extensions exceed the request framing limit");
        const size = std.fmt.parseInt(u64, size_text, 16) catch std.math.maxInt(u64);
        if (size > max or body.items.len + size > max) return refuse(c, too_big);
        if (size == 0) {
            var trailers: usize = 0;
            while (true) {
                const trailer = try metaLine(c, a) orelse return refuse(c, "invalid or oversized chunked request framing");
                if (trailer.len == 0) return .{ .ok = body.items };
                trailers += trailer.len + 2;
                if (trailers > metadata_limit or !trailerName(trailer)) return refuse(c, "invalid or oversized request trailers");
            }
        }
        const got = c.readExact(a, &body, size) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => 0,
        };
        if (got != size) return refuse(c, "incomplete request body");
        var crlf: std.ArrayList(u8) = .empty;
        const n = c.readExact(a, &crlf, 2) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => 0,
        };
        if (n != 2) return refuse(c, "incomplete request body");
        if (!std.mem.eql(u8, crlf.items, "\r\n")) return refuse(c, "invalid chunk terminator");
    }
}

/// A trailer opens with a token and a colon.
fn trailerName(line: []const u8) bool {
    var i: usize = 0;
    while (i < line.len) : (i += 1) {
        const ch = line[i];
        const token = std.ascii.isAlphanumeric(ch) or std.mem.indexOfScalar(u8, "!#$%&'*+.^_`|~-", ch) != null;
        if (!token) break;
    }
    return i > 0 and i < line.len and line[i] == ':';
}

/// The body as JSON (an empty body reads as ``{}``): a framing refusal, or the decoder's message as another exception.
pub fn readJson(conn: *Conn, cx: *errors.Cx) errors.Refused!json.Value {
    const raw = switch (try read(conn, cx.a, limit)) {
        .ok => |b| b,
        .refused => |message| return cx.refuse(message),
    };
    return switch (try json.parse(cx.a, if (raw.len == 0) "{}" else raw)) {
        .ok => |v| v,
        .err => |message| cx.other(message),
    };
}

//! One HTTP/1.1 connection as Python's BaseHTTPRequestHandler serves it: bounded heads, keep-alive, error pages.
const std = @import("std");
const posix = std.posix;
const Allocator = std.mem.Allocator;
const log = @import("log.zig");

pub const max_line = 65536;
pub const max_headers = 100;

pub const Header = struct { name: []const u8, value: []const u8 };

pub const Timeouts = struct {
    idle_ms: i32 = 900_000, // a kept-alive connection waiting for its next request
    read_ms: i32 = 120_000, // a request whose next bytes stop arriving
    write_ms: i32 = 300_000, // a client that stops reading
};

pub const IoError = error{ Closed, Timeout };

/// What reading a request head ended with.
pub const Outcome = enum { ready, handled, closed };

/// Fired once per request at its first final status (>= 200), for the key-labelled request counter.
pub const StatusHook = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, conn: *Conn, code: u16) void,
};

pub const Conn = struct {
    fd: posix.socket_t,
    peer: []const u8,
    timeouts: Timeouts = .{},
    buf: []u8,
    start: usize = 0,
    end: usize = 0,
    out: std.ArrayList(u8) = .empty,
    gpa: Allocator,
    /// Per request, reset by ``readRequest``.
    close: bool = true,
    version: []const u8 = "HTTP/0.9",
    method: []const u8 = "",
    path: []const u8 = "",
    requestline: []const u8 = "",
    headers: []Header = &.{},
    key_label: ?[]const u8 = null,
    auth_enabled: bool = false,
    counted: bool = false,
    hook: ?StatusHook = null,
    broken: bool = false,
    quiet_log: bool = false,

    pub fn init(gpa: Allocator, fd: posix.socket_t, peer: []const u8) !Conn {
        return .{ .fd = fd, .peer = peer, .gpa = gpa, .buf = try gpa.alloc(u8, 2 * max_line + 8192) };
    }

    pub fn deinit(c: *Conn) void {
        c.gpa.free(c.buf);
        c.out.deinit(c.gpa);
    }

    fn fill(c: *Conn, timeout_ms: i32) IoError!usize {
        if (c.start == c.end) {
            c.start = 0;
            c.end = 0;
        } else if (c.end == c.buf.len) {
            std.mem.copyForwards(u8, c.buf, c.buf[c.start..c.end]);
            c.end -= c.start;
            c.start = 0;
        }
        var fds = [_]posix.pollfd{.{ .fd = c.fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, timeout_ms) catch return error.Closed;
        if (ready == 0) return error.Timeout;
        const got = posix.read(c.fd, c.buf[c.end..]) catch return error.Closed;
        c.end += got;
        return got;
    }

    /// ``rfile.readline(limit)``: through the newline, ``limit`` bytes, or what is left before EOF.
    pub fn readLine(c: *Conn, a: Allocator, limit: usize, timeout_ms: i32) (IoError || Allocator.Error)![]u8 {
        var line: std.ArrayList(u8) = .empty;
        var wait = timeout_ms;
        while (true) {
            const avail = c.buf[c.start..c.end];
            const scan = avail[0..@min(avail.len, limit - line.items.len)];
            if (std.mem.indexOfScalar(u8, scan, '\n')) |i| {
                try line.appendSlice(a, scan[0 .. i + 1]);
                c.start += i + 1;
                return line.items;
            }
            try line.appendSlice(a, scan);
            c.start += scan.len;
            if (line.items.len >= limit) return line.items;
            if (try c.fill(wait) == 0) return line.items;
            wait = c.timeouts.read_ms;
        }
    }

    /// ``rfile.read(n)``: ``n`` bytes, fewer only at EOF.
    pub fn readExact(c: *Conn, a: Allocator, list: *std.ArrayList(u8), n: usize) (IoError || Allocator.Error)!usize {
        var want = n;
        while (want > 0) {
            if (c.start == c.end and try c.fill(c.timeouts.read_ms) == 0) break;
            const take = @min(want, c.end - c.start);
            try list.appendSlice(a, c.buf[c.start .. c.start + take]);
            c.start += take;
            want -= take;
        }
        return n - want;
    }

    /// Whether the peer has gone: readable with nothing to read (Python's socket_cancellation).
    pub fn peerGone(c: *Conn) bool {
        if (c.broken) return true;
        var fds = [_]posix.pollfd{.{ .fd = c.fd, .events = posix.POLL.IN | posix.POLL.PRI, .revents = 0 }};
        const ready = posix.poll(&fds, 0) catch return true;
        if (ready == 0) return false;
        var probe: [1]u8 = undefined;
        const rc = posix.system.recvfrom(c.fd, &probe, 1, posix.MSG.PEEK | posix.MSG.DONTWAIT, null, null);
        return switch (posix.errno(rc)) {
            .SUCCESS => rc == 0,
            .AGAIN, .INTR => false,
            else => true,
        };
    }

    /// Writes ``data`` whole; a write that cannot progress for ``write_ms`` marks the client gone.
    pub fn writeAll(c: *Conn, data: []const u8) error{Closed}!void {
        if (c.broken) return error.Closed;
        var sent: usize = 0;
        while (sent < data.len) {
            var fds = [_]posix.pollfd{.{ .fd = c.fd, .events = posix.POLL.OUT, .revents = 0 }};
            const ready = posix.poll(&fds, c.timeouts.write_ms) catch return c.fail();
            if (ready == 0) return c.fail();
            const rc = posix.system.write(c.fd, data[sent..].ptr, data.len - sent);
            switch (posix.errno(rc)) {
                .SUCCESS => sent += @intCast(rc),
                .AGAIN, .INTR => {},
                else => return c.fail(),
            }
        }
    }

    fn fail(c: *Conn) error{Closed} {
        c.broken = true;
        c.close = true;
        return error.Closed;
    }

    /// The request path as Python holds it: latin-1 text.
    pub fn requestPath(c: *const Conn, a: Allocator) []const u8 {
        return latin1(a, c.path) catch c.path;
    }

    /// Python's ``headers.get(name)``: the first value, any case.
    pub fn header(c: *const Conn, name: []const u8) ?[]const u8 {
        for (c.headers) |h| if (std.ascii.eqlIgnoreCase(h.name, name)) return h.value;
        return null;
    }

    pub fn headerCount(c: *const Conn, name: []const u8) usize {
        var n: usize = 0;
        for (c.headers) |h| n += @intFromBool(std.ascii.eqlIgnoreCase(h.name, name));
        return n;
    }

    /// Reads and parses one request head; error replies are sent here, as ``parse_request`` sends them.
    pub fn readRequest(c: *Conn, a: Allocator, idle: bool) Allocator.Error!Outcome {
        c.key_label = null;
        c.counted = false;
        const raw = c.readLine(a, max_line + 1, if (idle) c.timeouts.idle_ms else c.timeouts.read_ms) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                c.close = true;
                return .closed;
            },
        };
        if (raw.len > max_line) {
            c.requestline = "";
            c.version = "";
            c.method = "";
            try c.sendError(a, 414, null, null);
            return .handled;
        }
        if (raw.len == 0) {
            c.close = true;
            return .closed;
        }
        return c.parseRequest(a, raw);
    }

    fn parseRequest(c: *Conn, a: Allocator, raw: []const u8) Allocator.Error!Outcome {
        c.method = "";
        c.version = "HTTP/0.9";
        c.close = true;
        c.headers = &.{};
        const line = std.mem.trimEnd(u8, raw, "\r\n");
        c.requestline = try latin1(a, line);
        var words: std.ArrayList([]const u8) = .empty;
        var it = std.mem.tokenizeAny(u8, line, " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0");
        while (it.next()) |w| try words.append(a, w);
        if (words.items.len == 0) return .closed;
        if (words.items.len >= 3) {
            const version = words.items[words.items.len - 1];
            const number = parseVersion(version) orelse {
                try c.sendError(a, 400, try std.fmt.allocPrint(a, "Bad request version ({s})", .{try pyRepr(a, version)}), null);
                return .handled;
            };
            if (number[0] > 1 or (number[0] == 1 and number[1] >= 1)) c.close = false;
            if (number[0] >= 2) {
                try c.sendError(a, 505, try std.fmt.allocPrint(a, "Invalid HTTP version ({s})", .{try latin1(a, version[5..])}), null);
                return .handled;
            }
            c.version = version;
        }
        if (words.items.len < 2 or words.items.len > 3) {
            try c.sendError(a, 400, try std.fmt.allocPrint(a, "Bad request syntax ({s})", .{try pyRepr(a, line)}), null);
            return .handled;
        }
        c.method = words.items[0];
        c.path = words.items[1];
        if (std.mem.startsWith(u8, c.path, "//")) c.path = try std.mem.concat(a, u8, &.{ "/", std.mem.trimStart(u8, c.path, "/") });
        if (words.items.len == 2) {
            c.close = true;
            if (!std.mem.eql(u8, c.method, "GET")) {
                try c.sendError(a, 400, try std.fmt.allocPrint(a, "Bad HTTP/0.9 request type ({s})", .{try pyRepr(a, c.method)}), null);
                return .handled;
            }
            return .ready;
        }
        if (!try c.readHeaders(a)) return .handled;
        const connection = c.header("Connection") orelse "";
        if (std.ascii.eqlIgnoreCase(connection, "close")) c.close = true else if (std.ascii.eqlIgnoreCase(connection, "keep-alive")) c.close = false;
        return .ready;
    }

    fn readHeaders(c: *Conn, a: Allocator) Allocator.Error!bool {
        var lines: std.ArrayList([]const u8) = .empty;
        while (true) {
            const line = c.readLine(a, max_line + 1, c.timeouts.read_ms) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => "",
            };
            if (line.len > max_line) {
                try c.sendError(a, 431, "Line too long", "got more than 65536 bytes when reading header line");
                return false;
            }
            if (lines.items.len > max_headers) {
                try c.sendError(a, 431, "Too many headers", "got more than 100 headers");
                return false;
            }
            try lines.append(a, line);
            if (line.len == 0 or std.mem.eql(u8, line, "\r\n") or std.mem.eql(u8, line, "\n")) break;
        }
        var headers: std.ArrayList(Header) = .empty;
        for (lines.items) |line| {
            if (line.len == 0 or line[0] == '\r' or line[0] == '\n') break;
            if (line[0] == ' ' or line[0] == '\t') {
                if (headers.items.len == 0) continue;
                const last = &headers.items[headers.items.len - 1];
                last.value = try std.mem.concat(a, u8, &.{ last.value, "\r\n", std.mem.trimEnd(u8, line, "\r\n") });
                continue;
            }
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse break;
            const name = line[0..colon];
            if (!headerName(name)) break; // email's parser ends the head at a line that is not a header
            const value = std.mem.trimEnd(u8, std.mem.trimStart(u8, line[colon + 1 ..], " \t"), "\r\n");
            try headers.append(a, .{ .name = name, .value = value });
        }
        c.headers = headers.items;
        return true;
    }

    /// Python's ``send_response`` head: status line, Server and Date (nothing for HTTP/0.9).
    pub fn startResponse(c: *Conn, code: u16, phrase: ?[]const u8) Allocator.Error!void {
        c.out.clearRetainingCapacity();
        if (c.quiet_log) c.quiet_log = false else log.request(c, code);
        if (code >= 200 and !c.counted) {
            c.counted = true;
            if (c.hook) |h| h.call(h.ctx, c, code);
        }
        if (std.mem.eql(u8, c.version, "HTTP/0.9")) return;
        var date: [40]u8 = undefined;
        try c.out.print(c.gpa, "HTTP/1.1 {d} {s}\r\nServer: TensorFold\r\nDate: {s}\r\n", .{ code, phrase orelse reason(code), httpDate(&date) });
    }

    pub fn addHeader(c: *Conn, name: []const u8, value: []const u8) Allocator.Error!void {
        if (std.ascii.eqlIgnoreCase(name, "connection")) {
            if (std.ascii.eqlIgnoreCase(value, "close")) c.close = true else if (std.ascii.eqlIgnoreCase(value, "keep-alive")) c.close = false;
        }
        if (std.mem.eql(u8, c.version, "HTTP/0.9")) return;
        try c.out.print(c.gpa, "{s}: {s}\r\n", .{ name, value });
    }

    /// Ends the head and sends it with ``body``.
    pub fn finish(c: *Conn, body: []const u8) error{Closed}!void {
        if (!std.mem.eql(u8, c.version, "HTTP/0.9")) c.out.appendSlice(c.gpa, "\r\n") catch return c.fail();
        c.out.appendSlice(c.gpa, body) catch return c.fail();
        try c.writeAll(c.out.items);
        c.out.clearRetainingCapacity();
    }

    /// A JSON reply with Python's ``_send_json`` headers.
    pub fn sendJson(c: *Conn, code: u16, body: []const u8) void {
        c.sendJsonOr(code, body) catch {};
    }

    fn sendJsonOr(c: *Conn, code: u16, body: []const u8) !void {
        try c.startResponse(code, null);
        try c.addHeader("Content-Type", "application/json");
        var len: [24]u8 = undefined;
        try c.addHeader("Content-Length", try std.fmt.bufPrint(&len, "{d}", .{body.len}));
        if (c.close) try c.addHeader("Connection", "close");
        try c.finish(body);
    }

    /// ``send_error``: an HTML page that closes the connection.
    pub fn sendError(c: *Conn, a: Allocator, code: u16, message: ?[]const u8, explain: ?[]const u8) Allocator.Error!void {
        const msg = message orelse reason(code);
        log.line("{s} code {d}, message {s}{s}", .{ c.peer, code, msg, log.keySuffix(c) });
        try c.startResponse(code, message);
        try c.addHeader("Connection", "close");
        const page = try std.fmt.allocPrint(a, error_page, .{ code, try htmlEscape(a, msg), code, try htmlEscape(a, explain orelse description(code)) });
        try c.addHeader("Content-Type", "text/html;charset=utf-8");
        var len: [24]u8 = undefined;
        try c.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{page.len}) catch unreachable);
        c.finish(if (std.mem.eql(u8, c.method, "HEAD")) "" else page) catch {};
    }

    /// ``Expect: 100-continue`` before the body is read.
    pub fn sendContinue(c: *Conn) void {
        c.writeAll("HTTP/1.1 100 Continue\r\n\r\n") catch {};
    }
};

const error_page =
    \\<!DOCTYPE HTML>
    \\<html lang="en">
    \\    <head>
    \\        <meta charset="utf-8">
    \\        <style type="text/css">
    \\            :root {{
    \\                color-scheme: light dark;
    \\            }}
    \\        </style>
    \\        <title>Error response</title>
    \\    </head>
    \\    <body>
    \\        <h1>Error response</h1>
    \\        <p>Error code: {d}</p>
    \\        <p>Message: {s}.</p>
    \\        <p>Error code explanation: {d} - {s}.</p>
    \\    </body>
    \\</html>
    \\
;

fn headerName(name: []const u8) bool {
    for (name) |ch| if (ch < 0x21 or ch > 0x7e) return false;
    return true;
}

fn parseVersion(v: []const u8) ?[2]u64 {
    if (!std.mem.startsWith(u8, v, "HTTP/")) return null;
    var parts = std.mem.splitScalar(u8, v[5..], '.');
    const major = parts.next() orelse return null;
    const minor = parts.next() orelse return null;
    if (parts.next() != null) return null;
    var out: [2]u64 = undefined;
    for ([_][]const u8{ major, minor }, 0..) |part, i| {
        if (part.len == 0 or part.len > 10) return null;
        for (part) |ch| if (!std.ascii.isDigit(ch)) return null;
        out[i] = std.fmt.parseInt(u64, part, 10) catch return null;
    }
    return out;
}

pub fn reason(code: u16) []const u8 {
    return switch (code) {
        100 => "Continue",
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        414 => "URI Too Long",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        else => "",
    };
}

fn description(code: u16) []const u8 {
    return switch (code) {
        400 => "Bad request syntax or unsupported method",
        404 => "Nothing matches the given URI",
        414 => "URI is too long",
        431 => "The server is unwilling to process the request because its header fields are too large",
        501 => "Server does not support this operation",
        505 => "Cannot fulfill request",
        else => "",
    };
}

/// RFC 1123 date in GMT, as ``email.utils.formatdate(usegmt=True)``.
pub fn httpDate(buf: []u8) []const u8 {
    const now = std.Io.Clock.real.now(log.io()).toSeconds();
    const es: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(0, now)) };
    const day = es.getEpochDay();
    const yd = day.calculateYearDay();
    const md = yd.calculateMonthDay();
    const secs = es.getDaySeconds();
    const days = [_][]const u8{ "Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed" };
    const months = [_][]const u8{ "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
    return std.fmt.bufPrint(buf, "{s}, {d:0>2} {s} {d} {d:0>2}:{d:0>2}:{d:0>2} GMT", .{
        days[@intCast(day.day % 7)], md.day_index + 1,          months[@backingInt(md.month) - 1], yd.year,
        secs.getHoursIntoDay(),      secs.getMinutesIntoHour(), secs.getSecondsIntoMinute(),
    }) catch "";
}

/// Bytes read as ISO-8859-1, re-encoded as UTF-8 (Python decodes request lines so).
pub fn latin1(a: Allocator, bytes: []const u8) Allocator.Error![]const u8 {
    for (bytes) |b| {
        if (b >= 0x80) break;
    } else return bytes;
    var out: std.ArrayList(u8) = .empty;
    for (bytes) |b| {
        if (b < 0x80) try out.append(a, b) else try out.appendSlice(a, &.{ 0xc0 | (b >> 6), 0x80 | (b & 0x3f) });
    }
    return out.items;
}

/// Python's ``repr`` of a latin-1 decoded string.
pub fn pyRepr(a: Allocator, bytes: []const u8) Allocator.Error![]const u8 {
    const single = std.mem.indexOfScalar(u8, bytes, '\'') != null;
    const double = std.mem.indexOfScalar(u8, bytes, '"') != null;
    const q: u8 = if (single and !double) '"' else '\'';
    var out: std.ArrayList(u8) = .empty;
    try out.append(a, q);
    for (bytes) |b| {
        switch (b) {
            '\\' => try out.appendSlice(a, "\\\\"),
            '\t' => try out.appendSlice(a, "\\t"),
            '\n' => try out.appendSlice(a, "\\n"),
            '\r' => try out.appendSlice(a, "\\r"),
            else => if (b == q) {
                try out.appendSlice(a, &.{ '\\', b });
            } else if (b < 0x20 or (b >= 0x7f and b <= 0xa0) or b == 0xad) {
                try out.print(a, "\\x{x:0>2}", .{b});
            } else if (b >= 0x80) {
                try out.appendSlice(a, &.{ 0xc0 | (b >> 6), 0x80 | (b & 0x3f) });
            } else try out.append(a, b),
        }
    }
    try out.append(a, q);
    return out.items;
}

fn htmlEscape(a: Allocator, s: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (s) |ch| switch (ch) {
        '&' => try out.appendSlice(a, "&amp;"),
        '<' => try out.appendSlice(a, "&lt;"),
        '>' => try out.appendSlice(a, "&gt;"),
        else => try out.append(a, ch),
    };
    return out.items;
}

test "versions and repr" {
    try std.testing.expectEqual([2]u64{ 1, 1 }, parseVersion("HTTP/1.1").?);
    try std.testing.expect(parseVersion("HTTP/1") == null);
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("\"a'b\"", try pyRepr(arena.allocator(), "a'b"));
    try std.testing.expectEqualStrings("'\\x7f\\xa0\xc3\xa9\\t'", try pyRepr(arena.allocator(), "\x7f\xa0\xe9\t"));
}

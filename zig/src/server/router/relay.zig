//! One request forwarded to a child, and its reply relayed to the client byte for byte as the bytes arrive.
const std = @import("std");
const http_conn = @import("../http_conn.zig");
const Conn = http_conn.Conn;
const Allocator = std.mem.Allocator;
const posix = std.posix;

/// How an exchange ended: the whole reply relayed, the client gone, no reply head from the child, or a reply cut short.
pub const Outcome = enum { done, client_gone, no_reply, cut };

pub const Result = struct { outcome: Outcome, status: u16 = 0 };

/// Milliseconds the child may send nothing, before its reply's head or between its bytes.
pub const Limits = struct { idle_ms: i64 = 3_600_000 };

/// Request headers the router sets or ends itself: hop-by-hop, framing, the host, the keys and the origin it checked.
const dropped = [_][]const u8{ "host", "content-length", "transfer-encoding", "connection", "keep-alive", "proxy-connection", "te", "trailer", "upgrade", "expect", "authorization", "x-api-key", "proxy-authorization", "origin" };

/// Sends ``conn``'s request with ``body`` to the child on ``port`` under its ``key``, then relays the reply.
pub fn forward(conn: *Conn, a: Allocator, port: u16, key: []const u8, body: []const u8, limits: Limits) Result {
    const fd = connect(port) orelse return .{ .outcome = .no_reply };
    defer _ = posix.system.close(fd);
    const head = requestHead(a, conn, port, key, body.len) catch return .{ .outcome = .no_reply };
    if (!send(fd, head) or !send(fd, body)) return .{ .outcome = .no_reply };
    return relay(conn, fd, limits);
}

/// The forwarded head: the client's method, target, version and end-to-end headers, the child's host and key.
pub fn requestHead(a: Allocator, conn: *const Conn, port: u16, key: []const u8, length: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{s} {s} {s}\r\nHost: 127.0.0.1:{d}\r\n", .{ conn.method, conn.path, conn.version, port });
    for (conn.headers) |h| {
        if (listed(&dropped, h.name) or connectionNames(conn, h.name)) continue;
        try out.print(a, "{s}: {s}\r\n", .{ h.name, h.value });
    }
    try out.print(a, "Authorization: Bearer {s}\r\nContent-Length: {d}\r\n", .{ key, length });
    if (conn.close) try out.appendSlice(a, "Connection: close\r\n");
    try out.appendSlice(a, "\r\n");
    return out.items;
}

fn listed(names: []const []const u8, name: []const u8) bool {
    for (names) |n| if (std.ascii.eqlIgnoreCase(n, name)) return true;
    return false;
}

/// Whether the client's Connection header names ``name`` as hop-by-hop.
fn connectionNames(conn: *const Conn, name: []const u8) bool {
    for (conn.headers) |h| if (std.ascii.eqlIgnoreCase(h.name, "connection") and hasToken(h.value, name)) return true;
    return false;
}

fn hasToken(value: []const u8, token: []const u8) bool {
    var it = std.mem.splitScalar(u8, value, ',');
    while (it.next()) |part| if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, part, " \t"), token)) return true;
    return false;
}

/// Keeps ``fd`` out of the children the router starts, which would otherwise hold its connection open.
pub fn closeOnExec(fd: posix.fd_t) void {
    _ = @import("sys").c.fcntl(fd, posix.F.SETFD, @as(c_int, posix.FD_CLOEXEC));
}

fn connect(port: u16) ?posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return null;
    const fd: posix.socket_t = @intCast(rc);
    closeOnExec(fd);
    var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) {
        _ = posix.system.close(fd);
        return null;
    }
    return fd;
}

fn send(fd: posix.socket_t, bytes: []const u8) bool {
    var at: usize = 0;
    while (at < bytes.len) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.OUT, .revents = 0 }};
        if ((posix.poll(&fds, 30_000) catch return false) == 0) return false;
        const rc = posix.system.write(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => at += @intCast(rc),
            .AGAIN, .INTR => {},
            else => return false,
        }
    }
    return true;
}

/// The reply head's status and framing.
pub const Head = struct { status: u16, length: ?u64 = null, chunked: bool = false, close: bool = false };

/// A reply head through its blank line; null when it is not an HTTP status line with valid framing.
pub fn parseHead(head: []const u8) ?Head {
    var lines = std.mem.splitSequence(u8, head, "\r\n");
    const first = lines.next() orelse return null;
    if (!std.mem.startsWith(u8, first, "HTTP/1.")) return null;
    var words = std.mem.tokenizeScalar(u8, first, ' ');
    _ = words.next();
    const code = words.next() orelse return null;
    var out: Head = .{ .status = std.fmt.parseInt(u16, code, 10) catch return null, .close = std.mem.startsWith(u8, first, "HTTP/1.0") };
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line[0..colon];
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            out.length = std.fmt.parseInt(u64, value, 10) catch return null;
        } else if (std.ascii.eqlIgnoreCase(name, "transfer-encoding")) {
            out.chunked = hasToken(value, "chunked");
        } else if (std.ascii.eqlIgnoreCase(name, "connection")) {
            if (hasToken(value, "close")) out.close = true;
            if (hasToken(value, "keep-alive")) out.close = false;
        }
    }
    return out;
}

/// Where a chunked body ends, read as it passes: the relayed bytes are never decoded or changed.
pub const Chunked = struct {
    phase: enum { size, extension, data, data_end, trailer, done } = .size,
    size: u64 = 0,
    digits: u8 = 0,
    left: u64 = 0,
    line: usize = 0,
    bad: bool = false,

    /// How many of ``bytes`` belong to the body; fewer than all once it has ended.
    pub fn feed(c: *Chunked, bytes: []const u8) usize {
        var i: usize = 0;
        while (i < bytes.len and c.phase != .done) {
            if (c.phase == .data) {
                const take: usize = @intCast(@min(c.left, bytes.len - i));
                c.left -= take;
                i += take;
                if (c.left == 0) c.phase = .data_end;
                continue;
            }
            const b = bytes[i];
            i += 1;
            switch (c.phase) {
                .size => if (std.fmt.charToDigit(b, 16)) |d| {
                    c.digits += 1;
                    if (c.digits > 15) c.stopBad();
                    c.size = c.size * 16 + d;
                } else |_| {
                    if (b == '\n') c.sized() else if (b != '\r') c.phase = .extension;
                },
                .extension => if (b == '\n') c.sized(),
                .data_end => if (b == '\n') {
                    c.phase = .size;
                    c.size = 0;
                    c.digits = 0;
                },
                .trailer => if (b == '\n') {
                    if (c.line == 0) c.phase = .done;
                    c.line = 0;
                } else if (b != '\r') {
                    c.line += 1;
                },
                .data, .done => unreachable,
            }
        }
        return i;
    }

    fn sized(c: *Chunked) void {
        if (c.digits == 0) return c.stopBad();
        if (c.size == 0) {
            c.phase = .trailer;
            c.line = 0;
        } else {
            c.phase = .data;
            c.left = c.size;
        }
    }

    fn stopBad(c: *Chunked) void {
        c.bad = true;
        c.phase = .done;
    }
};

/// A reply body's end: none, a length, chunks, or the child closing its socket.
const Framing = union(enum) {
    none,
    length: u64,
    chunked: Chunked,
    close,

    fn of(head: Head, method: []const u8) Framing {
        if (std.mem.eql(u8, method, "HEAD") or head.status / 100 == 1 or head.status == 204 or head.status == 304) return .none;
        if (head.chunked) return .{ .chunked = .{} };
        if (head.length) |n| return .{ .length = n };
        return .close;
    }

    fn feed(f: *Framing, bytes: []const u8) usize {
        switch (f.*) {
            .none => return 0,
            .length => |*left| {
                const take: usize = @intCast(@min(left.*, bytes.len));
                left.* -= take;
                return take;
            },
            .chunked => |*c| return c.feed(bytes),
            .close => return bytes.len,
        }
    }

    fn done(f: Framing) bool {
        return switch (f) {
            .none => true,
            .length => |left| left == 0,
            .chunked => |c| c.phase == .done,
            .close => false,
        };
    }
};

const Read = union(enum) { data: usize, eof, failed, client_gone };

/// The child's next bytes, while watching the client for a hangup; nothing for ``idle_ms`` is a failure.
fn read(conn: *Conn, fd: posix.socket_t, into: []u8, limits: Limits, watch: *bool) Read {
    var waited: i64 = 0;
    while (true) {
        var fds = [_]posix.pollfd{
            .{ .fd = fd, .events = posix.POLL.IN, .revents = 0 },
            .{ .fd = if (watch.*) conn.fd else -1, .events = posix.POLL.IN, .revents = 0 },
        };
        const ready = posix.poll(&fds, 1000) catch return .failed;
        if (fds[0].revents != 0) {
            const n = posix.read(fd, into) catch return .failed;
            return if (n == 0) .eof else .{ .data = n };
        }
        // a client that sends more without leaving is checked each second, not polled, so its bytes cannot spin this
        if (fds[1].revents != 0) watch.* = false;
        if (conn.peerGone()) return .client_gone;
        if (ready == 0) waited += 1000;
        if (waited >= limits.idle_ms) return .failed;
    }
}

/// The reply from the child on ``fd``: its head and body written to the client as they come, then where it ended.
fn relay(conn: *Conn, fd: posix.socket_t, limits: Limits) Result {
    var buf: [64 << 10]u8 = undefined;
    var got: usize = 0;
    var watch = true;
    const head_end = while (true) {
        if (std.mem.indexOf(u8, buf[0..got], "\r\n\r\n")) |i| break i + 4;
        if (got == buf.len) return .{ .outcome = .no_reply };
        switch (read(conn, fd, buf[got..], limits, &watch)) {
            .data => |n| got += n,
            .eof, .failed => return .{ .outcome = .no_reply },
            .client_gone => return .{ .outcome = .client_gone },
        }
    };
    const head = parseHead(buf[0..head_end]) orelse return .{ .outcome = .no_reply };
    var framing = Framing.of(head, conn.method);
    const first = head_end + framing.feed(buf[head_end..got]);
    conn.writeAll(buf[0..first]) catch return .{ .outcome = .client_gone, .status = head.status };
    while (!framing.done()) {
        switch (read(conn, fd, &buf, limits, &watch)) {
            .data => |n| conn.writeAll(buf[0..framing.feed(buf[0..n])]) catch return .{ .outcome = .client_gone, .status = head.status },
            .eof => if (framing == .close) break else return .{ .outcome = .cut, .status = head.status },
            .failed => return .{ .outcome = .cut, .status = head.status },
            .client_gone => return .{ .outcome = .client_gone, .status = head.status },
        }
    }
    if (framing == .chunked and framing.chunked.bad) return .{ .outcome = .cut, .status = head.status };
    if (head.close or framing == .close) conn.close = true;
    conn.response_complete = true;
    return .{ .outcome = .done, .status = head.status };
}

test "serve --router: a chunked body ends after its last chunk and trailers, however the bytes are split" {
    const body = "4\r\nWiki\r\n5;x=1\r\npedia\r\n0\r\nX-T: 1\r\n\r\n";
    const then = body ++ "next";
    for (1..body.len + 1) |cut| {
        var c: Chunked = .{};
        const first = c.feed(then[0..cut]);
        try std.testing.expectEqual(cut, first);
        const rest = c.feed(then[cut..]);
        try std.testing.expectEqual(body.len - cut, rest);
        try std.testing.expect(c.phase == .done and !c.bad);
    }
    var bad: Chunked = .{};
    _ = bad.feed("zz\r\n");
    try std.testing.expect(bad.bad);
}

test "serve --router: a reply head gives its status and framing" {
    const sse = parseHead("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").?;
    try std.testing.expect(sse.status == 200 and sse.close and sse.length == null and !sse.chunked);
    const json = parseHead("HTTP/1.1 404 Not Found\r\ncontent-length: 12\r\n\r\n").?;
    try std.testing.expect(json.status == 404 and json.length.? == 12 and !json.close);
    try std.testing.expect(parseHead("HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, chunked\r\n\r\n").?.chunked);
    try std.testing.expect(parseHead("HTTP/1.0 200 OK\r\n\r\n").?.close);
    try std.testing.expect(parseHead("SSH-2.0\r\n\r\n") == null);
    try std.testing.expect(parseHead("HTTP/1.1 200 OK\r\nContent-Length: x\r\n\r\n") == null);
}

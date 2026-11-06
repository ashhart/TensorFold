//! The listening socket: SO_REUSEADDR only (never REUSEPORT, so a second server on the port fails), backlog 128.
const std = @import("std");
const platform = @import("sys");
const posix = platform.posix;
const net = std.Io.net;
const log = @import("log.zig");
const Threaded = std.Io.Threaded;

/// The run has said once that accepts ran out of descriptors; a successful accept arms it again.
var nofile_logged = std.atomic.Value(bool).init(false);

pub const Listener = struct {
    fd: posix.socket_t,
    unix_path: ?[:0]const u8 = null,

    pub fn open(address: net.IpAddress) !Listener {
        var storage: Threaded.PosixAddress = undefined;
        const len = Threaded.addressToPosix(&address, &storage);
        const family: u32 = if (address == .ip4) posix.AF.INET else posix.AF.INET6;
        const rc = posix.system.socket(family, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const fd: posix.socket_t = @intCast(rc);
        errdefer _ = posix.system.close(fd);
        const one: c_int = 1;
        try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
        switch (posix.errno(posix.system.bind(fd, &storage.any, len))) {
            .SUCCESS => {},
            .ADDRINUSE => return error.AddressInUse,
            .ACCES => return error.AccessDenied,
            else => return error.BindFailed,
        }
        if (posix.errno(posix.system.listen(fd, 128)) != .SUCCESS) return error.ListenFailed;
        return .{ .fd = fd };
    }

    /// A Unix stream socket only its owner can connect to; a path already there is refused, never replaced.
    pub fn openUnix(path: [:0]const u8) !Listener {
        if (comptime platform.windows) return error.Unsupported;
        const c = platform.c;
        var addr: c.sockaddr.un = .{ .path = undefined };
        if (path.len == 0 or path.len >= addr.path.len) return error.NameTooLong;
        @memset(&addr.path, 0);
        @memcpy(addr.path[0..path.len], path);
        const fd = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        errdefer _ = c.close(fd);
        if (c.bind(fd, @ptrCast(&addr), @sizeOf(c.sockaddr.un)) != 0) return error.BindFailed;
        errdefer _ = c.unlink(path);
        // Set before listen: no one can connect while the socket still has the default mode.
        if (c.chmod(path, 0o600) != 0) return error.AccessDenied;
        if (c.listen(fd, 128) != 0) return error.ListenFailed;
        return .{ .fd = fd, .unix_path = path };
    }

    pub fn boundAddress(l: Listener) ?net.IpAddress {
        if (l.unix_path != null) return null;
        var storage: Threaded.PosixAddress = undefined;
        var len: posix.socklen_t = @sizeOf(Threaded.PosixAddress);
        if (posix.errno(posix.system.getsockname(l.fd, &storage.any, &len)) != .SUCCESS) return null;
        return Threaded.addressFromPosix(&storage);
    }
    pub fn port(l: Listener) u16 {
        return if (l.boundAddress()) |a| a.getPort() else 0;
    }

    pub const Accepted = struct { fd: posix.socket_t, peer: net.IpAddress };

    /// The next connection; null once ``stop`` is set or the socket fails.
    pub fn accept(l: Listener, stop: *const std.atomic.Value(bool)) ?Accepted {
        while (true) {
            if (stop.load(.acquire)) return null;
            var fds = [_]posix.pollfd{.{ .fd = l.fd, .events = posix.POLL.IN, .revents = 0 }};
            const ready = posix.poll(&fds, 200) catch return null;
            if (ready == 0) continue;
            var storage: Threaded.PosixAddress = undefined;
            var len: posix.socklen_t = @sizeOf(Threaded.PosixAddress);
            const rc = posix.system.accept(l.fd, &storage.any, &len);
            const e = posix.errno(rc);
            switch (e) {
                .SUCCESS => {
                    const fd: posix.socket_t = @intCast(rc);
                    // A Unix peer has no address or Nagle; it is local, so it reads as loopback.
                    if (l.unix_path != null) return .{ .fd = fd, .peer = .{ .ip4 = .loopback(0) } };
                    const one: c_int = 1;
                    posix.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, std.mem.asBytes(&one)) catch |err| {
                        _ = posix.system.close(fd);
                        log.line("could not configure accepted TCP connection: {s}", .{@errorName(err)});
                        continue;
                    };
                    nofile_logged.store(false, .release);
                    return .{ .fd = fd, .peer = Threaded.addressFromPosix(&storage) };
                },
                .INTR, .AGAIN, .CONNABORTED, .MFILE, .NFILE, .NOBUFS, .NOMEM => {
                    if (e != .INTR) {
                        if (firstNofile(e)) log.line("out of file descriptors ({s}): new connections wait until open ones close; raise the open-file limit (ulimit -n)", .{@tagName(e)});
                        std.Io.sleep(log.io(), .fromMilliseconds(10), .awake) catch {};
                    }
                    continue;
                },
                else => return null,
            }
        }
    }

    pub fn close(l: Listener) void {
        _ = posix.system.shutdown(l.fd, posix.SHUT.RDWR);
        _ = posix.system.close(l.fd);
        if (!platform.windows) if (l.unix_path) |path| {
            _ = platform.c.unlink(path);
        };
    }
};

/// Whether this failure is the burst's first descriptor exhaustion (EMFILE, ENFILE): one log line per burst.
fn firstNofile(e: posix.E) bool {
    return (e == .MFILE or e == .NFILE) and !nofile_logged.swap(true, .acq_rel);
}

test "the descriptor log fires once per burst" {
    defer nofile_logged.store(false, .release);
    try std.testing.expect(firstNofile(.MFILE));
    try std.testing.expect(!firstNofile(.NFILE));
    try std.testing.expect(!firstNofile(.AGAIN));
    nofile_logged.store(false, .release); // what a successful accept does
    try std.testing.expect(firstNofile(.NFILE));
}

/// The peer's address as Python's ``client_address[0]`` prints it.
pub fn peerText(buf: []u8, peer: net.IpAddress) []const u8 {
    switch (peer) {
        .ip4 => |a| return std.fmt.bufPrint(buf, "{d}.{d}.{d}.{d}", .{ a.bytes[0], a.bytes[1], a.bytes[2], a.bytes[3] }) catch "",
        .ip6 => |a| {
            var groups: [8]u16 = undefined;
            for (0..8) |i| groups[i] = std.mem.readInt(u16, a.bytes[2 * i ..][0..2], .big);
            var best_start: usize = 8;
            var best_len: usize = 0;
            var i: usize = 0;
            while (i < 8) {
                if (groups[i] != 0) {
                    i += 1;
                    continue;
                }
                const start = i;
                while (i < 8 and groups[i] == 0) i += 1;
                if (i - start > best_len and i - start > 1) {
                    best_start = start;
                    best_len = i - start;
                }
            }
            var w: std.Io.Writer = .fixed(buf);
            i = 0;
            while (i < 8) {
                if (i == best_start) {
                    w.writeAll(if (i == 0) "::" else ":") catch {};
                    i += best_len;
                    continue;
                }
                w.print("{x}", .{groups[i]}) catch {};
                if (i < 7) w.writeAll(":") catch {};
                i += 1;
            }
            return w.buffered();
        },
    }
}

test "peer text" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("127.0.0.1", peerText(&buf, .{ .ip4 = .loopback(0) }));
    try std.testing.expectEqualStrings("::1", peerText(&buf, .{ .ip6 = .loopback(0) }));
}

test "accepted IPv4 and IPv6 TCP streams disable Nagle for incremental responses" {
    for ([_]net.IpAddress{ .{ .ip4 = .loopback(0) }, .{ .ip6 = .loopback(0) } }) |address| {
        const listener = try Listener.open(address);
        defer listener.close();
        const rc = posix.system.socket(if (address == .ip4) posix.AF.INET else posix.AF.INET6, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const client: posix.socket_t = @intCast(rc);
        defer _ = posix.system.close(client);
        const peer: net.IpAddress = if (address == .ip4) .{ .ip4 = .loopback(listener.port()) } else .{ .ip6 = .loopback(listener.port()) };
        var storage: Threaded.PosixAddress = undefined;
        const len = Threaded.addressToPosix(&peer, &storage);
        if (posix.errno(posix.system.connect(client, &storage.any, len)) != .SUCCESS) return error.ConnectFailed;
        const accepted = acceptWithin(listener, 2000) orelse return error.AcceptFailed;
        defer _ = posix.system.close(accepted.fd);
        if (!platform.windows) { // sys wraps no getsockopt to read the option back on Windows
            var value: c_int = 0;
            var size: posix.socklen_t = @sizeOf(c_int);
            try std.testing.expect(std.c.getsockopt(accepted.fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, &value, &size) == 0);
            try std.testing.expect(value != 0);
        }
    }
}

fn acceptWithin(listener: Listener, timeout_ms: u32) ?Listener.Accepted {
    const Watchdog = struct {
        stop: std.atomic.Value(bool) = .init(false),
        done: std.atomic.Value(bool) = .init(false),

        fn run(w: *@This(), milliseconds: u32) void {
            var elapsed: u32 = 0;
            while (elapsed < milliseconds and !w.done.load(.acquire)) : (elapsed += 10) {
                std.Io.sleep(log.io(), .fromMilliseconds(10), .awake) catch break;
            }
            if (!w.done.load(.acquire)) w.stop.store(true, .release);
        }
    };
    var watchdog: Watchdog = .{};
    const thread = std.Thread.spawn(.{}, Watchdog.run, .{ &watchdog, timeout_ms }) catch return null;
    defer {
        watchdog.done.store(true, .release);
        thread.join();
    }
    return listener.accept(&watchdog.stop);
}

test "the accepted-stream test stops when no usable connection arrives" {
    const listener = try Listener.open(.{ .ip4 = .loopback(0) });
    defer listener.close();
    try std.testing.expect(acceptWithin(listener, 20) == null);
}

test "a Unix socket listener is owner-only, accepts, refuses an existing path and removes its path" {
    if (platform.windows) return error.SkipZigTest;
    const c = platform.c;
    var buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "/tmp/tf-listener-{d}.sock", .{c.getpid()}, 0);
    const listener = try Listener.openUnix(path);
    var closed = false;
    defer if (!closed) listener.close();
    try std.testing.expectError(error.BindFailed, Listener.openUnix(path));
    var st: c.Stat = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.stat(path, &st));
    try std.testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.mode)) & 0o777);
    var addr: c.sockaddr.un = .{ .path = undefined };
    @memset(&addr.path, 0);
    @memcpy(addr.path[0..path.len], path);
    const client = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
    try std.testing.expect(client >= 0);
    defer _ = c.close(client);
    try std.testing.expectEqual(@as(c_int, 0), c.connect(client, @ptrCast(&addr), @sizeOf(c.sockaddr.un)));
    const accepted = acceptWithin(listener, 2000) orelse return error.AcceptFailed;
    _ = posix.system.close(accepted.fd);
    try std.testing.expect(listener.boundAddress() == null);
    listener.close();
    closed = true;
    try std.testing.expect(c.access(path, c.F_OK) != 0);
}

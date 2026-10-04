//! The listening socket: SO_REUSEADDR only (never REUSEPORT, so a second server on the port fails), backlog 128.
const std = @import("std");
const posix = std.posix;
const net = std.Io.net;
const Threaded = std.Io.Threaded;

pub const Listener = struct {
    fd: posix.socket_t,

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

    pub fn port(l: Listener) u16 {
        var storage: Threaded.PosixAddress = undefined;
        var len: posix.socklen_t = @sizeOf(Threaded.PosixAddress);
        if (posix.errno(posix.system.getsockname(l.fd, &storage.any, &len)) != .SUCCESS) return 0;
        return Threaded.addressFromPosix(&storage).getPort();
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
            switch (posix.errno(rc)) {
                .SUCCESS => return .{ .fd = @intCast(rc), .peer = Threaded.addressFromPosix(&storage) },
                .INTR, .AGAIN, .CONNABORTED, .MFILE, .NFILE, .NOBUFS, .NOMEM => {
                    if (posix.errno(rc) != .INTR) std.Io.sleep(@import("log.zig").io(), .fromMilliseconds(10), .awake) catch {};
                    continue;
                },
                else => return null,
            }
        }
    }

    pub fn close(l: Listener) void {
        _ = posix.system.shutdown(l.fd, posix.SHUT.RDWR);
        _ = posix.system.close(l.fd);
    }
};

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

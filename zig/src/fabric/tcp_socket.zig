//! Nonblocking sockets with absolute deadlines. Numeric addresses only, including scoped IPv6.
const std = @import("std");
const builtin = @import("builtin");
const c = std.c;
const words = @import("words.zig");
const Error = @import("rdma.zig").Error;

extern "c" fn if_nametoindex([*:0]const u8) c_uint;

pub fn end(timeout_ns: u64) Error!u64 {
    return (try words.checkedNowNs()) +| timeout_ns;
}

pub fn close(fd: c.fd_t) void {
    _ = c.shutdown(fd, c.SHUT.RDWR);
    _ = c.close(fd);
}

pub fn prepare(fd: c.fd_t, tcp: bool) Error!void {
    const flags = c.fcntl(fd, c.F.GETFL);
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(c.O{ .NONBLOCK = true })));
    if (flags < 0 or c.fcntl(fd, c.F.SETFL, flags | nonblock) < 0 or
        c.fcntl(fd, c.F.SETFD, @as(c_int, c.FD_CLOEXEC)) < 0) return error.PeerDown;
    const one: c_int = 1;
    if (builtin.os.tag == .macos and c.setsockopt(fd, c.SOL.SOCKET, c.SO.NOSIGPIPE, &one, @sizeOf(c_int)) != 0) return error.PeerDown;
    if (tcp and c.setsockopt(fd, c.IPPROTO.TCP, c.TCP.NODELAY, &one, @sizeOf(c_int)) != 0) return error.PeerDown;
}

fn wait(fd: c.fd_t, events: i16, until: ?u64) Error!void {
    while (true) {
        const now = try words.checkedNowNs();
        if (until) |e| if (now >= e) return error.Timeout;
        const ms: c_int = if (until) |e| @intCast(@min(1000, (e - now + std.time.ns_per_ms - 1) / std.time.ns_per_ms)) else 1000;
        var fds = [_]c.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
        const n = c.poll(&fds, 1, ms);
        switch (c.errno(n)) {
            .SUCCESS => {
                if (n == 0) continue;
                if (fds[0].revents & events != 0) return;
                return error.PeerDown;
            },
            .INTR => continue,
            else => return error.PeerDown,
        }
    }
}

pub fn send(fd: c.fd_t, bytes: []const u8, until: u64) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        if (try words.checkedNowNs() >= until) return error.Timeout;
        const flags = if (builtin.os.tag == .linux) c.MSG.NOSIGNAL else 0;
        const n = c.send(fd, bytes.ptr + at, bytes.len - at, flags);
        switch (c.errno(n)) {
            .SUCCESS => {
                if (n == 0) return error.PeerDown;
                at += @intCast(n);
            },
            .INTR => continue,
            .AGAIN => try wait(fd, c.POLL.OUT, until),
            else => return error.PeerDown,
        }
    }
}

pub fn receive(fd: c.fd_t, bytes: []u8, until: ?u64) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        if (until) |e| if (try words.checkedNowNs() >= e) return error.Timeout;
        const n = c.recv(fd, bytes.ptr + at, bytes.len - at, 0);
        switch (c.errno(n)) {
            .SUCCESS => {
                if (n == 0) return error.PeerDown;
                at += @intCast(n);
            },
            .INTR => continue,
            .AGAIN => try wait(fd, c.POLL.IN, until),
            else => return error.PeerDown,
        }
    }
}

/// Both channels use an explicitly selected interface address; wildcards are refused.
pub fn open(host: []const u8, port: u16, rank: u32, until: u64) !c.fd_t {
    if (rank > 1 or port == 0) return error.BadTcpSettings;
    const percent = std.mem.indexOfScalar(u8, host, '%');
    const address = try std.Io.net.IpAddress.parse(host[0 .. percent orelse host.len], port);
    var storage: c.sockaddr.storage = undefined;
    var len: c.socklen_t = undefined;
    switch (address) {
        .ip4 => |ip| {
            if (percent != null or std.mem.allEqual(u8, &ip.bytes, 0)) return error.BadTcpSettings;
            const sa: *c.sockaddr.in = @ptrCast(@alignCast(&storage));
            sa.* = std.mem.zeroes(c.sockaddr.in);
            sa.family = c.AF.INET;
            sa.port = std.mem.nativeToBig(u16, port);
            sa.addr = @bitCast(ip.bytes);
            if (@hasField(c.sockaddr.in, "len")) sa.len = @sizeOf(c.sockaddr.in);
            len = @sizeOf(c.sockaddr.in);
        },
        .ip6 => |ip| {
            if (std.mem.allEqual(u8, &ip.bytes, 0)) return error.BadTcpSettings;
            const sa: *c.sockaddr.in6 = @ptrCast(@alignCast(&storage));
            sa.* = std.mem.zeroes(c.sockaddr.in6);
            sa.family = c.AF.INET6;
            sa.port = std.mem.nativeToBig(u16, port);
            sa.addr = ip.bytes;
            if (@hasField(c.sockaddr.in6, "len")) sa.len = @sizeOf(c.sockaddr.in6);
            if (percent) |at| {
                const scope = host[at + 1 ..];
                var name: [64:0]u8 = @splat(0);
                if (scope.len == 0 or scope.len >= name.len or std.mem.indexOfScalar(u8, scope, 0) != null) return error.BadTcpSettings;
                @memcpy(name[0..scope.len], scope);
                sa.scope_id = if_nametoindex(&name);
                if (sa.scope_id == 0) return error.BadTcpSettings;
            }
            len = @sizeOf(c.sockaddr.in6);
        },
    }
    while (try words.checkedNowNs() < until) {
        const fd = c.socket(@as(c_uint, storage.family), c.SOCK.STREAM, 0);
        if (fd < 0) return error.PeerDown;
        var keep = false;
        defer if (!keep) close(fd);
        try prepare(fd, true);
        if (rank == 0) {
            const one: c_int = 1;
            if (c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, &one, @sizeOf(c_int)) != 0 or
                c.bind(fd, @ptrCast(&storage), len) != 0 or c.listen(fd, 1) != 0) return error.TcpBindFailed;
            while (true) {
                try wait(fd, c.POLL.IN, until);
                const peer = c.accept(fd, null, null);
                switch (c.errno(peer)) {
                    .SUCCESS => {
                        errdefer close(peer);
                        try prepare(peer, true);
                        return peer;
                    },
                    .INTR, .AGAIN => continue,
                    else => return error.PeerDown,
                }
            }
        }
        const rc = c.connect(fd, @ptrCast(&storage), len);
        if (rc == 0) {
            keep = true;
            return fd;
        }
        if (c.errno(rc) == .INPROGRESS or c.errno(rc) == .INTR) {
            wait(fd, c.POLL.OUT, until) catch |err| {
                if (err == error.Timeout) return err;
                words.pause(10 * std.time.ns_per_ms);
                continue;
            };
            var status: c_int = 0;
            var size: c.socklen_t = @sizeOf(c_int);
            if (c.getsockopt(fd, c.SOL.SOCKET, c.SO.ERROR, &status, &size) == 0 and status == 0) {
                keep = true;
                return fd;
            }
        }
        words.pause(10 * std.time.ns_per_ms);
    }
    return error.Timeout;
}

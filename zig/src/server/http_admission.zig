//! Reserve the connection slot before creating its thread; an overload reply never waits for a slow peer.
const std = @import("std");
const platform = @import("sys");
const posix = platform.posix;
pub const default_limit: u32 = 128;
pub const response = "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 1\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";

pub fn acquire(count: *std.atomic.Value(u32), limit: u32) bool {
    var n = count.load(.acquire);
    while (n < limit) {
        n = count.cmpxchgWeak(n, n + 1, .acq_rel, .acquire) orelse return true;
    }
    return false;
}

fn nonblocking(fd: posix.socket_t) bool {
    if (platform.windows) {
        const ws = struct {
            extern "ws2_32" fn ioctlsocket(socket: usize, cmd: i32, arg: *u32) callconv(.winapi) i32;
        };
        var one: u32 = 1;
        return ws.ioctlsocket(@bitCast(fd), @bitCast(@as(u32, 0x8004667e)), &one) == 0;
    }
    const c = platform.c;
    const flags = c.fcntl(fd, c.F.GETFL);
    const flag: c_int = @bitCast(@as(u32, @bitCast(c.O{ .NONBLOCK = true })));
    return flags >= 0 and c.fcntl(fd, c.F.SETFL, flags | flag) >= 0;
}

/// Best effort on an unshared accepted socket; close even if the peer cannot receive the overload reply.
pub fn reject(fd: posix.socket_t) void {
    defer _ = posix.system.close(fd);
    if (!nonblocking(fd)) return;
    var sent: usize = 0;
    while (sent < response.len) {
        const n = posix.system.write(fd, response[sent..].ptr, response.len - sent);
        if (posix.errno(n) != .SUCCESS or n == 0) return;
        sent += @intCast(n);
    }
}

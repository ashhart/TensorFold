//! TCP for the bootstrap and the fate channel: listen, accept and connect with deadlines, whole sends and receives.
const std = @import("std");
const c = std.c;
const posix = std.posix;

pub const Fd = c.fd_t;

pub const Error = error{ SocketFailed, AddressUnresolved, AddressInUse, Timeout, Closed, Interrupted };

pub fn nowNs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub fn sleepNs(ns: u64) void {
    var ts: c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    while (c.nanosleep(&ts, &ts) != 0) {}
}

/// TCP_NODELAY (frames are tiny and latency matters) and keepalive (a vanished host fails a read in ~90 s, not never).
fn tune(fd: Fd) void {
    const one: c_int = 1;
    _ = c.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, &one, @sizeOf(c_int));
    _ = c.setsockopt(fd, posix.SOL.SOCKET, posix.SO.KEEPALIVE, &one, @sizeOf(c_int));
    const idle: c_int = 30;
    const intvl: c_int = 10;
    const cnt: c_int = 6;
    _ = c.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.KEEPIDLE, &idle, @sizeOf(c_int));
    _ = c.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.KEEPINTVL, &intvl, @sizeOf(c_int));
    _ = c.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.KEEPCNT, &cnt, @sizeOf(c_int));
}

/// Rank 0's listening socket on every IPv4 address at `port`.
pub fn listen(port: u16) Error!Fd {
    const fd = c.socket(posix.AF.INET, posix.SOCK.STREAM | posix.SOCK.CLOEXEC, 0);
    if (fd < 0) return error.SocketFailed;
    errdefer _ = c.close(fd);
    const one: c_int = 1;
    _ = c.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, &one, @sizeOf(c_int));
    var addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = 0 };
    if (c.bind(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in)) != 0) return error.AddressInUse;
    if (c.listen(fd, 64) != 0) return error.SocketFailed;
    return fd;
}

/// One connection before `deadline_ns` (monotonic).
pub fn accept(fd: Fd, deadline_ns: u64) Error!Fd {
    while (true) {
        try waitReadable(fd, deadline_ns);
        const conn = c.accept(fd, null, null);
        if (conn >= 0) {
            tune(conn);
            return conn;
        }
    }
}

/// Connects to `host:port`, retrying every 50 ms (the master may not listen yet) until `deadline_ns`.
pub fn connect(host: []const u8, port: u16, deadline_ns: u64) Error!Fd {
    var name: [256]u8 = undefined;
    if (host.len >= name.len) return error.AddressUnresolved;
    @memcpy(name[0..host.len], host);
    name[host.len] = 0;
    var service: [8]u8 = undefined;
    const svc = std.fmt.bufPrintSentinel(&service, "{d}", .{port}, 0) catch unreachable;
    const hints: c.addrinfo = .{ .flags = .{}, .family = posix.AF.INET, .socktype = posix.SOCK.STREAM, .protocol = 0, .addrlen = 0, .addr = null, .canonname = null, .next = null };
    while (true) {
        var res: ?*c.addrinfo = null;
        if (@backingInt(c.getaddrinfo(name[0..host.len :0], svc, &hints, &res)) == 0) {
            defer c.freeaddrinfo(res.?);
            const fd = c.socket(posix.AF.INET, posix.SOCK.STREAM | posix.SOCK.CLOEXEC, 0);
            if (fd < 0) return error.SocketFailed;
            if (c.connect(fd, res.?.addr.?, res.?.addrlen) == 0) {
                tune(fd);
                return fd;
            }
            _ = c.close(fd);
        }
        if (nowNs() >= deadline_ns) return error.Timeout;
        sleepNs(50 * std.time.ns_per_ms);
    }
}

fn waitReadable(fd: Fd, deadline_ns: u64) Error!void {
    while (true) {
        const t = nowNs();
        if (t >= deadline_ns) return error.Timeout;
        const ms: c_int = @intCast(@min((deadline_ns - t) / std.time.ns_per_ms + 1, 1000));
        var p = [_]c.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const n = c.poll(&p, 1, ms);
        if (n > 0) return;
    }
}

/// Every byte of `bytes`, or `error.Closed`.
pub fn sendAll(fd: Fd, bytes: []const u8) Error!void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = c.send(fd, bytes[off..].ptr, bytes.len - off, posix.MSG.NOSIGNAL);
        if (n <= 0) {
            if (n < 0 and posix.errno(n) == .INTR) continue;
            return error.Closed;
        }
        off += @intCast(n);
    }
}

/// Fills `out` before `deadline_ns` (null: no deadline), or `error.Closed` at the end of the stream.
pub fn recvExact(fd: Fd, out: []u8, deadline_ns: ?u64) Error!void {
    var off: usize = 0;
    while (off < out.len) {
        if (deadline_ns) |d| try waitReadable(fd, d);
        const n = c.recv(fd, out[off..].ptr, out.len - off, 0);
        if (n <= 0) {
            if (n < 0 and posix.errno(n) == .INTR) continue;
            return error.Closed;
        }
        off += @intCast(n);
    }
}

/// A non-blocking send of the whole of `bytes` if the socket buffer has room (false: dropped, or the peer is gone).
pub fn trySend(fd: Fd, bytes: []const u8) bool {
    const n = c.send(fd, bytes.ptr, bytes.len, posix.MSG.NOSIGNAL | posix.MSG.DONTWAIT);
    return n == @as(isize, @intCast(bytes.len));
}

pub fn hangup(fd: Fd) void {
    _ = c.shutdown(fd, posix.SHUT.RDWR);
}

pub fn close(fd: Fd) void {
    _ = c.close(fd);
}

/// This host's identity (the kernel's boot id, else the host name): ranks with equal ids share memory.
pub fn hostId(out: *[64]u8) []const u8 {
    @memset(out, 0);
    const f = c.open("/proc/sys/kernel/random/boot_id", .{ .ACCMODE = .RDONLY });
    if (f >= 0) {
        defer _ = c.close(f);
        const n = c.read(f, out, out.len);
        if (n > 0) return std.mem.trim(u8, out[0..@intCast(n)], " \n");
    }
    if (c.gethostname(out, out.len) != 0) return "";
    return std.mem.sliceTo(out, 0);
}

test "a listener and a client trade bytes and see the hang-up" {
    const l = try listen(0);
    defer close(l);
    var addr: posix.sockaddr.in = undefined;
    var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
    try std.testing.expectEqual(@as(c_int, 0), c.getsockname(l, @ptrCast(&addr), &len));
    const port = std.mem.bigToNative(u16, addr.port);
    const deadline = nowNs() + 5 * std.time.ns_per_s;
    const a = try connect("localhost", port, deadline);
    defer close(a);
    const b = try accept(l, deadline);
    defer close(b);
    try sendAll(a, "hello");
    var got: [5]u8 = undefined;
    try recvExact(b, &got, deadline);
    try std.testing.expectEqualStrings("hello", &got);
    try std.testing.expectError(error.Timeout, recvExact(b, &got, nowNs() + 20 * std.time.ns_per_ms));
    hangup(a);
    try std.testing.expectError(error.Closed, recvExact(b, &got, deadline));
}

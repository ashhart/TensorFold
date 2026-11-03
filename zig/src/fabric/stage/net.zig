//! TCP for the stage link on Linux and macOS: listen on one address, connect and accept with deadlines, whole sends.
const std = @import("std");
const c = std.c;
const posix = std.posix;

pub const Fd = c.fd_t;

pub const Error = error{ SocketFailed, AddressUnresolved, AddressInUse, WildcardAddress, Timeout, Closed };

/// MSG_NOSIGNAL where the platform has it; macOS sets SO_NOSIGPIPE on the socket instead.
const nosignal: u32 = if (@hasDecl(posix.MSG, "NOSIGNAL")) posix.MSG.NOSIGNAL else 0;

pub fn nowNs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub fn sleepNs(ns: u64) void {
    var ts: c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    while (c.nanosleep(&ts, &ts) != 0) {}
}

/// No Nagle (a decode row is one small frame), keepalive (a vanished host fails a read), no SIGPIPE on a dead peer.
fn tune(fd: Fd) void {
    const one: c_int = 1;
    _ = c.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, &one, @sizeOf(c_int));
    _ = c.setsockopt(fd, posix.SOL.SOCKET, posix.SO.KEEPALIVE, &one, @sizeOf(c_int));
    if (@hasDecl(posix.SO, "NOSIGPIPE")) _ = c.setsockopt(fd, posix.SOL.SOCKET, posix.SO.NOSIGPIPE, &one, @sizeOf(c_int));
    const idle: c_int = 30;
    const intvl: c_int = 10;
    const cnt: c_int = 6;
    _ = c.setsockopt(fd, posix.IPPROTO.TCP, (if (@hasDecl(posix.TCP, "KEEPIDLE")) posix.TCP.KEEPIDLE else posix.TCP.KEEPALIVE), &idle, @sizeOf(c_int));
    _ = c.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.KEEPINTVL, &intvl, @sizeOf(c_int));
    _ = c.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.KEEPCNT, &cnt, @sizeOf(c_int));
}

fn closeOnExec(fd: Fd) void {
    _ = c.fcntl(fd, posix.F.SETFD, @as(c_int, posix.FD_CLOEXEC));
}

/// The IPv4 address `host` names, in network order.
fn resolve(host: []const u8) Error!u32 {
    var name: [256]u8 = undefined;
    if (host.len >= name.len) return error.AddressUnresolved;
    @memcpy(name[0..host.len], host);
    name[host.len] = 0;
    const hints: c.addrinfo = .{ .flags = .{}, .family = posix.AF.INET, .socktype = posix.SOCK.STREAM, .protocol = 0, .addrlen = 0, .addr = null, .canonname = null, .next = null };
    var res: ?*c.addrinfo = null;
    if (@backingInt(c.getaddrinfo(name[0..host.len :0], null, &hints, &res)) != 0) return error.AddressUnresolved;
    defer c.freeaddrinfo(res.?);
    const in: *const posix.sockaddr.in = @ptrCast(@alignCast(res.?.addr.?));
    return in.addr;
}

/// A listening socket on `host`'s address only (never every interface), at `port` (0: the kernel picks one).
pub fn listen(host: []const u8, at: u16) Error!Fd {
    const ip = try resolve(host);
    if (ip == 0) return error.WildcardAddress; // 0.0.0.0 would listen on every interface
    const fd = c.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    closeOnExec(fd);
    errdefer _ = c.close(fd);
    const one: c_int = 1;
    _ = c.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, &one, @sizeOf(c_int));
    var addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, at), .addr = ip };
    if (c.bind(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in)) != 0) return error.AddressInUse;
    if (c.listen(fd, 8) != 0) return error.SocketFailed;
    return fd;
}

/// The port a listening socket holds.
pub fn port(fd: Fd) Error!u16 {
    var addr: posix.sockaddr.in = undefined;
    var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
    if (c.getsockname(fd, @ptrCast(&addr), &len) != 0) return error.SocketFailed;
    return std.mem.bigToNative(u16, addr.port);
}

/// One connection before `deadline` (monotonic ns).
pub fn accept(fd: Fd, deadline: u64) Error!Fd {
    while (true) {
        try wait(fd, posix.POLL.IN, deadline);
        const conn = c.accept(fd, null, null);
        if (conn >= 0) {
            closeOnExec(conn);
            tune(conn);
            return conn;
        }
    }
}

/// Connects to `host:port`, retrying every 50 ms (the peer may not listen yet) until `deadline`.
pub fn connect(host: []const u8, to: u16, deadline: u64) Error!Fd {
    const ip = try resolve(host);
    var addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, to), .addr = ip };
    while (true) {
        const fd = c.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        closeOnExec(fd);
        if (connectBefore(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in), deadline)) {
            tune(fd);
            return fd;
        }
        _ = c.close(fd);
        if (nowNs() >= deadline) return error.Timeout;
        sleepNs(50 * std.time.ns_per_ms);
    }
}

fn connectBefore(fd: Fd, address: *const posix.sockaddr, len: posix.socklen_t, deadline: u64) bool {
    const flags = c.fcntl(fd, posix.F.GETFL);
    if (flags < 0) return false;
    if (c.fcntl(fd, posix.F.SETFL, flags | @as(c_int, @bitCast(posix.O{ .NONBLOCK = true }))) < 0) return false;
    defer _ = c.fcntl(fd, posix.F.SETFL, flags);
    const rc = c.connect(fd, address, len);
    if (rc == 0) return true;
    if (posix.errno(rc) != .INPROGRESS) return false;
    wait(fd, posix.POLL.OUT, deadline) catch return false;
    var err: c_int = 0;
    var size: posix.socklen_t = @sizeOf(c_int);
    return c.getsockopt(fd, posix.SOL.SOCKET, posix.SO.ERROR, @ptrCast(&err), &size) == 0 and err == 0;
}

/// Until `fd` is ready for `events`, or `error.Timeout` at `deadline`; a hung-up peer counts as ready (the read fails).
fn wait(fd: Fd, events: i16, deadline: u64) Error!void {
    while (true) {
        const now = nowNs();
        if (now >= deadline) return error.Timeout;
        var p = [_]c.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
        const ms: c_int = @intCast(@min((deadline - now) / std.time.ns_per_ms + 1, 1000));
        if (c.poll(&p, 1, ms) > 0) return;
    }
}

/// Every byte of `bytes` before `deadline`, or `error.Closed` when the peer is gone.
pub fn sendAll(fd: Fd, bytes: []const u8, deadline: u64) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        try wait(fd, posix.POLL.OUT, deadline);
        const n = c.send(fd, bytes[at..].ptr, bytes.len - at, nosignal | posix.MSG.DONTWAIT);
        if (n < 0 and (posix.errno(n) == .INTR or posix.errno(n) == .AGAIN)) continue;
        if (n <= 0) return error.Closed;
        at += @intCast(n);
    }
}

/// Fills `out` before `deadline`, or `error.Closed` at the end of the stream.
pub fn recvExact(fd: Fd, out: []u8, deadline: u64) Error!void {
    var at: usize = 0;
    while (at < out.len) {
        try wait(fd, posix.POLL.IN, deadline);
        const n = c.recv(fd, out[at..].ptr, out.len - at, posix.MSG.DONTWAIT);
        if (n < 0 and (posix.errno(n) == .INTR or posix.errno(n) == .AGAIN)) continue;
        if (n <= 0) return error.Closed;
        at += @intCast(n);
    }
}

/// Ends the stream both ways, the descriptor kept: the peer sees `error.Closed`, and `close` still frees it.
pub fn hangUp(fd: Fd) void {
    _ = c.shutdown(fd, posix.SHUT.RDWR);
}

pub fn close(fd: Fd) void {
    hangUp(fd);
    _ = c.close(fd);
}

test "stage net: a listener on loopback trades bytes, times out and sees the hang-up" {
    const l = try listen("127.0.0.1", 0);
    defer close(l);
    const deadline = nowNs() + 5 * std.time.ns_per_s;
    const a = try connect("127.0.0.1", try port(l), deadline);
    const b = try accept(l, deadline);
    defer close(b);
    try sendAll(a, "hello", deadline);
    var got: [5]u8 = undefined;
    try recvExact(b, &got, deadline);
    try std.testing.expectEqualStrings("hello", &got);
    try std.testing.expectError(error.Timeout, recvExact(b, &got, nowNs() + 20 * std.time.ns_per_ms));
    close(a);
    try std.testing.expectError(error.Closed, recvExact(b, &got, deadline));
}

test "stage net: an unknown host is refused, and a connect with no listener times out" {
    try std.testing.expectError(error.AddressUnresolved, listen("no-such-host.invalid", 0));
    try std.testing.expectError(error.WildcardAddress, listen("0.0.0.0", 0));
    const l = try listen("127.0.0.1", 0);
    const free = try port(l);
    close(l);
    try std.testing.expectError(error.Timeout, connect("127.0.0.1", free, nowNs() + 120 * std.time.ns_per_ms));
}

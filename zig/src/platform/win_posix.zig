//! std.posix as the server calls it, on Windows: Winsock sockets, WSAPoll, console events as SIGINT and SIGTERM.

const std = @import("std");
const c = @import("win_c.zig");

pub const E = c.E;
pub const errno = c.errno;
pub const fd_t = c.fd_t;
pub const AT = c.AT;
pub const S = c.S;

// ---- sockets ----

pub const socket_t = isize;
pub const socklen_t = c_int;
pub const sockaddr = extern struct {
    family: u16,
    data: [14]u8,

    pub const in = extern struct { family: u16 = AF.INET, port: u16, addr: u32, zero: [8]u8 = @splat(0) };
};
pub const timeval = c.timeval;

pub const AF = struct {
    pub const UNIX: u32 = 1;
    pub const INET: u32 = 2;
    pub const INET6: u32 = 23;
};
pub const SOCK = struct {
    pub const STREAM: u32 = 1;
    pub const CLOEXEC: u32 = 0; // Winsock handles are not inherited by children unless asked
};
pub const SOL = struct {
    pub const SOCKET: i32 = 0xffff;
};
pub const SO = struct {
    /// SO_EXCLUSIVEADDRUSE: Windows' SO_REUSEADDR would let a second server take the port, the opposite of POSIX's.
    pub const REUSEADDR: u32 = @bitCast(@as(i32, -5));
    pub const RCVTIMEO: u32 = 0x1006;
    pub const SNDTIMEO: u32 = 0x1005;
    pub const KEEPALIVE: u32 = 0x0008;
};
pub const IPPROTO = struct {
    pub const TCP: i32 = 6;
};
pub const TCP = struct {
    pub const NODELAY: u32 = 1;
    pub const KEEPIDLE: u32 = 3; // TCP_KEEPALIVE: seconds idle before the first probe
    pub const KEEPINTVL: u32 = 17;
    pub const KEEPCNT: u32 = 16;
};
pub const SHUT = struct {
    pub const RD: i32 = 0;
    pub const WR: i32 = 1;
    pub const RDWR: i32 = 2;
};
pub const MSG = struct {
    pub const PEEK: u32 = 0x2;
    /// Winsock has no per-call nonblocking flag; every caller polls first, so the receive never waits.
    pub const DONTWAIT: u32 = 0;
};
pub const POLL = struct {
    pub const IN: i16 = 0x0300; // POLLRDNORM | POLLRDBAND
    /// WSAPoll refuses POLLPRI; out-of-band data is not a signal the server uses.
    pub const PRI: i16 = 0;
    pub const OUT: i16 = 0x0010;
    pub const ERR: i16 = 0x0001;
    pub const HUP: i16 = 0x0002;
    pub const NVAL: i16 = 0x0004;
};
pub const pollfd = extern struct { fd: socket_t, events: i16, revents: i16 };

const WSAData = extern struct { version: u16, high: u16, rest: [400]u8 };
const ws = struct {
    extern "ws2_32" fn WSAStartup(version: u16, data: *WSAData) callconv(.winapi) c_int;
    extern "ws2_32" fn WSAGetLastError() callconv(.winapi) c_int;
    extern "ws2_32" fn WSAPoll(fds: [*]pollfd, n: u32, timeout: c_int) callconv(.winapi) c_int;
    extern "ws2_32" fn socket(af: c_int, kind: c_int, proto: c_int) callconv(.winapi) isize;
    extern "ws2_32" fn bind(s: isize, addr: *const anyopaque, len: c_int) callconv(.winapi) c_int;
    extern "ws2_32" fn listen(s: isize, backlog: c_int) callconv(.winapi) c_int;
    extern "ws2_32" fn connect(s: isize, addr: *const anyopaque, len: c_int) callconv(.winapi) c_int;
    extern "ws2_32" fn accept(s: isize, addr: ?*anyopaque, len: ?*c_int) callconv(.winapi) isize;
    extern "ws2_32" fn getsockname(s: isize, addr: *anyopaque, len: *c_int) callconv(.winapi) c_int;
    extern "ws2_32" fn setsockopt(s: isize, level: c_int, opt: c_int, val: [*]const u8, len: c_int) callconv(.winapi) c_int;
    extern "ws2_32" fn shutdown(s: isize, how: c_int) callconv(.winapi) c_int;
    extern "ws2_32" fn closesocket(s: isize) callconv(.winapi) c_int;
    extern "ws2_32" fn send(s: isize, buf: [*]const u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
    extern "ws2_32" fn recv(s: isize, buf: [*]u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
};
extern "c" fn _errno() *c_int;

var started = std.atomic.Value(u8).init(0);

/// WSAStartup once per process, before the first socket call.
fn start() void {
    if (started.load(.acquire) == 2) return;
    if (started.cmpxchgStrong(0, 1, .acq_rel, .acquire) == null) {
        var data: WSAData = undefined;
        _ = ws.WSAStartup(0x0202, &data);
        started.store(2, .release);
        return;
    }
    while (started.load(.acquire) != 2) std.atomic.spinLoopHint();
}

/// -1 with errno from the thread's Winsock error.
fn failed() c_int {
    const e: E = switch (ws.WSAGetLastError()) {
        10004 => .INTR,
        10013 => .ACCES,
        10014 => .FAULT,
        10022 => .INVAL,
        10024 => .MFILE,
        10035 => .AGAIN,
        10038 => .NOTSOCK,
        10048 => .ADDRINUSE,
        10049 => .ADDRNOTAVAIL,
        10050 => .NETDOWN,
        10053 => .CONNABORTED,
        10054 => .CONNRESET,
        10055 => .NOBUFS,
        10057 => .NOTCONN,
        10058 => .PIPE,
        10060 => .TIMEDOUT,
        10061 => .CONNREFUSED,
        else => .IO,
    };
    _errno().* = @backingInt(e);
    return -1;
}

fn len32(n: usize) c_int {
    return @intCast(@min(n, std.math.maxInt(c_int)));
}

pub const system = struct {
    pub fn socket(family: u32, kind: u32, proto: u32) isize {
        start();
        const fd = ws.socket(@intCast(family), @intCast(kind), @intCast(proto));
        return if (fd == -1) failed() else fd;
    }
    /// `len` as std.Io.Threaded's addressToPosix gives it (its socklen_t is unsigned on Windows).
    pub fn bind(fd: socket_t, addr: *const anyopaque, len: anytype) c_int {
        return if (ws.bind(fd, addr, @intCast(len)) != 0) failed() else 0;
    }
    pub fn connect(fd: socket_t, addr: *const anyopaque, len: anytype) c_int {
        return if (ws.connect(fd, addr, @intCast(len)) != 0) failed() else 0;
    }
    pub fn listen(fd: socket_t, backlog: u31) c_int {
        return if (ws.listen(fd, backlog) != 0) failed() else 0;
    }
    pub fn accept(fd: socket_t, addr: ?*anyopaque, len: ?*socklen_t) isize {
        const s = ws.accept(fd, addr, len);
        return if (s == -1) failed() else s;
    }
    pub fn getsockname(fd: socket_t, addr: *anyopaque, len: *socklen_t) c_int {
        return if (ws.getsockname(fd, addr, len) != 0) failed() else 0;
    }
    pub fn shutdown(fd: socket_t, how: i32) c_int {
        return if (ws.shutdown(fd, how) != 0) failed() else 0;
    }
    /// Closes a socket; files close through `sys.c.close`.
    pub fn close(fd: socket_t) c_int {
        return if (ws.closesocket(fd) != 0) failed() else 0;
    }
    /// Sends on a socket.
    pub fn write(fd: socket_t, buf: [*]const u8, len: usize) isize {
        const n = ws.send(fd, buf, len32(len), 0);
        return if (n < 0) failed() else n;
    }
    pub fn read(fd: socket_t, buf: [*]u8, len: usize) isize {
        const n = ws.recv(fd, buf, len32(len), 0);
        return if (n < 0) failed() else n;
    }
    pub fn poll(fds: [*]pollfd, n: usize, timeout_ms: i32) isize {
        const got = posixPoll(fds[0..n], timeout_ms) catch return -1;
        return @intCast(got);
    }
    pub fn recvfrom(fd: socket_t, buf: [*]u8, len: usize, flags: u32, addr: ?*anyopaque, alen: ?*socklen_t) isize {
        _ = addr;
        _ = alen;
        const n = ws.recv(fd, buf, len32(len), @intCast(flags));
        return if (n < 0) failed() else n;
    }
};

pub fn setsockopt(fd: socket_t, level: i32, opt: u32, value: []const u8) error{SetSockOptFailed}!void {
    if (opt == SO.RCVTIMEO or opt == SO.SNDTIMEO) {
        // POSIX gives a timeval; Winsock takes the milliseconds as a DWORD
        const tv: *align(1) const timeval = @ptrCast(value.ptr);
        const ms: u32 = @intCast(@as(i64, tv.sec) * 1000 + @divTrunc(@as(i64, tv.usec), 1000));
        if (ws.setsockopt(fd, level, @bitCast(opt), @ptrCast(&ms), @sizeOf(u32)) != 0) return error.SetSockOptFailed;
        return;
    }
    if (ws.setsockopt(fd, level, @bitCast(opt), value.ptr, @intCast(value.len)) != 0) return error.SetSockOptFailed;
}

/// WSAPoll, with POSIX's rule that an entry whose fd is negative is skipped (WSAPoll would fail the call).
const posixPoll = poll;

pub fn poll(fds: []pollfd, timeout_ms: i32) error{PollFailed}!usize {
    start();
    var live: [64]pollfd = undefined;
    var where: [64]usize = undefined;
    var n: usize = 0;
    for (fds, 0..) |*f, i| {
        f.revents = 0;
        if (f.fd < 0) continue;
        if (n == live.len) return error.PollFailed;
        live[n] = f.*;
        where[n] = i;
        n += 1;
    }
    if (n == 0) {
        if (timeout_ms > 0) Sleep(@intCast(timeout_ms));
        return 0;
    }
    const rc = ws.WSAPoll(&live, @intCast(n), timeout_ms);
    if (rc < 0) return error.PollFailed;
    for (live[0..n], where[0..n]) |f, i| fds[i].revents = f.revents;
    return @intCast(rc);
}

extern "kernel32" fn Sleep(ms: u32) callconv(.winapi) void;

/// Receives from a socket; 0 is the peer's orderly close, and a receive timeout (SO.RCVTIMEO) is WouldBlock.
pub fn read(fd: socket_t, buf: []u8) error{ ConnectionResetByPeer, WouldBlock }!usize {
    const n = ws.recv(fd, buf.ptr, len32(buf.len), 0);
    if (n < 0) return switch (ws.WSAGetLastError()) {
        10035, 10060 => error.WouldBlock,
        else => error.ConnectionResetByPeer,
    };
    return @intCast(n);
}

// ---- processes ----

extern "kernel32" fn TerminateProcess(process: *anyopaque, code: c_uint) callconv(.winapi) c_int;

/// kill for a spawned child's handle: TERM ends it with exit code 1; HUP and the rest have no Windows equivalent.
pub fn kill(process: std.os.windows.HANDLE, sig: SIG) error{ Unsupported, Unexpected }!void {
    if (sig != .TERM) return error.Unsupported;
    if (TerminateProcess(process, 1) == 0) return error.Unexpected;
}

// ---- signals: console control events ----

pub const SIG = enum(c_int) {
    HUP = 1,
    INT = 2,
    PIPE = 13,
    TERM = 15,
    _,

    pub const IGN: ?Handler = @ptrFromInt(1);
    pub const DFL: ?Handler = null;
};
pub const Handler = *align(1) const fn (SIG) callconv(.c) void; // align(1): IGN is address 1, as std.posix has it
pub const sigset_t = u32;
pub const Sigaction = struct {
    handler: extern union { handler: ?Handler, sigaction: ?*const anyopaque },
    mask: sigset_t,
    flags: u32,
};

pub fn sigemptyset() sigset_t {
    return 0;
}

var on_int: ?Handler = null;
var on_term: ?Handler = null;
var console_hooked = std.atomic.Value(bool).init(false);

extern "kernel32" fn SetConsoleCtrlHandler(routine: ?*const fn (u32) callconv(.winapi) c_int, add: c_int) callconv(.winapi) c_int;

/// Ctrl+C and Ctrl+Break are SIGINT; closing the console, logoff and shutdown are SIGTERM.
fn consoleEvent(kind: u32) callconv(.winapi) c_int {
    const h = switch (kind) {
        0, 1 => on_int,
        else => on_term,
    } orelse return 0;
    if (h == SIG.IGN.?) return 1;
    h(if (kind <= 1) .INT else .TERM);
    return 1;
}

/// SIGINT and SIGTERM through console events; Windows has no SIGHUP or SIGPIPE (a closed socket fails the send).
pub fn sigaction(sig: SIG, act: ?*const Sigaction, old: ?*Sigaction) void {
    if (old) |o| o.* = .{ .handler = .{ .handler = SIG.DFL }, .mask = 0, .flags = 0 };
    const a = act orelse return;
    switch (sig) {
        .INT => on_int = a.handler.handler,
        .TERM => on_term = a.handler.handler,
        else => return,
    }
    if (!console_hooked.swap(true, .acq_rel)) _ = SetConsoleCtrlHandler(consoleEvent, 1);
}

// ---- descriptor limits: Windows has none to raise ----

pub const rlim_t = u64;
pub const rlimit = struct { cur: rlim_t, max: rlim_t };
pub const rlimit_resource = enum { NOFILE };

pub fn getrlimit(_: rlimit_resource) error{Unsupported}!rlimit {
    return error.Unsupported;
}

pub fn setrlimit(_: rlimit_resource, _: rlimit) error{Unsupported}!void {
    return error.Unsupported;
}

/// A connected stream pair over loopback TCP (Windows has no socketpair); for tests that drive a connection.
pub fn socketpair(pair: *[2]socket_t) error{SocketPair}!void {
    start();
    const l = ws.socket(@intCast(AF.INET), @intCast(SOCK.STREAM), 0);
    if (l == -1) return error.SocketPair;
    defer _ = ws.closesocket(l);
    var addr: extern struct { family: u16 = 2, port: u16 = 0, addr: [4]u8 = .{ 127, 0, 0, 1 }, zero: [8]u8 = @splat(0) } = .{};
    var len: c_int = @sizeOf(@TypeOf(addr));
    if (ws.bind(l, &addr, len) != 0 or ws.listen(l, 1) != 0 or ws.getsockname(l, &addr, &len) != 0) return error.SocketPair;
    const a = ws.socket(@intCast(AF.INET), @intCast(SOCK.STREAM), 0);
    if (a == -1) return error.SocketPair;
    errdefer _ = ws.closesocket(a);
    if (ws.connect(a, &addr, len) != 0) return error.SocketPair;
    const b = ws.accept(l, null, null);
    if (b == -1) return error.SocketPair;
    pair.* = .{ a, b };
}

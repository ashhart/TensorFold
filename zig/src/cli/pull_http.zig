//! Pull's sockets: a bounded connect, reads that give up after the stall timeout, and proxied names as tunnels.
const std = @import("std");
const builtin = @import("builtin");
const route = @import("pull_route.zig");
const Io = std.Io;
const net = Io.net;
const posix = std.posix;
const Error = net.IpAddress.ConnectError;

const threaded_vtable = blk: {
    var threaded: Io.Threaded = .init_single_threaded;
    break :blk threaded.io().vtable;
};
const vtable = blk: {
    var table = threaded_vtable.*;
    table.netConnectIp = connectIp;
    table.netLookup = lookup;
    table.operate = operate;
    break :blk table;
};

pub fn client(a: std.mem.Allocator, io: Io) std.http.Client {
    return .{ .allocator = a, .io = forClient(io) };
}

/// The threaded I/O with pull's connector, lookups and reads; macOS also waits out an interrupted connect.
pub fn forClient(io: Io) Io {
    if (io.vtable != threaded_vtable) return io;
    return .{ .userdata = io.userdata, .vtable = &vtable };
}

fn inner(userdata: ?*anyopaque) Io {
    return .{ .userdata = userdata, .vtable = threaded_vtable };
}

fn connectIp(userdata: ?*anyopaque, address: *const net.IpAddress, options: net.IpAddress.ConnectOptions) Error!net.Socket {
    if (options.mode != .stream or (options.protocol != null and options.protocol != .tcp) or options.timeout != .none)
        return threaded_vtable.netConnectIp(userdata, address, options);
    const io = inner(userdata);
    if (route.tunnelOf(io, address)) |t| return @import("pull_tunnel.zig").open(.{ .userdata = userdata, .vtable = &vtable }, t);
    const deadline = route.addressDeadline(io, address) orelse connectDeadline(io);
    return timed(io, address, null, deadline);
}

pub fn connectDeadline(io: Io) Io.Clock.Timestamp {
    return .fromNow(io, .{ .raw = .fromSeconds(route.settings.connect_s), .clock = .awake });
}

/// The millisecond budget left at `now`, or null once the absolute deadline passes.
pub fn remainingMsAt(deadline: Io.Clock.Timestamp, now: Io.Clock.Timestamp) ?u64 {
    const left = now.raw.durationTo(deadline.raw).nanoseconds;
    if (left <= 0) return null;
    return @intCast(@max(1, @divFloor(left, std.time.ns_per_ms)));
}

/// A TCP connect using only time left on `deadline`; `via` names the proxy it reaches, if any.
pub fn timed(io: Io, address: *const net.IpAddress, via: ?route.Proxy, deadline: Io.Clock.Timestamp) Error!net.Socket {
    const limit_ms = remainingMsAt(deadline, .now(io, .awake)) orelse {
        route.fail(io, .{ .connect_timeout = via });
        return error.Timeout;
    };
    const SocketOps = if (builtin.os.tag == .windows) @import("pull_windows.zig") else Posix;
    return connectWith(SocketOps, inner(io.userdata), address, limit_ms) catch |err| {
        if (err == error.Timeout) route.fail(io, .{ .connect_timeout = via });
        return err;
    };
}

pub const Resolver = *const fn (?*anyopaque, net.HostName, *Io.Queue(net.HostName.LookupResult), net.HostName.LookupOptions) net.HostName.LookupError!void;
pub const TimedLookupError = net.HostName.LookupError || Io.Timeout.Error || Io.ConcurrentError;

pub fn nativeLookup(userdata: ?*anyopaque, name: net.HostName, results: *Io.Queue(net.HostName.LookupResult), options: net.HostName.LookupOptions) net.HostName.LookupError!void {
    return threaded_vtable.netLookup(userdata, name, results, options);
}

fn runResolver(resolve: Resolver, userdata: ?*anyopaque, name: net.HostName, results: *Io.Queue(net.HostName.LookupResult), options: net.HostName.LookupOptions, done: *Io.Event, io: Io) net.HostName.LookupError!void {
    defer done.set(io);
    return resolve(userdata, name, results, options);
}

/// Resolves until `deadline`, then cancels and drains; an uncooperative resolver delays reporting but is awaited.
pub fn lookupTimed(resolve: Resolver, userdata: ?*anyopaque, io: Io, name: net.HostName, results: *Io.Queue(net.HostName.LookupResult), options: net.HostName.LookupOptions, deadline: Io.Clock.Timestamp) TimedLookupError!void {
    defer results.close(io);
    var buf: [16]net.HostName.LookupResult = undefined;
    var native_results: Io.Queue(net.HostName.LookupResult) = .init(&buf);
    var done: Io.Event = .unset;
    var future = try io.concurrent(runResolver, .{ resolve, userdata, name, &native_results, options, &done, io });
    done.waitTimeout(io, .{ .deadline = deadline }) catch |err| {
        future.cancel(io) catch {};
        future.await(io) catch {}; // the resolver has returned and every borrowed buffer is dead here
        return err;
    };
    try future.await(io);
    while (native_results.getOneUncancelable(io)) |result| {
        results.putOne(io, result) catch |err| switch (err) {
            error.Closed, error.Canceled => return error.Canceled,
        };
    } else |_| {}
}

/// A proxied host's name stands for a tunnel: it is not resolved here (the proxy resolves it).
fn lookup(userdata: ?*anyopaque, name: net.HostName, results: *Io.Queue(net.HostName.LookupResult), options: net.HostName.LookupOptions) net.HostName.LookupError!void {
    const io = inner(userdata);
    if (route.standIn(io, name.bytes, options.port)) |stand_in| {
        defer results.close(io);
        results.putOneUncancelable(io, .{ .address = stand_in }) catch {};
        results.putOneUncancelable(io, .{ .canonical_name = name }) catch {};
        return;
    }
    const deadline = route.hostDeadline(io, name.bytes, options.port) orelse connectDeadline(io);
    var buf: [16]net.HostName.LookupResult = undefined;
    var found: Io.Queue(net.HostName.LookupResult) = .init(&buf);
    lookupTimed(nativeLookup, userdata, io, name, &found, options, deadline) catch |err| {
        results.close(io);
        switch (err) {
            error.Timeout => {
                route.fail(io, .{ .connect_timeout = null });
                return error.UnknownHostName;
            },
            error.ConcurrencyUnavailable => return error.DetectingNetworkConfigurationFailed,
            else => return @errorCast(err),
        }
    };
    defer results.close(io);
    while (found.getOne(io)) |result| {
        switch (result) {
            .address => |address| route.rememberAddress(io, address, deadline) catch return error.DetectingNetworkConfigurationFailed,
            .canonical_name => {},
        }
        results.putOne(io, result) catch return error.Canceled;
    } else |err| switch (err) {
        error.Closed => {},
        error.Canceled => return error.Canceled,
    }
}

/// A socket read waits for bytes at most the stall timeout, then fails as timed out instead of waiting on.
fn operate(userdata: ?*anyopaque, op: Io.Operation) Io.Cancelable!Io.Operation.Result {
    const available = if (op == .net_read) blk: {
        if (builtin.os.tag != .windows) break :blk ready(op.net_read.socket_handle, route.settings.stall_s);
        break :blk @import("pull_windows.zig").ready(inner(userdata), op.net_read.socket_handle, route.settings.stall_s) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            else => |e| return .{ .net_read = e },
        };
    } else true;
    if (!available) {
        route.stalled();
        return .{ .net_read = error.ConnectionTimedOut };
    }
    return threaded_vtable.operate(userdata, op);
}

fn ready(fd: posix.socket_t, seconds: u32) bool {
    const ms: c_int = @intCast(@min(@as(u64, seconds) * 1000, std.math.maxInt(c_int)));
    var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
    while (true) {
        const rc = std.c.poll(&fds, 1, ms);
        if (rc < 0 and posix.errno(rc) == .INTR) continue;
        return rc != 0; // an error or a ready socket: the read itself reports it
    }
}

/// Connects without connecting the same socket again after an interrupt; error.Timeout after `limit_ms`.
pub fn connectWith(comptime Ops: type, io: Io, address: *const net.IpAddress, limit_ms: u64) Error!net.Socket {
    try io.checkCancel();
    const start = Io.Clock.awake.now(io);
    const fd = try Ops.open(io, address);
    errdefer Ops.close(fd);
    switch (Ops.begin(fd, address)) {
        .SUCCESS => {},
        .INTR, .INPROGRESS, .ALREADY => while (true) {
            try io.checkCancel();
            if (!try Ops.wait(io, fd)) {
                if (start.durationTo(Io.Clock.awake.now(io)).toNanoseconds() >= @as(i96, limit_ms) * std.time.ns_per_ms) return error.Timeout;
                continue;
            }
            try check(try Ops.finish(io, fd));
            break;
        },
        else => |err| try check(err),
    }
    try io.checkCancel();
    try Ops.blocking(io, fd);
    return Ops.local(io, fd);
}

fn check(err: posix.E) Error!void {
    return switch (err) {
        .SUCCESS => {},
        .ADDRNOTAVAIL => error.AddressUnavailable,
        .AFNOSUPPORT => error.AddressFamilyUnsupported,
        .AGAIN, .INPROGRESS => error.WouldBlock,
        .ALREADY => error.ConnectionPending,
        .CONNREFUSED => error.ConnectionRefused,
        .CONNRESET => error.ConnectionResetByPeer,
        .HOSTUNREACH => error.HostUnreachable,
        .NETUNREACH => error.NetworkUnreachable,
        .TIMEDOUT => error.Timeout,
        .ACCES => error.AccessDenied,
        .NETDOWN => error.NetworkDown,
        .MFILE => error.ProcessFdQuotaExceeded,
        .NFILE => error.SystemFdQuotaExceeded,
        .NOBUFS, .NOMEM => error.SystemResources,
        .PROTONOSUPPORT => error.ProtocolUnsupportedByAddressFamily,
        .PROTOTYPE => error.SocketModeUnsupported,
        else => error.Unexpected,
    };
}

const Posix = struct {
    const c = std.c;
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(c.O{ .NONBLOCK = true })));

    pub fn open(io: Io, address: *const net.IpAddress) Error!posix.socket_t {
        const fd = while (true) {
            const rc = c.socket(Io.Threaded.posixAddressFamily(address), c.SOCK.STREAM, c.IPPROTO.TCP);
            const err = posix.errno(rc);
            if (err == .INTR) {
                try io.checkCancel();
                continue;
            }
            try check(err);
            break rc;
        };
        errdefer close(fd);
        _ = try fcntl(io, fd, c.F.SETFD, c.FD_CLOEXEC);
        const flags = try fcntl(io, fd, c.F.GETFL, 0);
        _ = try fcntl(io, fd, c.F.SETFL, flags | nonblock);
        return fd;
    }

    pub fn close(fd: posix.socket_t) void {
        _ = c.close(fd);
    }

    pub fn begin(fd: posix.socket_t, address: *const net.IpAddress) posix.E {
        var storage: Io.Threaded.PosixAddress = undefined;
        const len = Io.Threaded.addressToPosix(address, &storage);
        return posix.errno(c.connect(fd, &storage.any, len));
    }

    pub fn wait(_: Io, fd: posix.socket_t) Error!bool {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.OUT, .revents = 0 }};
        const rc = c.poll(&fds, 1, 50);
        const err = posix.errno(rc);
        if (err == .INTR) return false;
        try check(err);
        if (fds[0].revents & posix.POLL.NVAL != 0) return error.Unexpected;
        return rc != 0;
    }

    pub fn finish(io: Io, fd: posix.socket_t) Error!posix.E {
        while (true) {
            var err: c_int = 0;
            var len: posix.socklen_t = @sizeOf(c_int);
            const rc = c.getsockopt(fd, c.SOL.SOCKET, c.SO.ERROR, @ptrCast(&err), &len);
            const result = posix.errno(rc);
            if (result == .INTR) {
                try io.checkCancel();
                continue;
            }
            try check(result);
            return @fromBackingInt(@intCast(err));
        }
    }

    pub fn blocking(io: Io, fd: posix.socket_t) Error!void {
        const flags = try fcntl(io, fd, c.F.GETFL, 0);
        _ = try fcntl(io, fd, c.F.SETFL, flags & ~nonblock);
    }

    pub fn local(io: Io, fd: posix.socket_t) Error!net.Socket {
        while (true) {
            var storage: Io.Threaded.PosixAddress = undefined;
            var len: posix.socklen_t = @sizeOf(Io.Threaded.PosixAddress);
            const rc = c.getsockname(fd, &storage.any, &len);
            const err = posix.errno(rc);
            if (err == .INTR) {
                try io.checkCancel();
                continue;
            }
            try check(err);
            return .{ .handle = fd, .address = Io.Threaded.addressFromPosix(&storage) };
        }
    }

    fn fcntl(io: Io, fd: posix.socket_t, cmd: c_int, arg: c_int) Error!c_int {
        while (true) {
            const rc = c.fcntl(fd, cmd, arg);
            const err = posix.errno(rc);
            if (err == .INTR) {
                try io.checkCancel();
                continue;
            }
            try check(err);
            return rc;
        }
    }
};

test {
    _ = @import("pull_http_test.zig");
    _ = @import("pull_proxy_test.zig");
    _ = @import("pull_windows_test.zig");
}

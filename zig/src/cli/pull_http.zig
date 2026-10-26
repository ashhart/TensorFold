//! Pull's TCP connects on macOS wait for completion after an interrupt, without connecting the same socket again.
const std = @import("std");
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
    break :blk table;
};

pub fn client(a: std.mem.Allocator, io: Io) std.http.Client {
    return .{ .allocator = a, .io = forClient(io) };
}

pub fn forClient(io: Io) Io {
    if (@import("builtin").os.tag != .macos or io.vtable != threaded_vtable) return io;
    return .{ .userdata = io.userdata, .vtable = &vtable };
}

fn connectIp(userdata: ?*anyopaque, address: *const net.IpAddress, options: net.IpAddress.ConnectOptions) Error!net.Socket {
    if (options.mode != .stream or (options.protocol != null and options.protocol != .tcp) or options.timeout != .none)
        return threaded_vtable.netConnectIp(userdata, address, options);
    return connectWith(Posix, .{ .userdata = userdata, .vtable = threaded_vtable }, address);
}

pub fn connectWith(comptime Ops: type, io: Io, address: *const net.IpAddress) Error!net.Socket {
    try io.checkCancel();
    const fd = try Ops.open(io, address);
    errdefer Ops.close(fd);
    switch (Ops.begin(fd, address)) {
        .SUCCESS => {},
        .INTR, .INPROGRESS, .ALREADY => while (true) {
            try io.checkCancel();
            if (!try Ops.wait(io, fd)) continue;
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
}

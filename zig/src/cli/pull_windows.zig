//! Windows sockets for pull: native asynchronous AFD handles, compatible with std.Io's HTTP and TLS streams.
const std = @import("std");
const win = std.os.windows;
const Io = std.Io;
const net = Io.net;
const Error = net.IpAddress.ConnectError;
const Request = @import("pull_windows_io.zig").Request(@import("pull_windows_io.zig").Native);

const ConnectPacket = extern struct {
    reserved: [3]usize = @splat(0),
    address: Io.Threaded.PosixAddress,
};

const Attempt = struct {
    socket: net.Socket,
    request: Request,
    packet: ConnectPacket = .{ .address = undefined },
};

// The request's IOSB and address must stay at one address while AFD owns them.
pub fn open(io: Io, address: *const net.IpAddress) Error!*Attempt {
    const any: net.IpAddress = switch (address.*) {
        .ip4 => .{ .ip4 = .unspecified(0) },
        .ip6 => .{ .ip6 = .unspecified(0) },
    };
    const socket = any.bind(io, .{ .mode = .stream, .protocol = .tcp }) catch |err| switch (err) {
        error.AddressInUse => return error.AddressUnavailable,
        else => |e| return e,
    };
    errdefer socket.close(io);
    const a = std.heap.page_allocator.create(Attempt) catch return error.SystemResources;
    errdefer std.heap.page_allocator.destroy(a);
    a.* = .{ .socket = socket, .request = try Request.init(socket.handle) };
    return a;
}

pub fn begin(a: *Attempt, address: *const net.IpAddress) std.posix.E {
    const len = Io.Threaded.addressToPosix(address, &a.packet.address);
    const bytes = @as([*]const u8, @ptrCast(&a.packet))[0 .. @offsetOf(ConnectPacket, "address") + len];
    return result(a.request.begin(win.IOCTL.AFD.CONNECT, bytes, &.{}));
}

pub fn wait(io: Io, a: *Attempt) Error!bool {
    return a.request.wait(io);
}

pub fn finish(_: Io, a: *Attempt) Error!std.posix.E {
    return result(a.request.status);
}

pub fn blocking(_: Io, _: *Attempt) Error!void {
    // std.Io waits for each overlapped operation; the underlying Windows handle stays asynchronous.
}

pub fn local(_: Io, a: *Attempt) Error!net.Socket {
    const socket = a.socket;
    a.request.deinit();
    std.heap.page_allocator.destroy(a);
    return socket;
}

pub fn close(a: *Attempt) void {
    a.request.deinit();
    win.CloseHandle(a.socket.handle);
    std.heap.page_allocator.destroy(a);
}

const PollHandle = extern struct { handle: win.HANDLE, events: u32, status: win.NTSTATUS = .SUCCESS };
const read_events: u32 = 0x39;
const Poll = extern struct {
    timeout: i64,
    count: u32 = 1,
    exclusive: win.BOOLEAN = .FALSE,
    reserved: [3]u8 = @splat(0),
    handles: [1]PollHandle,
};

pub fn ready(io: Io, socket: net.Socket.Handle, seconds: u32) net.Stream.Reader.Error!bool {
    // Wait for normal bytes or closure; expedited-only bytes cannot satisfy std.Io's normal receive.
    var poll: Poll = .{ .timeout = -@as(i64, seconds) * 10_000_000, .handles = .{.{ .handle = socket, .events = read_events }} };
    var request = try Request.init(socket);
    defer request.deinit();
    const start = Io.Clock.awake.now(io);
    _ = request.begin(win.IOCTL.AFD.POLL, std.mem.asBytes(&poll), std.mem.asBytes(&poll));
    while (!try request.wait(io)) {
        if (start.durationTo(Io.Clock.awake.now(io)).toNanoseconds() >= @as(i96, seconds) * std.time.ns_per_s) return false;
    }
    return readable(request.status, poll.count, poll.handles[0].events, poll.handles[0].status);
}

pub fn readable(status: win.NTSTATUS, count: u32, events: u32, handle_status: win.NTSTATUS) net.Stream.Reader.Error!bool {
    return switch (status) {
        .SUCCESS => count != 0 and (events & read_events != 0 or handle_status != .SUCCESS),
        .TIMEOUT, .IO_TIMEOUT => false,
        .CONNECTION_RESET, .CONNECTION_ABORTED, .REMOTE_DISCONNECT, .LOCAL_DISCONNECT => true,
        else => error.Unexpected,
    };
}

pub fn result(status: win.NTSTATUS) std.posix.E {
    return switch (status) {
        .SUCCESS => .SUCCESS,
        .PENDING => .INPROGRESS,
        .CONNECTION_REFUSED => .CONNREFUSED,
        .CONNECTION_RESET, .CONNECTION_ABORTED, .REMOTE_DISCONNECT => .CONNRESET,
        .TIMEOUT, .IO_TIMEOUT => .TIMEDOUT,
        .NETWORK_UNREACHABLE => .NETUNREACH,
        .HOST_UNREACHABLE => .HOSTUNREACH,
        .NETWORK_BUSY => .NETDOWN,
        .ACCESS_DENIED => .ACCES,
        .INSUFFICIENT_RESOURCES, .NO_MEMORY => .NOBUFS,
        .INVALID_ADDRESS, .INVALID_ADDRESS_COMPONENT => .ADDRNOTAVAIL,
        else => .IO,
    };
}

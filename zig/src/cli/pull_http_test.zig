const std = @import("std");
const http = @import("pull_http.zig");
const Io = std.Io;
const net = Io.net;
const Error = net.IpAddress.ConnectError;

const Fake = struct {
    var first: std.posix.E = .INTR;
    var last: std.posix.E = .SUCCESS;
    var cancel: bool = false;
    var begins: usize = 0;
    var waits: usize = 0;
    var closes: usize = 0;
    var restored: bool = false;
    var fail_restore: bool = false;
    var fail_local: bool = false;

    fn reset() void {
        first = .INTR;
        last = .SUCCESS;
        cancel = false;
        begins = 0;
        waits = 0;
        closes = 0;
        restored = false;
        fail_restore = false;
        fail_local = false;
    }

    pub fn open(_: Io, _: *const net.IpAddress) Error!std.posix.socket_t {
        return 7;
    }
    pub fn begin(_: std.posix.socket_t, _: *const net.IpAddress) std.posix.E {
        begins += 1;
        return if (begins == 1) first else .ISCONN;
    }
    pub fn wait(_: Io, _: std.posix.socket_t) Error!bool {
        waits += 1;
        if (cancel) return error.Canceled;
        return waits > 1;
    }
    pub fn finish(_: Io, _: std.posix.socket_t) Error!std.posix.E {
        return last;
    }
    pub fn blocking(_: Io, _: std.posix.socket_t) Error!void {
        if (fail_restore) return error.Unexpected;
        restored = true;
    }
    pub fn local(_: Io, fd: std.posix.socket_t) Error!net.Socket {
        if (fail_local) return error.Unexpected;
        return .{ .handle = fd, .address = .{ .ip4 = .loopback(1234) } };
    }
    pub fn close(_: std.posix.socket_t) void {
        closes += 1;
    }
};

const address: net.IpAddress = .{ .ip4 = .loopback(80) };

test "pull connect: an interrupted connect waits without reconnecting the connected socket" {
    Fake.reset();
    const socket = try http.connectWith(Fake, std.testing.io, &address);
    try std.testing.expectEqual(@as(std.posix.socket_t, 7), socket.handle);
    try std.testing.expectEqual(@as(usize, 1), Fake.begins);
    try std.testing.expectEqual(@as(usize, 2), Fake.waits);
    try std.testing.expectEqual(@as(usize, 0), Fake.closes);
    try std.testing.expect(Fake.restored);
}

test "pull connect: pending completion preserves refusal and cancellation with one close" {
    for ([_]std.posix.E{ .INTR, .INPROGRESS, .ALREADY }) |first| {
        Fake.reset();
        Fake.first = first;
        Fake.last = .CONNREFUSED;
        try std.testing.expectError(error.ConnectionRefused, http.connectWith(Fake, std.testing.io, &address));
        try std.testing.expectEqual(@as(usize, 1), Fake.begins);
        try std.testing.expectEqual(@as(usize, 1), Fake.closes);
        try std.testing.expect(!Fake.restored);
        Fake.reset();
        Fake.first = first;
        Fake.cancel = true;
        try std.testing.expectError(error.Canceled, http.connectWith(Fake, std.testing.io, &address));
        try std.testing.expectEqual(@as(usize, 1), Fake.closes);
    }
}

test "pull connect: immediate success and refusal do not wait, and unrelated I/O is retained" {
    Fake.reset();
    Fake.first = .SUCCESS;
    _ = try http.connectWith(Fake, std.testing.io, &address);
    try std.testing.expectEqual(@as(usize, 0), Fake.waits);
    try std.testing.expect(Fake.restored);
    Fake.reset();
    Fake.first = .CONNREFUSED;
    try std.testing.expectError(error.ConnectionRefused, http.connectWith(Fake, std.testing.io, &address));
    try std.testing.expectEqual(@as(usize, 0), Fake.waits);
    try std.testing.expectEqual(@as(usize, 1), Fake.closes);
    var table = std.testing.io.vtable.*;
    const custom: Io = .{ .userdata = std.testing.io.userdata, .vtable = &table };
    const kept = http.forClient(custom);
    try std.testing.expectEqual(custom.vtable, kept.vtable);
    try std.testing.expectEqual(custom.userdata, kept.userdata);
}

test "pull connect: the macOS adapter changes only the threaded TCP connector" {
    const original = std.testing.io;
    const adapted = http.forClient(original);
    try std.testing.expectEqual(original.userdata, adapted.userdata);
    inline for (@typeInfo(Io.VTable).@"struct".field_names) |name| {
        if (comptime std.mem.eql(u8, name, "netConnectIp") and @import("builtin").os.tag == .macos) {
            try std.testing.expect(@field(original.vtable, name) != @field(adapted.vtable, name));
        } else {
            try std.testing.expect(@field(original.vtable, name) == @field(adapted.vtable, name));
        }
    }
}

test "pull connect: real TCP returns a blocking close-on-exec socket and preserves refusal" {
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest;
    const io = http.forClient(std.testing.io);
    const loopback: net.IpAddress = .{ .ip4 = .loopback(0) };
    var server = try loopback.listen(io, .{});
    defer server.deinit(io);
    const stream = try server.socket.address.connect(io, .{ .mode = .stream });
    defer stream.close(io);
    const accepted = try server.accept(io);
    defer accepted.close(io);
    const c = std.c;
    const flags = c.fcntl(stream.socket.handle, c.F.GETFL);
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(c.O{ .NONBLOCK = true })));
    try std.testing.expect(flags >= 0 and flags & nonblock == 0);
    const fd_flags = c.fcntl(stream.socket.handle, c.F.GETFD);
    try std.testing.expect(fd_flags >= 0 and fd_flags & c.FD_CLOEXEC != 0);
    try std.testing.expect(stream.socket.address.getPort() != 0);
    var refused = try loopback.listen(io, .{});
    const closed_address = refused.socket.address;
    refused.deinit(io);
    try std.testing.expectError(error.ConnectionRefused, closed_address.connect(io, .{ .mode = .stream }));
}

test "pull connect: completion cleanup keeps one owner on restoration or address failure" {
    Fake.reset();
    Fake.fail_restore = true;
    try std.testing.expectError(error.Unexpected, http.connectWith(Fake, std.testing.io, &address));
    try std.testing.expectEqual(@as(usize, 1), Fake.closes);
    Fake.reset();
    Fake.fail_local = true;
    try std.testing.expectError(error.Unexpected, http.connectWith(Fake, std.testing.io, &address));
    try std.testing.expectEqual(@as(usize, 1), Fake.closes);
}

test "pull connect: a pending attempt observes real I/O cancellation and closes once" {
    const Pending = struct {
        var entered: std.atomic.Value(bool) = .init(false);
        var closes: std.atomic.Value(usize) = .init(0);

        pub const open = Fake.open;
        pub const finish = Fake.finish;
        pub const blocking = Fake.blocking;
        pub const local = Fake.local;

        pub fn begin(_: std.posix.socket_t, _: *const net.IpAddress) std.posix.E {
            return .INPROGRESS;
        }
        pub fn wait(_: Io, _: std.posix.socket_t) Error!bool {
            entered.store(true, .release);
            std.atomic.spinLoopHint();
            return false;
        }
        pub fn close(_: std.posix.socket_t) void {
            _ = closes.fetchAdd(1, .monotonic);
        }
        fn run(io: Io) Error!net.Socket {
            return http.connectWith(@This(), io, &address);
        }
    };
    Pending.entered.store(false, .monotonic);
    Pending.closes.store(0, .monotonic);
    const io = std.testing.io;
    var future = try io.concurrent(Pending.run, .{io});
    while (!Pending.entered.load(.acquire)) try Io.sleep(io, .fromMilliseconds(1), .awake);
    try std.testing.expectError(error.Canceled, future.cancel(io));
    try std.testing.expectEqual(@as(usize, 1), Pending.closes.load(.monotonic));
}

//! Native-request ownership and Windows status handling, exercised with a fake kernel on every host.
const std = @import("std");
const win = std.os.windows;
const Io = std.Io;
const socket = @import("pull_windows.zig");
const Request = @import("pull_windows_io.zig").Request(Kernel);

const Kernel = struct {
    var pending: bool = true;
    var waits: usize = 0;
    var cancellations: usize = 0;
    var drains: usize = 0;
    var closes: usize = 0;
    var iosb: ?*win.IO_STATUS_BLOCK = null;
    var completion: win.NTSTATUS = .SUCCESS;
    var ordered: bool = true;

    fn reset() void {
        pending = true;
        waits = 0;
        cancellations = 0;
        drains = 0;
        closes = 0;
        iosb = null;
        completion = .SUCCESS;
        ordered = true;
    }

    pub fn event() error{ SystemResources, Unexpected }!win.HANDLE {
        return @ptrFromInt(8);
    }

    pub fn control(_: win.HANDLE, _: win.HANDLE, block: *win.IO_STATUS_BLOCK, _: win.CTL_CODE, _: []const u8, _: []u8) win.NTSTATUS {
        iosb = block;
        block.* = .{ .u = .{ .Status = completion }, .Information = 0 };
        return if (pending) .PENDING else completion;
    }

    pub fn wait(_: win.HANDLE, ticks: i64) win.NTSTATUS {
        std.debug.assert(ticks == -500_000);
        waits += 1;
        return if (waits == 1) .TIMEOUT else .SUCCESS;
    }

    pub fn cancel(_: win.HANDLE, block: *win.IO_STATUS_BLOCK) void {
        ordered = ordered and iosb == block and drains == 0 and closes == 0;
        cancellations += 1;
    }

    pub fn drain(_: win.HANDLE) void {
        ordered = ordered and cancellations == 1 and closes == 0;
        drains += 1;
    }

    pub fn close(_: win.HANDLE) void {
        ordered = ordered and (cancellations == 0 or drains == 1);
        closes += 1;
    }
};

test "pull Windows: pending completion waits in bounded slices and releases the event once" {
    Kernel.reset();
    var request = try Request.init(@ptrFromInt(7));
    try std.testing.expectEqual(win.NTSTATUS.PENDING, request.begin(win.IOCTL.AFD.CONNECT, &.{}, &.{}));
    try std.testing.expect(!try request.wait(std.testing.io));
    try std.testing.expect(request.pending);
    try std.testing.expect(try request.wait(std.testing.io));
    try std.testing.expect(!request.pending);
    try std.testing.expectEqual(win.NTSTATUS.SUCCESS, request.status);
    request.deinit();
    try std.testing.expectEqual(@as(usize, 0), Kernel.cancellations);
    try std.testing.expectEqual(@as(usize, 1), Kernel.closes);
}

test "pull Windows: abandoning a pending request cancels and drains before releasing storage" {
    Kernel.reset();
    var request = try Request.init(@ptrFromInt(7));
    _ = request.begin(win.IOCTL.AFD.POLL, &.{}, &.{});
    try std.testing.expect(!try request.wait(std.testing.io));
    request.deinit();
    try std.testing.expectEqual(@as(usize, 1), Kernel.cancellations);
    try std.testing.expectEqual(@as(usize, 1), Kernel.drains);
    try std.testing.expectEqual(@as(usize, 1), Kernel.closes);
    try std.testing.expect(Kernel.ordered);
}

test "pull Windows: I/O cancellation retains ownership until native cancellation has completed" {
    Kernel.reset();
    var request = try Request.init(@ptrFromInt(7));
    _ = request.begin(win.IOCTL.AFD.CONNECT, &.{}, &.{});
    var table = std.testing.io.vtable.*;
    table.checkCancel = struct {
        fn check(_: ?*anyopaque) Io.Cancelable!void {
            return error.Canceled;
        }
    }.check;
    try std.testing.expectError(error.Canceled, request.wait(.{ .userdata = std.testing.io.userdata, .vtable = &table }));
    try std.testing.expectEqual(@as(usize, 0), Kernel.waits);
    request.deinit();
    try std.testing.expect(Kernel.ordered);
    try std.testing.expectEqual(@as(usize, 1), Kernel.drains);
}

test "pull Windows: immediate refusal and asynchronous refusal retain the same connect error" {
    for ([_]bool{ false, true }) |pending| {
        Kernel.reset();
        Kernel.pending = pending;
        Kernel.completion = .CONNECTION_REFUSED;
        var request = try Request.init(@ptrFromInt(7));
        _ = request.begin(win.IOCTL.AFD.CONNECT, &.{}, &.{});
        while (!try request.wait(std.testing.io)) {}
        try std.testing.expectEqual(std.posix.E.CONNREFUSED, socket.result(request.status));
        request.deinit();
        try std.testing.expectEqual(@as(usize, 0), Kernel.cancellations);
    }
    try std.testing.expectEqual(std.posix.E.TIMEDOUT, socket.result(.IO_TIMEOUT));
    try std.testing.expectEqual(std.posix.E.ACCES, socket.result(.ACCESS_DENIED));
    try std.testing.expectEqual(std.posix.E.NETUNREACH, socket.result(.NETWORK_UNREACHABLE));
}

test "pull Windows: readability distinguishes a stall from data, disconnect and native failure" {
    try std.testing.expect(try socket.readable(.SUCCESS, 1, 1, .SUCCESS));
    try std.testing.expect(try socket.readable(.SUCCESS, 1, 0x10, .SUCCESS));
    try std.testing.expect(try socket.readable(.SUCCESS, 1, 0, .CONNECTION_RESET));
    try std.testing.expect(try socket.readable(.CONNECTION_RESET, 0, 0, .SUCCESS));
    try std.testing.expect(!try socket.readable(.IO_TIMEOUT, 0, 0, .SUCCESS));
    try std.testing.expect(!try socket.readable(.TIMEOUT, 0, 0, .SUCCESS));
    try std.testing.expect(!try socket.readable(.SUCCESS, 0, 0, .SUCCESS));
    try std.testing.expect(!try socket.readable(.SUCCESS, 1, 2, .SUCCESS));
    try std.testing.expectError(error.Unexpected, socket.readable(.INVALID_HANDLE, 0, 0, .SUCCESS));
}

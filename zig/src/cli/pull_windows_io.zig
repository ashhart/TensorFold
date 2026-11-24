//! Owns one asynchronous AFD request until completion, including cancellation before its buffers go away.
const std = @import("std");
const win = std.os.windows;
const Io = std.Io;
const InitError = error{ SystemResources, Unexpected };

pub fn Request(comptime Kernel: type) type {
    return struct {
        socket: win.HANDLE,
        event: win.HANDLE,
        iosb: win.IO_STATUS_BLOCK = undefined,
        pending: bool = false,
        status: win.NTSTATUS = .SUCCESS,

        const Self = @This();

        pub fn init(socket: win.HANDLE) InitError!Self {
            return .{ .socket = socket, .event = try Kernel.event() };
        }

        pub fn begin(r: *Self, code: win.CTL_CODE, input: []const u8, output: []u8) win.NTSTATUS {
            r.status = Kernel.control(r.socket, r.event, &r.iosb, code, input, output);
            r.pending = r.status == .PENDING;
            return r.status;
        }

        pub fn wait(r: *Self, io: Io) (Io.Cancelable || Io.UnexpectedError)!bool {
            try io.checkCancel();
            if (!r.pending) return true;
            return switch (Kernel.wait(r.event, -50 * 10_000)) {
                .SUCCESS => blk: {
                    r.pending = false;
                    r.status = r.iosb.u.Status;
                    break :blk true;
                },
                .TIMEOUT, .ALERTED, .USER_APC => false,
                else => error.Unexpected,
            };
        }

        pub fn deinit(r: *Self) void {
            if (r.pending) {
                Kernel.cancel(r.socket, &r.iosb);
                // Cancellation is asynchronous too; retain the IOSB and input/output storage until it finishes.
                Kernel.drain(r.event);
                r.pending = false;
            }
            Kernel.close(r.event);
        }
    };
}

pub const Native = struct {
    pub fn event() InitError!win.HANDLE {
        var handle: win.HANDLE = undefined;
        const status = win.ntdll.NtCreateEvent(&handle, .{
            .STANDARD = .{ .SYNCHRONIZE = true },
            .SPECIFIC = .{ .EVENT = .{ .MODIFY_STATE = true } },
        }, null, .Notification, .FALSE);
        return switch (status) {
            .SUCCESS => handle,
            .INSUFFICIENT_RESOURCES, .NO_MEMORY => error.SystemResources,
            else => error.Unexpected,
        };
    }

    pub fn control(socket: win.HANDLE, e: win.HANDLE, iosb: *win.IO_STATUS_BLOCK, code: win.CTL_CODE, input: []const u8, output: []u8) win.NTSTATUS {
        return win.ntdll.NtDeviceIoControlFile(socket, e, null, null, iosb, code, if (input.len == 0) null else input.ptr, @intCast(input.len), if (output.len == 0) null else output.ptr, @intCast(output.len));
    }

    pub fn wait(e: win.HANDLE, ticks: i64) win.NTSTATUS {
        return win.ntdll.NtWaitForSingleObject(e, .FALSE, &ticks);
    }

    pub fn cancel(socket: win.HANDLE, iosb: *win.IO_STATUS_BLOCK) void {
        var cancelled: win.IO_STATUS_BLOCK = undefined;
        _ = win.ntdll.NtCancelIoFileEx(socket, iosb, &cancelled);
    }

    pub fn drain(e: win.HANDLE) void {
        while (true) switch (win.ntdll.NtWaitForSingleObject(e, .FALSE, null)) {
            .SUCCESS => return,
            .ALERTED, .USER_APC => continue,
            else => unreachable,
        };
    }

    pub fn close(e: win.HANDLE) void {
        win.CloseHandle(e);
    }
};

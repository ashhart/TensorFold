//! The operating system as TensorFold calls it: `c` is libc's surface (std.c on POSIX, win_c.zig on Windows) and
//! `DynLib` opens shared libraries (std.DynLib on POSIX, LoadLibraryW on Windows). Every module imports this as `sys`.

const std = @import("std");
const builtin = @import("builtin");

pub const windows = builtin.os.tag == .windows;

pub const c = if (windows) @import("win_c.zig") else std.c;

/// std.posix, or its sockets, signals and limits over Winsock and console events.
pub const posix = if (windows) @import("win_posix.zig") else std.posix;

pub const DynLib = if (windows) WinDynLib else std.DynLib;

/// The CUDA driver's file name: nvcuda.dll comes with the display driver on Windows.
pub const cuda_driver = if (windows) "nvcuda.dll" else "libcuda.so.1";

extern "kernel32" fn LoadLibraryW(name: [*:0]const u16) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn FreeLibrary(module: *anyopaque) callconv(.winapi) c_int;

const WinDynLib = struct {
    module: *anyopaque,

    pub const Error = error{ FileNotFound, NameTooLong, InvalidWtf8 };

    pub fn open(path: []const u8) Error!WinDynLib {
        var buf: [1024:0]u16 = undefined;
        const n = std.unicode.wtf8ToWtf16Le(buf[0 .. buf.len - 1], path) catch return error.InvalidWtf8;
        buf[n] = 0;
        return .{ .module = LoadLibraryW(buf[0..n :0].ptr) orelse return error.FileNotFound };
    }

    pub fn openZ(path: [*:0]const u8) Error!WinDynLib {
        return open(std.mem.span(path));
    }

    pub fn close(self: *WinDynLib) void {
        _ = FreeLibrary(self.module);
    }

    pub fn lookup(self: *WinDynLib, comptime T: type, name: [:0]const u8) ?T {
        const p = GetProcAddress(self.module, name.ptr) orelse return null;
        return @ptrCast(@alignCast(p));
    }
};

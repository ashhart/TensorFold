//! The OS as TensorFold calls it, imported as `sys`: `c` is std.c or win_c.zig, `DynLib` is std.DynLib or LoadLibraryW.

const std = @import("std");
const builtin = @import("builtin");

pub const windows = builtin.os.tag == .windows;

pub const c = if (windows) @import("win_c.zig") else std.c;

/// std.posix, or its sockets, signals and limits over Winsock and console events.
pub const posix = if (windows) @import("win_posix.zig") else std.posix;

pub const DynLib = if (windows) WinDynLib else std.DynLib;
pub const optional_libraries = @import("optional_libraries.zig");

/// An environment variable's value, through libc (std.process.Environ.getPosix does not build on Windows).
pub fn getenv(name: [*:0]const u8) ?[:0]const u8 {
    return std.mem.span(c.getenv(name) orelse return null);
}

/// The CUDA driver's file name: nvcuda.dll comes with the display driver on Windows.
pub const cuda_driver = if (windows) "nvcuda.dll" else "libcuda.so.1";

/// The file a loaded library's `symbol` comes from, as dladdr's dli_fname, in `buf` on Windows.
pub fn libraryOf(symbol: *const anyopaque, buf: []u8) ?[]const u8 {
    if (!windows) @compileError("libraryOf: dladdr gives the path elsewhere");
    var module: ?*anyopaque = null;
    // GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | UNCHANGED_REFCOUNT: the module holding the address, not pinned
    if (GetModuleHandleExA(0x4 | 0x2, @ptrCast(symbol), &module) == 0) return null;
    const n = GetModuleFileNameA(module, buf.ptr, @intCast(@min(buf.len, std.math.maxInt(u32))));
    return if (n == 0 or n >= buf.len) null else buf[0..n];
}

extern "kernel32" fn GetModuleHandleExA(flags: u32, name: ?[*]const u8, module: *?*anyopaque) callconv(.winapi) c_int;
extern "kernel32" fn GetModuleFileNameA(module: ?*anyopaque, name: [*]u8, size: u32) callconv(.winapi) u32;
extern "kernel32" fn GetFullPathNameW(name: [*:0]const u16, len: u32, buf: [*]u16, part: ?*?[*:0]u16) callconv(.winapi) u32;
extern "kernel32" fn LoadLibraryW(name: [*:0]const u16) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn FreeLibrary(module: *anyopaque) callconv(.winapi) c_int;

const WinDynLib = struct {
    module: *anyopaque,

    pub const Error = error{ FileNotFound, NameTooLong, InvalidWtf8 };

    pub fn open(path: []const u8) Error!WinDynLib {
        return .{ .module = try @import("windows_library.zig").open(Kernel, path) };
    }
    const Kernel = struct {
        pub const Handle = *anyopaque;
        pub fn fullPath(path: [:0]const u16, out: []u16) usize {
            return GetFullPathNameW(path.ptr, @intCast(out.len), out.ptr, null);
        }
        pub fn load(path: [:0]const u16) ?Handle {
            return LoadLibraryW(path.ptr);
        }
    };

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

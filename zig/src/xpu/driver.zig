//! The Level Zero loader opened at run time: libze_loader.so.1 entry points in one table, and checked calls.

const std = @import("std");
const abi = @import("abi.zig");

pub const Error = error{ DriverUnavailable, MissingSymbol, ZeFailed, OutOfDeviceMemory, NoDevice, BuildFailed, OutOfMemory, Invalid };

pub const Driver = struct {
    lib: std.DynLib,
    api: abi.Api,

    pub fn open() Error!Driver {
        return openPath("libze_loader.so.1");
    }

    /// Resolves every field of `abi.Api` by its exact name; a missing symbol refuses the whole loader.
    pub fn openPath(path: []const u8) Error!Driver {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        var api: abi.Api = undefined;
        const info = @typeInfo(abi.Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        const d: Driver = .{ .lib = lib, .api = api };
        try d.check(api.zeInit(0), "zeInit");
        return d;
    }

    pub fn close(self: *Driver) void {
        self.lib.close();
    }

    /// Logs a failed call with its result code; out of device memory is told apart.
    pub fn check(_: *const Driver, res: abi.Result, what: []const u8) Error!void {
        if (res == abi.success) return;
        std.log.err("{s}: ze result 0x{x}", .{ what, res });
        return if (res == abi.error_out_of_device_memory) error.OutOfDeviceMemory else error.ZeFailed;
    }
};

//! The CUDA driver opened at run time: libcuda.so.1's versioned entry points in one table, and checked calls.

const std = @import("std");
const abi = @import("abi.zig");

pub const Error = error{ DriverUnavailable, MissingSymbol, CudaFailed, OutOfDeviceMemory, NotReady, NotFound, Invalid };

pub const Driver = struct {
    lib: std.DynLib,
    api: abi.Api,

    pub fn open() Error!Driver {
        return openPath("libcuda.so.1");
    }

    /// Resolves every field of `abi.Api` by its exact name; a missing symbol refuses the whole driver.
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
        try d.check(api.cuInit(0), "cuInit");
        return d;
    }

    pub fn close(self: *Driver) void {
        self.lib.close();
    }

    /// Logs a failed call with the driver's own name and text for the code; NOT_READY is a state, not a failure.
    pub fn check(self: *const Driver, res: abi.Result, what: []const u8) Error!void {
        if (res == abi.success) return;
        if (res == abi.error_not_ready) return error.NotReady;
        std.log.err("{s}: {s} ({d}) {s}", .{ what, self.errorName(res), res, self.errorText(res) });
        return switch (res) {
            2 => error.OutOfDeviceMemory,
            abi.error_not_found => error.NotFound,
            else => error.CudaFailed,
        };
    }

    pub fn errorName(self: *const Driver, res: abi.Result) []const u8 {
        var s: ?[*:0]const u8 = null;
        if (self.api.cuGetErrorName(res, &s) != abi.success) return "CUDA_ERROR_UNKNOWN_CODE";
        return if (s) |p| std.mem.span(p) else "CUDA_ERROR_UNKNOWN_CODE";
    }

    pub fn errorText(self: *const Driver, res: abi.Result) []const u8 {
        var s: ?[*:0]const u8 = null;
        if (self.api.cuGetErrorString(res, &s) != abi.success) return "";
        return if (s) |p| std.mem.span(p) else "";
    }

    /// The driver's CUDA version as 1000 * major + 10 * minor.
    pub fn version(self: *const Driver) Error!c_int {
        var v: c_int = 0;
        try self.check(self.api.cuDriverGetVersion(&v), "cuDriverGetVersion");
        return v;
    }

    pub fn deviceCount(self: *const Driver) Error!c_int {
        var n: c_int = 0;
        try self.check(self.api.cuDeviceGetCount(&n), "cuDeviceGetCount");
        return n;
    }

    /// Device `ordinal`'s compute capability as 10 * major + minor, read without making a context.
    pub fn capability(self: *const Driver, ordinal: c_int) Error!u32 {
        var dev: abi.Device = 0;
        try self.check(self.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var major: c_int = 0;
        var minor: c_int = 0;
        try self.check(self.api.cuDeviceGetAttribute(&major, .compute_capability_major, dev), "cuDeviceGetAttribute");
        try self.check(self.api.cuDeviceGetAttribute(&minor, .compute_capability_minor, dev), "cuDeviceGetAttribute");
        return @intCast(10 * major + minor);
    }
};

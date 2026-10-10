//! The CUDA driver opened at run time: libcuda.so.1's versioned entry points in one table, and checked calls.

const std = @import("std");
const abi = @import("abi.zig");

pub const Error = error{ DriverUnavailable, MissingSymbol, CudaFailed, OutOfDeviceMemory, NotReady, NotFound, Invalid };

pub const Driver = struct {
    lib: std.DynLib,
    api: abi.Api,
    /// cuFuncGetName (CUDA 12.3+; optional: an older driver opens without it): a failed launch names its kernel
    func_name: ?FuncGetName = null,

    pub const FuncGetName = *const fn (*?[*:0]const u8, abi.Function) callconv(.c) abi.Result;

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
        const d: Driver = .{ .lib = lib, .api = api, .func_name = lib.lookup(FuncGetName, "cuFuncGetName") };
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
        std.log.err("{s}: {s} ({d}) {s}{s}", .{ what, self.errorName(res), res, self.errorText(res), shutdownNote(res) });
        return errorOf(res);
    }

    /// The error a failed call returns (`check` without its log line, for callers that log more).
    pub fn errorOf(res: abi.Result) Error {
        return switch (res) {
            2 => error.OutOfDeviceMemory,
            abi.error_not_found => error.NotFound,
            abi.error_not_ready => error.NotReady,
            else => error.CudaFailed,
        };
    }

    /// The driver's name for the code, else knownName's: during the exit's teardown cuGetErrorName fails too.
    pub fn errorName(self: *const Driver, res: abi.Result) []const u8 {
        var s: ?[*:0]const u8 = null;
        if (self.api.cuGetErrorName(res, &s) == abi.success) if (s) |p| return std.mem.span(p);
        return knownName(res) orelse "CUDA_ERROR_UNKNOWN_CODE";
    }

    /// A kernel's symbol name (cuFuncGetName), or "?" when the driver cannot say.
    pub fn functionName(self: *const Driver, f: abi.Function) []const u8 {
        const get = self.func_name orelse return "?";
        var s: ?[*:0]const u8 = null;
        if (get(&s, f) != abi.success) return "?";
        return if (s) |p| std.mem.span(p) else "?";
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

/// CUresult names (cuda.h) for when cuGetErrorName cannot answer.
pub fn knownName(res: abi.Result) ?[]const u8 {
    return switch (res) {
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        200 => "CUDA_ERROR_INVALID_IMAGE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        209 => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        400 => "CUDA_ERROR_INVALID_HANDLE",
        500 => "CUDA_ERROR_NOT_FOUND",
        600 => "CUDA_ERROR_NOT_READY",
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        701 => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        702 => "CUDA_ERROR_LAUNCH_TIMEOUT",
        709 => "CUDA_ERROR_CONTEXT_IS_DESTROYED",
        710 => "CUDA_ERROR_ASSERT",
        714 => "CUDA_ERROR_HARDWARE_STACK_ERROR",
        715 => "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        716 => "CUDA_ERROR_MISALIGNED_ADDRESS",
        719 => "CUDA_ERROR_LAUNCH_FAILED",
        999 => "CUDA_ERROR_UNKNOWN",
        else => null,
    };
}

/// CUDA_ERROR_DEINITIALIZED: the process is exiting (a stop signal, then exit) while a thread still calls the driver.
pub fn shutdownNote(res: abi.Result) []const u8 {
    return if (res == 4) " (the CUDA driver is shutting down: the process is exiting)" else "";
}

test "CUresult names without the driver: DEINITIALIZED is 4, unknown codes stay unknown" {
    try std.testing.expectEqualStrings("CUDA_ERROR_DEINITIALIZED", knownName(4).?);
    try std.testing.expectEqualStrings("CUDA_ERROR_ILLEGAL_ADDRESS", knownName(700).?);
    try std.testing.expect(knownName(12345) == null);
    try std.testing.expect(shutdownNote(4).len > 0 and shutdownNote(1).len == 0);
}

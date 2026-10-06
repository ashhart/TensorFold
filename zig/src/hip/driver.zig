//! First HIP admission boundary; no CUDA symbol aliases or GPU kernels.
//! Signatures: ROCm/HIP include/hip/hip_runtime_api.h.
const std = @import("std");

pub const Error = error{ DriverUnavailable, MissingSymbol, HipFailed, InvalidDeviceCount };
pub const Api = struct {
    hipInit: *const fn (c_uint) callconv(.c) c_int,
    hipGetDeviceCount: *const fn (*c_int) callconv(.c) c_int,
};

pub const Driver = struct {
    lib: std.DynLib,
    api: Api,

    pub fn openPath(path: []const u8) Error!Driver {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const api = try @import("symbols.zig").resolve(Api, &lib);
        try initialize(api);
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Driver) void {
        self.lib.close();
    }

    pub fn deviceCount(self: *const Driver) Error!c_int {
        return countDevices(self.api);
    }
};

fn countDevices(api: Api) Error!c_int {
    var count: c_int = 0;
    try check(api.hipGetDeviceCount(&count));
    if (count < 0) return error.InvalidDeviceCount;
    return count;
}

fn check(result: c_int) Error!void {
    if (result != 0) return error.HipFailed;
}

fn initialize(api: Api) Error!void {
    try check(api.hipInit(0));
}

test "initialization forwards zero flags and refuses HIP failure" {
    const Mock = struct {
        fn init(flags: c_uint) callconv(.c) c_int {
            return if (flags == 0) 0 else 1;
        }
        fn failed(_: c_uint) callconv(.c) c_int {
            return 1;
        }
        fn count(out: *c_int) callconv(.c) c_int {
            out.* = 0;
            return 0;
        }
    };
    try initialize(.{ .hipInit = Mock.init, .hipGetDeviceCount = Mock.count });
    try std.testing.expectError(error.HipFailed, initialize(.{ .hipInit = Mock.failed, .hipGetDeviceCount = Mock.count }));
}

test "a real non-HIP library cannot satisfy admission" {
    const builtin = @import("builtin");
    const path = switch (builtin.os.tag) {
        .macos => "/usr/lib/libSystem.B.dylib",
        .linux => "libc.so.6",
        else => return error.SkipZigTest,
    };
    try std.testing.expectError(error.MissingSymbol, Driver.openPath(path));
}

test "HIP failure cannot become success" {
    try check(0);
    try std.testing.expectError(error.HipFailed, check(1));
    try std.testing.expectError(error.HipFailed, check(-1));
}

test "absent HIP library fails closed" {
    try std.testing.expectError(error.DriverUnavailable, Driver.openPath("/nonexistent/tensorfold-test/libamdhip64.so"));
}

test "device enumeration rejects failed calls and negative counts" {
    const Mock = struct {
        fn init(_: c_uint) callconv(.c) c_int {
            return 0;
        }
        fn negative(count: *c_int) callconv(.c) c_int {
            count.* = -1;
            return 0;
        }
        fn failed(count: *c_int) callconv(.c) c_int {
            count.* = 9;
            return 1;
        }
        fn empty(count: *c_int) callconv(.c) c_int {
            count.* = 0;
            return 0;
        }
    };
    try std.testing.expectError(error.InvalidDeviceCount, countDevices(.{ .hipInit = Mock.init, .hipGetDeviceCount = Mock.negative }));
    try std.testing.expectError(error.HipFailed, countDevices(.{ .hipInit = Mock.init, .hipGetDeviceCount = Mock.failed }));
    try std.testing.expectEqual(@as(c_int, 0), try countDevices(.{ .hipInit = Mock.init, .hipGetDeviceCount = Mock.empty }));
}

//! Complete model-free HIP runtime admission; partial symbol tables never escape.
const std = @import("std");
const abi = @import("abi.zig");
pub const Error = error{ DriverUnavailable, MissingSymbol, HipFailed, Invalid };

pub const Runtime = struct {
    lib: std.DynLib,
    api: abi.Api,

    pub fn open() Error!Runtime {
        return openPath("libamdhip64.so");
    }

    pub fn openPath(path: []const u8) Error!Runtime {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const api = try @import("symbols.zig").resolve(abi.Api, &lib);
        try check(api.hipInit(0));
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Runtime) void {
        self.lib.close();
        self.* = undefined;
    }
};

pub fn check(result: abi.Result) Error!void {
    if (result != 0) return error.HipFailed;
}

test "complete runtime refuses absent library" {
    try std.testing.expectError(error.DriverUnavailable, Runtime.openPath("/nonexistent/hip-runtime.so"));
}

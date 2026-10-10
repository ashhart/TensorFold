//! A retained HIP primary context; release dependent resources before deinit.
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const std = @import("std");

pub const Context = struct {
    r: *const runtime.Runtime,
    device: abi.Device,
    handle: abi.Context,

    pub fn init(r: *const runtime.Runtime, ordinal: c_int) runtime.Error!Context {
        if (ordinal < 0) return error.Invalid;
        var device: abi.Device = 0;
        try runtime.check(r.api.hipDeviceGet(&device, ordinal));
        var handle: abi.Context = null;
        try runtime.check(r.api.hipDevicePrimaryCtxRetain(&handle, device));
        errdefer _ = r.api.hipDevicePrimaryCtxRelease(device);
        if (handle == null) return error.Invalid;
        try runtime.check(r.api.hipCtxSetCurrent(handle));
        return .{ .r = r, .device = device, .handle = handle };
    }

    pub fn makeCurrent(self: Context) runtime.Error!void {
        try runtime.check(self.r.api.hipCtxSetCurrent(self.handle));
    }

    pub fn synchronize(self: Context) runtime.Error!void {
        try runtime.check(self.r.api.hipDeviceSynchronize());
    }

    pub fn attribute(self: Context, a: abi.DeviceAttribute) runtime.Error!c_int {
        var v: c_int = 0;
        try runtime.check(self.r.api.hipDeviceGetAttribute(&v, a, self.device));
        return v;
    }

    /// The device's marketing name ("AMD Radeon PRO W7800" ...).
    pub fn name(self: Context, buf: []u8) runtime.Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        try runtime.check(self.r.api.hipDeviceGetName(buf.ptr, @intCast(buf.len), self.device));
        return std.mem.sliceTo(buf, 0);
    }

    pub const MemInfo = struct { free: usize, total: usize };

    /// The current device's free and total bytes (hipMemGetInfo, looked up on its own: the admission set lacks it).
    pub fn memInfo(self: Context) runtime.Error!MemInfo {
        const Get = *const fn (*usize, *usize) callconv(.c) abi.Result;
        const lib: *std.DynLib = @constCast(&self.r.lib);
        const get = lib.lookup(Get, "hipMemGetInfo") orelse return error.MissingSymbol;
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try runtime.check(get(&m.free, &m.total));
        return m;
    }

    pub fn deinit(self: *Context) void {
        _ = self.r.api.hipDeviceSynchronize();
        _ = self.r.api.hipDevicePrimaryCtxRelease(self.device);
        self.* = undefined;
    }
};

test "context synchronization uses the device API and propagates errors" {
    const Mock = struct {
        var calls: usize = 0;
        var result: c_int = 0;
        fn synchronize() callconv(.c) c_int {
            calls += 1;
            return result;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipDeviceSynchronize = Mock.synchronize;
    const ctx = Context{ .r = &r, .device = 0, .handle = null };
    Mock.calls = 0;
    Mock.result = 0;
    try ctx.synchronize();
    Mock.result = 801;
    try std.testing.expectError(error.HipFailed, ctx.synchronize());
    try std.testing.expectEqual(@as(usize, 2), Mock.calls);
}

test "failed context selection releases the primary-context retain" {
    const Mock = struct {
        var released: usize = 0;
        fn device(out: *abi.Device, _: c_int) callconv(.c) c_int {
            out.* = 0;
            return 0;
        }
        fn retain(out: *abi.Context, _: abi.Device) callconv(.c) c_int {
            out.* = @ptrFromInt(16);
            return 0;
        }
        fn release(_: abi.Device) callconv(.c) c_int {
            released += 1;
            return 0;
        }
        fn current(_: abi.Context) callconv(.c) c_int {
            return 1;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipDeviceGet = Mock.device;
    r.api.hipDevicePrimaryCtxRetain = Mock.retain;
    r.api.hipDevicePrimaryCtxRelease = Mock.release;
    r.api.hipCtxSetCurrent = Mock.current;
    Mock.released = 0;
    try std.testing.expectError(error.Invalid, Context.init(&r, -1));
    try std.testing.expectError(error.HipFailed, Context.init(&r, 0));
    try std.testing.expectEqual(@as(usize, 1), Mock.released);
}

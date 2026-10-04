//! One device's primary context, the one CUDA's runtime and PyTorch share, made current on the calling thread.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Context = struct {
    d: *const Driver,
    device: abi.Device,
    handle: abi.Context,

    pub fn init(d: *const Driver, ordinal: c_int) Error!Context {
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        errdefer _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
        try d.check(d.api.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        return .{ .d = d, .device = dev, .handle = ctx };
    }

    /// Waits for all work, then drops this retain; resources made in it must be released first.
    pub fn deinit(self: *Context) void {
        _ = self.d.api.cuCtxSynchronize();
        _ = self.d.api.cuDevicePrimaryCtxRelease_v2(self.device);
        self.* = undefined;
    }

    /// Makes the context current on another thread before it calls the driver.
    pub fn makeCurrent(self: *const Context) Error!void {
        try self.d.check(self.d.api.cuCtxSetCurrent(self.handle), "cuCtxSetCurrent");
    }

    pub fn synchronize(self: *const Context) Error!void {
        try self.d.check(self.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
    }

    pub fn attribute(self: *const Context, a: abi.DeviceAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.cuDeviceGetAttribute(&v, a, self.device), "cuDeviceGetAttribute");
        return v;
    }

    /// Compute capability as 10 * major + minor (GB10: 121).
    pub fn capability(self: *const Context) Error!u32 {
        const major = try self.attribute(.compute_capability_major);
        const minor = try self.attribute(.compute_capability_minor);
        return @intCast(10 * major + minor);
    }

    pub fn name(self: *const Context, buf: []u8) Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        try self.d.check(self.d.api.cuDeviceGetName(buf.ptr, @intCast(buf.len), self.device), "cuDeviceGetName");
        return std.mem.sliceTo(buf, 0);
    }

    pub const MemInfo = struct { free: usize, total: usize };

    pub fn memInfo(self: *const Context) Error!MemInfo {
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try self.d.check(self.d.api.cuMemGetInfo_v2(&m.free, &m.total), "cuMemGetInfo");
        return m;
    }
};

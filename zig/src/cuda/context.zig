//! One device's primary context, the one CUDA's runtime and PyTorch share, made current on the calling thread.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Context = struct {
    d: *const Driver,
    device: abi.Device,
    handle: abi.Context,

    /// How the host waits for the device (CU_CTX_SCHED_*); host scheduling only, no launch or value changes.
    pub const Schedule = enum(c_uint) { auto = 0, spin = 1, yield = 2, blocking = 4 };

    /// The device's primary context's schedule, before any `init` of it (the driver ignores it on an active context).
    pub fn setSchedule(d: *const Driver, ordinal: c_int, s: Schedule) Error!void {
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        try d.check(d.api.cuDevicePrimaryCtxSetFlags_v2(dev, @backingInt(s)), "cuDevicePrimaryCtxSetFlags");
    }

    pub fn init(d: *const Driver, ordinal: c_int) Error!Context {
        // a -Dgpu server's images run on that one SM: refuse another GPU here, before any engine loads an image
        if (@import("kernels.zig").built_sm) |sm| {
            var buf: [192]u8 = undefined;
            if (@import("driver.zig").gpuRefusal(sm, try d.capability(ordinal), ordinal, &buf)) |why| {
                std.log.err("{s}", .{why});
                return error.WrongGpu;
            }
        } else if (@import("kernels.zig").target_sms.len > 0) {
            const actual_sm = try d.capability(ordinal);
            var gpu_name: [256]u8 = undefined;
            var message: [1024]u8 = undefined;
            if (@import("kernels.zig").supports(actual_sm) == false) {
                const gpu = d.deviceName(ordinal, &gpu_name) catch "unknown NVIDIA GPU";
                std.log.err("{s}", .{releaseRefusal(gpu, actual_sm, ordinal, @import("kernels.zig").target_sms, &message)});
                return error.WrongGpu;
            }
        }
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        errdefer _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
        try d.check(d.api.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        try @import("context_health.zig").retain(d, ctx);
        @import("copy_source.zig").setCurrent(ctx);
        return .{ .d = d, .device = dev, .handle = ctx };
    }

    /// For a library in a host's process: the current device's primary context (device 0's if none), or error.Invalid.
    pub fn shared(d: *const Driver) Error!Context {
        var current: abi.Context = null;
        try d.check(d.api.cuCtxGetCurrent(&current), "cuCtxGetCurrent");
        if (current == null) return init(d, 0);
        var dev: abi.Device = 0;
        try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        if (ctx != current) {
            _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
            return error.Invalid;
        }
        @import("context_health.zig").retain(d, ctx) catch |err| {
            _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
            return err;
        };
        @import("copy_source.zig").setCurrent(ctx); // the host made it current: copy sources are tagged with it
        return .{ .d = d, .device = dev, .handle = ctx };
    }

    /// Waits for all work, then drops this retain; resources made in it must be released first.
    pub fn deinit(self: *Context) void {
        self.d.synchronize() catch {};
        @import("copy_source.zig").dropSpares(self.d, self.handle); // their pinned bytes go with the context
        _ = self.d.api.cuDevicePrimaryCtxRelease_v2(self.device);
        @import("context_health.zig").release(self.d, self.handle);
        self.* = undefined;
    }

    /// Makes the context current on another thread before it calls the driver.
    pub fn makeCurrent(self: *const Context) Error!void {
        try self.d.check(self.d.api.cuCtxSetCurrent(self.handle), "cuCtxSetCurrent");
        @import("copy_source.zig").setCurrent(self.handle);
    }

    /// Readiness can be read from HTTP threads without making the context current.
    pub fn healthy(self: *const Context) bool {
        return @import("context_health.zig").healthy(self.d, self.handle);
    }

    pub fn synchronize(self: *const Context) Error!void {
        try self.d.synchronize();
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

    pub fn uuid(self: *const Context) Error![16]u8 {
        var id: [16]u8 = undefined;
        try self.d.check(self.d.api.cuDeviceGetUuid_v2(&id, self.device), "cuDeviceGetUuid");
        return id;
    }

    pub const MemInfo = struct { free: usize, total: usize };

    pub fn memInfo(self: *const Context) Error!MemInfo {
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try self.d.check(self.d.api.cuMemGetInfo_v2(&m.free, &m.total), "cuMemGetInfo");
        return m;
    }
};

/// The release refusal is composed before retaining a context or loading any module.
pub fn releaseRefusal(name: []const u8, capability: u32, ordinal: c_int, sms: []const u32, buffer: []u8) []const u8 {
    var w = std.Io.Writer.fixed(buffer);
    w.print("CUDA device {d} ({s}) has compute capability {d}.{d} (sm_{d}); this build has kernels for SMs [", .{ ordinal, name, capability / 10, capability % 10, capability }) catch return "this build has no kernels for the selected GPU";
    for (sms, 0..) |sm, i| w.print("{s}{d}", .{ if (i == 0) "" else ",", sm }) catch break;
    w.writeAll("]: use a build with kernels for this GPU") catch {};
    return w.buffered();
}

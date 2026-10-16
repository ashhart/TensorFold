//! One Level Zero device and its context on the XPU backend's loader table (ze.zig); the Arc card preferred.

const std = @import("std");
const ze = @import("ze");

pub const Error = ze.Error || error{ OutOfDeviceMemory, BuildFailed, OutOfMemory, Invalid };

const out_of_device_memory: u32 = 0x70000002;

/// Entries this runtime calls beyond ze.Api: teardown, fills and the SPIR-V build log.
pub const Extra = struct {
    zeContextDestroy: *const fn (ze.H) callconv(.c) u32,
    zeCommandListDestroy: *const fn (ze.H) callconv(.c) u32,
    zeCommandListAppendMemoryFill: *const fn (ze.H, ?*anyopaque, *const anyopaque, usize, usize, ze.H, u32, ?[*]ze.H) callconv(.c) u32,
    zeModuleDestroy: *const fn (ze.H) callconv(.c) u32,
    zeModuleBuildLogGetString: *const fn (ze.H, *usize, ?[*]u8) callconv(.c) u32,
    zeModuleBuildLogDestroy: *const fn (ze.H) callconv(.c) u32,
    zeKernelDestroy: *const fn (ze.H) callconv(.c) u32,
};

/// The loader: ze.Api plus `Extra`, every symbol resolved or the whole loader refused.
pub const Driver = struct {
    lib: std.DynLib,
    api: ze.Api,
    x: Extra,

    pub fn open() Error!Driver {
        var lib = std.DynLib.open("libze_loader.so.1") catch return error.LoaderUnavailable;
        errdefer lib.close();
        const d: Driver = .{ .lib = lib, .api = try ze.resolve(ze.Api, &lib), .x = try ze.resolve(Extra, &lib) };
        try d.check(d.api.zeInit(1), "zeInit"); // GPU drivers only
        return d;
    }

    pub fn close(self: *Driver) void {
        self.lib.close();
    }

    /// Logs a failed call with its result code; out of device memory is told apart.
    pub fn check(_: *const Driver, res: u32, what: []const u8) Error!void {
        if (res == 0) return;
        std.log.err("{s}: ze result 0x{x}", .{ what, res });
        return if (res == out_of_device_memory) error.OutOfDeviceMemory else error.ZeFailed;
    }
};

pub const max_devices = 16;

pub const Entry = struct { driver: ze.H, device: ze.H, props: ze.DeviceProperties };

pub fn entryName(e: *const Entry) []const u8 {
    return std.mem.sliceTo(&e.props.name, 0);
}

/// Every device of every driver in the loader's order (at most `max_devices`), with its properties.
pub fn enumerate(d: *const Driver, out: *[max_devices]Entry) Error![]Entry {
    var n: usize = 0;
    var nd: u32 = 8;
    var drivers: [8]ze.H = @splat(null);
    try d.check(d.api.zeDriverGet(&nd, &drivers), "zeDriverGet");
    for (drivers[0..nd]) |drv| {
        var count: u32 = max_devices;
        var devs: [max_devices]ze.H = @splat(null);
        try d.check(d.api.zeDeviceGet(drv, &count, &devs), "zeDeviceGet");
        for (devs[0..count]) |dev| {
            if (n == out.len) return out[0..n];
            out[n] = .{ .driver = drv, .device = dev, .props = .{} };
            try d.check(d.api.zeDeviceGetProperties(dev, &out[n].props), "zeDeviceGetProperties");
            n += 1;
        }
    }
    return out[0..n];
}

/// The index `--device` names, else the first name containing "Arc", else the first device.
pub fn select(names: []const []const u8, ordinal: ?u32) Error!usize {
    if (names.len == 0) return error.NoDevice;
    if (ordinal) |o| return if (o < names.len) o else error.NoDevice;
    for (names, 0..) |n, i| if (std.mem.indexOf(u8, n, "Arc") != null) return i;
    return 0;
}

pub const Context = struct {
    d: *const Driver,
    /// The device and context as the XPU backend's ze.Gpu, so its helpers serve this device too.
    gpu: ze.Gpu,

    pub fn init(d: *const Driver, ordinal: ?u32) Error!Context {
        var all: [max_devices]Entry = undefined;
        const list = try enumerate(d, &all);
        var names: [max_devices][]const u8 = undefined;
        for (list, 0..) |*e, i| names[i] = entryName(e);
        const e = list[try select(names[0..list.len], ordinal)];
        var ctx: ze.H = null;
        try d.check(d.api.zeContextCreate(e.driver, &.{}, &ctx), "zeContextCreate");
        return .{ .d = d, .gpu = .{ .lib = d.lib, .api = d.api, .driver = e.driver, .device = e.device, .context = ctx, .props = e.props } };
    }

    pub fn deinit(self: *Context) void {
        _ = self.d.x.zeContextDestroy(self.gpu.context);
        self.* = undefined;
    }

    pub fn name(self: *const Context) []const u8 {
        return std.mem.sliceTo(&self.gpu.props.name, 0);
    }

    /// The largest single allocation the device accepts.
    pub fn maxAlloc(self: *const Context) u64 {
        return self.gpu.props.maxMemAllocSize;
    }
};

test "the Arc card is preferred, an ordinal wins, an unknown one is refused" {
    const names = [_][]const u8{ "Intel(R) Graphics [0x7d67]", "Intel(R) Arc(TM) Pro B70", "Intel(R) Arc(TM) A380" };
    try std.testing.expectEqual(@as(usize, 1), try select(&names, null));
    try std.testing.expectEqual(@as(usize, 0), try select(&names, 0));
    try std.testing.expectEqual(@as(usize, 2), try select(&names, 2));
    try std.testing.expectError(error.NoDevice, select(&names, 3));
    try std.testing.expectEqual(@as(usize, 0), try select(names[0..1], null));
    try std.testing.expectError(error.NoDevice, select(&.{}, null));
}

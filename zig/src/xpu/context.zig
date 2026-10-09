//! One Level Zero device and its context; devices are numbered across all drivers, the Arc card preferred.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const max_devices = 16;

pub const Entry = struct { driver: abi.DriverHandle, device: abi.DeviceHandle, props: abi.DeviceProperties };

pub fn entryName(e: *const Entry) []const u8 {
    return std.mem.sliceTo(&e.props.name, 0);
}

/// Every device of every driver in the loader's order (at most `max_devices`), with its properties.
pub fn enumerate(d: *const Driver, out: *[max_devices]Entry) Error![]Entry {
    var n: usize = 0;
    var nd: u32 = 0;
    try d.check(d.api.zeDriverGet(&nd, null), "zeDriverGet");
    var drivers: [8]abi.DriverHandle = undefined;
    nd = @min(nd, drivers.len);
    try d.check(d.api.zeDriverGet(&nd, &drivers), "zeDriverGet");
    for (drivers[0..nd]) |drv| {
        var count: u32 = 0;
        try d.check(d.api.zeDeviceGet(drv, &count, null), "zeDeviceGet");
        var devs: [max_devices]abi.DeviceHandle = undefined;
        count = @min(count, devs.len);
        try d.check(d.api.zeDeviceGet(drv, &count, &devs), "zeDeviceGet");
        for (devs[0..count]) |dev| {
            if (n == out.len) return out[0..n];
            out[n] = .{ .driver = drv, .device = dev, .props = std.mem.zeroes(abi.DeviceProperties) };
            out[n].props.stype = abi.structure_type_device_properties;
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
    entry: Entry,
    handle: abi.ContextHandle,

    pub fn init(d: *const Driver, ordinal: ?u32) Error!Context {
        var all: [max_devices]Entry = undefined;
        const list = try enumerate(d, &all);
        var names: [max_devices][]const u8 = undefined;
        for (list, 0..) |*e, i| names[i] = entryName(e);
        const pick = try select(names[0..list.len], ordinal);
        var ctx: abi.ContextHandle = null;
        try d.check(d.api.zeContextCreate(list[pick].driver, &.{}, &ctx), "zeContextCreate");
        return .{ .d = d, .entry = list[pick], .handle = ctx };
    }

    pub fn deinit(self: *Context) void {
        _ = self.d.api.zeContextDestroy(self.handle);
        self.* = undefined;
    }

    pub fn name(self: *const Context) []const u8 {
        return entryName(&self.entry);
    }

    /// The largest single allocation the device accepts.
    pub fn maxAlloc(self: *const Context) u64 {
        return self.entry.props.max_mem_alloc_size;
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

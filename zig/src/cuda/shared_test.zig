//! Context.shared and Stream.borrowed against a fake driver: which contexts they retain, make current and release.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;

/// The driver as the calls see it: device n's primary context is 0x1000 + n, and ordinal k names device k + 5.
const Fake = struct {
    var current: abi.Context = null;
    var device_of_current: abi.Device = 0;
    var set_result: abi.Result = abi.success;
    var retains: usize = 0;
    var releases: usize = 0;
    var released: abi.Device = -1;
    var switches: usize = 0;
    var device_reads: usize = 0;
    var streams: usize = 0;

    fn primary(dev: abi.Device) abi.Context {
        return @ptrFromInt(0x1000 + @as(usize, @intCast(dev)));
    }

    fn getCurrent(ctx: *abi.Context) callconv(.c) abi.Result {
        ctx.* = current;
        return abi.success;
    }

    fn getDevice(dev: *abi.Device) callconv(.c) abi.Result {
        device_reads += 1;
        dev.* = device_of_current;
        return abi.success;
    }

    fn deviceGet(dev: *abi.Device, ordinal: c_int) callconv(.c) abi.Result {
        dev.* = ordinal + 5;
        return abi.success;
    }

    fn retain(ctx: *abi.Context, dev: abi.Device) callconv(.c) abi.Result {
        retains += 1;
        ctx.* = primary(dev);
        return abi.success;
    }

    fn release(dev: abi.Device) callconv(.c) abi.Result {
        releases += 1;
        released = dev;
        return abi.success;
    }

    fn setCurrent(ctx: abi.Context) callconv(.c) abi.Result {
        switches += 1;
        if (set_result == abi.success) current = ctx;
        return set_result;
    }

    fn synchronize() callconv(.c) abi.Result {
        return abi.success;
    }

    fn streamCreate(s: *abi.Stream, _: c_uint) callconv(.c) abi.Result {
        streams += 1;
        s.* = null;
        return abi.success;
    }

    /// A driver whose thread has `ctx` current, on device `dev`; every count starts at zero.
    fn driver(ctx: abi.Context, dev: abi.Device) Driver {
        current = ctx;
        device_of_current = dev;
        set_result = abi.success;
        retains = 0;
        releases = 0;
        released = -1;
        switches = 0;
        device_reads = 0;
        streams = 0;
        var d: Driver = undefined;
        d.api.cuCtxGetCurrent = getCurrent;
        d.api.cuCtxGetDevice = getDevice;
        d.api.cuDeviceGet = deviceGet;
        d.api.cuDevicePrimaryCtxRetain = retain;
        d.api.cuDevicePrimaryCtxRelease_v2 = release;
        d.api.cuCtxSetCurrent = setCurrent;
        d.api.cuCtxSynchronize = synchronize;
        d.api.cuStreamCreate = streamCreate;
        return d;
    }
};

test "Context.shared keeps the host's current primary context, on its own device, and makes nothing current" {
    var d = Fake.driver(Fake.primary(3), 3);
    var c = try Context.shared(&d);
    try std.testing.expectEqual(@as(abi.Device, 3), c.device);
    try std.testing.expectEqual(Fake.primary(3), c.handle);
    try std.testing.expectEqual(@as(usize, 1), Fake.retains);
    try std.testing.expectEqual(@as(usize, 0), Fake.switches);
    c.deinit();
    try std.testing.expectEqual(@as(usize, 1), Fake.releases);
    try std.testing.expectEqual(@as(abi.Device, 3), Fake.released);
    try std.testing.expectEqual(Fake.primary(3), Fake.current);
}

test "Context.shared refuses a current context that is not a primary one and drops the retain it took" {
    const foreign: abi.Context = @ptrFromInt(0x9000);
    var d = Fake.driver(foreign, 2);
    try std.testing.expectError(error.Invalid, Context.shared(&d));
    try std.testing.expectEqual(@as(usize, 1), Fake.retains);
    try std.testing.expectEqual(@as(usize, 1), Fake.releases);
    try std.testing.expectEqual(@as(abi.Device, 2), Fake.released);
    try std.testing.expectEqual(@as(usize, 0), Fake.switches);
    try std.testing.expectEqual(foreign, Fake.current);
}

test "Context.shared with no current context makes device 0's primary current" {
    var d = Fake.driver(null, 0);
    var c = try Context.shared(&d);
    try std.testing.expectEqual(@as(abi.Device, 5), c.device);
    try std.testing.expectEqual(Fake.primary(5), c.handle);
    try std.testing.expectEqual(Fake.primary(5), Fake.current);
    try std.testing.expectEqual(@as(usize, 1), Fake.switches);
    try std.testing.expectEqual(@as(usize, 0), Fake.device_reads);
    c.deinit();
    try std.testing.expectEqual(@as(usize, 1), Fake.releases);
}

test "Context.shared drops device 0's primary when making it current fails" {
    var d = Fake.driver(null, 0);
    Fake.set_result = abi.error_not_ready; // a failure check() returns without logging: an error log fails a test
    try std.testing.expectError(error.NotReady, Context.shared(&d));
    try std.testing.expectEqual(@as(usize, 1), Fake.retains);
    try std.testing.expectEqual(@as(usize, 1), Fake.releases);
    try std.testing.expectEqual(@as(abi.Device, 5), Fake.released);
    try std.testing.expectEqual(@as(abi.Context, null), Fake.current);
}

test "Stream.borrowed carries the caller's handle, null for the legacy stream, and creates no stream" {
    var d = Fake.driver(null, 0);
    const theirs: abi.Stream = @ptrFromInt(0x7000);
    const s = Stream.borrowed(&d, theirs);
    try std.testing.expectEqual(theirs, s.handle);
    try std.testing.expect(s.d == &d);
    try std.testing.expectEqual(@as(abi.Stream, null), Stream.borrowed(&d, null).handle);
    try std.testing.expectEqual(@as(usize, 0), Fake.streams);
}

//! Shared Level Zero device selection and peer opening for the Xe2 runners.
const std = @import("std");
const ze = @import("ze");

/// Discover the first Level Zero device's PCI ID; XPU_DEVICE_ID bypasses discovery (for example 0xe211 B60).
pub fn detectDeviceId() !u32 {
    if (try @import("sys").environment.read(?u32, "XPU_DEVICE_ID", null)) |id| return id;
    var lib = std.DynLib.open("libze_loader.so.1") catch return error.LoaderUnavailable;
    defer lib.close();
    const api = try ze.resolve(ze.Api, &lib);
    try ze.check(api.zeInit(1));
    var nd: u32 = 0;
    try ze.check(api.zeDriverGet(&nd, null));
    var drivers: [8]ze.H = @splat(null);
    nd = @min(nd, drivers.len);
    try ze.check(api.zeDriverGet(&nd, &drivers));
    for (drivers[0..nd]) |drv| {
        if (drv == null) continue;
        var nv: u32 = 0;
        try ze.check(api.zeDeviceGet(drv, &nv, null));
        var devs: [16]ze.H = @splat(null);
        nv = @min(nv, devs.len);
        try ze.check(api.zeDeviceGet(drv, &nv, &devs));
        for (devs[0..nv]) |dev| {
            if (dev == null) continue;
            var p: ze.DeviceProperties = .{};
            try ze.check(api.zeDeviceGetProperties(dev, &p));
            if (p.deviceId != 0) return p.deviceId;
        }
    }
    return error.NoDevice;
}

pub fn openPeers(out: []ze.Gpu) !void {
    return openPeersWith(out, detectDeviceId, ze.Gpu.openPeers);
}

fn openPeersWith(out: []ze.Gpu, detect: anytype, open: anytype) !void {
    const id = try detect();
    try open(id, out);
}

test "XPU device selection: B60 and B70 IDs reach the peer opener" {
    const Fake = struct {
        var selected: u32 = 0;
        var opened: u32 = 0;
        fn detect() !u32 {
            return selected;
        }
        fn open(id: u32, out: []ze.Gpu) !void {
            opened = id;
            try std.testing.expectEqual(@as(usize, 2), out.len);
        }
    };
    var out: [2]ze.Gpu = undefined;
    for ([_]u32{ 0xe211, 0xe223 }) |id| {
        Fake.selected = id;
        Fake.opened = 0;
        try openPeersWith(&out, Fake.detect, Fake.open);
        try std.testing.expectEqual(id, Fake.opened);
    }
}

test "XPU device selection: the environment override needs no device or driver" {
    const sys = @import("sys");
    const name = "XPU_DEVICE_ID";
    const before = if (sys.getenv(name)) |raw| try std.testing.allocator.dupeSentinel(u8, raw, 0) else null;
    defer {
        if (before) |raw| {
            _ = sys.env.setenv(name, raw, 1);
            std.testing.allocator.free(raw);
        } else _ = sys.env.unsetenv(name);
    }
    for ([_][:0]const u8{ "0xe211", "0xe223" }, [_]u32{ 0xe211, 0xe223 }) |raw, id| {
        try std.testing.expectEqual(@as(c_int, 0), sys.env.setenv(name, raw, 1));
        try std.testing.expectEqual(id, try detectDeviceId());
    }
}

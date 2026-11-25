//! Shared Level Zero device selection and peer opening for the Xe2 runners.
const std = @import("std");
const builtin = @import("builtin");
const ze = @import("ze");

/// Discover the first Level Zero device's PCI ID; XPU_DEVICE_ID bypasses discovery (for example 0xe211 B60).
pub fn detectDeviceId() !u32 {
    if (try @import("sys").environment.read(?u32, "XPU_DEVICE_ID", null)) |id| return id;
    // Level Zero's loader is Linux's libze_loader.so.1; elsewhere std.DynLib cannot open it, so no device is found
    if (comptime builtin.os.tag != .linux) return error.LoaderUnavailable;
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
    if (try @import("sys").environment.read(?u32, "XPU_DEVICE_ID", null)) |id| return ze.Gpu.openPeers(id, out);
    return openPeersWith(out, detectDeviceId, ze.Gpu.openPeers);
}

// Qwen27/TP preserve their incumbent B70 default; only NoDevice permits discovery.
fn openPeersWith(out: []ze.Gpu, detect: anytype, open: anytype) !void {
    open(0xe223, out) catch |err| {
        if (err != error.NoDevice) return err;
        const id = try detect();
        if (id == 0xe223) return error.NoDevice;
        return open(id, out);
    };
}

test "XPU device selection: B60 discovery follows only a missing B70 group" {
    const Fake = struct {
        var allow_b60: bool = true;
        var detected: u32 = 0;
        var detect_calls: usize = 0;
        var attempts: [2]u32 = .{ 0, 0 };
        var opened: usize = 0;
        fn detect() !u32 {
            detect_calls += 1;
            return detected;
        }
        fn open(id: u32, out: []ze.Gpu) !void {
            try std.testing.expectEqual(@as(usize, 2), out.len);
            attempts[opened] = id;
            opened += 1;
            if (id == 0xe223 or !allow_b60) return error.NoDevice;
        }
    };
    var out: [2]ze.Gpu = undefined;
    const Case = struct { allow_b60: bool, detected: u32 };
    for ([_]Case{
        .{ .allow_b60 = true, .detected = 0xe211 },
        .{ .allow_b60 = false, .detected = 0xe211 },
        .{ .allow_b60 = false, .detected = 0xe223 },
    }) |case| {
        Fake.allow_b60 = case.allow_b60;
        Fake.detected = case.detected;
        Fake.detect_calls = 0;
        Fake.attempts = .{ 0, 0 };
        Fake.opened = 0;
        if (case.allow_b60) {
            try openPeersWith(&out, Fake.detect, Fake.open);
        } else {
            try std.testing.expectError(error.NoDevice, openPeersWith(&out, Fake.detect, Fake.open));
        }
        try std.testing.expectEqual(if (case.detected == 0xe211) @as(usize, 2) else 1, Fake.opened);
        try std.testing.expectEqual(@as(u32, 0xe223), Fake.attempts[0]);
        if (case.detected == 0xe211) try std.testing.expectEqual(@as(u32, 0xe211), Fake.attempts[1]);
        try std.testing.expectEqual(@as(usize, 1), Fake.detect_calls);
    }
}

test "XPU device selection: opener errors other than NoDevice propagate" {
    const Fake = struct {
        var detect_calls: usize = 0;
        var open_calls: usize = 0;
        fn detect() !u32 {
            detect_calls += 1;
            return 0xe211;
        }
        fn open(id: u32, out: []ze.Gpu) !void {
            try std.testing.expectEqual(@as(u32, 0xe223), id);
            try std.testing.expectEqual(@as(usize, 1), out.len);
            open_calls += 1;
            return error.ZeFailed;
        }
    };
    var out: [1]ze.Gpu = undefined;
    try std.testing.expectError(error.ZeFailed, openPeersWith(&out, Fake.detect, Fake.open));
    try std.testing.expectEqual(@as(usize, 1), Fake.open_calls);
    try std.testing.expectEqual(@as(usize, 0), Fake.detect_calls);
}

test "XPU device selection: mixed enumerations keep B70 for TP=1 and TP=2" {
    const Fake = struct {
        var detected: u32 = 0;
        var detect_calls: usize = 0;
        var opened: u32 = 0;
        var peers: usize = 0;
        fn detect() !u32 {
            detect_calls += 1;
            return detected;
        }
        fn open(id: u32, out: []ze.Gpu) !void {
            opened = id;
            peers = out.len;
        }
    };
    var out: [2]ze.Gpu = undefined;
    for ([_]u32{ 0xe211, 0xe223 }) |first_detected| {
        for ([_]usize{ 1, 2 }) |tp| {
            Fake.detected = first_detected;
            Fake.detect_calls = 0;
            Fake.opened = 0;
            Fake.peers = 0;
            try openPeersWith(out[0..tp], Fake.detect, Fake.open);
            try std.testing.expectEqual(@as(u32, 0xe223), Fake.opened);
            try std.testing.expectEqual(tp, Fake.peers);
            try std.testing.expectEqual(@as(usize, 0), Fake.detect_calls);
        }
    }
}

test "XPU device selection: the environment override needs no device or driver" {
    const sys = @import("sys");
    const name = "XPU_DEVICE_ID";
    const before = try sys.environment.read(?u32, name, null);
    defer {
        if (before) |id| {
            var storage: [16]u8 = undefined;
            const text = std.fmt.bufPrint(&storage, "0x{x}", .{id}) catch unreachable;
            storage[text.len] = 0;
            const raw: [:0]const u8 = storage[0..text.len :0];
            _ = sys.env.setenv(name, raw, 1);
        } else _ = sys.env.unsetenv(name);
    }
    for ([_][:0]const u8{ "0xe211", "0xe223" }, [_]u32{ 0xe211, 0xe223 }) |raw, id| {
        try std.testing.expectEqual(@as(c_int, 0), sys.env.setenv(name, raw, 1));
        try std.testing.expectEqual(id, try detectDeviceId());
    }
}

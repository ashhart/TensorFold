//! The CUDA driver's peer queries for P2P diagnostics: visible devices, bus ids, peer access and P2P attributes.
const std = @import("std");
const platform = @import("sys");
const nvml = @import("p2p_nvml.zig");

const R = c_int;

/// CU_DEVICE_P2P_ATTRIBUTE_* (cuda.h).
pub const attr_performance_rank: c_int = 1;
pub const attr_access_supported: c_int = 2;
pub const attr_native_atomic_supported: c_int = 3;

/// Looked up apart from `abi.Api`, so a driver without one of these still opens for inference.
pub const Api = struct {
    cuDriverGetVersion: *const fn (*c_int) callconv(.c) R,
    cuCtxGetDevice: *const fn (*c_int) callconv(.c) R,
    cuDeviceGetCount: *const fn (*c_int) callconv(.c) R,
    cuDeviceGet: *const fn (*c_int, c_int) callconv(.c) R,
    cuDeviceGetName: *const fn ([*]u8, c_int, c_int) callconv(.c) R,
    cuDeviceGetPCIBusId: *const fn ([*]u8, c_int, c_int) callconv(.c) R,
    cuDeviceCanAccessPeer: *const fn (*c_int, c_int, c_int) callconv(.c) R,
    cuDeviceGetP2PAttribute: *const fn (*c_int, c_int, c_int, c_int) callconv(.c) R,
};

pub const Library = struct {
    lib: platform.DynLib,
    api: Api,

    /// The process's CUDA driver again (the loader shares the one already open), or null without every symbol.
    pub fn open() ?Library {
        var lib = platform.DynLib.open(platform.cuda_driver) catch return null;
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                lib.close();
                return null;
            };
        }
        return .{ .lib = lib, .api = api };
    }
};

pub const max_devices = 16;

/// A device this process sees; the peer fields are from this rank's device to it (null: not asked or failed).
pub const Dev = struct {
    ordinal: u32,
    handle: c_int = 0,
    name: nvml.Text(96) = .{},
    bus: nvml.Text(32) = .{},
    key: ?u64 = null,
    can_access: ?c_int = null,
    access_supported: ?c_int = null,
    atomics: ?c_int = null,
    performance_rank: ?c_int = null,
};

/// The driver's CUDA version, the visible devices and which one is current on this thread.
pub const Survey = struct {
    driver_api: ?c_int = null,
    devs: [max_devices]Dev = undefined,
    count: usize = 0,
    mine: ?usize = null,

    pub fn list(s: *const Survey) []const Dev {
        return s.devs[0..s.count];
    }

    pub fn current(s: *const Survey) ?*const Dev {
        return &s.devs[s.mine orelse return null];
    }

    /// The visible device on the bus `key` names, if this process sees it.
    pub fn onBus(s: *const Survey, key: ?u64) ?*const Dev {
        const k = key orelse return null;
        for (s.list()) |*d| if (d.key == k) return d;
        return null;
    }
};

/// Visible devices and, with more than one, peer access from the current context's device to each other.
pub fn survey(api: *const Api, out: *Survey) void {
    out.* = .{};
    var v: c_int = 0;
    if (api.cuDriverGetVersion(&v) == 0) out.driver_api = v;
    var current: c_int = -1;
    if (api.cuCtxGetDevice(&current) != 0) current = -1;
    var n: c_int = 0;
    if (api.cuDeviceGetCount(&n) != 0) return;
    for (0..@min(@as(usize, @intCast(@max(n, 0))), max_devices)) |i| {
        var d: Dev = .{ .ordinal = @intCast(i) };
        if (api.cuDeviceGet(&d.handle, @intCast(i)) != 0) continue;
        var name: [96]u8 = @splat(0);
        if (api.cuDeviceGetName(&name, name.len, d.handle) == 0) d.name = .ofC(&name);
        var bus: [32]u8 = @splat(0);
        if (api.cuDeviceGetPCIBusId(&bus, bus.len, d.handle) == 0) {
            d.bus = .ofC(&bus);
            d.key = nvml.busKey(d.bus.slice());
        }
        if (d.handle == current) out.mine = out.count;
        out.devs[out.count] = d;
        out.count += 1;
    }
    const m = out.mine orelse return;
    const me = out.devs[m].handle;
    for (out.devs[0..out.count], 0..) |*d, i| if (i != m) {
        var x: c_int = 0;
        if (api.cuDeviceCanAccessPeer(&x, me, d.handle) == 0) d.can_access = x;
        if (api.cuDeviceGetP2PAttribute(&x, attr_access_supported, me, d.handle) == 0) d.access_supported = x;
        if (api.cuDeviceGetP2PAttribute(&x, attr_native_atomic_supported, me, d.handle) == 0) d.atomics = x;
        if (api.cuDeviceGetP2PAttribute(&x, attr_performance_rank, me, d.handle) == 0) d.performance_rank = x;
    };
}

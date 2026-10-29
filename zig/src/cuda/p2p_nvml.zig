//! NVML (libnvidia-ml.so.1) opened at run time: the host's GPUs, each pair's P2P read/write status and PCIe path.
const std = @import("std");
const platform = @import("sys");

pub const Device = ?*opaque {};
const R = c_int;

/// nvmlPciInfo_t: the legacy bus id, four ids, the subsystem id, then the full bus id (NVML 11 and later).
pub const PciInfo = extern struct {
    bus_id_legacy: [16]u8 = @splat(0),
    domain: c_uint = 0,
    bus: c_uint = 0,
    device: c_uint = 0,
    pci_device_id: c_uint = 0,
    pci_subsystem_id: c_uint = 0,
    bus_id: [32]u8 = @splat(0),
};

/// NVML_P2P_CAPS_INDEX_READ and NVML_P2P_CAPS_INDEX_WRITE.
pub const caps_read: c_int = 0;
pub const caps_write: c_int = 1;

/// Each field is the exact exported symbol of libnvidia-ml.so.1.
pub const Api = struct {
    nvmlInit_v2: *const fn () callconv(.c) R,
    nvmlShutdown: *const fn () callconv(.c) R,
    nvmlSystemGetDriverVersion: *const fn ([*]u8, c_uint) callconv(.c) R,
    nvmlDeviceGetCount_v2: *const fn (*c_uint) callconv(.c) R,
    nvmlDeviceGetHandleByIndex_v2: *const fn (c_uint, *Device) callconv(.c) R,
    nvmlDeviceGetName: *const fn (Device, [*]u8, c_uint) callconv(.c) R,
    nvmlDeviceGetPciInfo_v3: *const fn (Device, *PciInfo) callconv(.c) R,
    nvmlDeviceGetP2PStatus: *const fn (Device, Device, c_int, *c_int) callconv(.c) R,
    nvmlDeviceGetTopologyCommonAncestor: *const fn (Device, Device, *c_int) callconv(.c) R,
};

pub const Library = struct {
    lib: platform.DynLib,
    api: Api,

    /// The driver's NVML, or null when it is missing or lacks a symbol (the diagnostics then say so).
    pub fn open() ?Library {
        return openPath("libnvidia-ml.so.1");
    }

    pub fn openPath(path: []const u8) ?Library {
        var lib = platform.DynLib.open(path) catch return null;
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

/// nvmlGpuP2PStatus_t by its NVML name.
pub fn statusName(s: c_int) []const u8 {
    return switch (s) {
        0 => "OK",
        1 => "CHIPSET_NOT_SUPPORTED",
        2 => "GPU_NOT_SUPPORTED",
        3 => "IOH_TOPOLOGY_NOT_SUPPORTED",
        4 => "DISABLED_BY_REGKEY",
        5 => "NOT_SUPPORTED",
        6 => "UNKNOWN",
        else => "UNRECOGNIZED",
    };
}

/// nvmlGpuTopologyLevel_t by `nvidia-smi topo -m`'s names (INTERNAL: two GPUs on one board).
pub fn levelName(l: c_int) []const u8 {
    return switch (l) {
        0 => "INTERNAL",
        10 => "PIX",
        20 => "PXB",
        30 => "PHB",
        40 => "NODE",
        50 => "SYS",
        else => "UNRECOGNIZED",
    };
}

/// A short NUL-free string kept by value.
pub fn Text(comptime n: usize) type {
    return struct {
        buf: [n]u8 = undefined,
        len: usize = 0,

        pub fn of(s: []const u8) @This() {
            var t: @This() = .{};
            t.len = @min(s.len, n);
            @memcpy(t.buf[0..t.len], s[0..t.len]);
            return t;
        }

        /// The text up to its first NUL, trimmed of spaces.
        pub fn ofC(raw: []const u8) @This() {
            return of(std.mem.trim(u8, std.mem.sliceTo(raw, 0), " "));
        }

        pub fn slice(t: *const @This()) []const u8 {
            return t.buf[0..t.len];
        }
    };
}

/// A PCI bus id ("0000:01:00.0" from CUDA, "00000000:01:00.0" from NVML) as one comparable number.
pub fn busKey(text: []const u8) ?u64 {
    const t = std.mem.trim(u8, std.mem.sliceTo(text, 0), " ");
    const colon2 = std.mem.lastIndexOfScalar(u8, t, ':') orelse return null;
    const colon1 = std.mem.lastIndexOfScalar(u8, t[0..colon2], ':') orelse return null;
    const dot = std.mem.lastIndexOfScalar(u8, t, '.') orelse return null;
    if (dot < colon2) return null;
    const domain = std.fmt.parseInt(u32, t[0..colon1], 16) catch return null;
    const bus = std.fmt.parseInt(u8, t[colon1 + 1 .. colon2], 16) catch return null;
    const device = std.fmt.parseInt(u8, t[colon2 + 1 .. dot], 16) catch return null;
    const function = std.fmt.parseInt(u8, t[dot + 1 ..], 16) catch return null;
    return @as(u64, domain) << 32 | @as(u64, bus) << 16 | @as(u64, device) << 8 | function;
}

pub const max_gpus = 16;

/// One GPU NVML lists; `read`, `write` and `path` are against this rank's GPU (null: not asked or the call failed).
pub const Gpu = struct {
    index: u32,
    handle: Device = null,
    name: Text(96) = .{},
    bus: Text(32) = .{},
    key: ?u64 = null,
    read: ?c_int = null,
    write: ?c_int = null,
    path: ?c_int = null,
};

/// What NVML says about this host: the driver, every GPU, which is this rank's, or why it said nothing.
pub const Survey = struct {
    driver: Text(80) = .{},
    gpus: [max_gpus]Gpu = undefined,
    count: usize = 0,
    mine: ?usize = null,
    pair_count: usize = 0,
    /// The NVML call that failed (with its code), when the survey stopped early.
    failed: ?[]const u8 = null,
    code: c_int = 0,

    pub fn list(s: *const Survey) []const Gpu {
        return s.gpus[0..s.count];
    }

    fn fail(s: *Survey, what: []const u8, code: c_int) void {
        s.failed = what;
        s.code = code;
    }
};

/// The host's GPUs (CUDA_VISIBLE_DEVICES does not narrow NVML) against `mine`'s bus id; every pair into `all`.
pub fn survey(api: *const Api, mine: ?u64, out: *Survey, all: []Pair) void {
    out.* = .{};
    const init = api.nvmlInit_v2();
    if (init != 0) return out.fail("nvmlInit_v2", init);
    defer _ = api.nvmlShutdown();
    var driver: [80]u8 = @splat(0);
    if (api.nvmlSystemGetDriverVersion(&driver, driver.len) == 0) out.driver = .ofC(&driver);
    var n: c_uint = 0;
    const counted = api.nvmlDeviceGetCount_v2(&n);
    if (counted != 0) return out.fail("nvmlDeviceGetCount_v2", counted);
    for (0..@min(n, max_gpus)) |i| {
        var g: Gpu = .{ .index = @intCast(i) };
        const got = api.nvmlDeviceGetHandleByIndex_v2(@intCast(i), &g.handle);
        if (got != 0) return out.fail("nvmlDeviceGetHandleByIndex_v2", got);
        var name: [96]u8 = @splat(0);
        if (api.nvmlDeviceGetName(g.handle, &name, name.len) == 0) g.name = .ofC(&name);
        var pci: PciInfo = .{};
        if (api.nvmlDeviceGetPciInfo_v3(g.handle, &pci) == 0) {
            g.bus = .ofC(&pci.bus_id);
            g.key = busKey(g.bus.slice());
        }
        if (mine != null and g.key == mine) out.mine = out.count;
        out.gpus[out.count] = g;
        out.count += 1;
    }
    if (out.mine) |m| for (out.gpus[0..out.count], 0..) |*g, i| if (i != m) {
        const p = pair(api, out.gpus[m], g.*);
        g.read = p.read;
        g.write = p.write;
        g.path = p.path;
    };
    for (out.gpus[0..out.count], 0..) |a, i| for (out.gpus[i + 1 .. out.count]) |b| {
        if (out.pair_count == all.len) return;
        all[out.pair_count] = pair(api, a, b);
        out.pair_count += 1;
    };
}

/// One pair's statuses (read, write, path; null where NVML would not say), GPUs by NVML index.
pub const Pair = struct { a: u32, b: u32, read: ?c_int = null, write: ?c_int = null, path: ?c_int = null };

fn pair(api: *const Api, a: Gpu, b: Gpu) Pair {
    var p: Pair = .{ .a = a.index, .b = b.index };
    var s: c_int = 0;
    if (api.nvmlDeviceGetP2PStatus(a.handle, b.handle, caps_read, &s) == 0) p.read = s;
    if (api.nvmlDeviceGetP2PStatus(a.handle, b.handle, caps_write, &s) == 0) p.write = s;
    if (api.nvmlDeviceGetTopologyCommonAncestor(a.handle, b.handle, &s) == 0) p.path = s;
    return p;
}

test "nvmlPciInfo_t keeps NVML's layout: the full bus id after five ids" {
    try std.testing.expectEqual(@as(usize, 68), @sizeOf(PciInfo));
    try std.testing.expectEqual(@as(usize, 36), @offsetOf(PciInfo, "bus_id"));
}

test "bus ids from CUDA and NVML compare equal; malformed ids do not parse" {
    try std.testing.expectEqual(busKey("0000:01:00.0").?, busKey("00000000:01:00.0").?);
    try std.testing.expect(busKey("0000:C1:00.0").? != busKey("0000:01:00.0").?);
    try std.testing.expectEqual(@as(?u64, null), busKey("01:00"));
    try std.testing.expectEqual(@as(?u64, null), busKey(""));
}

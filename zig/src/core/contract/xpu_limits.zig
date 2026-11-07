//! Gate 5 for XPU launches: the Xe2 card's limits, the compiled usage, and every launch against both.
const std = @import("std");

/// A card class we serve: what one work-group may take, and what one Xe-core holds.
pub const Xe = struct {
    name: []const u8,
    ocloc_device: []const u8,
    /// The PCI id the device runner finds the card by.
    pci: u32,
    max_threads: u32 = 1024,
    max_dims: [3]u32 = .{ 1024, 1024, 1024 },
    /// Bytes one work-group may reserve.
    slm_workgroup: u32 = 131072,
    /// Bytes of SLM one Xe-core holds.
    slm_xe_core: u32 = 131072,
    sub_groups: []const u32 = &.{ 16, 32 },
    /// Vector engines one Xe-core.
    eus_xe_core: u32 = 8,
    /// Hardware threads one EU runs at 128 GRF; a 256-GRF kernel halves it.
    threads_eu: u32 = 8,
    /// Xe-cores the card has, from the device read's compute units and EUs a core.
    xe_cores: u32 = 32,
    grf_modes: []const u32 = &.{ 128, 256 },
};

/// BMG-G31 (Arc Pro B70), the one card class the XPU backend serves.
pub const xe = [_]Xe{
    .{ .name = "bmg-g31", .ocloc_device = "bmg", .pci = 0xe223 },
};

/// The image kind the launcher loads: a native zebin (format 1), or SPIR-V IL the driver builds (format 0).
pub const Format = enum { native, spirv };

/// An XPU launch as the launcher sets it, with what the compile declared.
pub const XpuLaunch = struct {
    name: []const u8,
    symbol: []const u8,
    image: []const u8,
    format: Format,
    threads: [3]u32,
    slm: u32 = 0,
    sub_group: u32,
    grf: u32,

    pub fn workItems(l: XpuLaunch) u64 {
        return @as(u64, l.threads[0]) * l.threads[1] * l.threads[2];
    }
};

/// What ocloc made of one kernel: its execution_env from the image's .ze_info (0: the field is absent).
pub const XpuUsage = struct {
    threads: [3]u32 = .{ 0, 0, 0 },
    slm: u32 = 0,
    sub_group: u32 = 0,
    grf: u32 = 0,
    spill: u32 = 0,
    private: u32 = 0,
};

/// Work-items one Xe-core holds for a kernel built in `grf` mode with `sub_group` lanes.
pub fn resident(t: Xe, grf: u32, sub_group: u32) u64 {
    const threads = if (grf == 256) t.threads_eu / 2 else t.threads_eu;
    return @as(u64, t.eus_xe_core) * threads * sub_group;
}

fn note(out: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !usize {
    try out.print(a, fmt ++ "\n", args);
    return 1;
}

/// Every problem one launch has on one card from its declaration alone; returns how many.
pub fn fits(l: XpuLaunch, t: Xe, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n: usize = 0;
    const at = .{ l.name, t.name };
    const items = l.workItems();
    if (l.threads[0] == 0 or l.threads[1] == 0 or l.threads[2] == 0) n += try note(out, a, "{s} on {s}: a work-group dimension of 0", at);
    if (items > t.max_threads) n += try note(out, a, "{s} on {s}: {d} work-items a work-group, the most is {d}", at ++ .{ items, t.max_threads });
    for (l.threads, t.max_dims) |asked, most| {
        if (asked > most) n += try note(out, a, "{s} on {s}: a work-group dimension of {d}, the most is {d}", at ++ .{ asked, most });
    }
    if (l.slm > t.slm_workgroup) n += try note(out, a, "{s} on {s}: {d} SLM bytes a work-group, the most is {d}", at ++ .{ l.slm, t.slm_workgroup });
    if (std.mem.indexOfScalar(u32, t.sub_groups, l.sub_group) == null) n += try note(out, a, "{s} on {s}: sub-group {d}, the device runs {any}", at ++ .{ l.sub_group, t.sub_groups });
    if (l.sub_group == 0 or items % l.sub_group != 0) n += try note(out, a, "{s} on {s}: {d} work-items are not a multiple of sub-group {d}", at ++ .{ items, l.sub_group });
    if (std.mem.indexOfScalar(u32, t.grf_modes, l.grf) == null) n += try note(out, a, "{s} on {s}: {d} GRF a thread, the modes are {any}", at ++ .{ l.grf, t.grf_modes });
    if (l.grf == 256 and l.sub_group > 16) n += try note(out, a, "{s} on {s}: sub-group {d} at 256 GRF (the most is 16)", at ++ .{l.sub_group});
    const most = resident(t, l.grf, l.sub_group);
    if (items > most) n += try note(out, a, "{s} on {s}: {d} work-items at {d} GRF and sub-group {d}, a Xe-core holds {d}", at ++ .{ items, l.grf, l.sub_group, most });
    return n;
}

/// Every way the record differs from what the compile made, so a stale record cannot pass; returns how many.
pub fn stale(l: XpuLaunch, u: XpuUsage, t: Xe, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n: usize = 0;
    const at = .{ l.name, t.name };
    const items = l.workItems();
    const say = "{s} on {s}: the record says {d}, the compile {d}";
    if (u.threads[0] != 0 and !std.meta.eql(u.threads, l.threads)) n += try note(out, a, say ++ " work-items", at ++ .{ items, @as(u64, u.threads[0]) * u.threads[1] * u.threads[2] });
    if (u.slm != l.slm) n += try note(out, a, say ++ " SLM bytes", at ++ .{ l.slm, u.slm });
    if (u.sub_group != l.sub_group) n += try note(out, a, say ++ " sub-group", at ++ .{ l.sub_group, u.sub_group });
    if (u.grf != l.grf) n += try note(out, a, say ++ " GRF", at ++ .{ l.grf, u.grf });
    return n;
}

/// The record's GRF mode against the zebin's `grf_count`, which ocloc writes for every kernel; returns 0 or 1.
pub fn grfBuilt(l: XpuLaunch, built: u32, t: Xe, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    if (built == 0) return note(out, a, "{s} on {s}: the zebin gives no grf_count for {s}", .{ l.name, t.name, l.symbol });
    if (built == l.grf) return 0;
    return note(out, a, "{s} on {s}: the record says {d} GRF, the zebin was built at {d}", .{ l.name, t.name, l.grf, built });
}

/// Every problem one launch has on one card from its declaration and compile; returns how many.
pub fn xpu(l: XpuLaunch, u: XpuUsage, t: Xe, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n = try fits(l, t, out, a);
    n += try stale(l, u, t, out, a);
    if (u.spill != 0) _ = try note(out, a, "{s} on {s}: spills {d} bytes (recorded)", .{ l.name, t.name, u.spill });
    return n;
}

/// What the device's own properties say, as `zeDeviceGetProperties` reports them.
pub const Device = struct {
    deviceId: u32,
    numEUsPerSubslice: u32,
    numThreadsPerEU: u32,
    physicalEUSimdWidth: u32,
    numSlices: u32,
    numSubslicesPerSlice: u32,
};

/// Every way the device differs from the row, so a wrong row is a finding; returns how many.
pub fn liveRow(d: Device, t: Xe, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n: usize = 0;
    const say = "{s} live: the device says {d}, the row {d}";
    if (d.deviceId != t.pci) n += try note(out, a, say ++ " (PCI id)", .{ t.name, d.deviceId, t.pci });
    if (d.numEUsPerSubslice != t.eus_xe_core) n += try note(out, a, say ++ " (EUs a Xe-core)", .{ t.name, d.numEUsPerSubslice, t.eus_xe_core });
    if (d.numThreadsPerEU != t.threads_eu) n += try note(out, a, say ++ " (threads an EU)", .{ t.name, d.numThreadsPerEU, t.threads_eu });
    if (d.physicalEUSimdWidth != t.sub_groups[0]) n += try note(out, a, say ++ " (SIMD width)", .{ t.name, d.physicalEUSimdWidth, t.sub_groups[0] });
    const cores = @as(u64, d.numSlices) * d.numSubslicesPerSlice;
    if (cores != t.xe_cores) n += try note(out, a, "{s} live: the device has {d} Xe-cores, the row {d}", .{ t.name, cores, t.xe_cores });
    return n;
}

/// What a loaded kernel's own properties say, as `zeKernelGetProperties` reports them.
pub const Loaded = struct {
    threads: [3]u32,
    sub_group: u32,
    max_sub_group: u32,
    slm: u32,
    spill: u32,
    args: u32,
};

/// Every way a loaded kernel differs from its record, and a launch that does not fit; returns how many.
pub fn liveKernel(l: XpuLaunch, u: XpuUsage, k: Loaded, args: u32, t: Xe, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n: usize = 0;
    const say = "{s} live on {s}: the kernel says {d}, the record {d}";
    const items = l.workItems();
    if (!std.meta.eql(k.threads, u.threads)) n += try note(out, a, say ++ " work-items", .{ l.name, t.name, @as(u64, k.threads[0]) * k.threads[1] * k.threads[2], @as(u64, u.threads[0]) * u.threads[1] * u.threads[2] });
    if (k.sub_group != u.sub_group) n += try note(out, a, say ++ " sub-group", .{ l.name, t.name, k.sub_group, u.sub_group });
    if (k.max_sub_group < u.sub_group) n += try note(out, a, "{s} live on {s}: sub-group {d}, the kernel's most {d}", .{ l.name, t.name, u.sub_group, k.max_sub_group });
    if (k.slm != u.slm) n += try note(out, a, say ++ " SLM bytes", .{ l.name, t.name, k.slm, u.slm });
    if (k.spill != u.spill) n += try note(out, a, say ++ " spill bytes", .{ l.name, t.name, k.spill, u.spill });
    if (k.args != args) n += try note(out, a, say ++ " arguments", .{ l.name, t.name, k.args, args });
    if (items > resident(t, l.grf, l.sub_group)) n += try note(out, a, "{s} live on {s}: {d} work-items at {d} GRF and sub-group {d}, a Xe-core holds {d}", .{ l.name, t.name, items, l.grf, l.sub_group, resident(t, l.grf, l.sub_group) });
    return n;
}

test "the device's own properties match the row, and each differing field is named" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const t = xe[0];
    const dev = Device{ .deviceId = 0xe223, .numEUsPerSubslice = 8, .numThreadsPerEU = 8, .physicalEUSimdWidth = 16, .numSlices = 8, .numSubslicesPerSlice = 4 };
    try std.testing.expectEqual(@as(usize, 0), try liveRow(dev, t, &out, a));
    inline for (.{ "deviceId", "numEUsPerSubslice", "numThreadsPerEU", "physicalEUSimdWidth", "numSlices" }) |field| {
        var wrong = dev;
        @field(wrong, field) += 1;
        try std.testing.expectEqual(@as(usize, 1), try liveRow(wrong, t, &out, a));
    }
    var other = dev;
    other.deviceId = 0x7d67;
    try std.testing.expectEqual(@as(usize, 1), try liveRow(other, t, &out, a));
}

test "a loaded kernel that matches its record passes, and each difference is named" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const t = xe[0];
    const l = XpuLaunch{ .name = "ok", .symbol = "ok", .image = "glue", .format = .native, .threads = .{ 256, 1, 1 }, .slm = 2048, .sub_group = 16, .grf = 256 };
    const u = XpuUsage{ .threads = .{ 256, 1, 1 }, .slm = 2048, .sub_group = 16, .grf = 256 };
    const k = Loaded{ .threads = .{ 256, 1, 1 }, .sub_group = 16, .max_sub_group = 16, .slm = 2048, .spill = 0, .args = 11 };
    try std.testing.expectEqual(@as(usize, 0), try liveKernel(l, u, k, 11, t, &out, a));
    var wide = k;
    wide.threads = .{ 512, 1, 1 };
    try std.testing.expectEqual(@as(usize, 1), try liveKernel(l, u, wide, 11, t, &out, a));
    var sg = k;
    sg.sub_group = 32;
    try std.testing.expectEqual(@as(usize, 1), try liveKernel(l, u, sg, 11, t, &out, a));
    var low = k;
    low.max_sub_group = 8;
    try std.testing.expectEqual(@as(usize, 1), try liveKernel(l, u, low, 11, t, &out, a));
    var slm = k;
    slm.slm = 10272;
    try std.testing.expectEqual(@as(usize, 1), try liveKernel(l, u, slm, 11, t, &out, a));
    var spill = k;
    spill.spill = 64;
    try std.testing.expectEqual(@as(usize, 1), try liveKernel(l, u, spill, 11, t, &out, a));
    try std.testing.expectEqual(@as(usize, 1), try liveKernel(l, u, k, 10, t, &out, a));
    // a launch past the 256-GRF resident limit is named
    const big = XpuLaunch{ .name = "big", .symbol = "big", .image = "glue", .format = .native, .threads = .{ 1024, 1, 1 }, .sub_group = 16, .grf = 256 };
    const big_u = XpuUsage{ .threads = .{ 1024, 1, 1 }, .sub_group = 16, .grf = 256 };
    const big_k = Loaded{ .threads = .{ 1024, 1, 1 }, .sub_group = 16, .max_sub_group = 16, .slm = 0, .spill = 0, .args = 11 };
    try std.testing.expectEqual(@as(usize, 1), try liveKernel(big, big_u, big_k, 11, t, &out, a));
}

test "the BMG row holds one work-group on one Xe-core" {
    const t = xe[0];
    try std.testing.expectEqual(@as(u32, 131072), t.slm_xe_core);
    try std.testing.expect(t.slm_workgroup <= t.slm_xe_core);
    try std.testing.expectEqual(@as(u64, 1024), resident(t, 128, 16));
    try std.testing.expectEqual(@as(u64, 512), resident(t, 256, 16));
    try std.testing.expectEqual(@as(u64, 2048), resident(t, 128, 32));
    // the device read's 256 compute units are 8 EUs a Xe-core over 32 Xe-cores
    try std.testing.expectEqual(@as(u32, 256), t.eus_xe_core * t.xe_cores);
}

test "a launch inside the row passes; each planted excess is named" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const t = xe[0];
    const ok = XpuLaunch{ .name = "ok", .symbol = "ok", .image = "glue", .format = .native, .threads = .{ 256, 1, 1 }, .slm = 2048, .sub_group = 16, .grf = 256 };
    const usage = XpuUsage{ .threads = .{ 256, 1, 1 }, .slm = 2048, .sub_group = 16, .grf = 256 };
    try std.testing.expectEqual(@as(usize, 0), try xpu(ok, usage, t, &out, a));
    var edge = ok;
    edge.threads = .{ 512, 1, 1 };
    var edge_usage = usage;
    edge_usage.threads = .{ 512, 1, 1 };
    try std.testing.expectEqual(@as(usize, 0), try xpu(edge, edge_usage, t, &out, a));
    var wide = edge;
    wide.threads = .{ 1024, 1, 1 };
    var wide_usage = edge_usage;
    wide_usage.threads = .{ 1024, 1, 1 };
    try std.testing.expectEqual(@as(usize, 1), try xpu(wide, wide_usage, t, &out, a));
    var over = ok;
    over.threads = .{ 2048, 1, 1 };
    var over_usage = usage;
    over_usage.threads = .{ 256, 1, 1 };
    try std.testing.expect(try xpu(over, over_usage, t, &out, a) >= 1);
    var big = ok;
    big.slm = 160 * 1024;
    var big_usage = usage;
    big_usage.slm = 160 * 1024;
    try std.testing.expectEqual(@as(usize, 1), try xpu(big, big_usage, t, &out, a));
    var odd = ok;
    odd.threads = .{ 200, 1, 1 };
    var odd_usage = usage;
    odd_usage.threads = .{ 256, 1, 1 };
    try std.testing.expect(try xpu(odd, odd_usage, t, &out, a) >= 1);
    var sg = ok;
    sg.sub_group = 64;
    var sg_usage = usage;
    sg_usage.sub_group = 64;
    try std.testing.expectEqual(@as(usize, 2), try xpu(sg, sg_usage, t, &out, a));
    var grf = ok;
    grf.grf = 64;
    var grf_usage = usage;
    grf_usage.grf = 64;
    try std.testing.expectEqual(@as(usize, 1), try xpu(grf, grf_usage, t, &out, a));
    var off = ok;
    off.slm = 4096;
    try std.testing.expectEqual(@as(usize, 1), try xpu(off, usage, t, &out, a));
    const none = XpuUsage{};
    try std.testing.expect(try xpu(ok, none, t, &out, a) >= 1);
}

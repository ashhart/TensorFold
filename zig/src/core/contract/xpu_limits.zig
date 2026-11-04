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

/// Every problem one launch has on one card from its declaration and compile; returns how many.
pub fn xpu(l: XpuLaunch, u: XpuUsage, t: Xe, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n = try fits(l, t, out, a);
    n += try stale(l, u, t, out, a);
    if (u.spill != 0) _ = try note(out, a, "{s} on {s}: spills {d} bytes (recorded)", .{ l.name, t.name, u.spill });
    return n;
}

test "the BMG row holds one work-group on one Xe-core" {
    const t = xe[0];
    try std.testing.expectEqual(@as(u32, 131072), t.slm_xe_core);
    try std.testing.expect(t.slm_workgroup <= t.slm_xe_core);
    try std.testing.expectEqual(@as(u64, 1024), resident(t, 128, 16));
    try std.testing.expectEqual(@as(u64, 512), resident(t, 256, 16));
    try std.testing.expectEqual(@as(u64, 2048), resident(t, 128, 32));
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

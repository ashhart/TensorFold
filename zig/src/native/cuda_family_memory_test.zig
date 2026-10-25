const std = @import("std");
const cuda = @import("cuda");
const poolFor = @import("cuda_family_memory.zig").poolFor;
const t = std.testing;
const Fake = struct {
    var integrated: c_int = 1;
    var missing = false;
    var text: ?[]const u8 = null;
    var zones: ?[]const u8 = null;
    var probes: usize = 0;
    var zone_probes: usize = 0;
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: c_int) callconv(.c) cuda.abi.Result {
        out.* = integrated;
        return 0;
    }
    fn memInfo(free: *usize, total: *usize) callconv(.c) cuda.abi.Result {
        if (missing) return cuda.abi.error_not_ready;
        total.* = 64 << 30;
        free.* = 40 << 30;
        return 0;
    }
    pub fn procText(a: std.mem.Allocator, _: std.Io, path: []const u8) ?[]u8 {
        if (std.mem.eql(u8, path, "/proc/zoneinfo")) {
            zone_probes += 1;
            return a.dupe(u8, zones orelse return null) catch null;
        }
        probes += 1;
        return a.dupe(u8, text orelse return null) catch null;
    }
    pub fn env(_: [*:0]const u8) ?[:0]const u8 {
        return null;
    }
};
test "Windows pool: real CUDA counts, OS selection, unified reserves and missing sources" {
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetAttribute = Fake.attribute;
    driver.api.cuMemGetInfo_v2 = Fake.memInfo;
    const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    var problem: []const u8 = "";
    Fake.missing = false;
    Fake.integrated = 1;
    Fake.probes = 0;
    Fake.text = null;
    const maybe_win = try poolFor(true, Fake, t.allocator, t.io, &ctx, &problem);
    try t.expect(maybe_win != null);
    const win = maybe_win.?;
    try t.expect(win.unified);
    try t.expectEqual(@as(u64, 64 << 30), win.total);
    try t.expectEqual(@as(u64, 40 << 30), win.free);
    try t.expectEqual(@as(u64, (64 << 30) / 10), win.reserve);
    try t.expectEqual(@as(usize, 0), Fake.probes);
    try t.expect((try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)) == null);
    Fake.text = "MemTotal: broken\n";
    try t.expect((try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)) == null);
    Fake.text = "MemTotal: 134217728 kB\nMemAvailable: 100663296 kB\n";
    const linux = (try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)).?;
    try t.expect(linux.unified);
    try t.expectEqual(@as(u64, 128 << 30), linux.total);
    try t.expectEqual(@as(u64, 96 << 30), linux.free);
    try t.expectEqual(@as(usize, 3), Fake.probes);
    Fake.integrated = 0;
    inline for (.{ false, true }) |windows| {
        const discrete = (try poolFor(windows, Fake, t.allocator, t.io, &ctx, &problem)).?;
        try t.expect(!discrete.unified);
        try t.expectEqual(@as(u64, 64 << 30), discrete.total);
        try t.expectEqual(@as(u64, 2 << 30), discrete.reserve);
    }
    try t.expectEqual(@as(usize, 3), Fake.probes);
    Fake.missing = true;
    Fake.integrated = 1;
    try t.expectError(error.NotReady, poolFor(true, Fake, t.allocator, t.io, &ctx, &problem));
    try t.expectError(error.NotReady, poolFor(false, Fake, t.allocator, t.io, &ctx, &problem));
}

/// One zone whose CPU list holds `parked` bytes beyond its high_min, as kernel 6.7+ prints /proc/zoneinfo.
fn zoneinfo(a: std.mem.Allocator, parked: u64) ![]u8 {
    const hm = 2146;
    const fmt = "Node 0, zone   Normal\n  pages free     10\n  pagesets\n    cpu: 0\n              count:    {d}\n" ++
        "              high:     5000\n              batch:    63\n              high_min: {d}\n  vm stats threshold: 110\n" ++
        "  start_pfn:           4096\n";
    return std.fmt.allocPrint(a, fmt, .{ hm + parked / std.heap.pageSize(), hm });
}

test "a Linux unified pool counts the free pages parked on the per-CPU lists, at most MemTotal" {
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetAttribute = Fake.attribute;
    driver.api.cuMemGetInfo_v2 = Fake.memInfo;
    const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    var problem: []const u8 = "";
    const g: u64 = 1 << 30;
    Fake.missing = false;
    Fake.integrated = 1;
    Fake.text = "MemTotal: 134217728 kB\nMemAvailable: 100663296 kB\n";
    defer Fake.zones = null;
    const six = try zoneinfo(t.allocator, 6 * g);
    defer t.allocator.free(six);
    Fake.zones = six;
    try t.expectEqual(102 * g, (try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)).?.free);
    const forty = try zoneinfo(t.allocator, 40 * g);
    defer t.allocator.free(forty);
    Fake.zones = forty;
    try t.expectEqual(128 * g, (try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)).?.free); // MemTotal
    for ([_]?[]const u8{ null, "", "Node 0, zone Normal\n  pagesets\n    cpu: 0\n      count: x\n" }) |zones| {
        Fake.zones = zones;
        try t.expectEqual(96 * g, (try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)).?.free);
    }
    Fake.zones = six;
    Fake.zone_probes = 0;
    const windows = (try poolFor(true, Fake, t.allocator, t.io, &ctx, &problem)).?;
    try t.expectEqual(@as(u64, 40 << 30), windows.free);
    Fake.integrated = 0;
    const card = (try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)).?;
    try t.expectEqual(@as(u64, 40 << 30), card.free);
    try t.expectEqual(@as(usize, 0), Fake.zone_probes);
}

test "the pool names the parked share it counted, for the start-up log line" {
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetAttribute = Fake.attribute;
    driver.api.cuMemGetInfo_v2 = Fake.memInfo;
    const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    var problem: []const u8 = "";
    const g: u64 = 1 << 30;
    Fake.missing = false;
    Fake.integrated = 1;
    Fake.text = "MemTotal: 134217728 kB\nMemAvailable: 130023424 kB\n";
    defer Fake.zones = null;
    const six = try zoneinfo(t.allocator, 6 * g);
    defer t.allocator.free(six);
    Fake.zones = six;
    try t.expectEqual(4 * g, (try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)).?.parked); // to MemTotal
    Fake.text = "MemTotal: 134217728 kB\nMemAvailable: 100663296 kB\n";
    try t.expectEqual(6 * g, (try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)).?.parked);
    Fake.zones = null;
    try t.expectEqual(@as(u64, 0), (try poolFor(false, Fake, t.allocator, t.io, &ctx, &problem)).?.parked);
}

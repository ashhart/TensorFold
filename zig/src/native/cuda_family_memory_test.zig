const std = @import("std");
const cuda = @import("cuda");
const poolFor = @import("cuda_family_memory.zig").poolFor;
const t = std.testing;
const Fake = struct {
    var integrated: c_int = 1;
    var missing = false;
    var text: ?[]const u8 = null;
    var probes: usize = 0;
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
    pub fn procText(a: std.mem.Allocator, _: std.Io, _: []const u8) ?[]u8 {
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

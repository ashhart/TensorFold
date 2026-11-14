//! Real planted proc files through the Linux reader and the existing governor and cache operations.
const std = @import("std");
const builtin = @import("builtin");
const mp = @import("memory_pressure.zig");
const linux = @import("sys").memory_policy.linux_pressure;
const pc = @import("prompt_cache.zig");
const shed = @import("prompt_cache_shed.zig").shed;
const Fake = @import("prompt_cache_fake.zig").Fake;
const a = std.testing.allocator;
const io = std.testing.io;
const quiet = "some avg10=0.00 avg60=0 avg300=0 total=0\nfull avg10=0.00 avg60=0 avg300=0 total=0\n";
const healthy = "MemTotal: 1000000 kB\nMemFree: 1 kB\nMemAvailable: 400000 kB\nCached: 500000 kB\n";

fn psi(out: []u8, some: f64, full: f64) ![]const u8 {
    return std.fmt.bufPrint(out, "some avg10={d:.3} total=0\nfull avg10={d:.3} total=0\n", .{ some, full });
}
fn mem(out: []u8, available: u64) ![]const u8 {
    return std.fmt.bufPrint(out, "MemTotal: 1000000 kB\nMemAvailable: {d} kB\n", .{available});
}
fn plant(dir: std.Io.Dir, p: []const u8, m: []const u8) !void {
    try dir.createDirPath(io, "pressure");
    try dir.writeFile(io, .{ .sub_path = "pressure/memory", .data = p });
    try dir.writeFile(io, .{ .sub_path = "meminfo", .data = m });
}

test "memory pressure: a normally budgeted Linux host at low MemAvailable stays normal" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const full = "MemTotal: 126877696 kB\nMemAvailable: 4194304 kB\nMemFree: 1 kB\nCached: 2097152 kB\n";
    try plant(tmp.dir, quiet, full);
    const reading = linux.readAt(root).?;
    try std.testing.expectEqual(@as(u64, 4) << 30, reading.available);
    try std.testing.expectEqual(@as(u32, 1), reading.level());
    var files: Files = .{ .root = root };
    var g: mp.Governor = .{ .source = .{ .ctx = &files, .read = Files.read } };
    try std.testing.expect(g.poll(0) == null);
    try std.testing.expectEqual(mp.Level.normal, g.level);
}

test "memory pressure: Linux planted files classify PSI boundaries and retain available-memory diagnostics" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    try plant(tmp.dir, quiet, healthy);
    try std.testing.expectEqual(@as(u32, 1), linux.readAt(root).?.level());
    const Case = struct { available: u64, some: f64 = 0, full: f64 = 0, want: ?u32, p: ?[]const u8 = null, m: ?[]const u8 = null };
    const cases = [_]Case{
        .{ .available = 100001, .want = 1 },                                                                                                                            .{ .available = 100000, .want = 1 },
        .{ .available = 50001, .want = 1 },                                                                                                                             .{ .available = 50000, .want = 1 },
        .{ .available = 0, .want = 1 },                                                                                                                                 .{ .available = 1000000, .want = 1 },
        .{ .available = 400000, .some = 9.999, .want = 1 },                                                                                                             .{ .available = 400000, .some = 10, .want = 2 },
        .{ .available = 400000, .full = 0.999, .want = 1 },                                                                                                             .{ .available = 400000, .full = 1, .want = 2 },
        .{ .available = 400000, .full = 9.999, .want = 2 },                                                                                                             .{ .available = 400000, .full = 10, .want = 4 },
        .{ .available = 400000, .want = 4, .p = "some avg10=12.34 avg60=5.67 avg300=1.23 total=987654321\nfull avg60=5.67 total=987654321 avg300=1.23 avg10=12.34\n" }, .{ .available = 400000, .want = null, .p = "some avg10=0.00 avg60=0.00 avg300=0.00 total=987654321\n" },
        .{ .available = 400000, .want = null, .m = "MemTotal: 1000000 kB\nMemFree: 1 kB\nBuffers: 500 kB\nCached: 500000 kB\n" },
    };
    var p: [256]u8 = undefined;
    var m: [128]u8 = undefined;
    for (cases) |c| {
        try plant(tmp.dir, c.p orelse try psi(&p, c.some, c.full), c.m orelse try mem(&m, c.available));
        const reading = linux.readAt(root);
        if (c.want) |want| {
            const got = reading.?;
            try std.testing.expectEqual(want, got.level());
            try std.testing.expectEqual(c.available * 1024, got.available);
        } else try std.testing.expect(reading == null);
    }
}

test "memory pressure: Linux malformed, missing, truncated and oversized proc files never reuse a prior level" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    try std.testing.expect(linux.readAt(root) == null);
    for ([_][]const u8{
        "",                               "some avg10=1\n",                       "some avg10=1\nfull total=0\n",
        "some avg10=nan\nfull avg10=0\n", "some avg10=inf\nfull avg10=0\n",       "some avg10=-1\nfull avg10=0\n",
        "some avg10=101\nfull avg10=0\n", "some avg10=0 avg10=1\nfull avg10=0\n", "some avg10=0\nsome avg10=1\nfull avg10=0\n",
    }) |bad| {
        try plant(tmp.dir, "some avg10=10\nfull avg10=10\n", healthy);
        try std.testing.expectEqual(@as(u32, 4), linux.readAt(root).?.level());
        try plant(tmp.dir, bad, healthy);
        try std.testing.expect(linux.readAt(root) == null);
    }
    for ([_][]const u8{
        "",                                                        "MemTotal: 100 kB\n",                                         "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
        "MemTotal: 100 kB\nMemAvailable: 101 kB\n",                "MemTotal: 100 MB\nMemAvailable: 1 kB\n",                     "MemTotal: 100 kB\nMemAvailable: -1 kB\n",
        "MemTotal: 18446744073709551615 kB\nMemAvailable: 1 kB\n", "MemTotal: 100 kB\nMemAvailable: 1 kB\nMemAvailable: 2 kB\n",
    }) |bad| {
        try plant(tmp.dir, quiet, bad);
        try std.testing.expect(linux.readAt(root) == null);
    }
    var huge: [16385]u8 = @splat(' ');
    @memcpy(huge[0..healthy.len], healthy);
    try plant(tmp.dir, quiet, &huge);
    try std.testing.expect(linux.readAt(root) == null);
    @memset(&huge, ' ');
    @memcpy(huge[0..quiet.len], quiet);
    try plant(tmp.dir, huge[0..1025], healthy);
    try std.testing.expect(linux.readAt(root) == null);
    try plant(tmp.dir, quiet, healthy);
    try std.testing.expectEqual(@as(u32, 1), linux.readAt(root).?.level());
    try tmp.dir.deleteFile(io, "pressure/memory");
    try std.testing.expect(linux.readAt(root) == null);
}

test "memory pressure: Linux is explicitly enabled while macOS and disabled/unsupported paths keep their settings" {
    try std.testing.expect(mp.Settings.forOs(.linux, null, null) == null);
    for ([_]?[]const u8{ "0", "off", "OFF", "", "true", "2" }) |v| try std.testing.expect(mp.Settings.forOs(.linux, v, null) == null);
    for ([_][]const u8{ "1", "on", "ON", " on " }) |v| {
        const s = mp.Settings.forOs(.linux, v, "25").?;
        try std.testing.expect(s.on and s.percent == 25);
    }
    try std.testing.expectEqualDeep(mp.Settings.fromText(null, null), mp.Settings.forOs(.macos, null, null).?);
    try std.testing.expect(mp.Settings.forOs(.macos, "off", null) == null);
    try std.testing.expect(mp.Settings.forOs(.windows, "1", null) == null);
    try std.testing.expect(mp.Governor.fromEnv() == null);
}

const Files = struct {
    root: []const u8,
    reads: usize = 0,
    fn read(ctx: ?*anyopaque) ?mp.Level {
        const f: *Files = @ptrCast(@alignCast(ctx.?));
        f.reads += 1;
        return mp.ofKernel((linux.readAt(f.root) orelse return null).level());
    }
};

test "memory pressure: planted Linux pressure sheds optional state, protects borrowed state, recovers without flapping" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var files: Files = .{ .root = root };
    var g: mp.Governor = .{ .source = .{ .ctx = &files, .read = Files.read } };
    var fake: Fake = .{ .gpa = a };
    var store = pc.Store.init(a, fake.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem_a = arena.allocator();
    const one = [_]u32{ 1, 1, 1, 1, 9 };
    const two = [_]u32{ 2, 2, 2, 2, 2, 2, 2, 2, 9 };
    const next = [_]u32{ 2, 2, 2, 2, 2, 2, 2, 2, 9, 4, 9 };
    _ = fake.pass(&store, &one, try store.begin(mem_a, &one, 4, &.{}, &.{}, null, &.{}));
    _ = fake.pass(&store, &two, try store.begin(mem_a, &two, 8, &.{}, &.{}, null, &.{}));
    const borrowed = try store.lookup(mem_a, &next, 9, &.{}, &.{}, &.{});
    store.budget = store.held;
    try plant(tmp.dir, quiet, healthy);
    try std.testing.expect(g.poll(0) == null);
    try plant(tmp.dir, "some avg10=10\nfull avg10=0\n", healthy);
    try std.testing.expect(g.poll(100 * std.time.ns_per_ms) == null);
    try std.testing.expectEqual(@as(usize, 1), files.reads);
    try std.testing.expectEqual(mp.Level.warning, g.poll(250 * std.time.ns_per_ms).?.to);
    try plant(tmp.dir, "some avg10=10\nfull avg10=10\n", healthy);
    try std.testing.expectEqual(mp.Level.critical, g.poll(500 * std.time.ns_per_ms).?.to);
    store.hold = g.level != .normal;
    const held = store.held;
    while (shed(&store, g.mark(store.budget))) {}
    try std.testing.expect(store.held < held);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expect(!shed(&store, 0));
    const resumed = fake.pass(&store, &next, try store.begin(mem_a, &next, 9, &.{}, &.{}, null, &.{}));
    var fresh: u64 = 0;
    for (next) |id| fresh = fresh *% 31 +% id;
    try std.testing.expectEqual(fresh, resumed);
    store.resumed(borrowed.entry.?, &next, true);
    try plant(tmp.dir, quiet, healthy);
    try std.testing.expect(g.poll(std.time.ns_per_s) == null);
    try plant(tmp.dir, "some avg10=10\nfull avg10=10\n", healthy);
    try std.testing.expect(g.poll(2 * std.time.ns_per_s) == null);
    try plant(tmp.dir, quiet, healthy);
    try std.testing.expect(g.poll(3 * std.time.ns_per_s) == null);
    try std.testing.expect(g.poll(7 * std.time.ns_per_s) == null);
    try std.testing.expectEqual(mp.Level.normal, g.poll(8 * std.time.ns_per_s).?.to);
    store.hold = g.level != .normal;
    try std.testing.expect(!store.hold);
}

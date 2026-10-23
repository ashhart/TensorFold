//! VerbsWire on fake verbs whose completions lag: what it holds in flight and what a failed transport leaves behind.
const std = @import("std");
const rv = @import("roce_verbs.zig");
const fake = @import("roce_fake.zig");
const Roce = @import("roce.zig").Roce;
const Fake = fake.Fake;
const Halts = fake.Halts;
const slot = fake.slot;

test "chunked posts: the wire's own in-flight count and peak match the HCAs', and reach the 2 MiB cap" {
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(2, slot));
    defer std.testing.allocator.free(region);
    Fake.region = region;
    Fake.slot_bytes = slot;
    Fake.reset(3);
    var hw: fake.Hw = .{};
    var w = hw.wire(region);
    for (1..9) |seq| {
        try w.post(@intCast(seq & 1), @intCast(seq), slot);
        try std.testing.expectEqual(Fake.inflight, w.inflight());
        try std.testing.expect(w.inflight() <= rv.max_inflight);
    }
    while (w.inflight() > 0) try w.reap();
    try std.testing.expectEqual(@as(u64, 0), Fake.inflight);
    try std.testing.expectEqual(Fake.max_inflight, w.peak.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 2 << 20), w.peak.load(.monotonic));
}

test "a failed transport: both QPs go to error before the first flush and every posted write is reaped" {
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(2, slot));
    defer std.testing.allocator.free(region);
    @memset(region, 0);
    Fake.region = region;
    Fake.slot_bytes = slot;
    Fake.reset(1);
    Halts.reset();
    var hw: fake.Hw = .{};
    var w = hw.wire(region);
    Halts.install(&hw, &w);
    // two 1 MiB ops posted, neither completed: 2 MiB in flight at the HCAs, the link busy
    try std.testing.expect(w.idle());
    try w.post(1, 1, slot);
    try w.post(0, 2, slot);
    try std.testing.expectEqual(@as(u64, 2 * slot), w.inflight());
    try std.testing.expect(!w.idle());
    try std.testing.expect(w.quiesce(std.time.ns_per_s));
    try std.testing.expect(Halts.halted[0] and Halts.halted[1]);
    try std.testing.expectEqual(@as(u32, 0), Halts.flushed_before_halt);
    try std.testing.expectEqual(@as(u32, 4), Halts.flushes);
    try std.testing.expectEqual(@as(u64, 0), w.inflight());
    try std.testing.expectEqual(@as(u64, 0), Fake.inflight);
    // the line tf-tp-test's transport run reads: the device's states and every reap, all of them flushes
    var line: [256]u8 = undefined;
    try std.testing.expect(w.idle());
    try std.testing.expectEqualStrings("HCA 0 QP ERR; posted 2, completed 2; HCA 1 QP ERR; posted 2, completed 2; reaped 4 (flushed 4), still posted 0", w.report(&line));
    // an aborted transport reports itself failed: its deinit keeps the region
    Halts.reset();
    var r: Roce = .{ .me = 0, .world = 2, .layout = rv.Layout.of(2, slot), .slot_bytes = slot, .timeout_ns = 1, .region = region, .mapping = region, .wire = .{ .verbs = w }, .cpu = null };
    try std.testing.expect(!r.failed());
    r.abort();
    try std.testing.expect(Halts.halted[0] and Halts.halted[1]);
    try std.testing.expect(r.failed());
}

test "a QP whose inline capacity cannot take the 4-byte flag post is refused" {
    try std.testing.expect(!rv.inlineFits(.{ .max_send_wr = 1, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 3 }));
    try std.testing.expect(rv.inlineFits(.{ .max_send_wr = 1, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 4 }));
}

/// /proc/self/smaps' VmFlags of the mapping that contains `at`.
fn vmFlags(at: usize, buf: []u8) ![]const u8 {
    const fd = std.c.open("/proc/self/smaps", .{});
    if (fd < 0) return error.NoSmaps;
    defer _ = std.c.close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const got = std.c.read(fd, buf[n..].ptr, buf.len - n);
        if (got <= 0) break;
        n += @intCast(got);
    }
    var lines = std.mem.splitScalar(u8, buf[0..n], '\n');
    var inside = false;
    while (lines.next()) |l| {
        const sp = std.mem.indexOfScalar(u8, l, ' ') orelse l.len;
        if (std.mem.indexOfScalar(u8, l[0..sp], '-')) |dash| {
            const lo = std.fmt.parseInt(usize, l[0..dash], 16) catch continue;
            const hi = std.fmt.parseInt(usize, l[dash + 1 .. sp], 16) catch continue;
            inside = lo <= at and at < hi;
        }
        if (inside and std.mem.startsWith(u8, l, "VmFlags:")) return l;
    }
    return error.NotFound;
}

test "the region is DONTFORK and NOHUGEPAGE before registration" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const len = Roce.regionBytes(2, slot);
    const m = try std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(m);
    const buf = try std.testing.allocator.alloc(u8, 4 << 20);
    defer std.testing.allocator.free(buf);
    // smaps: dc = MADV_DONTFORK, nh = MADV_NOHUGEPAGE
    const before = try vmFlags(@intFromPtr(m.ptr), buf);
    try std.testing.expect(std.mem.indexOf(u8, before, " dc") == null and std.mem.indexOf(u8, before, " nh") == null);
    try rv.prepareRegion(m);
    // An interior page stays inside the VMA even when a neighbouring mapping merges with it.
    const flags = try vmFlags(@intFromPtr(m.ptr) + 4096, buf);
    try std.testing.expect(std.mem.indexOf(u8, flags, " dc") != null);
    try std.testing.expect(std.mem.indexOf(u8, flags, " nh") != null);
}

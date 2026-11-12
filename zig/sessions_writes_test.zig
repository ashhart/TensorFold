//! The write bench's accounting: a park counts Disk.plan's bytes; on Linux the real tier's files and reads equal it.
const std = @import("std");
const testing = std.testing;
const tier = @import("sessions_writes_tier.zig");
const load = @import("sessions_writes_load.zig");

fn ids(gpa: std.mem.Allocator, first: i32, n: usize) ![]i32 {
    const v = try gpa.alloc(i32, n);
    for (v, 0..) |*t, i| t.* = first + @as(i32, @intCast(i));
    return v;
}

test "sessions writes: an evicted entry is parked whole at Disk.plan's bytes and read back from the tier" {
    const gpa = testing.allocator;
    var cfg = tier.dsv41();
    cfg.policy.ram_bytes = cfg.snapshot.bytes();
    const b = try tier.Bench.init(gpa, testing.io, cfg);
    defer b.deinit();
    const a = try ids(gpa, 1, 5600);
    defer gpa.free(a);
    const c = try ids(gpa, 100_000, 3100);
    defer gpa.free(c);
    try b.request(a[0..5100], 5000);
    try testing.expectEqual(@as(u64, 0), b.now().files);
    // a second chat's entry puts the first past the RAM budget: its 4,992 tokens (ced.zig:47) parked whole
    try b.request(c, 3000);
    const parked = b.now();
    try testing.expectEqual(@as(u64, 1), parked.files);
    try testing.expectEqual(@as(u64, 0), parked.deltas);
    try testing.expectEqual(try b.fileBytes(4992, 0), parked.written);
    try testing.expectEqual(@as(u64, 0), parked.read);
    // the first chat comes back: its prompt resumes those 4,992 tokens from the tier, and the second chat's goes
    try b.request(a, 5500);
    const back = b.now();
    try testing.expectEqual(@as(u64, 4992), back.disk_tokens);
    try testing.expect(back.read > 0 and back.read < parked.written);
    try testing.expectEqual(parked.written + try b.fileBytes(2992, 0), back.written);
}

fn agent(skip_covered: bool) !tier.Counts {
    var cfg = tier.dsv41();
    cfg.policy.ram_bytes = 4 * cfg.snapshot.bytes();
    cfg.policy.skip_covered = skip_covered;
    const b = try tier.Bench.init(testing.allocator, testing.io, cfg);
    defer b.deinit();
    var sh = load.shape(.agent);
    sh.requests = 30;
    return (try load.run(testing.allocator, b, .agent, sh, 7)).end;
}

test "sessions writes: DeepSeek's defaults park a live agent's older turns, which SKIP_COVERED drops unwritten" {
    const parked = try agent(false);
    try testing.expect(parked.files >= 20);
    try testing.expectEqual(@as(u64, 0), parked.read);
    try testing.expectEqual(@as(u64, 0), parked.disk_tokens);
    const dropped = try agent(true);
    try testing.expectEqual(@as(u64, 0), dropped.files);
    try testing.expect(dropped.covered_drops >= 20);
    try testing.expectEqual(parked.ram_tokens, dropped.ram_tokens);
}

fn mixed(root: ?[]const u8, covered: bool) !struct { counts: tier.Counts, peak: u64 } {
    var cfg = tier.dsv41();
    cfg.root = root;
    cfg.policy.skip_covered = covered;
    cfg.policy.delta = covered;
    cfg.snapshot.layers = 4;
    cfg.policy.ram_bytes = 6 * cfg.snapshot.bytes();
    cfg.capacity = 1 << 15;
    // Limit the pool to one slot's capacity so the mixed workload exercises spills.
    cfg.pool_tokens = cfg.capacity;
    cfg.chunk = 1 << 20;
    cfg.staging = 4;
    const b = try tier.Bench.init(testing.allocator, testing.io, cfg);
    defer b.deinit();
    const sh: load.Shape = .{ .requests = 150, .system = 1024, .first = .{ .lo = 256, .hi = 2048 }, .turn = .{ .lo = 128, .hi = 1536 }, .reply = .{ .lo = 64, .hi = 512 }, .cap = 16_384, .users = 20, .hot = 4, .fresh = 10, .hot_odds = 60 };
    const r = try load.run(testing.allocator, b, .mix, sh, 11);
    return .{ .counts = r.end, .peak = b.peak_used };
}

test "sessions writes: on real files the tier writes and reads back exactly what the bench accounts" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const gpa = testing.allocator;
    for ([_]bool{ false, true }) |covered| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        const root = try tmp.dir.realPathFileAlloc(testing.io, ".", gpa);
        defer gpa.free(root);
        const real = try mixed(root, covered);
        const accounted = try mixed(null, covered);
        try testing.expect(real.counts.read > 0 and real.counts.spills > 0);
        if (covered) try testing.expect(real.counts.deltas > 0);
        inline for (@typeInfo(tier.Counts).@"struct".field_names) |n| try testing.expectEqual(@field(real.counts, n), @field(accounted.counts, n));
        try testing.expectEqual(real.peak, accounted.peak);
    }
}

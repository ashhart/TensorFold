//! Retirement bookkeeping follows the four rows leaving a live window, independent of its reserved context.
const std = @import("std");
const kv = @import("kv_pool.zig");
const t = std.testing;

fn firstTrim(context: usize, high: bool) !usize {
    var counted = t.FailingAllocator.init(t.allocator, .{});
    var pool = try kv.Pool.init(counted.allocator(), context, 2);
    defer pool.deinit();
    var lease: kv.Lease = .{};
    defer pool.finish(&lease);
    const used = 8192;
    const dropped = 4;
    try pool.promise(&lease, used);
    if (high) try pool.coverRun(&lease, used, 1, null) else try pool.cover(&lease, used, null);
    const base: u32 = if (high) @intCast(context - used) else 0;
    const before = counted.allocated_bytes;
    for (used - dropped..used) |i| try kv.retirement.retire(&pool, &lease, i, 7);
    pool.trim(&lease, used - dropped);
    const bytes = counted.allocated_bytes - before;
    std.debug.print("E98 trim context={d} used={d} dropped={d} high={any} allocated_bytes={d}\n", .{ context, used, dropped, high, bytes });
    try t.expectEqual(@as(usize, used - dropped), lease.ids.items.len);
    try t.expectEqual(@as(usize, dropped), pool.census().pending);
    for (lease.ids.items, 0..) |id, i| {
        try t.expectEqual(base + @as(u32, @intCast(i)), id);
        try t.expectEqual(@as(u32, 1), pool.holders(id));
    }
    for (used - dropped..used) |i| {
        try t.expect(kv.retirement.held(&pool, base + i));
        try t.expectEqual(@as(u32, 0), pool.holders(base + @as(u32, @intCast(i))));
    }
    try kv.retirement.settle(&pool, 6);
    try t.expectEqual(@as(usize, dropped), pool.census().pending);
    try kv.retirement.settle(&pool, 7);
    try t.expectEqual(@as(usize, 0), pool.census().pending);
    try t.expectEqual(@as(usize, dropped), lease.promised);
    for (0..dropped) |i| try t.expectEqual(base + @as(u32, @intCast(used - i - 1)), try pool.take(&lease));
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u32{ pool.refs, pool.leased, pool.free[0..pool.nfree], lease.ids.items }) |part|
        hash.update(std.mem.sliceAsBytes(part));
    const digest = hash.finalResult();
    std.debug.print("E98 state context={d} high={any} sha256={s}\n", .{ context, high, std.fmt.bytesToHex(digest, .lower) });
    return bytes;
}

test "rollback trim cost: four rows at 8K live do not allocate a context-sized retirement table" {
    const small = try firstTrim(16_384, false);
    const large = try firstTrim(262_144, false);
    const high = try firstTrim(262_144, true);
    try t.expect(small < 8192 and large < 8192 and high < 8192);
    try t.expectEqual(large, high);
    try t.expectEqual(small, large);
}

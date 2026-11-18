//! Retirement bookkeeping follows the four rows leaving a live window, independent of its reserved context.
const std = @import("std");
const kv = @import("kv_pool.zig");
const t = std.testing;

fn firstTrim(context: usize) !usize {
    var counted = t.FailingAllocator.init(t.allocator, .{});
    var pool = try kv.Pool.init(counted.allocator(), context, 2);
    defer pool.deinit();
    var lease: kv.Lease = .{};
    defer pool.finish(&lease);
    const used = 8192;
    const dropped = 4;
    try pool.promise(&lease, used);
    try pool.cover(&lease, used, null);
    const before = counted.allocated_bytes;
    for (used - dropped..used) |i| try kv.retirement.retire(&pool, &lease, i, 7);
    pool.trim(&lease, used - dropped);
    const bytes = counted.allocated_bytes - before;
    std.debug.print("E98 trim context={d} used={d} dropped={d} allocated_bytes={d}\n", .{ context, used, dropped, bytes });
    try t.expectEqual(@as(usize, used - dropped), lease.ids.items.len);
    try t.expectEqual(@as(usize, dropped), pool.census().pending);
    for (lease.ids.items, 0..) |id, i| {
        try t.expectEqual(@as(u32, @intCast(i)), id);
        try t.expectEqual(@as(u32, 1), pool.holders(id));
    }
    for (used - dropped..used) |i| {
        try t.expect(kv.retirement.held(&pool, i));
        try t.expectEqual(@as(u32, 0), pool.holders(@intCast(i)));
    }
    try kv.retirement.settle(&pool, 6);
    try t.expectEqual(@as(usize, dropped), pool.census().pending);
    try kv.retirement.settle(&pool, 7);
    try t.expectEqual(@as(usize, 0), pool.census().pending);
    try t.expectEqual(@as(usize, dropped), lease.promised);
    for (0..dropped) |i| try t.expectEqual(@as(u32, @intCast(used - i - 1)), try pool.take(&lease));
    return bytes;
}

test "rollback trim cost: four rows at 8K live do not allocate a context-sized retirement table" {
    const small = try firstTrim(16_384);
    const large = try firstTrim(262_144);
    try t.expect(small < 8192 and large < 8192);
    try t.expectEqual(small, large);
}

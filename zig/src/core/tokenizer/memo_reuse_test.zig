//! Memo recency protects successful reads and avoids copying duplicate completed segments.
const std = @import("std");
const Memo = @import("memo.zig").Memo;
const t = std.testing;
const a = t.allocator;
fn expect(m: *Memo, text: []const u8, first: bool, wanted: ?[]const u32) !void {
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    const found = try m.append(text, first, a, &out);
    try t.expectEqual(wanted != null, found);
    if (wanted) |ids| try t.expectEqualSlices(u32, ids, out.items);
}
test "memo recency: a successful hit protects the oldest insertion from eviction" {
    const m = try Memo.create(a, 15);
    defer m.destroy();
    m.put("a", false, &.{1});
    m.put("b", false, &.{2});
    m.put("c", false, &.{3});
    try expect(m, "a", false, &.{1});
    m.put("d", false, &.{4});
    try expect(m, "b", false, null);
    try expect(m, "a", false, &.{1});
    try expect(m, "c", false, &.{3});
    try expect(m, "d", false, &.{4});
    try t.expectEqual(@as(usize, 15), m.held);
}
test "memo recency: duplicate inserts keep owned IDs with no replacement allocation" {
    var counted = t.FailingAllocator.init(a, .{});
    const m = try Memo.create(counted.allocator(), 10);
    defer m.destroy();
    var original = [_]u32{1};
    m.put("a", true, &original);
    m.put("b", true, &.{2});
    const before = counted.allocated_bytes;
    @memset(&original, 9);
    for (0..1000) |_| m.put("a", true, &original);
    try t.expectEqual(before, counted.allocated_bytes);
    m.put("c", true, &.{3});
    try expect(m, "b", true, null);
    try expect(m, "a", true, &.{1});
    try expect(m, "c", true, &.{3});
    try t.expectEqual(@as(usize, 10), m.held);
}
test "memo recency: failed reads and mismatched first mode do not promote an entry" {
    for ([_]bool{ false, true }) |allocation_fails| {
        const m = try Memo.create(a, 10);
        defer m.destroy();
        m.put("a", false, &.{1});
        m.put("b", false, &.{2});
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(a);
        if (allocation_fails) {
            var fail = t.FailingAllocator.init(a, .{ .fail_index = 0, .resize_fail_index = 0 });
            try t.expectError(error.OutOfMemory, m.append("a", false, fail.allocator(), &ids));
        } else try t.expect(!try m.append("a", true, a, &ids));
        try t.expectEqual(@as(u64, 0), m.hits);
        m.put("c", false, &.{3});
        try expect(m, "a", false, null);
        try expect(m, "b", false, &.{2});
        try expect(m, "c", false, &.{3});
    }
}
test "memo recency: links survive map growth and a thousand cold arrivals within the payload bound" {
    const m = try Memo.create(a, 48 * 8);
    defer m.destroy();
    for (0..48) |i| {
        var text: [4]u8 = undefined;
        m.put(try std.fmt.bufPrint(&text, "{d:0>4}", .{i}), false, &.{@intCast(i)});
    }
    for (48..1048) |i| {
        try expect(m, "0000", false, &.{0});
        var text: [4]u8 = undefined;
        m.put(try std.fmt.bufPrint(&text, "{d:0>4}", .{i}), false, &.{@intCast(i)});
        try t.expect(m.held <= m.cap);
    }
    try expect(m, "0001", false, null);
    try expect(m, "0000", false, &.{0});
    try expect(m, "1047", false, &.{1047});
    const held = m.held;
    m.put("an entry that does not fit", false, &(@as([128]u32, @splat(1))));
    try t.expectEqual(held, m.held);
    try expect(m, "0000", false, &.{0});
}

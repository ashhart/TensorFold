//! Per-row global-segment and modulo-slot IDs for batched Inkling2 attention.
const std = @import("std");

pub fn writeStreamIds(row_counts: []const usize, global: []i32, slot: []i32, comptime max_slots: usize) void {
    std.debug.assert(global.len == slot.len and max_slots > 0);
    var base: usize = 0;
    for (row_counts, 0..) |rows, si| {
        std.debug.assert(base + rows <= global.len);
        for (0..rows) |i| {
            global[base + i] = 0;
            slot[base + i] = @intCast(si % max_slots);
        }
        base += rows;
    }
    std.debug.assert(base == global.len);
}

test "three non-tree streams select the same metadata bytes as sequential execution" {
    const row_counts = [_]usize{ 2, 1, 3 };
    var global: [6]i32 = undefined;
    var slot_ids: [6]i32 = undefined;
    writeStreamIds(&row_counts, &global, &slot_ids, 8);
    const Seg = struct { p0: u64, off: u64 };
    const segs = [_]Seg{
        .{ .p0 = 11, .off = 5 },
        .{ .p0 = 23, .off = 17 },
        .{ .p0 = 47, .off = 31 },
        .{ .p0 = 71, .off = 43 },
        .{ .p0 = 89, .off = 59 },
    };
    var sequential: [6]u64 = undefined;
    var batched: [6]u64 = undefined;
    var base: usize = 0;
    for (row_counts, 0..) |rows, si| {
        for (0..rows) |i| {
            const ri = base + i;
            const one = segs[si];
            const many = segs[si + @as(usize, @intCast(global[ri]))];
            sequential[ri] = (one.off << 32) | one.p0;
            batched[ri] = (many.off << 32) | many.p0;
        }
        base += rows;
    }
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&sequential), std.mem.asBytes(&batched));
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 1, 2, 2, 2 }, &slot_ids);
}

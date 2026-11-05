//! Solo prompt passes (#1554 C), backend-free: wider chunks while no other stream decodes, the busy chunk else.
const std = @import("std");

/// A family's limits: its busy chunk, its widest solo pass, the grid a solo pass keeps (1: any row count).
pub const Limits = struct { busy: usize, most: usize, grid: usize = 1 };

/// A solo value's rows: unset, `l.busy` (off); `l.busy` to `l.most` on `l.grid`, the rest error.BadChunkSolo.
pub fn parse(v: ?[]const u8, l: Limits) !usize {
    const s = v orelse return l.busy;
    const n = std.fmt.parseInt(usize, s, 10) catch return error.BadChunkSolo;
    if (n < l.busy or n > l.most or n % l.grid != 0) return error.BadChunkSolo;
    return n;
}

/// The named variable's solo rows (parse).
pub fn rows(name: [:0]const u8, l: Limits) !usize {
    return parse(if (std.c.getenv(name)) |v| std.mem.span(v) else null, l);
}

/// Whether a stream besides `own` is past its prompt pass: its lane's `fill` field (where its pass stands) is null.
pub fn othersDecode(lanes: anytype, own: anytype, comptime fill: []const u8) bool {
    var it = lanes.iterator();
    while (it.next()) |kv| if (kv.key_ptr.* != own and @field(kv.value_ptr.*, fill) == null) return true;
    return false;
}

/// The rows the pass of stream `own` takes as it starts: `solo` unless another stream decodes, then `busy`.
pub fn passRows(lanes: anytype, own: anytype, comptime fill: []const u8, solo: usize, busy: usize) usize {
    if (solo == busy) return busy;
    return if (othersDecode(lanes, own, fill)) busy else solo;
}

test "a solo value is off unset, takes busy to most on the grid and refuses the rest" {
    const l: Limits = .{ .busy = 512, .most = 4096, .grid = 512 };
    try std.testing.expectEqual(@as(usize, 512), try parse(null, l));
    try std.testing.expectEqual(@as(usize, 2048), try parse("2048", l));
    try std.testing.expectEqual(@as(usize, 4096), try parse("4096", l));
    const bad = [_][]const u8{ "0", "256", "1000", "4608", "-1", "2k", "" };
    for (bad) |v| try std.testing.expectError(error.BadChunkSolo, parse(v, l));
    try std.testing.expectEqual(@as(usize, 1537), try parse("1537", .{ .busy = 1536, .most = 16384 }));
}

test "a pass takes the solo rows unless another stream decodes; prompts still filling do not count" {
    const L = struct { fill: ?usize };
    var m: std.AutoHashMapUnmanaged(u32, L) = .empty;
    defer m.deinit(std.testing.allocator);
    try m.put(std.testing.allocator, 0, .{ .fill = 0 });
    try std.testing.expectEqual(@as(usize, 2048), passRows(&m, 0, "fill", 2048, 512));
    try m.put(std.testing.allocator, 1, .{ .fill = 1024 });
    try std.testing.expectEqual(@as(usize, 2048), passRows(&m, 0, "fill", 2048, 512));
    m.getPtr(1).?.fill = null;
    try std.testing.expectEqual(@as(usize, 512), passRows(&m, 0, "fill", 2048, 512));
    try std.testing.expectEqual(@as(usize, 2048), passRows(&m, 1, "fill", 2048, 512)); // its own lane never counts
    try std.testing.expectEqual(@as(usize, 512), passRows(&m, 0, "fill", 512, 512));
}

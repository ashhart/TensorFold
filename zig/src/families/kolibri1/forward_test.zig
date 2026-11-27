const std = @import("std");
const fwd = @import("forward.zig");
const config = @import("config.zig");

test "kolibri1_1 plan: the two dense layers count their slot ids and weights once (8,192 rows: 65,536 bytes)" {
    var why: config.Why = .{};
    var m: fwd.Model = undefined;
    m.c = try config.parse(std.testing.allocator, config.kolibri11.pinned, &why);
    m.dense = .{ 0, 0 };
    m.planned = 0;
    try m.denseSlots(undefined, 8192);
    try std.testing.expectEqual(@as(usize, 2 * 8192 * 4), m.planned.?);
}

test "the ring holds a prompt chunk and a window, in whole key tiles" {
    try std.testing.expectEqual(@as(usize, 8768), fwd.ringSize(fwd.prompt_rows, 513));
    try std.testing.expect(fwd.ringSize(fwd.prompt_rows, 513) - 513 >= fwd.prompt_rows);
}

test "a page's bytes: the pool's layout and the kernels' stride agree" {
    var why: config.Why = .{};
    var m: fwd.Model = undefined;
    m.c = try config.parse(std.testing.allocator, config.pinned, &why);
    m.context = 32768;
    m.ring = fwd.ringSize(fwd.prompt_rows, m.c.window);
    try std.testing.expectEqual(m.pageBytes(), m.layout().pageBytes());
    try std.testing.expectEqual(@as(u32, 10), m.layout().group().layers);
}

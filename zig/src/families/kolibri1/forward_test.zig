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

test "kolibri1_1 sizing: a 2,048-token window's 3,072 positions take 3,072-row chunks and a 3,648-key ring, not 8,768" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, config.kolibri11.pinned, &why);
    try std.testing.expectEqual(@as(usize, 3072), fwd.rowsFor(3072));
    try std.testing.expectEqual(@as(usize, fwd.prompt_rows), fwd.rowsFor(17408)); // past the grid: whole chunks
    try std.testing.expectEqual(@as(usize, 3648), fwd.ringSize(fwd.rowsFor(3072), c.window));
    // the served stream's bytes: 40 sliding rings of 3,648 keys and 4 indexed owners of 3,072, K and V, bf16
    const per = 2 * c.kv_heads * c.head_dim * 2;
    try std.testing.expectEqual(per * (40 * 3648 + 4 * 3072), fwd.Model.streamBytes(c, 3072));
}

test "context sizing: every context's chunk fits its rows, a round fits the ring past the window, 8,192 on keeps the grid" {
    var c: usize = 1;
    while (c <= 20000) : (c += 1) {
        const rows = fwd.rowsFor(c);
        try std.testing.expect(rows >= @min(c, fwd.prompt_rows) and rows >= fwd.round_rows);
        try std.testing.expect(fwd.ringSize(rows, 513) - 513 >= rows); // Kolibri-1 and 1.1: a 513-key window
        if (c >= fwd.prompt_rows) try std.testing.expectEqual(@as(usize, fwd.prompt_rows), rows);
    }
}

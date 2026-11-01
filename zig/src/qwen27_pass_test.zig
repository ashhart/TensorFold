//! Real draft scratch allocation against a counted device; no model or GPU is created.
const std = @import("std");
const metal = @import("metal");
const Frame = @import("families/qwen27/dflash/runtime_frame.zig").Frame;

test "qwen27 pass: growing attention scratch reuses a bounded reservation across rounds" {
    const a = std.testing.allocator;
    var allocations: metal.Allocations = .{};
    var f = Frame{ .allocator = a, .device = .{ .allocations = &allocations }, .queue = undefined, .active = true, .epoch = 1 };
    defer {
        for (f.pool.items) |buffer| buffer.deinit();
        f.pool.deinit(a);
        f.entries.deinit(a);
    }
    const initial = try f.alloc(1, 64, .bf16);
    const first = try f.ref(initial);
    @memset(first.buf.data, 0x5a);
    for (2..33) |rows| {
        f.cursor = 0;
        f.epoch += 1;
        f.entries.clearRetainingCapacity();
        const view = try f.alloc(@intCast(rows), 64, .bf16);
        const current = try f.ref(view);
        try std.testing.expectEqual(@as(usize, rows * 128), @as(usize, view.rows) * view.stride * 2);
        try std.testing.expect(current.buf.length() >= rows * 128);
    }
    try std.testing.expectEqual(@as(usize, 1), allocations.calls);
    try std.testing.expectEqual(first.buf.id, f.pool.items[0].id);
    try std.testing.expect(std.mem.allEqual(u8, f.pool.items[0].data[0..128], 0x5a));
    try std.testing.expect(f.pool.items[0].length() <= 65536);
    const retained = f.pool.items[0].id;
    allocations.fail = allocations.calls;
    f.cursor = 0;
    try std.testing.expectError(error.NoBuffer, f.alloc(513, 64, .bf16));
    try std.testing.expectEqual(retained, f.pool.items[0].id);
    try std.testing.expectEqual(@as(usize, 1), allocations.live);
}

test {
    _ = @import("native/qwen27_keep_submit.zig");
}

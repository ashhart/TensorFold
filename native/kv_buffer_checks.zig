const std = @import("std");
const mx = @import("mlx.zig");
const Buffer = @import("kv_buffer.zig").Buffer;
const equal = @import("sampling_checks.zig").equal;
const address = @import("kv_buffer.zig").address;

fn data(s: *mx.Scope, rows: usize, axis: usize, value: f32) !mx.Array {
    const dims: []const i32 = if (axis == 2) &.{ 1, 2, @intCast(rows), 8 } else &.{ @intCast(rows), 8 };
    return s.cast(try s.binary(mx.c.mlx_add, try s.zeros(dims, mx.f32t), try s.scalar(value)), mx.bf16);
}
pub fn exercise() !void {
    var b = Buffer{};
    defer b.deinit();
    var view = mx.empty;
    defer mx.free(view);
    for (0..6) |i| {
        var s = mx.Scope{};
        defer s.deinit();
        const write = try b.append(&s, view, try data(&s, if (i == 0) 2047 else 3, 2, @floatFromInt(i)), 2);
        var next = try b.finish(&s, write, if (i == 0) 2047 else 2);
        errdefer next.deinit();
        try mx.eval(write.capacity);
        try mx.replace(&view, try s.slice(write.view, 2, 0, next.offset));
        b.deinit();
        b = next;
    }
    var saved = try b.clone();
    defer saved.deinit();
    var s = mx.Scope{};
    defer s.deinit();
    const before = try s.own(try mx.retain(view));
    const write = try b.append(&s, view, try data(&s, 16, 2, 99), 2);
    try mx.eval(write.capacity);
    try equal(&s, before, try s.slice(write.view, 2, 0, b.offset));
    const trimmed = try saved.prefix(&s, 31);
    var owned = try trimmed.clone();
    defer owned.deinit();
    try std.testing.expect(owned.spare.ctx == null);
    try std.testing.expectError(error.InvalidCachePrefix, saved.prefix(&s, saved.offset + 1));
    try std.testing.expectError(error.InvalidCachePrefix, b.finish(&s, write, 17));
}

/// Compare mutation, forks and rollback against independently concatenated rows.
/// Retained snapshots deliberately prevent donation; correctness must not depend
/// on MLX's completion-handler timing or whether it copies a buffer.
fn histories(axis: usize) !void {
    var b = Buffer{};
    defer b.deinit();
    var view = mx.empty;
    defer mx.free(view);
    var truth = mx.empty;
    defer mx.free(truth);
    var rng = std.Random.DefaultPrng.init(0x4b56434f57 + axis);
    for (0..128) |i| {
        var s = mx.Scope{};
        defer s.deinit();
        var saved = try b.clone();
        defer saved.deinit();
        const old = if (truth.ctx != null) try s.own(try mx.retain(truth)) else mx.empty;
        const rows: i32 = if (i == 0) 2047 else @intCast(rng.random().intRangeAtMost(u32, 1, 32));
        const keep: i32 = if (i == 0) rows else @intCast(rng.random().intRangeAtMost(u32, 0, @intCast(rows)));
        const added = try data(&s, @intCast(rows), axis, @floatFromInt(i + 1));
        const expected = if (truth.ctx != null) try s.cat(&.{ truth, added }, @intCast(axis)) else added;
        const write = try b.append(&s, view, added, axis);
        try equal(&s, expected, write.view);
        var next = try b.finish(&s, write, keep);
        errdefer next.deinit();
        try mx.replace(&view, try s.slice(write.view, axis, 0, next.offset));
        try mx.replace(&truth, try s.slice(expected, axis, 0, next.offset));
        b.deinit();
        b = next;
        if (saved.current.ctx != null) try equal(&s, old, try s.slice(saved.current, axis, 0, saved.offset));
        if (i % 8 == 7) {
            // Includes an empty rollback, rollback into the recent append, and
            // rollback before spare_end, which must discard the stale spare.
            const end: i32 = switch (i % 32) {
                7 => 0,
                15 => @max(0, b.offset - 1),
                else => @divTrunc(b.offset, 2),
            };
            const prefix = try b.prefix(&s, end);
            var restored = try prefix.clone();
            errdefer restored.deinit();
            try mx.replace(&view, try s.slice(view, axis, 0, end));
            try mx.replace(&truth, try s.slice(truth, axis, 0, end));
            b.deinit();
            b = restored;
        }
        try equal(&s, truth, view);
    }
}
pub fn check() !void {
    try mx.init();
    defer mx.shutdown();
    var reused: usize = 0;
    for ([_]usize{ 0, 2 }) |axis| {
        var b = Buffer{};
        defer b.deinit();
        var view = mx.empty;
        defer mx.free(view);
        for (0..256) |i| {
            var s = mx.Scope{};
            defer s.deinit();
            // MLX eval.cpp retains input Data in Metal completion callbacks.
            // An array host read need not release those callbacks yet. Drain
            // them for this isolated exact-allocation test, not in production.
            try mx.check(mx.c.mlx_synchronize(mx.stream));
            const donor = if (b.spare.ctx != null) address(b.spare) else 0;
            const added = try data(&s, 1, axis, @floatFromInt(i));
            const expected = if (view.ctx != null) try s.cat(&.{ view, added }, @intCast(axis)) else added;
            const write = try b.append(&s, view, added, axis);
            try mx.eval(write.capacity);
            if (donor != 0) {
                const actual = address(write.capacity);
                if (donor != actual) {
                    std.debug.print("Buffer reuse failure: axis={d}, write={d}, donor=0x{x}, output=0x{x}\n", .{ axis, i, donor, actual });
                    return error.BufferWasNotReused;
                }
                reused += 1;
            }
            try equal(&s, expected, write.view);
            var next = try b.finish(&s, write, 1);
            errdefer next.deinit();
            try mx.replace(&view, write.view);
            b.deinit();
            b = next;
        }
    }
    try exercise();
    try histories(0);
    try histories(2);
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var active: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&active));
    try std.testing.expectEqual(@as(usize, 0), active);
    std.debug.print("PASS: {d} writes reuse the exact drained donor allocation; 256 reference histories, growth, partial commits, snapshots and rollback pass\n", .{reused});
}

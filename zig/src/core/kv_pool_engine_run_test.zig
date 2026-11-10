//! First-run pool budgeting at exact single-window and multi-stream boundaries.
const std = @import("std");
const pool = @import("kv_pool.zig");
const Layout = pool.Layout;
const Budget = pool.Budget;
const plan = pool.plan;
const holds = pool.holds;

test "pooled first run: exact single-window boundary and one-byte refusal" {
    const l: Layout = .{ .fixed_bytes = 10, .token_bytes = 3, .page_tokens = 4, .snap_bytes = 0, .run_bytes = 13, .engine_run = true };
    const b: Budget = .{ .room = 70, .engine = 10, .lanes = 1, .fixed = true, .window = 9, .engine_paged = true };
    const p = try plan(l, b);
    try std.testing.expectEqual(@as(u32, 1), p.lanes);
    try std.testing.expectEqual(@as(usize, 5), p.pages);
    try std.testing.expect(holds(l, b));
    var short = b;
    short.room -= 1;
    try std.testing.expectError(error.NoStream, plan(l, short));
    try std.testing.expect(!holds(l, short));
}

test "pooled first run: C4 pays each fixed part and run exactly once" {
    const l: Layout = .{ .fixed_bytes = 10, .token_bytes = 3, .page_tokens = 4, .snap_bytes = 0, .run_bytes = 13, .engine_run = true };
    const b: Budget = .{ .room = 208, .engine = 10, .lanes = 4, .fixed = true, .window = 9, .engine_paged = true };
    const p = try plan(l, b);
    try std.testing.expectEqual(@as(u32, 4), p.lanes);
    try std.testing.expectEqual(@as(usize, 14), p.pages);
    var short = b;
    short.room -= 1;
    try std.testing.expectError(error.TooMany, plan(l, short));
    var full = b;
    full.room = 280;
    try std.testing.expectEqual(@as(usize, 20), (try plan(l, full)).pages);
    full.room = 1000;
    try std.testing.expectEqual(@as(usize, 20), (try plan(l, full)).pages);
    full.engine_paged = false;
    try std.testing.expectEqual(@as(usize, 15), (try plan(l, full)).pages);
}

//! Reusable per-row host prices, shared by current verify allocation and next-round holds.
const std = @import("std");
const Stream = @import("stream.zig").Stream;

/// Grammar routes can change between callbacks; only a stable request route has a reusable quote.
pub fn rowMs(s: *const Stream) f64 {
    return if (s.grammar == null) s.verify_row_ms orelse 0 else 0;
}

/// Snapshot positive quotes once; callers exclude confidence controls and complete-round pricing.
pub fn snapshot(a: std.mem.Allocator, plans: anytype, enabled: bool) ![]const f64 {
    if (!enabled) return &.{};
    var rows: []f64 = &.{};
    for (plans, 0..) |p, i| {
        const s = p.stream;
        const ms = if (s.finished or s.paused) 0 else rowMs(s);
        if (ms > 0 and rows.len == 0) {
            rows = try a.alloc(f64, plans.len);
            @memset(rows, 0);
        }
        if (rows.len > 0) rows[i] = ms;
    }
    return rows;
}

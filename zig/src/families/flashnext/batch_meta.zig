//! CPU validation and immutable row positions for a shared Flash Next forward.
const std = @import("std");
pub const max_rows = 16;
pub const Window = struct { slot: u32, pos: u32, rows: u32 };

/// The last forward remains a draft source after keep; only its unsettled state may be committed once.
pub const Round = struct {
    start: u64 = 0,
    rows: usize = 0,
    pending: bool = false,

    pub fn begin(r: *Round, start: u64, rows: usize) !void {
        if (r.pending or rows == 0 or rows > max_rows) return error.WindowOutOfStep;
        r.* = .{ .start = start, .rows = rows, .pending = true };
    }
    pub fn keep(r: *Round, n: usize) !void {
        if (!r.pending or n == 0 or n > r.rows) return error.KeepOutOfStep;
        r.pending = false;
    }
    pub fn draft(r: Round, start: u64, path: []const u32, follows: usize) !void {
        if (start != r.start or path.len == 0 or path.len > r.rows or path.len != follows)
            return error.DraftOutOfStep;
        for (path, 0..) |row, i| if (row != i) return error.TreeDraftsUnsupported;
    }
};

pub const Plan = struct {
    total: usize = 0,
    starts: [max_rows]usize = @splat(0),
    positions: [max_rows]u32 = @splat(0),
    keys: [max_rows]u32 = @splat(0),

    pub fn init(windows: []const Window, capacity: u32) !Plan {
        if (windows.len == 0) return error.EmptyBatch;
        if (windows.len > max_rows) return error.TooManyRows;
        var p: Plan = .{};
        for (windows, 0..) |w, i| {
            if (w.rows == 0) return error.EmptyWindow;
            if (w.rows > max_rows - p.total) return error.TooManyRows;
            if (w.pos > capacity or w.rows > capacity - w.pos) return error.ContextFull;
            for (windows[0..i]) |old| if (old.slot == w.slot) return error.DuplicateSlot;
            p.starts[i] = p.total;
            for (0..w.rows) |r| {
                p.positions[p.total + r] = w.pos + @as(u32, @intCast(r));
                p.keys[p.total + r] = p.positions[p.total + r] + 1;
            }
            p.total += w.rows;
        }
        return p;
    }
};

pub fn fit(want: u32, fixed: bool, room: u64, per_slot: u64) !u32 {
    if (per_slot == 0) return error.BadSlotBudget;
    const extra: u64 = @min(max_rows - 1, room / per_slot);
    const n: u32 = @intCast(@min(@as(u64, @min(@max(want, 1), max_rows)), 1 + extra));
    if (fixed and n < want) return error.OverMemoryLimit;
    return n;
}

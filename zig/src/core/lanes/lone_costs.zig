//! Price a lone window from its own complete rounds, retaining startup tile steps between measured widths.
const std = @import("std");
const Table = @import("table.zig").Table;

fn usable(ms: ?f64) bool {
    const x = ms orelse return false;
    return x > 0 and std.math.isFinite(x);
}

fn observed(s: anytype) Table {
    const state = s.depth orelse return .{};
    return state.ms orelse .{};
}

pub fn enabled(e: anytype, p: anytype) bool {
    return e.alone and e.cfg.lone_cost_tile > 0 and e.cfg.timed and e.cfg.plain_guard and
        e.cfg.draft_confidence == null and p.kind == .head and p.lanes == null;
}

/// Measured full-round time at this width; otherwise the nearest sample plus the startup forward-time difference.
pub fn estimate(startup: Table, samples: Table, rows: usize, overhead: f64) ?f64 {
    if (usable(samples.get(@intCast(rows - 1)))) return samples.get(@intCast(rows - 1));
    const base = startup.get(@intCast(rows)) orelse return null;
    var nearest: ?usize = null;
    var distance: usize = std.math.maxInt(usize);
    for (samples.slots, 1..) |ms, width| {
        if (!usable(ms) or !usable(startup.get(@intCast(width)))) continue;
        const delta = if (width > rows) width - rows else rows - width;
        if (delta < distance) {
            nearest = width;
            distance = delta;
        }
    }
    if (nearest) |width| {
        const price = samples.get(@intCast(width - 1)).? + base - startup.get(@intCast(width)).?;
        if (usable(price)) return price;
    }
    return base + overhead;
}

/// Arena-owned complete-round prices; callers must pass zero additional overhead to the allocator.
pub fn prices(e: anytype, p: anytype) !Table {
    var out: Table = .{};
    const overhead = e.rule.overheadMs(1, p.stream.sampling != null);
    const samples = observed(p.stream);
    for (1..p.count() + 2) |rows| {
        if (estimate(e.cfg.shared_costs, samples, rows, overhead)) |ms|
            try out.put(e.arena.allocator(), @intCast(rows), ms);
    }
    return out;
}

/// Keep overdue calibration through short offers; space later probes from the completed trial.
pub fn probe(e: anytype, p: anytype) ?u32 {
    const s = p.stream;
    if (s.rounds < s.cost_probe_after) return null;
    const samples = observed(s);
    const most: u32 = @intCast(@min(p.count() + 1, e.cfg.batch_rows));
    var retry: ?u32 = null;
    var earliest: u64 = std.math.maxInt(u64);
    var rows: u32 = 1;
    while (true) {
        if (rows < s.cost_probes.len) {
            const cd = s.cost_probes[rows];
            if (s.rounds >= cd.after) {
                if (!usable(samples.get(rows - 1))) return rows - 1;
                if (cd.retry and (retry == null or cd.after < earliest)) {
                    retry = rows - 1;
                    earliest = cd.after;
                }
            }
        }
        if (rows == most) return retry;
        rows = @min(most, if (rows == 1) @max(2, e.cfg.lone_cost_tile) else rows +| e.cfg.lone_cost_tile);
    }
}

pub const Cooldown = struct { after: u64 = 8, spacing: u64 = 8, retry: bool = false };

pub const Trial = struct {
    rows: u32,
    expected: f64,
    best: ?struct { expected: f64, ms: f64 } = null,
};

/// Snapshot landing probabilities and only this stream's measured, feasible alternatives before forcing a trial.
pub fn trial(s: anytype, chances: []const f64, rows: u32) Trial {
    var out: Trial = .{ .rows = rows, .expected = 1 };
    const samples = observed(s);
    var expected: f64 = 1;
    for (0..chances.len + 1) |i| {
        if (i > 0) expected += chances[i - 1];
        if (i + 1 == rows) out.expected = expected;
        const ms = samples.get(@intCast(i));
        if (i + 1 == rows or !usable(ms)) continue;
        if (out.best == null or expected / ms.? > out.best.?.expected / out.best.?.ms)
            out.best = .{ .expected = expected, .ms = ms.? };
    }
    return out;
}

/// A completed trial resets its own sample; losing widths retry after doubling and an expected repayment bound.
pub fn observe(s: anytype, requested: ?u32, rows: usize, ms: f64) void {
    if (rows == 0 or rows >= s.cost_probes.len or !usable(ms)) return;
    const samples = observed(s);
    if (rows > samples.slots.len) return;
    const cd = &s.cost_probes[rows];
    if (requested == null) {
        // Active widths refresh their age; a width abandoned after an ordinary bad sample remains retryable.
        cd.after = s.rounds +| cd.spacing;
        cd.retry = true;
        return;
    }
    if (requested) |width| if (width == rows and rows < s.cost_probes.len) {
        if (s.cost_probe_trial) |t| {
            if (t.rows != rows) return;
            const previous = cd.spacing;
            cd.spacing = 8;
            cd.retry = false;
            if (t.best) |best| {
                const expected_saving = best.expected * ms / t.expected - best.ms;
                if (expected_saving > 0) {
                    const repay = @ceil(ms / expected_saving);
                    const limit: f64 = @floatFromInt(std.math.maxInt(u64));
                    const spacing: u64 = if (repay >= limit) std.math.maxInt(u64) else @intFromFloat(repay);
                    cd.spacing = @max(previous *| 2, spacing);
                    cd.retry = true;
                }
            }
        }
        samples.slots[rows - 1] = ms;
        cd.after = s.rounds +| cd.spacing;
        s.cost_probe_trial = null;
        s.cost_probe_spacing = cd.spacing;
        s.cost_probe_after = cd.after;
    };
}

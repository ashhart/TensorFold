//! Shared rows by landing probability and forward costs between timed widths (Python allocate.py, extend_costs).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;

pub const Cost = struct { width: u32, ms: f64 };

/// Draft prefixes per stream with the most expected tokens a ms; fixed rows always run. Caller frees.
pub fn allocate(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, overhead_ms: f64, max_rows: u64) ![]u32 {
    const counts = try gpa.alloc(u32, probs.len);
    defer gpa.free(counts);
    const best_counts = try gpa.alloc(u32, probs.len);
    @memset(counts, 0);
    @memset(best_counts, 0);
    var rows: u64 = 0;
    for (fixed) |f| rows += f;
    var expected: f64 = @floatFromInt(rows);
    const timed = !costs.empty();
    var best = rate(costs, timed, rows, expected, overhead_ms);
    while (rows < max_rows) {
        // the likeliest next node, ties to the lowest stream (Python's heapq of (-p, stream))
        var pick: ?usize = null;
        var pick_p: f64 = 0.0;
        for (probs, 0..) |p, s| {
            if (counts[s] >= p.len) continue;
            const v = p[counts[s]];
            if (pick == null or v > pick_p) {
                pick = s;
                pick_p = v;
            }
        }
        const s = pick orelse break;
        counts[s] += 1;
        rows += 1;
        expected += pick_p;
        const now = rate(costs, timed, rows, expected, overhead_ms);
        if (now > best) {
            best = now;
            @memcpy(best_counts, counts);
        }
    }
    return best_counts;
}

fn rate(costs: *const Table, timed: bool, total: u64, expected: f64, overhead_ms: f64) f64 {
    var cost: f64 = 1.0;
    if (timed) cost = costs.get(@intCast(total)) orelse return -1.0;
    return expected / (cost + overhead_ms);
}

/// A chain's nodes' chances from per-depth acceptance (the j-th lands if every earlier one did). Caller frees.
pub fn chainProbabilities(gpa: Allocator, rates: []const f64, count: usize) ![]f64 {
    const out = try gpa.alloc(f64, count);
    var reach: f64 = 1.0;
    for (out, 0..) |*o, j| {
        reach *= if (j < rates.len) rates[j] else rates[rates.len - 1];
        o.* = reach;
    }
    return out;
}

/// Costs for 1..rows: timed widths as measured, interpolated between them, the widest's ms a row past it.
pub fn extendCosts(gpa: Allocator, costs: []const Cost, rows: u32) !Table {
    var out: Table = .{};
    if (costs.len == 0) return out;
    const measured = try gpa.dupe(Cost, costs);
    defer gpa.free(measured);
    std.mem.sort(Cost, measured, {}, struct {
        fn less(_: void, a: Cost, b: Cost) bool {
            return a.width < b.width;
        }
    }.less);
    const widest = measured[measured.len - 1];
    var total: u32 = 1;
    while (total <= rows) : (total += 1) {
        var value: f64 = undefined;
        if (find(measured, total)) |ms| {
            value = ms;
        } else if (total > widest.width) {
            value = widest.ms * @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(widest.width));
        } else {
            var above: Cost = undefined;
            for (measured) |c| {
                if (c.width > total) {
                    above = c;
                    break;
                }
            }
            var below: ?Cost = null;
            for (measured) |c| {
                if (c.width < total) below = c;
            }
            if (below) |b| {
                const span: f64 = @floatFromInt(above.width - b.width);
                value = b.ms + (above.ms - b.ms) * @as(f64, @floatFromInt(total - b.width)) / span;
            } else value = above.ms;
        }
        try out.put(gpa, total, value);
    }
    return out;
}

fn find(costs: []const Cost, width: u32) ?f64 {
    for (costs) |c| if (c.width == width) return c.ms;
    return null;
}

test "allocate takes the widest round when no cost is timed" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    const probs = [_][]const f64{ &.{ 0.9, 0.5 }, &.{0.8} };
    const counts = try allocate(gpa, &.{ 1, 1 }, &probs, &empty, 0.0, 8);
    defer gpa.free(counts);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1 }, counts);
}

test "extendCosts interpolates and extrapolates" {
    const gpa = std.testing.allocator;
    var t = try extendCosts(gpa, &.{ .{ .width = 4, .ms = 8.0 }, .{ .width = 2, .ms = 4.0 } }, 6);
    defer t.deinit(gpa);
    try std.testing.expectEqual(@as(?f64, 4.0), t.get(1));
    try std.testing.expectEqual(@as(?f64, 6.0), t.get(3));
    try std.testing.expectEqual(@as(?f64, 12.0), t.get(6));
}

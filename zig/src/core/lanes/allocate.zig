//! Shared rows by landing probability and forward costs between timed widths (Python allocate.py, extend_costs).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;

pub const Cost = struct { width: u32, ms: f64 };

/// Draft prefixes per stream with the most expected tokens a ms; fixed rows always run. Caller frees.
pub fn allocate(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, overhead_ms: f64, max_rows: u64) ![]u32 {
    return allocateDrafts(gpa, fixed, probs, costs, overhead_ms, max_rows, .{});
}

/// What drafting the counted nodes costs: a head step each, or the deepest stream's steps when one batch drafts all.
pub const HeadCost = struct {
    step_ms: f64 = 0.0,
    batched: bool = false,
    free_first: bool = false,
};

/// allocate, each stream's drafts past its first costing one more head step.
pub fn allocatePriced(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, overhead_ms: f64, max_rows: u64, step_ms: f64) ![]u32 {
    return allocateDrafts(gpa, fixed, probs, costs, overhead_ms, max_rows, .{ .step_ms = step_ms, .free_first = true });
}

/// allocate, each count also priced by the head steps that draft it.
pub fn allocateDrafts(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, overhead_ms: f64, max_rows: u64, head: HeadCost) ![]u32 {
    const counts = try gpa.alloc(u32, probs.len);
    defer gpa.free(counts);
    const best_counts = try gpa.alloc(u32, probs.len);
    @memset(counts, 0);
    @memset(best_counts, 0);
    var rows: u64 = 0;
    for (fixed) |f| rows += f;
    var expected: f64 = @floatFromInt(rows);
    var drafts: u64 = 0;
    var deepest: u32 = 0;
    var firsts: u64 = 0;
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
        firsts += @intFromBool(counts[s] == 1);
        rows += 1;
        drafts += 1;
        deepest = @max(deepest, counts[s]);
        expected += pick_p;
        const steps: f64 = @floatFromInt(if (head.batched) deepest else drafts - (if (head.free_first) firsts else 0));
        const now = rate(costs, timed, rows, expected, overhead_ms + head.step_ms * steps);
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

test "a priced draft step keeps a stream's later drafts out when they land too seldom to pay" {
    const gpa = std.testing.allocator;
    var costs = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 12.0 } }, 8);
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.6, 0.3, 0.15 }, &.{ 0.6, 0.3, 0.15 } };
    const free = try allocate(gpa, &.{ 1, 1 }, &probs, &costs, 2.0, 8);
    defer gpa.free(free);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, free);
    const priced = try allocatePriced(gpa, &.{ 1, 1 }, &probs, &costs, 2.0, 8, 1.5);
    defer gpa.free(priced);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, priced);
}

test "a priced draft step leaves each stream's first draft free" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 10.0 } }, 8);
    defer flat.deinit(gpa);
    const probs = [_][]const f64{&.{0.5}};
    // charged, the draft would cost 100 ms against 0.5 tokens: it stays because the first is free
    const counts = try allocatePriced(gpa, &.{1}, &probs, &flat, 0.0, 8, 100.0);
    defer gpa.free(counts);
    try std.testing.expectEqualSlices(u32, &.{1}, counts);
}

test "a priced draft step cuts the stream whose later drafts land seldom and keeps the other's" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 10.0 } }, 8);
    defer flat.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.9, 0.85, 0.8 }, &.{ 0.9, 0.2, 0.1 } };
    const priced = try allocatePriced(gpa, &.{ 1, 1 }, &probs, &flat, 0.0, 8, 1.0);
    defer gpa.free(priced);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1 }, priced);
    const free = try allocate(gpa, &.{ 1, 1 }, &probs, &flat, 0.0, 8);
    defer gpa.free(free);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, free);
}

test "a zero draft step prices rounds as allocate does" {
    const gpa = std.testing.allocator;
    var costs = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 12.0 } }, 8);
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.9, 0.85, 0.8 }, &.{ 0.9, 0.2, 0.1 }, &.{0.4} };
    const priced = try allocatePriced(gpa, &.{ 1, 1, 1 }, &probs, &costs, 2.0, 8, 0.0);
    defer gpa.free(priced);
    const free = try allocate(gpa, &.{ 1, 1, 1 }, &probs, &costs, 2.0, 8);
    defer gpa.free(free);
    try std.testing.expectEqualSlices(u32, free, priced);
}

test "extendCosts interpolates and extrapolates" {
    const gpa = std.testing.allocator;
    var t = try extendCosts(gpa, &.{ .{ .width = 4, .ms = 8.0 }, .{ .width = 2, .ms = 4.0 } }, 6);
    defer t.deinit(gpa);
    try std.testing.expectEqual(@as(?f64, 4.0), t.get(1));
    try std.testing.expectEqual(@as(?f64, 6.0), t.get(3));
    try std.testing.expectEqual(@as(?f64, 12.0), t.get(6));
}

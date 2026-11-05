//! Shared rows by landing probability, host draw prices and forward costs between timed widths.
const std = @import("std");
const hold_price = @import("hold_price.zig");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;

/// A batched head's extra drafts cost the deepest stream's steps; FZ_PRICE_BATCHED=0 prices each stream's.
pub fn pricedBatched() bool {
    const v = std.c.getenv("FZ_PRICE_BATCHED") orelse return true;
    return v[0] != '0';
}

pub const Cost = struct { width: u32, ms: f64 };

/// Draft prefixes per stream with the most expected tokens a ms; fixed rows always run. Caller frees.
pub fn allocate(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, overhead_ms: f64, max_rows: u64) ![]u32 {
    return allocateDrafts(gpa, fixed, probs, costs, overhead_ms, max_rows, .{});
}

/// Grid prices replace steps; empty `charged` means all streams, and `hold_rows == 0` means unbounded passes.
pub const HeadCost = struct {
    step_ms: f64 = 0.0,
    batched: bool = false,
    free_first: bool = false,
    hold_costs: []const hold_price.HoldCost = &.{},
    hold_context_costs: []const hold_price.ContextCost = &.{},
    hold_context_rows: []const u32 = &.{},
    charged: []const bool = &.{},
    row_ms: f64 = 0, // learned cost per row past each live stream's pending row, excluding quoted host work
    host_row_ms: []const f64 = &.{}, // each stream's host draw cost per verified row, independent of the neural mask
    hold_rows: u32 = 0,
    hold_block_rows: u32 = 0,
};

/// Prices growing blocks in stream order; batched blocks pad to each physical pass's deepest, excluding free nodes.
const GridPass = struct {
    costs: []const hold_price.HoldCost,
    contexts: []const hold_price.ContextCost,
    debts: []const u32,
    batched: bool,
    free: u32,
    rows: u32,
    block_rows: u32,
    counts: []const u32, // the allocation so far, which prices a split batch exactly
    charged: []const bool,
    blocks: usize = 0, // batched: streams with a node past the free one
    deepest: i64 = 0, // and the deepest of them past it
    context_rows: u32 = 0,
    ms: f64 = 0.0, // an empty pass costs nothing

    /// A stream's count grew to `count`: the pass's new price.
    fn grow(p: *GridPass, index: usize, count: u32) f64 {
        const depth: i64 = count - p.free;
        if (depth <= 0) return p.ms;
        if (p.batched) {
            p.blocks += @intFromBool(depth == 1);
            p.deepest = @max(p.deepest, depth);
            if (p.debts.len > 0) p.context_rows = @max(p.context_rows, p.debts[index]);
            p.ms = if (p.rows == 0 or @as(i64, @intCast(p.blocks)) * @max(p.deepest, p.block_rows) <= p.rows)
                hold_price.holdMs(p.costs, p.blocks, p.deepest) + hold_price.contextMs(p.contexts, p.blocks, p.context_rows)
            else
                hold_price.countsMs(p.costs, p.contexts, p.rows, p.block_rows, p.counts, p.charged, p.free, p.debts);
        } else {
            p.ms += hold_price.holdMs(p.costs, 1, depth) - (if (depth > 1) hold_price.holdMs(p.costs, 1, depth - 1) else 0.0);
            if (depth == 1 and p.debts.len > 0) p.ms += hold_price.contextMs(p.contexts, 1, p.debts[index]);
        }
        return p.ms;
    }
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
    @memset(best_counts, 0);
    var fixed_rows: u64 = 0;
    var live: u64 = 0;
    var fixed_host_ms: f64 = 0;
    for (fixed, 0..) |f, i| {
        fixed_rows += f;
        live += @intFromBool(f > 0);
        if (head.host_row_ms.len > 0) {
            fixed_host_ms += @as(f64, @floatFromInt(f)) * head.host_row_ms[i];
        }
    }
    const grid = head.hold_costs.len > 0;
    const timed = !costs.empty();
    const mixed_host = timed and head.host_row_ms.len > 0 and mixed: {
        var quote: ?f64 = null;
        for (probs, 0..) |p, s| {
            if (p.len == 0) continue;
            if (quote) |first| {
                if (head.host_row_ms[s] != first) break :mixed true;
            } else quote = head.host_row_ms[s];
        }
        break :mixed false;
    };
    const context_order = timed and head.hold_context_costs.len > 0 and head.hold_context_rows.len > 0 and context: {
        for (probs, 0..) |p, s| {
            if (p.len > @intFromBool(head.free_first) and head.hold_context_rows[s] > 0 and
                (head.charged.len == 0 or head.charged[s])) break :context true;
        }
        break :context false;
    };
    const mixed = mixed_host or context_order;
    var best = rate(costs, timed, fixed_rows, @floatFromInt(fixed_rows), overhead_ms + fixed_host_ms + head.row_ms * @as(f64, @floatFromInt(fixed_rows -| live)));
    while (true) {
        const incumbent = best;
        // Unknown/nonfinite rates cannot price marginal utility (in particular, infinity * a free row).
        const price_order = mixed and incumbent >= 0 and std.math.isFinite(incumbent);
        @memset(counts, 0);
        var rows = fixed_rows;
        var host_ms = fixed_host_ms;
        var expected: f64 = @floatFromInt(rows);
        var drafts: u64 = 0;
        var deepest: u32 = 0;
        var firsts: u64 = 0;
        var pass: GridPass = .{ .costs = head.hold_costs, .contexts = head.hold_context_costs, .debts = head.hold_context_rows, .batched = head.batched, .free = @intFromBool(head.free_first), .rows = head.hold_rows, .block_rows = head.hold_block_rows, .counts = counts, .charged = head.charged };
        var head_ms: f64 = 0.0; // the nodes counted so far: the head's price
        while (rows < max_rows) {
            // Solo first-block context quotes order candidates; the complete physical group decides funding.
            var pick: ?usize = null;
            var pick_p: f64 = 0.0;
            var pick_score: f64 = 0.0;
            for (probs, 0..) |p, s| {
                if (counts[s] >= p.len) continue;
                const v = p[counts[s]];
                const score = if (price_order) ordered: {
                    var ms = if (head.host_row_ms.len > 0) head.host_row_ms[s] else 0;
                    if (context_order and counts[s] == @intFromBool(head.free_first) and
                        (head.charged.len == 0 or head.charged[s]))
                        ms += hold_price.contextMs(head.hold_context_costs, 1, head.hold_context_rows[s]);
                    break :ordered if (std.math.isFinite(ms)) v - incumbent * ms else -std.math.inf(f64);
                } else v;
                if (pick == null or score > pick_score) {
                    pick = s;
                    pick_p = v;
                    pick_score = score;
                }
            }
            const s = pick orelse break;
            counts[s] += 1;
            rows += 1;
            expected += pick_p;
            if (head.host_row_ms.len > 0) host_ms += head.host_row_ms[s];
            if (head.charged.len == 0 or head.charged[s]) {
                firsts += @intFromBool(counts[s] == 1);
                drafts += 1;
                deepest = @max(deepest, counts[s]);
                if (grid) head_ms = pass.grow(s, counts[s]);
            }
            if (!grid) {
                const steps: f64 = @floatFromInt(if (head.batched) deepest -| @intFromBool(head.free_first) else drafts - (if (head.free_first) firsts else 0));
                head_ms = head.step_ms * steps;
            }
            const now = rate(costs, timed, rows, expected, overhead_ms + host_ms + head_ms + head.row_ms * @as(f64, @floatFromInt(rows -| live)));
            if (now > best or (!timed and head.step_ms == 0 and !grid)) {
                best = now;
                @memcpy(best_counts, counts);
            }
        }
        // Strict improvement over finite prefix orders terminates without claiming a global grid search.
        if (!mixed or !(best > incumbent) or !std.math.isFinite(best)) break;
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
    chainProbabilitiesInto(rates, out);
    return out;
}

/// Fill a caller-owned chain buffer without allocating a second copy.
pub fn chainProbabilitiesInto(rates: []const f64, out: []f64) void {
    var reach: f64 = 1.0;
    for (out, 0..) |*o, j| {
        reach *= if (j < rates.len) rates[j] else rates[rates.len - 1];
        o.* = reach;
    }
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

// SYNTHETIC grids and costs below pin the arithmetic of the pass price, not any device's timings.
test "a timed hold grid prices the pass the drafts make, not a linear step each" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 10.0 } }, 8);
    defer flat.deinit(gpa);
    const probs = [_][]const f64{ &.{0.5}, &.{0.4} };
    // nine drafts average 1.2 ms apiece, yet any hold at all costs its 6 ms pass
    const grid = [_]hold_price.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 6.0 }, .{ .streams = 1, .depth = 9, .ms = 10.8 }, .{ .streams = 2, .depth = 1, .ms = 6.2 }, .{ .streams = 2, .depth = 9, .ms = 11.0 } };
    const linear = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &flat, 0.0, 8, .{ .step_ms = 1.2, .batched = true });
    defer gpa.free(linear);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, linear); // 2.9 tokens in 11.2 ms beat 2 in 10
    const priced = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &flat, 0.0, 8, .{ .step_ms = 1.2, .batched = true, .hold_costs = &grid });
    defer gpa.free(priced);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0 }, priced); // 2.9 tokens in 16.2 ms lose
    // Scalar steps cannot represent cheap additional streams and expensive additional depths.
    const deep = [_]hold_price.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 2.0 }, .{ .streams = 1, .depth = 2, .ms = 5.0 }, .{ .streams = 2, .depth = 1, .ms = 2.2 }, .{ .streams = 2, .depth = 2, .ms = 5.2 } };
    const chain = [_][]const f64{ &.{ 0.9, 0.5 }, &.{ 0.8, 0.3 } };
    const stepped = try allocateDrafts(gpa, &.{ 1, 1 }, &chain, &flat, 0.0, 8, .{ .step_ms = 2.0, .batched = true });
    defer gpa.free(stepped);
    try std.testing.expectEqualSlices(u32, &.{ 2, 2 }, stepped);
    const shaped = try allocateDrafts(gpa, &.{ 1, 1 }, &chain, &flat, 0.0, 8, .{ .step_ms = 2.0, .batched = true, .hold_costs = &deep });
    defer gpa.free(shaped);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, shaped);
}

test "a hold grid prices a batch by its active blocks and deepest, or each stream's own block, past any free first" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 10.0 } }, 8);
    defer flat.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.9, 0.7 }, &.{ 0.8, 0.6 } };
    // Serial passes ignore multi-stream prices; free first nodes add no hold cost.
    const grid = [_]hold_price.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 3.0 }, .{ .streams = 1, .depth = 2, .ms = 4.0 }, .{ .streams = 2, .depth = 1, .ms = 9.0 }, .{ .streams = 2, .depth = 2, .ms = 14.0 } };
    const Case = struct { batched: bool, free_first: bool, want: [2]u32 };
    const cases = [_]Case{
        .{ .batched = false, .free_first = true, .want = .{ 1, 1 } }, // the firsts cost nothing; 3 ms a second draft does not pay
        .{ .batched = false, .free_first = false, .want = .{ 2, 2 } }, // 3 + 3 + 1 + 1 ms for all four: 5 tokens in 18 ms
        .{ .batched = true, .free_first = true, .want = .{ 1, 1 } }, // a free first holds no block: the second drafts' 3 and 9 ms do not pay
        .{ .batched = true, .free_first = false, .want = .{ 1, 0 } }, // the second block's 9 ms are too dear: one block of one draft
    };
    for (cases) |c| {
        const got = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &flat, 0.0, 8, .{ .step_ms = 50.0, .batched = c.batched, .free_first = c.free_first, .hold_costs = &grid });
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, &c.want, got);
    }
}

test "a grid of one linear price per draft allocates as the step model does, however its pass is shaped" {
    const gpa = std.testing.allocator;
    var costs = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 16, .ms = 14.0 } }, 16);
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.9, 0.8, 0.7, 0.6 }, &.{ 0.85, 0.4, 0.3 }, &.{ 0.6, 0.5 } };
    var grid: [8]hold_price.HoldCost = undefined;
    for (&grid, 0..) |*c, i| c.* = .{ .streams = if (i < 4) 1 else 4, .depth = @intCast(i % 4 + 1), .ms = 1.5 * @as(f64, @floatFromInt(i % 4 + 1)) };
    for ([_]bool{ false, true }) |batched| for ([_]bool{ false, true }) |free_first| {
        const step = try allocateDrafts(gpa, &.{ 1, 1, 1 }, &probs, &costs, 2.0, 16, .{ .step_ms = 1.5, .batched = batched, .free_first = free_first });
        defer gpa.free(step);
        const priced = try allocateDrafts(gpa, &.{ 1, 1, 1 }, &probs, &costs, 2.0, 16, .{ .step_ms = 99.0, .batched = batched, .free_first = free_first, .hold_costs = &grid });
        defer gpa.free(priced);
        try std.testing.expectEqualSlices(u32, step, priced);
    };
}

test "a hold grid charges only the streams the head drafts: a copy's nodes take rows but no hold" {
    const gpa = std.testing.allocator;
    var costs = try extendCosts(gpa, &.{ .{ .width = 2, .ms = 10.0 }, .{ .width = 4, .ms = 12.0 } }, 4);
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{0.99}, &.{0.8} }; // a copied token, then a head's draft
    const grid = [_]hold_price.HoldCost{.{ .streams = 1, .depth = 1, .ms = 20.0 }};
    const charged = [_]bool{ false, true };
    // the copy's row is nearly free (2.99 tokens in 11 ms); the head's hold would cost 20 ms for 0.8 more
    const mixed = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &costs, 0.0, 4, .{ .hold_costs = &grid, .batched = true, .charged = &charged });
    defer gpa.free(mixed);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, mixed);
    // charged as a head's, the copy would bear a 20 ms hold of its own: rejected with the head
    const all = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &costs, 0.0, 4, .{ .hold_costs = &grid, .batched = true });
    defer gpa.free(all);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0 }, all);
    // without a grid the head's steps are free and the head stays
    const free = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &costs, 0.0, 4, .{});
    defer gpa.free(free);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, free);
}

test "host rows price pending work once and each optional draw even when the head does not charge it" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{.{ .width = 8, .ms = 10 }}, 8);
    defer flat.deinit(gpa);
    const chances = [_][]const f64{&.{ 0.5, 0.2, 0.1 }};
    for ([_]bool{ false, true }) |neural| {
        const counts = try allocateDrafts(gpa, &.{1}, &chances, &flat, 0, 8, .{ .charged = &.{neural}, .host_row_ms = &.{8} });
        defer gpa.free(counts);
        // 1 / 18 < 1.5 / 26, but 1.7 / 34 loses. Omitting pending draw cost wrongly rejects the first.
        try std.testing.expectEqualSlices(u32, &.{1}, counts);
    }
    const free = try allocateDrafts(gpa, &.{1}, &chances, &flat, 0, 8, .{ .host_row_ms = &.{0} });
    defer gpa.free(free);
    try std.testing.expectEqualSlices(u32, &.{3}, free);
    const untimed: Table = .{};
    const priced = try allocateDrafts(gpa, &.{1}, &chances, &untimed, 0, 8, .{ .host_row_ms = &.{8} });
    defer gpa.free(priced);
    // Unknown target cost is not a measured-free forward: preserve the untimed exploration path.
    try std.testing.expectEqualSlices(u32, &.{3}, priced);
}

test "unequal mixed routes keep cheap rows in either order without charging all rows as sampled" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{.{ .width = 9, .ms = 10 }}, 9);
    defer flat.deinit(gpa);
    for ([_]bool{ false, true }) |swapped| {
        const greedy: usize = @intFromBool(swapped);
        const sampled = 1 - greedy;
        var fixed: [2]u32 = undefined;
        var probabilities: [2][]const f64 = undefined;
        var host: [2]f64 = undefined;
        fixed[greedy] = 1;
        fixed[sampled] = 3;
        probabilities[greedy] = &.{ 0.9, 0.8, 0.7 };
        probabilities[sampled] = &.{ 0.6, 0.2 };
        host[greedy] = 0;
        host[sampled] = 5;
        const counts = try allocateDrafts(gpa, &fixed, &probabilities, &flat, 0, 9, .{ .host_row_ms = &host });
        defer gpa.free(counts);
        try std.testing.expectEqual(@as(u32, 3), counts[greedy]);
        try std.testing.expectEqual(@as(u32, 0), counts[sampled]);
    }
}

test "copied rows pay host draws but never inherit the neural hold price" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{.{ .width = 4, .ms = 10 }}, 4);
    defer flat.deinit(gpa);
    const chances = [_][]const f64{ &.{0.99}, &.{0.8} };
    const grid = [_]hold_price.HoldCost{.{ .streams = 1, .depth = 1, .ms = 20 }};
    for ([_]f64{ 0, 30 }) |copy_ms| {
        const counts = try allocateDrafts(gpa, &.{ 1, 1 }, &chances, &flat, 0, 4, .{ .hold_costs = &grid, .batched = true, .charged = &.{ false, true }, .host_row_ms = &.{ copy_ms, 0 } });
        defer gpa.free(counts);
        try std.testing.expectEqualSlices(u32, if (copy_ms == 0) &.{ 1, 0 } else &.{ 0, 0 }, counts);
    }
}

test "a hold past the pass's rows is priced as the passes it takes: an asymmetric cohort stops where its blocks stop fitting" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 32, .ms = 10.0 } }, 32);
    defer flat.deinit(gpa);
    // every pass costs 10 ms whatever its shape, so the price is the number of passes
    const grid = [_]hold_price.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 10 }, .{ .streams = 1, .depth = 9, .ms = 10 }, .{ .streams = 32, .depth = 1, .ms = 10 }, .{ .streams = 32, .depth = 9, .ms = 10 } };
    const deep: [9]f64 = @splat(0.99);
    const one = [_]f64{0.9};
    const Case = struct { deep_at: usize, rows: u32, want: [8]u32 };
    const cases = [_]Case{
        // one padded pass of eight blocks of nine: 23.21 expected tokens in 10 + 10 ms
        .{ .deep_at = 0, .rows = 0, .want = .{ 9, 1, 1, 1, 1, 1, 1, 1 } },
        // 32 rows hold three such blocks: eight make two passes, 23.21 in 30 ms, below the three blocks' 18.71 in 20
        .{ .deep_at = 0, .rows = 32, .want = .{ 9, 1, 1, 0, 0, 0, 0, 0 } },
        // A deep fourth stream splits the preceding shallow group before that stream is added.
        .{ .deep_at = 3, .rows = 32, .want = .{ 1, 1, 0, 9, 0, 0, 0, 0 } },
        .{ .deep_at = 3, .rows = 0, .want = .{ 1, 1, 1, 9, 1, 1, 1, 1 } },
        // 36 rows fit four blocks of nine; the fifth would need a second pass
        .{ .deep_at = 0, .rows = 36, .want = .{ 9, 1, 1, 1, 0, 0, 0, 0 } },
    };
    for (cases) |c| {
        var probs: [8][]const f64 = @splat(@as([]const f64, &one));
        const fixed: [8]u32 = @splat(1);
        probs[c.deep_at] = &deep;
        const got = try allocateDrafts(gpa, &fixed, &probs, &flat, 0.0, 32, .{ .hold_costs = &grid, .batched = true, .hold_rows = c.rows });
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, &c.want, got);
    }
}

test "extendCosts interpolates and extrapolates" {
    const gpa = std.testing.allocator;
    var t = try extendCosts(gpa, &.{ .{ .width = 4, .ms = 8.0 }, .{ .width = 2, .ms = 4.0 } }, 6);
    defer t.deinit(gpa);
    try std.testing.expectEqual(@as(?f64, 4.0), t.get(1));
    try std.testing.expectEqual(@as(?f64, 6.0), t.get(3));
    try std.testing.expectEqual(@as(?f64, 12.0), t.get(6));
}

test "mixed host prices keep the cheaper lower-probability row in either stream order" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{.{ .width = 4, .ms = 10 }}, 4);
    defer flat.deinit(gpa);
    const untimed: Table = .{};
    for ([_]bool{ false, true }) |swapped| {
        const cheap: usize = @intFromBool(swapped);
        const expensive = 1 - cheap;
        var chances: [2][]const f64 = undefined;
        var host: [2]f64 = undefined;
        chances[cheap] = &.{0.8};
        chances[expensive] = &.{0.9};
        host[cheap] = 0;
        host[expensive] = 100;
        const counts = try allocateDrafts(gpa, &.{ 1, 1 }, &chances, &flat, 0, 4, .{ .host_row_ms = &host });
        defer gpa.free(counts);
        try std.testing.expectEqual(@as(u32, 1), counts[cheap]);
        try std.testing.expectEqual(@as(u32, 0), counts[expensive]);
        const unknown = try allocateDrafts(gpa, &.{ 1, 1 }, &chances, &untimed, 0, 4, .{ .host_row_ms = &host });
        defer gpa.free(unknown);
        try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, unknown);
    }
}

test "a later cheap cohort reprices an initially profitable expensive companion" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{.{ .width = 11, .ms = 10 }}, 11);
    defer flat.deinit(gpa);
    const cheap_chain: [8]f64 = @splat(0.3);
    for ([_]bool{ false, true }) |swapped| {
        const cheap: usize = @intFromBool(swapped);
        const expensive = 1 - cheap;
        var chances: [2][]const f64 = undefined;
        var host: [2]f64 = undefined;
        chances[cheap] = &cheap_chain;
        chances[expensive] = &.{0.9};
        host[cheap] = 0;
        host[expensive] = 3;
        // First 2.9/16 beats 2.3/13; after eight cheap rows, 4.4/13 beats 5.3/16 without the expensive row.
        const counts = try allocateDrafts(gpa, &.{ 1, 1 }, &chances, &flat, 0, 11, .{ .host_row_ms = &host });
        defer gpa.free(counts);
        try std.testing.expectEqual(@as(u32, 8), counts[cheap]);
        try std.testing.expectEqual(@as(u32, 0), counts[expensive]);
    }
}

test "uniform host quotes retain stream ties and the strict profit boundary" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{.{ .width = 4, .ms = 10 }}, 4);
    defer flat.deinit(gpa);
    const chances = [_][]const f64{ &.{0.75}, &.{0.75} };
    for ([_]f64{ 0, 5 }) |quote| {
        const counts = try allocateDrafts(gpa, &.{ 1, 1 }, &chances, &flat, 0, 3, .{ .host_row_ms = &.{ quote, quote } });
        defer gpa.free(counts);
        try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, counts);
    }
    const equal = [_][]const f64{ &.{0.5}, &.{0.5} };
    const boundary = try allocateDrafts(gpa, &.{ 1, 1 }, &equal, &flat, 0, 4, .{ .host_row_ms = &.{ 5, 5 } });
    defer gpa.free(boundary);
    // 2/20 == 2.5/25 == 3/30: equality never purchases more work.
    try std.testing.expectEqualSlices(u32, &.{ 0, 0 }, boundary);
}

test "an infinite measured baseline does not resweep mixed host quotes" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{.{ .width = 3, .ms = 0 }}, 3);
    defer flat.deinit(gpa);
    const chances = [_][]const f64{ &.{0.9}, &.{0.8} };
    const counts = try allocateDrafts(gpa, &.{ 0, 1 }, &chances, &flat, 0, 3, .{ .host_row_ms = &.{ 1, 0 } });
    defer gpa.free(counts);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0 }, counts);
}

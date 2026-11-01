//! Shared rows by landing probability and forward costs between timed widths (Python allocate.py, extend_costs).
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

/// What drafting the counted nodes costs: a head step each, or the deepest stream's steps when one batch drafts all.
/// A timed `hold_costs` grid replaces the steps: it prices the pass the counted nodes actually make. `charged`
/// (empty: every stream, else one flag a stream) names the streams whose nodes the head drafts; the others' nodes
/// (a copy's, say) still take rows and count as expected tokens but add no head cost. `hold_rows` (0: unbounded) is
/// the rows a drafter's hold pass holds at most: a batch past it runs as several passes (hold_price.rowsMs).
pub const HeadCost = struct {
    step_ms: f64 = 0.0,
    batched: bool = false,
    free_first: bool = false,
    hold_costs: []const hold_price.HoldCost = &.{},
    charged: []const bool = &.{},
    hold_rows: u32 = 0,
};

/// The grid's price of the counted nodes' hold pass, regrown a node at a time: one pass of the active streams' blocks
/// as deep as the deepest when batched (as several in stream order once their rows pass `rows`), else each stream's own
/// block. A `free_first` node is the round's own draft.
const GridPass = struct {
    costs: []const hold_price.HoldCost,
    batched: bool,
    free: u32,
    rows: u32,
    counts: []const u32, // the allocation so far, which prices a split batch exactly
    charged: []const bool,
    blocks: usize = 0, // batched: streams with a node past the free one
    deepest: i64 = 0, // and the deepest of them past it
    ms: f64 = 0.0, // an empty pass costs nothing

    /// A stream's count grew to `count`: the pass's new price.
    fn grow(p: *GridPass, count: u32) f64 {
        const depth: i64 = count - p.free;
        if (depth <= 0) return p.ms;
        if (p.batched) {
            p.blocks += @intFromBool(depth == 1);
            p.deepest = @max(p.deepest, depth);
            p.ms = if (p.rows == 0 or @as(i64, @intCast(p.blocks)) * p.deepest <= p.rows)
                hold_price.holdMs(p.costs, p.blocks, p.deepest)
            else
                hold_price.countsMs(p.costs, p.rows, p.counts, p.charged, p.free);
        } else {
            p.ms += hold_price.holdMs(p.costs, 1, depth) - (if (depth > 1) hold_price.holdMs(p.costs, 1, depth - 1) else 0.0);
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
    @memset(counts, 0);
    @memset(best_counts, 0);
    var rows: u64 = 0;
    for (fixed) |f| rows += f;
    var expected: f64 = @floatFromInt(rows);
    var drafts: u64 = 0;
    var deepest: u32 = 0;
    var firsts: u64 = 0;
    const grid = head.hold_costs.len > 0;
    var pass: GridPass = .{ .costs = head.hold_costs, .batched = head.batched, .free = @intFromBool(head.free_first), .rows = head.hold_rows, .counts = counts, .charged = head.charged };
    const timed = !costs.empty();
    var head_ms: f64 = 0.0; // the nodes counted so far: the head's price
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
        if (head.charged.len == 0 or head.charged[s]) {
            firsts += @intFromBool(counts[s] == 1);
            drafts += 1;
            deepest = @max(deepest, counts[s]);
            if (grid) head_ms = pass.grow(counts[s]);
        }
        if (!grid) {
            const steps: f64 = @floatFromInt(if (head.batched) deepest -| @intFromBool(head.free_first) else drafts - (if (head.free_first) firsts else 0));
            head_ms = head.step_ms * steps;
        }
        const now = rate(costs, timed, rows, expected, overhead_ms + head_ms);
        if (now > best or (!timed and head.step_ms == 0 and !grid)) {
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
    // a block deeper costs 3 ms but another block 0.2: second drafts pay at a linear 2 ms a step, not at their pass's price
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
    // two blocks are dear: a batch pays them, while a stream's own block is the one-block price (the 2-block ms are never read)
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
        // the deep stream fourth in order: the blocks of streams 0 and 1 fit beside it, a third small one splits the pass
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

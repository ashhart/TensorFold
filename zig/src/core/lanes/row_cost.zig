//! What a shared round's rows cost past the cost table on this machine, learned from the rounds it ran.
const std = @import("std");
const Allocator = std.mem.Allocator;

const most_rows = 64;
const rate = 0.1; // weight of the newest round at a width
const enough = 4; // rounds at a width before it counts
const heaviest = 16; // a width's rounds past this weigh no more in the fit
const least_widths = 3; // counted widths: two always fit a line, so noise could read as a slope
const least_span = 4; // rows between the narrowest and widest counted widths
const fresh_rounds = 2048; // a width not run for this long stops counting: drafts priced out return and measure it again
const sparse_rounds = 256; // a width back after this long says more than its old mean: it moves half way at once
const least_ms = 0.5; // a row's excess below this is the table's own noise
const spread = [_]u64{ 0, 2, 4 }; // rows past the streams' own a round measures while there is no fit
const most_ms = 25.0; // a row's excess past this is noise

/// The rounds at one stream count: ms past the table, by total rows.
const Widths = struct {
    streams: u32,
    ms: [most_rows + 1]f64 = @splat(0),
    rounds: [most_rows + 1]u32 = @splat(0),
    last: [most_rows + 1]u64 = @splat(0), // the clock when this width last ran

    fn counted(w: *const Widths, r: usize, now: u64) bool {
        return w.rounds[r] >= enough and now - w.last[r] <= fresh_rounds;
    }

    /// The least-squares ms a row across the counted widths; null until a fit clears its own noise.
    fn slope(w: *const Widths, now: u64) ?f64 {
        var weight: f64 = 0;
        var rows_sum: f64 = 0;
        var ms_sum: f64 = 0;
        var widths: u32 = 0;
        var narrowest: usize = most_rows;
        var widest: usize = 0;
        for (w.rounds, w.ms, 0..) |n, ms, r| {
            if (!w.counted(r, now)) continue;
            const v: f64 = @floatFromInt(@min(n, heaviest));
            weight += v;
            rows_sum += v * @as(f64, @floatFromInt(r));
            ms_sum += v * ms;
            widths += 1;
            narrowest = @min(narrowest, r);
            widest = @max(widest, r);
        }
        if (widths < least_widths or widest < narrowest + least_span) return null;
        const rows_mean = rows_sum / weight;
        const ms_mean = ms_sum / weight;
        var across: f64 = 0;
        var together: f64 = 0;
        for (w.rounds, w.ms, 0..) |n, ms, r| {
            if (!w.counted(r, now)) continue;
            const v: f64 = @floatFromInt(@min(n, heaviest));
            const d = @as(f64, @floatFromInt(r)) - rows_mean;
            across += v * d * d;
            together += v * d * (ms - ms_mean);
        }
        const fit = together / across;
        var left: f64 = 0;
        for (w.rounds, w.ms, 0..) |n, ms, r| {
            if (!w.counted(r, now)) continue;
            const v: f64 = @floatFromInt(@min(n, heaviest));
            const miss = ms - ms_mean - fit * (@as(f64, @floatFromInt(r)) - rows_mean);
            left += v * miss * miss;
        }
        // the widths' scatter about the line over their spread in rows: a slope inside twice this may be nothing
        return if (fit > 2 * @sqrt(left / across)) fit else null;
    }
};

/// One fit's rounds, by stream count.
const Fit = struct {
    widths: std.ArrayList(Widths) = .empty,
    clock: u64 = 0,

    fn observe(f: *Fit, gpa: Allocator, streams: u32, rows: u64, ms: f64) !void {
        f.clock += 1;
        const w = for (f.widths.items) |*x| {
            if (x.streams == streams) break x;
        } else blk: {
            try f.widths.append(gpa, .{ .streams = streams });
            break :blk &f.widths.items[f.widths.items.len - 1];
        };
        const at: usize = @intCast(rows);
        const gap = f.clock - w.last[at];
        if (gap > fresh_rounds) w.rounds[at] = 0; // what it measured then says nothing now
        w.last[at] = f.clock;
        w.rounds[at] +|= 1;
        // a plain mean of the first rounds, so one slow first round does not linger
        const weight = @max(@max(rate, 1 / @as(f64, @floatFromInt(w.rounds[at]))), @as(f64, if (gap > sparse_rounds) 0.5 else 0));
        w.ms[at] += weight * (ms - w.ms[at]);
    }

    /// While a stream count has no fit and has not measured every width of `spread`: the rows this round may take.
    fn cap(f: *const Fit, streams: usize) ?u64 {
        const w = for (f.widths.items) |*x| {
            if (x.streams == streams) break x;
        } else return streams; // nothing measured here yet: start from the plain width
        if (w.slope(f.clock) != null) return null;
        var measured = true;
        for (spread) |past| measured = measured and streams + past <= most_rows and w.counted(streams + past, f.clock);
        if (measured) return null; // every width counted and still no fit: the table fits
        return streams + spread[@intCast((f.clock / enough) % spread.len)];
    }

    /// The fit at the nearest stream count that has one.
    fn extra(f: *const Fit, streams: usize) f64 {
        var best: ?f64 = null;
        var gap: usize = std.math.maxInt(usize);
        for (f.widths.items) |*w| {
            const d = if (w.streams > streams) w.streams - streams else streams - w.streams;
            if (d >= gap) continue;
            const s = w.slope(f.clock) orelse continue;
            best = s;
            gap = d;
        }
        const ms = best orelse return 0;
        return if (ms < least_ms) 0 else @min(ms, most_ms);
    }
};

pub const RowCost = struct {
    all: Fit = .{},
    greedy: Fit = .{}, // rounds without a sampled stream: a sampled row costs more, as the overhead keeps apart

    pub fn deinit(c: *RowCost, gpa: Allocator) void {
        c.all.widths.deinit(gpa);
        c.greedy.widths.deinit(gpa);
    }

    /// A shared round of `rows` at `streams` took `ms` past the table; a lone stream prices by its own window costs.
    pub fn observe(c: *RowCost, gpa: Allocator, streams: u32, rows: u64, ms: f64, sampled: bool) !void {
        if (streams < 2 or rows > most_rows) return;
        try c.all.observe(gpa, streams, rows, ms);
        if (sampled) c.greedy.clock += 1 else try c.greedy.observe(gpa, streams, rows, ms);
    }

    /// The most rows a shared round may take: `most`, or a width to measure while learning (no fit forms at one width).
    pub fn capRows(c: *const RowCost, streams: usize, sampled: bool, most: u64) u64 {
        if (streams < 2 or c.all.clock == 0) return most;
        const width = (if (sampled) c.all else c.greedy).cap(streams) orelse return most;
        return @min(width, most);
    }

    /// The ms each row costs past the table; never below it, and nothing while no fit clears its noise.
    pub fn extraMs(c: *const RowCost, streams: usize, sampled: bool) f64 {
        if (streams < 2) return 0;
        return if (sampled) c.all.extra(streams) else c.greedy.extra(streams);
    }
};

const t = std.testing;

fn feed(c: *RowCost, streams: u32, rows: u64, ms: f64, times: usize) !void {
    for (0..times) |_| try c.observe(t.allocator, streams, rows, ms, false);
}

test "row cost: nothing is charged until three widths span enough rows, and a lone stream is never charged" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    try t.expectEqual(@as(f64, 0), c.extraMs(4, false));
    try feed(&c, 4, 4, 10, 8);
    try feed(&c, 4, 9, 35, 8);
    try t.expectEqual(@as(f64, 0), c.extraMs(4, false)); // two widths always fit a line
    try feed(&c, 4, 5, 15, 8);
    try t.expectApproxEqAbs(@as(f64, 5), c.extraMs(4, false), 1e-9);
    try feed(&c, 1, 2, 10, 8);
    try feed(&c, 1, 8, 50, 8);
    try t.expectEqual(@as(f64, 0), c.extraMs(1, false));
}

test "row cost: the excess is the fit across widths, at the nearest stream count" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    for ([_]u64{ 4, 6, 9 }) |rows| try feed(&c, 4, rows, 2 + 4.25 * @as(f64, @floatFromInt(rows)), 8);
    try t.expectApproxEqAbs(@as(f64, 4.25), c.extraMs(4, false), 1e-9);
    try t.expectApproxEqAbs(@as(f64, 4.25), c.extraMs(3, false), 1e-9);
    try t.expectApproxEqAbs(@as(f64, 4.25), c.extraMs(8, false), 1e-9);
}

test "row cost: a width seen too few times does not count, and a falling fit charges nothing" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    try feed(&c, 2, 2, 10, 8);
    try feed(&c, 2, 4, 12, 8);
    try feed(&c, 2, 9, 500, 3);
    try t.expectEqual(@as(f64, 0), c.extraMs(2, false));
    var falling: RowCost = .{};
    defer falling.deinit(t.allocator);
    for ([_]u64{ 2, 5, 9 }) |rows| try feed(&falling, 2, rows, 20 - @as(f64, @floatFromInt(rows)), 8);
    try t.expectEqual(@as(f64, 0), falling.extraMs(2, false));
}

test "row cost: jitter where the table fits charges nothing" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    for ([_]u64{ 4, 5, 6, 7, 8, 9 }, [_]f64{ 10, 13, 9, 12, 10, 12 }) |rows, ms| try feed(&c, 4, rows, ms, 8);
    try t.expectEqual(@as(f64, 0), c.extraMs(4, false));
    var small: RowCost = .{};
    defer small.deinit(t.allocator);
    for ([_]u64{ 4, 6, 8 }) |rows| try feed(&small, 4, rows, 0.2 * @as(f64, @floatFromInt(rows)), 8);
    try t.expectEqual(@as(f64, 0), small.extraMs(4, false)); // a clean fit, but too small to be the table's error
}

test "row cost: a wild fit stops at the cap, and rows past the end are ignored" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    for ([_]u64{ 4, 6, 8 }) |rows| try feed(&c, 4, rows, 1000 * @as(f64, @floatFromInt(rows)), 8);
    try t.expectEqual(@as(f64, most_ms), c.extraMs(4, false));
    try c.observe(t.allocator, 4, most_rows + 1, 1, false);
}

test "row cost: one slow first round does not outlast the rounds after it" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    try feed(&c, 4, 4, 10, 8);
    try feed(&c, 4, 6, 10, 8);
    try c.observe(t.allocator, 4, 8, 900, false);
    try feed(&c, 4, 8, 10, 80);
    try t.expectEqual(@as(f64, 0), c.extraMs(4, false));
}

test "row cost: widths that stopped running stop counting, so priced-out drafts come back and measure again" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    for ([_]u64{ 4, 6, 9 }) |rows| try feed(&c, 4, rows, 4 * @as(f64, @floatFromInt(rows)), 8);
    try t.expectApproxEqAbs(@as(f64, 4), c.extraMs(4, false), 1e-9);
    try feed(&c, 4, 4, 16, fresh_rounds + 1); // drafts off: only the plain width runs
    try t.expectEqual(@as(f64, 0), c.extraMs(4, false));
    // the wide widths measure from scratch: what they said before the gap is gone
    for ([_]u64{ 6, 9 }) |rows| try feed(&c, 4, rows, 16, 8);
    try t.expectEqual(@as(f64, 0), c.extraMs(4, false));
}

test "row cost: sampled rounds stay out of the greedy fit and count for sampled streams" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    for ([_]u64{ 4, 6, 9 }) |rows| {
        for (0..8) |_| try c.observe(t.allocator, 4, rows, 7 * @as(f64, @floatFromInt(rows)), true);
    }
    try t.expectEqual(@as(f64, 0), c.extraMs(4, false));
    try t.expectApproxEqAbs(@as(f64, 7), c.extraMs(4, true), 1e-9);
}

const depth = @import("depth.zig");
const alloc = @import("allocate.zig");
const Table = @import("table.zig").Table;

/// A shared forward's table as one chip measured it, and a rule pricing by it; rounds that really cost 8.4 ms a row.
fn fed(table: *Table, learn: bool) !depth.Rule {
    for (1..33) |w| try table.put(t.allocator, @intCast(w), 12.5 + 4.1 * @as(f64, @floatFromInt(w - 1)));
    var rule: depth.Rule = .{ .gpa = t.allocator, .k = .{ .learn_rows = learn }, .prior = &.{0.6}, .most_drafts = 4, .family_costs = table, .shared_costs = table, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 32 };
    for ([_]u64{ 4, 6, 8, 10 }) |rows| {
        for (0..8) |_| try rule.observeOverhead(4, rows, 31 + 8.4 * @as(f64, @floatFromInt(rows - 4)), false, 0);
    }
    return rule;
}

test "row cost: rows the table underprices stay out of the overhead, and a draft that loses is not taken" {
    var table: Table = .{};
    defer table.deinit(t.allocator);
    var learned = try fed(&table, true);
    defer learned.deinit();
    var told: Table = .{};
    defer told.deinit(t.allocator);
    var blind = try fed(&told, false);
    defer blind.deinit();
    try t.expectApproxEqAbs(@as(f64, 4.3), learned.row_cost.extraMs(4, false), 1e-6);
    try t.expectEqual(@as(f64, 0), blind.row_cost.extraMs(4, false));
    // the blind rule calls a wide round's excess fixed, which makes the next draft look cheaper
    try t.expect(blind.overheadMs(4, false) > 25);
    try t.expect(learned.overheadMs(4, false) < 8);

    const chance = [_]f64{ 0.6, 0.36 };
    const probs = [_][]const f64{ &chance, &chance, &chance, &chance };
    const ones = [_]u32{ 1, 1, 1, 1 };
    const wide = try alloc.allocate(t.allocator, &ones, &probs, &told, blind.overheadMs(4, false), 32);
    defer t.allocator.free(wide);
    try t.expect(wide[0] > 0);
    const plain = try alloc.allocateDrafts(t.allocator, &ones, &probs, &table, learned.overheadMs(4, false), 32, .{ .row_ms = learned.row_cost.extraMs(4, false) });
    defer t.allocator.free(plain);
    try t.expectEqualSlices(u32, &.{ 0, 0, 0, 0 }, plain);
}

test "row cost: a draft that pays at the learned price is still taken" {
    var table: Table = .{};
    defer table.deinit(t.allocator);
    var learned = try fed(&table, true);
    defer learned.deinit();
    const sure = [_]f64{ 1.0, 1.0 };
    const probs = [_][]const f64{ &sure, &sure };
    const ones = [_]u32{ 1, 1 };
    // two streams on a slow fixed cost: a certain token for one row's price beats the round's rate
    const counts = try alloc.allocateDrafts(t.allocator, &ones, &probs, &table, 100, 32, .{ .row_ms = learned.row_cost.extraMs(4, false) });
    defer t.allocator.free(counts);
    try t.expect(counts[0] > 0 and counts[1] > 0);
}

test "row cost: confidence holds retain host and context prices when the shared row price is learned" {
    var table: Table = .{};
    defer table.deinit(t.allocator);
    for (1..9) |rows| try table.put(t.allocator, @intCast(rows), 10 + @as(f64, @floatFromInt(rows)));
    const holds = [_]depth.HoldCost{.{ .streams = 1, .depth = 1, .ms = 0.1 }};
    const contexts = [_]@import("hold_price.zig").ContextCost{.{ .streams = 1, .rows = 3, .ms = 0.1 }};
    var rule: depth.Rule = .{ .gpa = t.allocator, .k = .{ .learn_rows = true, .hold_costs = &holds, .hold_context_costs = &contexts }, .prior = &.{0.99}, .most_drafts = 1, .family_costs = &table, .shared_costs = &table, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 8 };
    defer rule.deinit();
    var states: [2]?depth.State = .{ null, null };
    defer for (&states) |*state| if (state.*) |*s| s.deinit(t.allocator);
    const whos = [_]depth.Who{
        .{ .state = &states[0], .draft_room = 2, .finished = false, .forced = false, .granted = 1, .gain = 0.99, .prices = &.{0.99}, .row_ms = 2, .hold_context_rows = 3 },
        .{ .state = &states[1], .draft_room = 2, .finished = false, .forced = true },
    };
    var base = @import("hold_price.zig").Base.pending(2);
    base.host_ms = 2;
    var out = [_]i64{ 1, 0 };
    try rule.payHolds(&whos, false, &out, base);
    try t.expectEqualSlices(i64, &.{ 1, 0 }, &out);
    // Synthetic residuals exclude the separately quoted host draw and hold pass.
    for ([_]u64{ 2, 4, 6 }) |rows| try feed(&rule.row_cost, 2, rows, 3.8 * @as(f64, @floatFromInt(rows - 2)), 8);
    out = .{ 1, 0 };
    try rule.payHolds(&whos, false, &out, base);
    try t.expectEqualSlices(i64, &.{ 0, 0 }, &out);
}

test "row cost: a forced row pays the excess too, so a draft beside it is priced from the same base" {
    var table: Table = .{};
    defer table.deinit(t.allocator);
    for (1..33) |w| try table.put(t.allocator, @intCast(w), 10 + @as(f64, @floatFromInt(w)));
    const chance = [_]f64{0.5};
    const none = [_]f64{};
    const probs = [_][]const f64{ &none, &chance };
    // five forced rows at 8 ms past the table: the round is dear, so the half-likely row pays only if the base says so
    const forced = [_]u32{ 6, 1 };
    const counts = try alloc.allocateDrafts(t.allocator, &forced, &probs, &table, 0, 32, .{ .row_ms = 8 });
    defer t.allocator.free(counts);
    // base 7 / (17 + 40) is 0.123 tokens a ms, and the draft 7.5 / (18 + 48) is 0.114: it stays out
    try t.expectEqual(@as(u32, 0), counts[1]);
    const free = try alloc.allocateDrafts(t.allocator, &forced, &probs, &table, 0, 32, .{});
    defer t.allocator.free(free);
    try t.expectEqual(@as(u32, 1), free[1]);
}

test "row cost: a width seen now and then follows today's cost, not the one it learned long ago" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    for ([_]u64{ 4, 6, 8, 10 }) |rows| try feed(&c, 4, rows, 8 * @as(f64, @floatFromInt(rows)), 8);
    try t.expect(c.extraMs(4, false) > 7);
    // contention gone: only a wide round now and then among the plain ones, and rows cost the table's price again
    for (0..6) |_| {
        try feed(&c, 4, 4, 32, sparse_rounds + 1);
        for ([_]u64{ 6, 8, 10 }) |rows| try c.observe(t.allocator, 4, rows, 32, false);
    }
    try t.expectEqual(@as(f64, 0), c.extraMs(4, false));
}

test "row cost: a sampled stretch ages the greedy fit too" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    for ([_]u64{ 4, 6, 9 }) |rows| try feed(&c, 4, rows, 4 * @as(f64, @floatFromInt(rows)), 8);
    for (0..fresh_rounds + 1) |_| try c.observe(t.allocator, 4, 4, 16, true);
    try t.expectEqual(@as(f64, 0), c.extraMs(4, false));
}

/// Rounds costing `excess` a row past the table, priced by the rule they teach: tokens a ms, first charge's round.
fn closedLoop(excess: f64, learn: bool, rounds: usize, chain: [4]f64) !struct { rate: f64, first: ?usize, ms: f64 } {
    var table: Table = .{};
    defer table.deinit(t.allocator);
    for (1..17) |w| try table.put(t.allocator, @intCast(w), 12.5 + 4.1 * @as(f64, @floatFromInt(w - 1)));
    var rule: depth.Rule = .{ .gpa = t.allocator, .k = .{ .learn_rows = learn }, .prior = &.{0.6}, .most_drafts = 4, .family_costs = &table, .shared_costs = &table, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 16 };
    defer rule.deinit();
    var random = std.Random.DefaultPrng.init(4111);
    const next = random.random();
    const ones = [_]u32{ 1, 1, 1, 1 };
    var chances: [4][4]f64 = undefined;
    var tokens: f64 = 0;
    var spent: f64 = 0;
    var first: ?usize = null;
    for (0..rounds) |i| {
        var probs: [4][]const f64 = undefined;
        for (&chances, &probs) |*held, *p| {
            const scale = 0.5 + 0.65 * next.float(f64);
            for (held, chain) |*x, base| x.* = @min(1.0, base * scale);
            p.* = held;
        }
        const row_ms = rule.row_cost.extraMs(4, false);
        if (first == null and row_ms > 0) first = i;
        const counts = try alloc.allocateDrafts(t.allocator, &ones, &probs, &table, rule.overheadMs(4, false), rule.row_cost.capRows(4, false, 16), .{ .row_ms = row_ms });
        defer t.allocator.free(counts);
        var rows: u64 = 4;
        var landed: f64 = 4;
        for (counts, probs) |n, p| {
            rows += n;
            for (p[0..n]) |x| landed += x;
        }
        const ms = 31 + (4.1 + excess) * @as(f64, @floatFromInt(rows - 4)) + 3 * (next.float(f64) - 0.5);
        tokens += landed;
        spent += ms;
        try rule.observeOverhead(4, rows, ms, false, 0);
    }
    return .{ .rate = tokens / spent, .first = first, .ms = spent / @as(f64, @floatFromInt(rounds)) };
}

const usual = [_]f64{ 0.85, 0.6, 0.45, 0.3 };
const certain = [_]f64{ 1.0, 1.0, 1.0, 1.0 }; // every stream wants the cap: rounds never vary by themselves

test "row cost: the learner starts within a few rounds however dear the rows, and beats the rule it replaces" {
    for ([_][4]f64{ usual, certain }) |chain| for ([_]f64{ 4.3, 5, 6, 8, 12 }) |excess| {
        const learned = try closedLoop(excess, true, 2000, chain);
        const blind = try closedLoop(excess, false, 2000, chain);
        errdefer std.debug.print("excess {d}: learned {any}, blind {any}\n", .{ excess, learned, blind });
        try t.expect(learned.first != null and learned.first.? < 40);
        try t.expect(learned.rate > blind.rate);
    };
}

test "row cost: while learning, rounds visit each width, then stop being capped" {
    var c: RowCost = .{};
    defer c.deinit(t.allocator);
    try t.expectEqual(@as(u64, 16), c.capRows(4, false, 16)); // nothing seen: nothing known to measure yet
    try t.expectEqual(@as(u64, 16), c.capRows(1, false, 16)); // a lone stream is never capped
    var seen: [17]u32 = @splat(0);
    for (0..24) |_| {
        const rows = c.capRows(4, false, 16);
        seen[rows] += 1;
        try c.observe(t.allocator, 4, rows, 20, false);
    }
    try t.expect(seen[4] >= enough and seen[6] >= enough and seen[8] >= enough);
    try t.expectEqual(@as(u64, 16), c.capRows(4, false, 16)); // every width counted, no fit: the table fits
    try feed(&c, 4, 4, 20, fresh_rounds + 1);
    try t.expect(c.capRows(4, false, 16) < 16); // the wide widths went stale: measure them again
}

test "row cost: where the table fits, the learner charges nothing and matches the rule it replaces" {
    const learned = try closedLoop(0, true, 2000, usual);
    const blind = try closedLoop(0, false, 2000, usual);
    try t.expect(learned.first == null);
    try t.expect(learned.rate > blind.rate * 0.97);
}

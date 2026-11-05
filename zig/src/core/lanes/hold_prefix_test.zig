//! Synthetic consumer regressions for full-round hold pricing, retained prefixes and funded recovery.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const depth = @import("depth.zig");
const win = @import("windows.zig");
const shared_drafts = @import("shared_drafts.zig");

const gpa = std.testing.allocator;

/// A pointer-stable engine over the existing fake target: forward ms by rows 1..8 and a mutable hold-cost table.
pub const Fixture = struct {
    /// `forward`: ms by rows 1..8, or null for a target with no timed forward costs (nothing is read of them).
    pub const Setup = struct {
        drafts: u32,
        forward: ?[8]f64,
        holds: []const depth.HoldCost,
        prior: []const f64,
        contexts: []const @import("hold_price.zig").ContextCost = &.{},
    };

    holds: [3]depth.HoldCost = undefined,
    cfg: Config = undefined,
    target: fake.Fake = undefined,
    clock: fake.FixedClock = .{},
    engine: Engine = undefined,

    pub fn init(f: *Fixture, s: Setup) !void {
        @memcpy(f.holds[0..s.holds.len], s.holds);
        var costs: [8]Cost = undefined;
        if (s.forward) |forward| {
            for (&costs, forward, 1..) |*c, ms, rows| c.* = .{ .width = @intCast(rows), .ms = ms };
        }
        f.cfg = try Config.init(gpa, .{
            .exact_width = 8,
            .mtp = true,
            .speculate = true,
            .allocate_shared_drafts = true,
            .drafts = s.drafts,
            .draft_probabilities = true,
            .granted_depth = true,
            .plain_guard = true,
            .draft_prior = s.prior,
            .window_costs = if (s.forward != null) &costs else &.{},
            .hold_costs = f.holds[0..s.holds.len],
            .hold_context_costs = s.contexts,
            .batch_rows = 8,
        }, 8, 7);
        f.target = .{ .gpa = gpa };
        f.clock = .{};
        f.engine = Engine.init(gpa, &f.cfg, f.target.backend(), f.clock.clock());
    }

    pub fn deinit(f: *Fixture) void {
        f.engine.deinit();
        f.target.deinit();
        f.cfg.deinit(gpa);
    }

    /// The actual consumer: the rows the head at plan 0 may propose next round.
    fn budget(f: *Fixture, plans: []const win.Plan) !i64 {
        const heads = [_]usize{0};
        var out: [1]i64 = undefined;
        try shared_drafts.budgets(&f.engine, plans, &heads, &out);
        return out[0];
    }

    pub fn expectBudget(f: *Fixture, plans: []const win.Plan, want: i64) !void {
        try std.testing.expectEqual(want, try f.budget(plans));
    }

    /// The actual consumer for several head streams: plan indices `heads`, budgets in `out`.
    pub fn budgetsFor(f: *Fixture, plans: []const win.Plan, heads: []const usize, out: []i64) !void {
        try shared_drafts.budgets(&f.engine, plans, heads, out);
    }

    /// Isolate the hold consumer with fresh reservations, without re-running row allocation.
    fn freshHolds(f: *Fixture, prices: []const []const f64, row_ms: []const f64, out: []i64) !void {
        var states: [8]?depth.State = @splat(null);
        defer for (states[0..prices.len]) |*state| {
            if (state.*) |*s| s.deinit(gpa);
        };
        var whos: [8]depth.Who = undefined;
        var base = @import("hold_price.zig").Base.pending(prices.len);
        for (prices, row_ms, out, whos[0..prices.len], states[0..prices.len]) |chances, ms, *wanted, *who, *state| {
            var gain: f64 = 0;
            for (chances) |chance| gain += chance;
            wanted.* = @intCast(chances.len);
            who.* = .{
                .state = state,
                .draft_room = 64,
                .finished = false,
                .forced = false,
                .granted = @intCast(chances.len),
                .gain = gain,
                .prices = chances,
                .sampled = ms > 0,
                .row_ms = ms,
            };
            base.host_ms += ms;
        }
        try f.engine.rule.payHolds(whos[0..prices.len], true, out, base);
    }
};

fn rowsPlus(base: f64) [8]f64 {
    var out: [8]f64 = undefined;
    for (&out, 1..) |*o, rows| o.* = base + @as(f64, @floatFromInt(rows));
    return out;
}

fn sampling(on: bool, seed: u64) ?@import("sampling.zig").Sampling {
    return if (on) .{ .seed = seed, .temperature = 1 } else null;
}

test "a profitable observed head prefix survives when the full reserved hold cannot pay, and dropping it needs a real loss" {
    // F(rows) = 9 + rows; one hold .1 ms, two 40 ms; p(first) observed 1, p(second | first) observed 0.
    for ([_]bool{ false, true }) |sampled| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 2,
            .forward = rowsPlus(9),
            .holds = &.{ .{ .streams = 1, .depth = 1, .ms = 0.1 }, .{ .streams = 1, .depth = 2, .ms = 40 } },
            .prior = &.{ 0.9, 0.9 },
        });
        defer f.deinit();
        try f.engine.rule.observeOverhead(1, 1, 10, sampled, 0);
        var s = try sm.Stream.init(gpa, .{ .id = "prefix", .prompt = &.{ 2, 7, 13 }, .max_new = 64, .sampling = sampling(sampled, 1234) });
        defer s.deinit(gpa);
        s.granted = 2;
        try s.draft_chances.appendSlice(gpa, &.{ 0.95, 0.9025 });
        try f.engine.rule.observeDepth(win.who(&s), 2, 1);
        const plans = [_]win.Plan{.{ .stream = &s, .position = 0, .kind = .none }};
        try f.expectBudget(&plans, 1); // raw chances reserve two; only the first draft's hold pays
        f.holds[0].ms = 1000; // neither prefix pays now; retaining one draft would also be wrong
        try f.expectBudget(&plans, 0);
        f.holds[0].ms = 0.1;
        try f.expectBudget(&plans, 1);
        f.holds[1].ms = 0.1; // a cheap full hold is still kept whole: the shorter prefix is only a fallback
        try f.expectBudget(&plans, 2);
    }
}

test "a futile exploration never fires, and a real recovery is available the moment its hold becomes affordable" {
    // Perfect acceptance cannot pay at 1000 ms; an impossible round must preserve the due probe.
    for ([_]bool{ false, true }) |sampled| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 1,
            .forward = rowsPlus(9),
            .holds = &.{.{ .streams = 1, .depth = 1, .ms = 1000 }},
            .prior = &.{0.9},
        });
        defer f.deinit();
        try f.engine.rule.observeOverhead(1, 1, 10, sampled, 0);
        var s = try sm.Stream.init(gpa, .{ .id = "explore", .prompt = &.{ 2, 7, 13 }, .max_new = 64, .sampling = sampling(sampled, 77) });
        defer s.deinit(gpa);
        s.granted = 1;
        try s.draft_chances.appendSlice(gpa, &.{0.99});
        try f.engine.rule.observeDepth(win.who(&s), 1, 0);
        const plans = [_]win.Plan{.{ .stream = &s, .position = 0, .kind = .none }};
        for (0..40) |_| try f.expectBudget(&plans, 0); // well past the 8 / 16 round probe cadence
        f.holds[0].ms = 0.1;
        try f.expectBudget(&plans, 1); // the saved recovery proposes at once: impossible rounds did not spend or reset it
        try f.engine.rule.observeDepth(win.who(&s), 1, 1); // the proposal landed
        try f.expectBudget(&plans, 1); // and the recovered hold stays profitable
    }
}

test "a head's hold is priced against the whole cohort's rows: forced companion rows are base, not free" {
    // Companion rows make the marginal hold lose even though the isolated head and its verify row would pay.
    for ([_]bool{ false, true }) |sampled| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 1,
            .forward = .{ 10, 10, 10, 10, 10, 11, 11, 11 },
            .holds = &.{.{ .streams = 1, .depth = 1, .ms = 0 }},
            .prior = &.{0.9},
        });
        defer f.deinit();
        try f.engine.rule.observeOverhead(2, 5, 10, sampled, 0); // two streams, five rows, no fixed overhead
        var head = try sm.Stream.init(gpa, .{ .id = "head", .prompt = &.{ 2, 7, 13 }, .max_new = 64, .sampling = sampling(sampled, 5) });
        defer head.deinit(gpa);
        var forced = try sm.Stream.init(gpa, .{ .id = "forced", .prompt = &.{ 3, 1, 4 }, .max_new = 64, .sampling = sampling(sampled, 6) });
        defer forced.deinit(gpa);
        head.granted = 1;
        try head.draft_chances.appendSlice(gpa, &.{0.9});
        try forced.force.appendNTimes(gpa, 11, 3); // three forced rows ride in the verify beside the pending ones
        const plans = [_]win.Plan{
            .{ .stream = &head, .position = 0, .kind = .none },
            .{ .stream = &forced, .position = 0, .kind = .forced },
        };
        try f.expectBudget(&plans, 1); // a free hold: the extra row pays
        f.holds[0].ms = 1;
        try f.expectBudget(&plans, 0); // the same row with a 1 ms hold is a loss against five certain tokens
        f.holds[0].ms = 0;
        try f.expectBudget(&plans, 1);
    }
}

test "a historical-zero probe survives a fresh zero-row reservation until its first funded trial" {
    for ([_]bool{ false, true }) |sampled| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 1,
            .forward = rowsPlus(9),
            .holds = &.{.{ .streams = 1, .depth = 1, .ms = 0.1 }},
            .prior = &.{0.9},
        });
        defer f.deinit();
        f.engine.rule.k.depth_probe_every = 4;
        f.engine.rule.k.plain_wait_most = 16;
        try f.engine.rule.observeOverhead(2, 8, 17, sampled, 0);
        var head = try sm.Stream.init(gpa, .{ .id = "zero-grant", .prompt = &.{ 2, 7, 13 }, .max_new = 64, .sampling = sampling(sampled, 5) });
        defer head.deinit(gpa);
        var forced = try sm.Stream.init(gpa, .{ .id = "full-reservation", .prompt = &.{ 3, 1, 4 }, .max_new = 64, .sampling = sampling(sampled, 6) });
        defer forced.deinit(gpa);
        head.granted = 0;
        try head.draft_chances.appendSlice(gpa, &.{0.99});
        try forced.force.appendNTimes(gpa, 11, 6); // two pending rows plus these consume all eight rows
        const plans = [_]win.Plan{
            .{ .stream = &head, .position = 0, .kind = .none },
            .{ .stream = &forced, .position = 0, .kind = .forced },
        };
        for (0..7) |_| try f.expectBudget(&plans, 0); // the due fourth round never had a funded trial
        forced.force.clearRetainingCapacity();
        try f.expectBudget(&plans, 1); // its first positive reservation must not wait for a consumed probe
    }
}

test "a historical-zero probe fires on its fourth shared round with no timed forward costs" {
    // No forward table: nothing prices the rows, so no round may starve the probe clock by waiting for one.
    for ([_]bool{ false, true }) |sampled| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 1,
            .forward = null,
            .holds = &.{.{ .streams = 1, .depth = 1, .ms = 0.1 }},
            .prior = &.{0.9},
        });
        defer f.deinit();
        f.engine.rule.k.depth_probe_every = 4;
        f.engine.rule.k.plain_wait_most = 16;
        var s = try sm.Stream.init(gpa, .{ .id = "untimed-zero", .prompt = &.{ 2, 7, 13 }, .max_new = 64, .sampling = sampling(sampled, 9) });
        defer s.deinit(gpa);
        s.granted = 0;
        try s.draft_chances.appendSlice(gpa, &.{0.99});
        const plans = [_]win.Plan{.{ .stream = &s, .position = 0, .kind = .none }};
        for (0..3) |_| try f.expectBudget(&plans, 0);
        try f.expectBudget(&plans, 1);
    }
}

test "a due exploration is feasible when a deeper padded hold is cheaper than a shallower one" {
    // Nonmonotonic costs make padded [3, 1] cheaper than [2, 2]; the bound must retain that probe.
    for ([_]bool{ false, true }) |sampled| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 3,
            .forward = @as([8]f64, @splat(10)),
            .holds = &.{
                .{ .streams = 1, .depth = 1, .ms = 11 },
                .{ .streams = 2, .depth = 1, .ms = 12 },
                .{ .streams = 2, .depth = 3, .ms = 9 },
            },
            .prior = &.{ 0.9, 0.9, 0.9 },
        });
        defer f.deinit();
        f.engine.rule.k.depth_probe_every = 2;
        try f.engine.rule.observeOverhead(4, 4, 10, sampled, 0);
        // Real remaining-token room: draftRoom() is max_new here, so capacities are 3 and 1.
        const max_new = [4]u32{ 4, 2, 64, 64 };
        var streams: [4]sm.Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(gpa);
        for (&streams, max_new, 0..) |*s, room, i| {
            s.* = try sm.Stream.init(gpa, .{ .id = "cohort", .prompt = &.{ 2, 7, 13 }, .max_new = room, .drafts = i < 2, .sampling = sampling(sampled, 40 + i) });
            made += 1;
        }
        streams[0].granted = 3;
        streams[1].granted = 1;
        try streams[0].draft_chances.appendSlice(gpa, &.{ 0.99, 0.98, 0.97 });
        try streams[1].draft_chances.appendSlice(gpa, &.{0.99});
        // each head's actual first draft missed: its observed gain is zero, so the normal hold stands aside
        try f.engine.rule.observeDepth(win.who(&streams[0]), 3, 0);
        try f.engine.rule.observeDepth(win.who(&streams[1]), 1, 0);
        var plans: [4]win.Plan = undefined;
        for (&plans, &streams) |*p, *s| p.* = .{ .stream = s, .position = 0, .kind = .none };
        const heads = [_]usize{ 0, 1 };
        var out: [2]i64 = undefined;
        try f.budgetsFor(&plans, &heads, &out);
        try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &out);
        // Both due intervals must fund a positive probe, then stand aside after that probe misses.
        for (0..2) |_| {
            var probed = false;
            for (0..16) |_| {
                try f.budgetsFor(&plans, &heads, &out);
                if (out[0] > 0 or out[1] > 0) {
                    probed = true;
                    break;
                }
            }
            try std.testing.expect(probed);
            for (out, streams[0..2]) |offered, *s| {
                if (offered > 0) try f.engine.rule.observeDepth(win.who(s), offered, 0);
            }
        }
    }
}

test "a confidence hold pays for every physical pass, not one impossible padded cohort" {
    for ([_]bool{ false, true }) |sampled| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 3,
            .forward = @as([8]f64, @splat(10)),
            .holds = &.{
                .{ .streams = 1, .depth = 3, .ms = 20 },
                .{ .streams = 2, .depth = 3, .ms = 20 },
            },
            .prior = &.{ 0.9, 0.9, 0.9 },
        });
        defer f.deinit();
        f.engine.rule.k.hold_rows = 4;
        try f.engine.rule.observeOverhead(2, 2, 10, sampled, 0);
        var streams: [2]sm.Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(gpa);
        for (&streams, 0..) |*s, i| {
            s.* = try sm.Stream.init(gpa, .{
                .id = "physical-hold",
                .prompt = &.{ 2, 7, 13 },
                .max_new = 64,
                .sampling = sampling(sampled, 70 + i),
            });
            made += 1;
            s.granted = 3;
            try s.draft_chances.appendSlice(gpa, &.{ 0.99, 0.98, 0.97 });
        }
        const plans = [_]win.Plan{
            .{ .stream = &streams[0], .position = 0, .kind = .none },
            .{ .stream = &streams[1], .position = 0, .kind = .none },
        };
        var out: [2]i64 = undefined;
        // Two three-deep blocks split at four physical rows: 7.88 / (10 + 20 + 20) loses to 2 / 10.
        try f.budgetsFor(&plans, &.{ 0, 1 }, &out);
        try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &out);
        // An actually unbounded provider needs only one pass: 7.88 / (10 + 20) genuinely wins.
        f.engine.rule.k.hold_rows = 0;
        try f.budgetsFor(&plans, &.{ 0, 1 }, &out);
        try std.testing.expectEqualSlices(i64, &.{ 3, 3 }, &out);
    }
}

test "observed row prices shorten current and reserved sampled windows without changing zero or unknown routes" {
    var f: Fixture = .{};
    try f.init(.{ .drafts = 3, .forward = @as([8]f64, @splat(10)), .holds = &.{.{ .streams = 1, .depth = 1, .ms = 1 }}, .prior = &.{ 0.9, 0.5, 0.2 } });
    defer f.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "host-prefix", .prompt = &.{ 2, 7, 13 }, .max_new = 64, .sampling = sampling(true, 8) });
    defer s.deinit(gpa);
    const tokens = [_]u32{ 11, 12, 13 };
    for ([_]?f64{ null, 0, 5, null, 5, 0 }) |quote| {
        s.verify_row_ms = quote;
        var plans = [_]win.Plan{.{ .stream = &s, .position = 0, .kind = .head, .tokens = &tokens }};
        try win.allocate(&f.engine, &plans);
        const wanted: usize = if ((quote orelse 0) > 0) 1 else 3;
        try std.testing.expectEqual(wanted, plans[0].count());
        // Current allocation retained all the original chances; reserve the same offer independently.
        s.granted = 3;
        s.grant_short = false;
        try f.expectBudget(&plans, @intCast(wanted));
    }
}

test "forced host work belongs to the baseline and disappears with a paused or finished companion" {
    var f: Fixture = .{};
    try f.init(.{ .drafts = 1, .forward = @as([8]f64, @splat(10)), .holds = &.{.{ .streams = 1, .depth = 1, .ms = 2 }}, .prior = &.{0.9} });
    defer f.deinit();
    var head = try sm.Stream.init(gpa, .{ .id = "head", .prompt = &.{ 2, 7, 13 }, .max_new = 64 });
    defer head.deinit(gpa);
    var forced = try sm.Stream.init(gpa, .{ .id = "forced", .prompt = &.{ 3, 1, 4 }, .max_new = 64, .sampling = sampling(true, 9) });
    defer forced.deinit(gpa);
    head.granted = 1;
    try head.draft_chances.append(gpa, 0.9);
    try forced.force.appendNTimes(gpa, 11, 3);
    const plans = [_]win.Plan{ .{ .stream = &head, .position = 0, .kind = .none }, .{ .stream = &forced, .position = 0, .kind = .forced } };
    try f.expectBudget(&plans, 0); // 5.9 / 12 loses to 5 / 10.
    forced.verify_row_ms = 5;
    try f.expectBudget(&plans, 1); // all four forced/pending draws: 5.9 / 32 beats 5 / 30.
    head.verify_row_ms = 20;
    forced.paused = true;
    try f.expectBudget(&plans, 1); // only the head remains: 1.9 / 52 beats 1 / 30.
    forced.paused = false;
    forced.finished = true;
    try f.expectBudget(&plans, 1);
}

test "a live host quote invalidates futile recovery while an unknown route preserves the funded trial" {
    var f: Fixture = .{};
    try f.init(.{ .drafts = 1, .forward = @as([8]f64, @splat(10)), .holds = &.{.{ .streams = 1, .depth = 1, .ms = 1 }}, .prior = &.{0.9} });
    defer f.deinit();
    f.engine.rule.k.depth_probe_every = 1;
    var head = try sm.Stream.init(gpa, .{ .id = "recover", .prompt = &.{ 2, 7, 13 }, .max_new = 64, .sampling = sampling(true, 9) });
    defer head.deinit(gpa);
    var forced = try sm.Stream.init(gpa, .{ .id = "companion", .prompt = &.{ 3, 1, 4 }, .max_new = 64 });
    defer forced.deinit(gpa);
    head.granted = 0;
    head.verify_row_ms = 20;
    try forced.force.append(gpa, 11);
    var whos = [_]depth.Who{ win.who(&head), win.who(&forced) };
    var out: [2]i64 = undefined;
    const probe = win.probe(&f.engine);
    // Even a perfect first row loses: 3 / 51 < 2 / 30. These intervals cannot spend the saved trial.
    for (0..8) |_| {
        try f.engine.rule.budgets(&whos, probe, true, &out);
        try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &out);
    }
    head.verify_row_ms = null;
    whos[0] = win.who(&head);
    out = .{ 0, 0 };
    try f.engine.rule.payHolds(&whos, true, &out, @import("hold_price.zig").Base.pending(2));
    try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &out); // no fresh rows means no trial, even with an optimistic quote
    try f.engine.rule.budgets(&whos, probe, true, &out);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, &out);
    // A costly sampled head cannot make a different eligible, unobserved route's probe look futile.
    head.verify_row_ms = 20;
    forced.force.clearRetainingCapacity();
    forced.granted = 0;
    whos = .{ win.who(&head), win.who(&forced) };
    try f.engine.rule.budgets(&whos, probe, true, &out);
    try std.testing.expectEqual(@as(i64, 1), out[1]);
}

test "mixed host routes share physical holds without charging cheap drafts for sampled rows in either order" {
    for ([_]bool{ false, true }) |swapped| {
        var f: Fixture = .{};
        try f.init(.{ .drafts = 2, .forward = @as([8]f64, @splat(10)), .holds = &.{ .{ .streams = 1, .depth = 1, .ms = 2 }, .{ .streams = 2, .depth = 1, .ms = 4 } }, .prior = &.{ 0.9, 0.9 } });
        defer f.deinit();
        var greedy = try sm.Stream.init(gpa, .{ .id = "greedy", .prompt = &.{ 2, 7, 13 }, .max_new = 64 });
        defer greedy.deinit(gpa);
        var sampled = try sm.Stream.init(gpa, .{ .id = "sampled", .prompt = &.{ 3, 1, 4 }, .max_new = 64, .sampling = sampling(true, 17) });
        defer sampled.deinit(gpa);
        greedy.granted = 2;
        sampled.granted = 1;
        sampled.grant_short = true;
        try greedy.draft_chances.appendSlice(gpa, &.{ 0.99, 0.98 });
        try sampled.draft_chances.append(gpa, 0.9);
        const g: usize = @intFromBool(swapped);
        const s = 1 - g;
        var plans: [2]win.Plan = undefined;
        plans[g] = .{ .stream = &greedy, .position = 0, .kind = .none };
        plans[s] = .{ .stream = &sampled, .position = 0, .kind = .none };
        var out: [2]i64 = undefined;
        for ([_]f64{ 0, 1, 0 }) |row_ms| {
            sampled.verify_row_ms = row_ms;
            try f.budgetsFor(&plans, &.{ 0, 1 }, &out);
            try std.testing.expectEqual(@as(i64, 2), out[g]);
            // At 1 ms/draw both heads reserve rows, but 4.87 / 16 loses to the greedy hold's 3.97 / 13.
            try std.testing.expectEqual(@as(i64, if (row_ms == 0) 1 else 0), out[s]);
        }
    }
}

test "a reserved sampled row can pay for its draw but not for that draw together with the hold" {
    var f: Fixture = .{};
    try f.init(.{ .drafts = 1, .forward = @as([8]f64, @splat(10)), .holds = &.{.{ .streams = 1, .depth = 1, .ms = 9 }}, .prior = &.{0.9} });
    defer f.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "hold-plus-host", .prompt = &.{ 2, 7, 13 }, .max_new = 64, .sampling = sampling(true, 11) });
    defer s.deinit(gpa);
    s.verify_row_ms = 5;
    const tokens = [_]u32{11};
    var plans = [_]win.Plan{.{ .stream = &s, .position = 0, .kind = .head, .tokens = &tokens }};
    try win.allocate(&f.engine, &plans);
    try std.testing.expectEqual(@as(usize, 1), plans[0].count()); // 1.9 / 20 beats 1 / 15.
    try f.expectBudget(&plans, 0); // 1.9 / 29 loses to 1 / 15 once the full hold also gets paid.
    f.holds[0].ms = 1;
    try f.expectBudget(&plans, 1);
}

test "mixed hold prices re-sweep after a first improvement exposes a better cheap cohort in either order" {
    for ([_]bool{ false, true }) |reversed| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 1,
            .forward = @as([8]f64, @splat(10)),
            .holds = &.{
                .{ .streams = 1, .depth = 1, .ms = 0 },
                .{ .streams = 2, .depth = 1, .ms = 0 },
                .{ .streams = 3, .depth = 1, .ms = 20 },
            },
            .prior = &.{0.9},
        });
        defer f.deinit();
        const chances = [_][]const f64{ &.{0.99}, &.{0.9}, &.{0.8}, &.{0.8} };
        const quotes = [_]f64{ 80, 2, 0, 0 };
        var prices: [4][]const f64 = undefined;
        var row_ms: [4]f64 = undefined;
        for (&prices, &row_ms, 0..) |*p, *ms, i| {
            const source = if (reversed) 3 - i else i;
            p.* = chances[source];
            ms.* = quotes[source];
        }
        var out: [4]i64 = undefined;
        try f.freshHolds(&prices, &row_ms, &out);
        // Repricing 4/92 first keeps 5.7/94; a second cost-aware sweep must find both cheap rows at 5.6/92.
        for (row_ms, out) |ms, wanted|
            try std.testing.expectEqual(@as(i64, if (ms == 0) 1 else 0), wanted);
    }
}

test "mixed hold prices retain the strict profitable boundary in either stream order" {
    for ([_]bool{ false, true }) |swapped| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 1,
            .forward = @as([8]f64, @splat(10)),
            .holds = &.{ .{ .streams = 1, .depth = 1, .ms = 10 }, .{ .streams = 2, .depth = 1, .ms = 20 } },
            .prior = &.{0.9},
        });
        defer f.deinit();
        const prices = if (swapped) [2][]const f64{ &.{0.5}, &.{1} } else [2][]const f64{ &.{1}, &.{0.5} };
        const row_ms = if (swapped) [2]f64{ 0, 30 } else [2]f64{ 30, 0 };
        var out: [2]i64 = undefined;
        for ([_]f64{ 9, 10, 11, 9 }) |hold_ms| {
            f.holds[0].ms = hold_ms;
            try f.freshHolds(&prices, &row_ms, &out);
            // The cheap hold ties exactly at 2.5/50 == 2/40; only a strict win survives.
            for (row_ms, out) |ms, wanted|
                try std.testing.expectEqual(@as(i64, if (ms == 0 and hold_ms < 10) 1 else 0), wanted);
        }
    }
}

test "mixed hold reordering preserves an incumbent that is no longer a prefix and prices actual splits" {
    for ([_]bool{ false, true }) |swapped| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 2,
            .forward = @as([8]f64, @splat(10)),
            .holds = &.{
                .{ .streams = 1, .depth = 1, .ms = 1 },
                .{ .streams = 1, .depth = 2, .ms = 8 },
                .{ .streams = 2, .depth = 2, .ms = 16 },
            },
            .prior = &.{ 0.9, 0.9 },
        });
        defer f.deinit();
        const prices = if (swapped) [2][]const f64{ &.{ 0.8, 0.8 }, &.{0.9} } else [2][]const f64{ &.{0.9}, &.{ 0.8, 0.8 } };
        const row_ms = if (swapped) [2]f64{ 0, 2 } else [2]f64{ 2, 0 };
        var out: [2]i64 = undefined;
        try f.freshHolds(&prices, &row_ms, &out);
        // Reordering cannot replace shallow 2.9/15 with deep 3.6/20 or the padded pair's 4.5/30.
        for (row_ms, out) |ms, wanted|
            try std.testing.expectEqual(@as(i64, if (ms > 0) 1 else 0), wanted);
        f.engine.rule.k.hold_rows = 2;
        try f.freshHolds(&prices, &row_ms, &out);
        // Splitting into real one-block passes costs 1+8 instead of 16: 4.5/23 wins.
        for (prices, out) |chances, wanted|
            try std.testing.expectEqual(@as(i64, @intCast(chances.len)), wanted);
    }
}

test "zero and uniform hold quotes keep the original stream-index tie" {
    var f: Fixture = .{};
    try f.init(.{
        .drafts = 1,
        .forward = @as([8]f64, @splat(10)),
        .holds = &.{ .{ .streams = 1, .depth = 1, .ms = 0.1 }, .{ .streams = 2, .depth = 1, .ms = 20 } },
        .prior = &.{0.75},
    });
    defer f.deinit();
    var out: [2]i64 = undefined;
    for ([_]f64{ 0, 2, 0 }) |row_ms| {
        try f.freshHolds(&.{ &.{0.75}, &.{0.75} }, &.{ row_ms, row_ms }, &out);
        try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, &out);
    }
}

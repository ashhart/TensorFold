//! Reserved head holds price shorter prefixes, recovery trials, and companion rows against the whole round.
//! Every timing is SYNTHETIC and makes a shared_drafts.budgets decision unambiguous; none is device data.
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
const Fixture = struct {
    /// `forward`: ms by rows 1..8, or null for a target with no timed forward costs (nothing is read of them).
    const Setup = struct { drafts: u32, forward: ?[8]f64, holds: []const depth.HoldCost, prior: []const f64 };

    holds: [3]depth.HoldCost = undefined,
    cfg: Config = undefined,
    target: fake.Fake = undefined,
    clock: fake.FixedClock = .{},
    engine: Engine = undefined,

    fn init(f: *Fixture, s: Setup) !void {
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
            .batch_rows = 8,
        }, 8, 7);
        f.target = .{ .gpa = gpa };
        f.clock = .{};
        f.engine = Engine.init(gpa, &f.cfg, f.target.backend(), f.clock.clock());
    }

    fn deinit(f: *Fixture) void {
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

    fn expectBudget(f: *Fixture, plans: []const win.Plan, want: i64) !void {
        try std.testing.expectEqual(want, try f.budget(plans));
    }

    /// The actual consumer for several head streams: plan indices `heads`, budgets in `out`.
    fn budgetsFor(f: *Fixture, plans: []const win.Plan, heads: []const usize, out: []i64) !void {
        try shared_drafts.budgets(&f.engine, plans, heads, out);
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
    // F(rows) = 9 + rows; one draft at most. The first draft was observed to miss, so a hold pays only on exploration.
    // At 1000 ms even perfect acceptance cannot pay, so no round may spend its probe on it.
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
    // F(1..5) = 10 ms, F(6..8) = 11 ms. The head's draft row alone pays (5.9 / 11 > 5 / 10); adding its 1 ms hold does
    // not (5.9 / 12 < 5 / 10). Priced against the head alone the hold looks profitable (1.9 / 11 > 1 / 10).
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
    // F(rows) = 10 ms and no fixed overhead; four streams give four pending rows, two of them heads of
    // capacity 3 and 1. The hold pass costs 11 ms (1 block, 1 deep), 12 ms (2 blocks, 1 deep) and
    // 9 ms (2 blocks, 3 deep): the padded future [3, 1] shape is cheaper than balanced [2, 2].
    // Perfect acceptance pays: 8 / (10 + 9) > 4 / 10.
    for ([_]bool{ false, true }) |sampled| {
        var f: Fixture = .{};
        try f.init(.{
            .drafts = 3,
            .forward = [_]f64{10} ** 8,
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
        // Two due intervals: each must offer a real positive probe; both heads missed, so the next stands aside again.
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
            .forward = [_]f64{10} ** 8,
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

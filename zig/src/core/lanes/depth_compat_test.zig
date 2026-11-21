//! Unpriced depth decisions retain the shared-budget and forced-round behavior of main.
const std = @import("std");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const t = std.testing;
const gpa = t.allocator;
const cheap_hold = [_]depth.HoldCost{.{ .streams = 1, .depth = 1, .ms = 0.01 }};

fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

fn ruleFor(costs: *const Table, priced: bool) depth.Rule {
    return .{
        .gpa = gpa,
        .k = .{ .hold_costs = if (priced) &cheap_hold else &.{}, .depth_probe_every = 8 },
        .prior = &.{0.5},
        .most_drafts = 3,
        .family_costs = costs,
        .shared_costs = costs,
        .mtp_step_ms = 0,
        .plain_guard = true,
        .node_probabilities = false,
        .batch_rows = 5,
    };
}

fn mixed(rule: *depth.Rule, states: *[3]?depth.State, inactive_rate: f64) ![3]depth.Who {
    var whos: [3]depth.Who = undefined;
    for (&whos, states, 0..) |*w, *st, i| {
        w.* = .{ .state = st, .draft_room = if (i == 0) 4 else 2, .finished = i == 1, .forced = i == 2 };
        const p = try rule.rates(w.*);
        if (i != 0) @memset(p, inactive_rate);
    }
    return whos;
}

test "mixed depth budgets retain all streams only without hold prices" {
    var costs: Table = .{};
    defer costs.deinit(gpa);
    for (1..9) |rows| try costs.put(gpa, @intCast(rows), 10);
    for ([_]bool{ false, true }) |priced| {
        var rule = ruleFor(&costs, priced);
        defer rule.deinit();
        var states: [3]?depth.State = @splat(null);
        defer for (&states) |*st| if (st.*) |*s| s.deinit(gpa);
        var whos = try mixed(&rule, &states, 1);
        for (&states) |*st| {
            st.*.?.rounds = 19;
            st.*.?.plain = 5;
            st.*.?.wait = 8;
        }
        var unused: u8 = 0;
        var out: [3]i64 = undefined;
        try rule.budgets(&whos, .{ .ptr = &unused, .copy = noCopy }, false, &out);
        // Unpriced inactive streams keep main's rows; costed holds reserve only active draft work.
        try t.expectEqualSlices(i64, &.{ if (priced) 3 else 0, 0, 0 }, &out);
        try t.expectEqual(@as(u64, 20), states[0].?.rounds);
        try t.expectEqual(@as(?u64, if (priced) 0 else 6), states[0].?.plain);
        try t.expectEqual(@as(?u64, 8), states[0].?.wait);
        for (states[1..]) |st| {
            try t.expectEqual(@as(u64, if (priced) 19 else 20), st.?.rounds);
            try t.expectEqual(@as(?u64, if (priced) 5 else 0), st.?.plain);
            try t.expectEqual(@as(?u64, 8), st.?.wait);
        }
    }
}

test "depth overhead counts use all streams unpriced and live streams costed" {
    var costs: Table = .{};
    defer costs.deinit(gpa);
    for (1..17) |rows| try costs.put(gpa, @intCast(rows), @floatFromInt(rows));
    for ([_]bool{ false, true }) |priced| {
        for ([_]bool{ false, true }) |own_steps| {
            for ([_]bool{ false, true }) |sampled| {
                var rule = ruleFor(&costs, priced);
                defer rule.deinit();
                rule.batch_rows = 16;
                rule.draft_step_ms = if (own_steps) 0.1 else 0;
                try rule.observeOverhead(2, 2, 2, sampled, 0);
                try rule.observeOverhead(3, 3, 103, sampled, if (own_steps) 100 else 0);
                try t.expectEqual(@as(f64, 100), rule.overheadMs(3, sampled));
                try t.expectEqual(@as(f64, if (own_steps) 0 else 100), rule.fixedMs(3, sampled));
                var states: [3]?depth.State = @splat(null);
                defer for (&states) |*st| if (st.*) |*s| s.deinit(gpa);
                var whos = try mixed(&rule, &states, 0);
                var unused: u8 = 0;
                var out: [3]i64 = undefined;
                try rule.budgets(&whos, .{ .ptr = &unused, .copy = noCopy }, sampled, &out);
                try t.expectEqualSlices(i64, &.{ if (priced) 0 else 3, 0, 0 }, &out);
            }
        }
    }
}

test "learned depth row prices retain the unpriced all-stream fit" {
    var costs: Table = .{};
    defer costs.deinit(gpa);
    for (1..17) |rows| try costs.put(gpa, @intCast(rows), 10);
    for ([_]bool{ false, true }) |priced| {
        for ([_]bool{ false, true }) |sampled| {
            var rule = ruleFor(&costs, priced);
            defer rule.deinit();
            rule.k.learn_rows = true;
            rule.batch_rows = 16;
            for ([_]u32{ 2, 3 }, [_]f64{ 25, 1 }) |streams, row_ms| {
                for ([_]u64{ 0, 2, 4 }) |past| {
                    for (0..8) |_| try rule.observeOverhead(streams, streams + past, 10 + row_ms * @as(f64, @floatFromInt(past)), sampled, 0);
                }
            }
            try t.expectApproxEqAbs(@as(f64, 25), rule.row_cost.extraMs(2, sampled), 1e-9);
            try t.expectApproxEqAbs(@as(f64, 1), rule.row_cost.extraMs(3, sampled), 1e-9);
            try t.expectEqual(@as(u64, 16), rule.row_cost.capRows(3, sampled, 16));
            var states: [3]?depth.State = @splat(null);
            defer for (&states) |*st| if (st.*) |*s| s.deinit(gpa);
            var whos = try mixed(&rule, &states, 0);
            var unused: u8 = 0;
            var out: [3]i64 = undefined;
            try rule.budgets(&whos, .{ .ptr = &unused, .copy = noCopy }, sampled, &out);
            try t.expectEqualSlices(i64, &.{ if (priced) 0 else 1, 0, 0 }, &out);
        }
    }
}

test "learning depth row caps retain the unpriced all-stream count" {
    var costs: Table = .{};
    defer costs.deinit(gpa);
    for (1..17) |rows| try costs.put(gpa, @intCast(rows), 10);
    for ([_]bool{ false, true }) |priced| {
        var rule = ruleFor(&costs, priced);
        defer rule.deinit();
        rule.k.learn_rows = true;
        rule.batch_rows = 16;
        for (0..8) |_| try rule.observeOverhead(3, 3, 10, false, 0);
        try t.expectEqual(@as(f64, 0), rule.row_cost.extraMs(3, false));
        try t.expectEqual(@as(u64, 7), rule.row_cost.capRows(3, false, 16));
        try t.expectEqual(@as(u64, 2), rule.row_cost.capRows(2, false, 16));
        var states: [3]?depth.State = @splat(null);
        defer for (&states) |*st| if (st.*) |*s| s.deinit(gpa);
        var whos = try mixed(&rule, &states, 0);
        var unused: u8 = 0;
        var out: [3]i64 = undefined;
        try rule.budgets(&whos, .{ .ptr = &unused, .copy = noCopy }, false, &out);
        try t.expectEqualSlices(i64, &.{ if (priced) 0 else 3, 0, 0 }, &out);
    }
}

test "depth probes count finished fixed rows only without hold prices" {
    var costs: Table = .{};
    defer costs.deinit(gpa);
    for (1..6) |rows| try costs.put(gpa, @intCast(rows), if (rows == 5) 12 else 10);
    for ([_]bool{ false, true }) |priced| {
        var rule = ruleFor(&costs, priced);
        defer rule.deinit();
        rule.prior = &.{0.9};
        try rule.observeOverhead(2, 2, 10, false, 0);
        try rule.observeOverhead(3, 3, 10, false, 0);
        var states: [3]?depth.State = @splat(null);
        defer for (&states) |*st| if (st.*) |*s| s.deinit(gpa);
        var whos = try mixed(&rule, &states, 0);
        var unused: u8 = 0;
        const probe: depth.Probe = .{ .ptr = &unused, .copy = noCopy };
        var out: [3]i64 = undefined;
        try rule.budgets(&whos, probe, false, &out);
        try t.expectEqualSlices(i64, &.{ 2, 0, 0 }, &out);
        rule.shared_rounds = 7;
        try rule.budgets(&whos, probe, false, &out);
        try t.expectEqualSlices(i64, &.{ if (priced) 3 else 2, 0, 0 }, &out);
    }
}

test "forced granted depths perform main bookkeeping only without hold prices" {
    const costs: Table = .{};
    for ([_]bool{ false, true }) |priced| {
        for ([_]bool{ false, true }) |reserving| {
            for ([_]?u32{ null, 0, 1 }) |granted| {
                var rule = ruleFor(&costs, priced);
                defer rule.deinit();
                rule.node_probabilities = true;
                rule.granted_depth = true;
                var state: ?depth.State = null;
                defer if (state) |*st| st.deinit(gpa);
                const who: depth.Who = .{ .state = &state, .draft_room = 100, .finished = false, .forced = true, .granted = granted };
                _ = try rule.rates(who);
                state.?.rounds = 19;
                state.?.plain = 7;
                state.?.wait = 8;
                var unused: u8 = 0;
                const probe: depth.Probe = .{ .ptr = &unused, .copy = noCopy };
                const got = if (reserving) try rule.reserveDepth(&who, null, probe) else try rule.headDepth(&who, null, probe);
                try t.expectEqual(@as(i64, 0), got);
                const unchanged = priced or granted == null;
                try t.expectEqual(@as(?u64, if (unchanged) 7 else 0), state.?.plain);
                try t.expectEqual(@as(?u64, if (!unchanged and granted.? == 0) 16 else 8), state.?.wait);
                try t.expectEqual(@as(u64, 19), state.?.rounds);
            }
        }
    }
}

test "forced ordinary depth consumes main's plain probe only without hold prices" {
    var slots = [_]?f64{ null, 10, 10, 10, 10 };
    const costs: Table = .{ .slots = &slots };
    for ([_]bool{ false, true }) |priced| {
        for ([_]bool{ false, true }) |reserving| {
            var rule = ruleFor(&costs, priced);
            defer rule.deinit();
            rule.mtp_step_ms = 100;
            if (priced) rule.k.hold_costs = &.{.{ .streams = 1, .depth = 1, .ms = 100 }};
            var state: ?depth.State = null;
            defer if (state) |*st| st.deinit(gpa);
            var who: depth.Who = .{ .state = &state, .draft_room = 100, .finished = false, .forced = true };
            _ = try rule.rates(who);
            state.?.rounds = 19;
            state.?.plain = 7;
            state.?.wait = 8;
            var unused: u8 = 0;
            const probe: depth.Probe = .{ .ptr = &unused, .copy = noCopy };
            const got = if (reserving) try rule.reserveDepth(&who, null, probe) else try rule.headDepth(&who, null, probe);
            try t.expectEqual(@as(i64, 0), got);
            try t.expectEqual(@as(?u64, if (priced) 7 else 0), state.?.plain);
            try t.expectEqual(@as(?u64, if (priced) 8 else 16), state.?.wait);
            try t.expectEqual(@as(u64, if (priced) 19 else 20), state.?.rounds);
            who.forced = false;
            try t.expectEqual(@as(i64, if (priced) 1 else 0), try rule.headDepth(&who, null, probe));
            try t.expectEqual(@as(?u64, if (priced) 0 else 1), state.?.plain);
            try t.expectEqual(@as(?u64, 16), state.?.wait);
            try t.expectEqual(@as(u64, if (priced) 20 else 21), state.?.rounds);
        }
    }
}

//! The original depth rule regressions.
const std = @import("std");
const d = @import("depth.zig");
const Rule = d.Rule;
const State = d.State;
const Who = d.Who;
const Probe = d.Probe;
const HoldCost = d.HoldCost;
const Table = @import("table.zig").Table;

test "a depth's acceptance estimate moves by the configured weight of the newest round" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .k = .{ .depth_rate = 0.08 }, .prior = &.{ 0.5, 0.5 }, .most_drafts = 2, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = true, .node_probabilities = false, .batch_rows = 8 };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try rule.observeDepth(who, 2, 1); // the first draft landed, the second did not
    try std.testing.expectApproxEqAbs(@as(f64, 0.54), st.?.p[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.46), st.?.p[1], 1e-12);
}

test "depth picks the most tokens a ms and probes deeper" {
    const gpa = std.testing.allocator;
    var fc: Table = .{};
    defer fc.deinit(gpa);
    for (1..5) |w| try fc.put(gpa, @intCast(w), 5.0 + @as(f64, @floatFromInt(w)));
    const shared: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.9, 0.9 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = false, .batch_rows = 8 };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try std.testing.expectEqual(@as(i64, 3), try rule.depth(who));
}

test "under a confidence floor a greedy stream drafts the most, and a sampled one keeps its rates' depth" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = No.copy };
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.5, 0.5, 0.5 }, .most_drafts = 3, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .confidence_floor = 0.5, .batch_rows = 8 };
    defer rule.deinit();
    var states: [2]?State = .{ null, null };
    defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
    const greedy: Who = .{ .state = &states[0], .draft_room = 100, .finished = false, .forced = false };
    const sampled: Who = .{ .state = &states[1], .draft_room = 100, .finished = false, .forced = false, .sampled = true };
    try std.testing.expectEqual(@as(i64, 3), try rule.headDepth(&greedy, null, probe));
    try std.testing.expectEqual(@as(i64, 1), try rule.headDepth(&sampled, null, probe)); // a first rate under 0.8: one draft
}

test "a shared round's drafts come out the same whichever depths its overhead was measured at" {
    const gpa = std.testing.allocator;
    var shared: Table = .{};
    defer shared.deinit(gpa);
    for (1..17) |w| try shared.put(gpa, @intCast(w), 10.0 + 2.0 * @as(f64, @floatFromInt(w)));
    const empty: Table = .{};
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = No.copy };
    var got: [2][2]i64 = undefined;
    for ([_][2]i64{ .{ 1, 1 }, .{ 3, 3 } }, &got) |measured, *out| {
        var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.8, 0.7, 0.6 }, .most_drafts = 4, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 3.0, .plain_guard = false, .node_probabilities = false, .batch_rows = 16 };
        defer rule.deinit();
        // two streams' round: their pending rows and the drafts verified, 3 ms of its own, then each drafted step
        const rows: u64 = @intCast(2 + measured[0] + measured[1]);
        const head = rule.headMs(&measured);
        try rule.observeOverhead(2, rows, shared.get(@intCast(rows)).? + 3.0 + head, false, head);
        var states: [2]?State = .{ null, null };
        defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
        var whos = [_]Who{ .{ .state = &states[0], .draft_room = 100, .finished = false, .forced = false }, .{ .state = &states[1], .draft_room = 100, .finished = false, .forced = false } };
        try rule.budgets(&whos, probe, false, out);
    }
    try std.testing.expectEqualSlices(i64, &got[0], &got[1]);
}

test "granted depths in a shared round fit one hold pass: every block is as deep as the deepest" {
    const empty: Table = .{};
    var rule: Rule = .{ .gpa = std.testing.allocator, .prior = &.{0.8}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
    defer rule.deinit();
    var states: [8]?State = @splat(null);
    defer for (&states) |*st| if (st.*) |*x| x.deinit(std.testing.allocator);
    var whos: [8]Who = undefined;
    for (&whos, &states, 0..) |*w, *st, i| w.* = .{ .state = st, .draft_room = 100, .finished = false, .forced = false, .granted = @intCast(i) };
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    var out: [8]i64 = undefined;
    try rule.budgets(&whos, .{ .ptr = &unused, .copy = No.copy }, false, &out);
    // a grant of none probes two; grants 1..7 that took every held draft, plus one; cut to 32 rows over eight blocks
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 3, 4, 4, 4, 4, 4 }, &out);
    try rule.budgets(whos[0..2], .{ .ptr = &unused, .copy = No.copy }, false, out[0..2]);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, out[0..2]); // two blocks fit whole
    for (whos[0..2]) |*w| w.grant_short = true;
    try rule.budgets(whos[0..2], .{ .ptr = &unused, .copy = No.copy }, false, out[0..2]);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1 }, out[0..2]); // a grant short of the held drafts: no slack
}

test "a copy ahead: one head draft beside it, or none when the head's drafts cost a pass of their own" {
    const empty: Table = .{};
    const Yes = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return true;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = Yes.copy };
    var st: ?State = null;
    defer if (st) |*x| x.deinit(std.testing.allocator);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false, .granted = 3 };
    for ([_]bool{ false, true }) |first| {
        var rule: Rule = .{ .gpa = std.testing.allocator, .k = .{ .copy_first = first }, .prior = &.{0.8}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
        defer rule.deinit();
        try std.testing.expectEqual(@as(i64, @intFromBool(!first)), try rule.headDepth(&who, null, probe));
        var out: [1]i64 = undefined;
        var whos = [_]Who{who};
        try rule.budgets(&whos, probe, false, &out);
        try std.testing.expectEqual(@as(i64, @intFromBool(!first)), out[0]);
    }
}

test "a granted hold stays only while its stream's expected drafts pay for the verify rows and the hold pass" {
    const gpa = std.testing.allocator;
    var shared: Table = .{};
    defer shared.deinit(gpa);
    for (1..33) |w| try shared.put(gpa, @intCast(w), 10.0 + @as(f64, @floatFromInt(w)));
    const empty: Table = .{};
    const cheap = [_]HoldCost{ .{ .streams = 1, .depth = 1, .ms = 1.0 }, .{ .streams = 8, .depth = 1, .ms = 4.5 } };
    var states: [4]?State = @splat(null);
    defer for (&states) |*st| if (st.*) |*x| x.deinit(gpa);
    const granted = [_]?u32{ 2, 2, 0, 1 };
    const gains = [_]?f64{ 1.8, 0.2, null, 0.9 };
    var whos: [4]Who = undefined;
    for (&whos, &states, granted, gains) |*w, *st, g, x| w.* = .{ .state = st, .draft_room = 100, .finished = false, .forced = false, .granted = g, .gain = x };
    var rule: Rule = .{ .gpa = gpa, .k = .{ .hold_costs = &cheap }, .prior = &.{0.8}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 0.5, .plain_guard = true, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
    defer rule.deinit();
    // 8 ms overhead: plain 4 / 22; stream 0 then 3 raise the rate (5.8 / 25, 6.7 / 26.5), stream 1 would lower it
    var out = [_]i64{ 4, 4, 0, 3 };
    try rule.payHolds(&whos, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 4, 0, 0, 3 }, &out);
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    rule.gpa = failing.allocator();
    out = .{ 4, 4, 0, 3 };
    try rule.payHolds(&whos, false, &out);
    try std.testing.expect(!failing.has_induced_failure);
    try std.testing.expectEqualSlices(i64, &.{ 4, 0, 0, 3 }, &out);
    rule.gpa = gpa;
    const dear = [_]HoldCost{.{ .streams = 1, .depth = 1, .ms = 20.0 }};
    rule.k.hold_costs = &dear;
    out = .{ 4, 4, 0, 3 };
    try rule.payHolds(&whos, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, &out); // no gain pays a 20 ms pass
    whos[2].granted = 0;
    out = .{ 4, 4, 2, 3 };
    try rule.payHolds(&whos, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2, 0 }, &out); // a probe after a grant of none keeps its hold
}

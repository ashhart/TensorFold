//! A shared hold's tests: blocks cut to one hold pass, and holds kept only while they pay (depth.zig's Rule).
const std = @import("std");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const Rule = depth.Rule;
const State = depth.State;
const Who = depth.Who;
const HoldCost = depth.HoldCost;

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

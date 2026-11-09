//! A mixed round retains the legacy head-only rule until an independent plain observation exists.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;

fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

fn table() !Table {
    var out: Table = .{};
    for (1..17) |n| try out.put(a, @intCast(n), 10 + 2 * @as(f64, @floatFromInt(n)));
    return out;
}

fn rule(costs: *Table) depth.Rule {
    return .{ .gpa = a, .prior = &.{0.3}, .most_drafts = 1, .family_costs = costs, .shared_costs = costs, .mtp_step_ms = 0, .plain_guard = true, .node_probabilities = false, .batch_rows = 16 };
}

test "shared prices: one head among four unmeasured streams keeps the legacy draft and probe" {
    var costs = try table();
    defer costs.deinit(a);
    var r = rule(&costs);
    defer r.deinit();
    var state: ?depth.State = null;
    defer if (state) |*s| s.deinit(a);
    var who = [_]depth.Who{.{ .state = &state, .draft_room = 20, .finished = false, .forced = false }};
    var out: [1]i64 = undefined;
    try r.budgetsFor(&who, .{ .ptr = &r, .copy = noCopy }, false, &out, 4);
    try t.expectEqual(@as(i64, 1), out[0]);
    try t.expectEqual(@as(u64, 1), r.shared_rounds);
    try t.expect(r.sharedPrice(4, false) == null);
}

test "shared prices: unmeasured holds use the head count and its original fixed cost" {
    var costs = try table();
    defer costs.deinit(a);
    var r = rule(&costs);
    defer r.deinit();
    r.k.hold_costs = &.{.{ .streams = 1, .depth = 1, .ms = 2 }};
    try r.observeOverhead(1, 1, 112, false, 0);
    try r.observeOverhead(4, 4, 18, false, 0);
    var state: ?depth.State = null;
    const who = [_]depth.Who{.{ .state = &state, .draft_room = 20, .finished = false, .forced = false, .granted = 1, .gain = 0.3 }};
    var out = [_]i64{1};
    try r.payHoldsFor(&who, false, &out, 4);
    try t.expectEqual(@as(i64, 1), out[0]);
}

test "shared prices: a measured different stream count cannot reprice a warm mixed round" {
    var costs = try table();
    defer costs.deinit(a);
    var r = rule(&costs);
    defer r.deinit();
    try r.observeShared(2, 2, 1002, false, 0, true, false);
    var states: [2]?depth.State = @splat(null);
    defer for (&states) |*state| if (state.*) |*st| st.deinit(a);
    var who: [2]depth.Who = undefined;
    for (&who, &states) |*w, *state| w.* = .{ .state = state, .draft_room = 20, .finished = false, .forced = false };
    var out: [2]i64 = undefined;
    try r.budgetsFor(&who, .{ .ptr = &r, .copy = noCopy }, false, &out, 4);
    try t.expectEqual(@as(u64, 1), r.shared_rounds);
    try t.expect(r.sharedPrice(4, false) == null);
}

//! Invalid copy-pricing inputs refuse or keep the unpriced width without moving its backoff.
const std = @import("std");
const depth = @import("depth.zig");
const price = @import("copy_price.zig");
const Table = @import("table.zig").Table;
const gpa = std.testing.allocator;

fn startup() !Table {
    var t: Table = .{};
    errdefer t.deinit(gpa);
    for (1..6) |width| try t.put(gpa, @intCast(width), 9 + @as(f64, @floatFromInt(width)));
    return t;
}

fn rule(costs: *const Table) depth.Rule {
    return .{ .gpa = gpa, .prior = &.{0.8}, .most_drafts = 4, .family_costs = costs, .shared_costs = costs, .mtp_step_ms = 0, .plain_guard = true, .copy_guard = true, .node_probabilities = false, .batch_rows = 32 };
}

test "copy edges: empty chances have a named refusal before state allocation" {
    var costs = try startup();
    defer costs.deinit(gpa);
    var r = rule(&costs);
    defer r.deinit();
    var state: ?depth.State = null;
    defer if (state) |*s| s.deinit(gpa);
    const who: depth.Who = .{ .state = &state, .draft_room = 16, .finished = false, .forced = false };
    for ([_]bool{ false, true }) |take| try std.testing.expectError(error.EmptyCopyChances, r.copyRows(who, &.{}, 4, take));
    try std.testing.expect(state == null);
}

test "copy edges: inactive, empty-table and empty-offer paths keep their existing early returns" {
    var costs = try startup();
    defer costs.deinit(gpa);
    var empty: Table = .{};
    defer empty.deinit(gpa);
    var r = rule(&costs);
    defer r.deinit();
    var state: ?depth.State = null;
    defer if (state) |*s| s.deinit(gpa);
    const who: depth.Who = .{ .state = &state, .draft_room = 16, .finished = false, .forced = false };
    try std.testing.expectEqual(@as(i64, 0), try r.copyRows(who, &.{}, 0, true));
    r.copy_guard = false;
    try std.testing.expectEqual(@as(i64, 4), try r.copyRows(who, &.{}, 4, true));
    r.copy_guard = true;
    r.family_costs = &empty;
    try std.testing.expectEqual(@as(i64, 4), try r.copyRows(who, &.{}, 4, true));
    try std.testing.expect(state == null);
}

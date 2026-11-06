//! Unpriced streams retain main's depth decisions; priced holds may exclude paused streams from head work.
const std = @import("std");
const depth = @import("depth.zig");
const win = @import("windows.zig");
const Stream = @import("stream.zig").Stream;
const Table = @import("table.zig").Table;

fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

test "shared compatibility: paused depth exclusion requires hold costs and ends on resume" {
    const a = std.testing.allocator;
    const table: Table = .{};
    for ([_]bool{ false, true }) |priced| {
        var rule: depth.Rule = .{
            .gpa = a,
            .prior = &.{ 0.8, 0.7, 0.6 },
            .most_drafts = 3,
            .family_costs = &table,
            .shared_costs = &table,
            .mtp_step_ms = 0,
            .plain_guard = false,
            .node_probabilities = true,
            .batch_rows = 8,
            .k = .{ .hold_costs = if (priced) &.{.{ .streams = 1, .depth = 1, .ms = 1 }} else &.{} },
        };
        defer rule.deinit();
        var s = try Stream.init(a, .{ .id = "paused", .prompt = &.{ 1, 2, 3 }, .max_new = 32 });
        defer s.deinit(a);
        const probe: depth.Probe = .{ .ptr = &rule, .copy = noCopy };
        s.paused = true;
        var who = win.who(&s, rule.k.hold_costs.len > 0);
        try std.testing.expectEqual(@as(i64, if (priced) 0 else 3), try rule.headDepth(&who, null, probe));
        s.paused = false;
        who = win.who(&s, rule.k.hold_costs.len > 0);
        try std.testing.expectEqual(@as(i64, 3), try rule.headDepth(&who, null, probe));
        s.finished = true;
        who = win.who(&s, rule.k.hold_costs.len > 0);
        try std.testing.expectEqual(@as(i64, 0), try rule.headDepth(&who, null, probe));
    }
}

//! A greedy stream under a confidence floor drafts the most, unless its model sets Constants.floor_rates.
const std = @import("std");
const depth = @import("depth.zig");
const Config = @import("config.zig").Config;
const Table = @import("table.zig").Table;

fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

test "depth floor: a greedy stream under a floor drafts the most by default and its rates' depth with floor_rates" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = noCopy };
    const prior: [15]f64 = @splat(0.5);
    for ([_]bool{ false, true }) |keep| {
        // fifteen drafts a chain, a first rate of 0.5 and no family costs: the rates' depth is one draft
        var rule: depth.Rule = .{ .gpa = gpa, .k = .{ .floor_rates = keep }, .prior = &prior, .most_drafts = 15, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .confidence_floor = 0.5, .batch_rows = 8 };
        defer rule.deinit();
        var states: [2]?depth.State = .{ null, null };
        defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
        const greedy: depth.Who = .{ .state = &states[0], .draft_room = 100, .finished = false, .forced = false };
        const sampled: depth.Who = .{ .state = &states[1], .draft_room = 100, .finished = false, .forced = false, .sampled = true };
        const want: i64 = if (keep) 1 else 15;
        try std.testing.expectEqual(want, try rule.headDepth(&greedy, null, probe));
        try std.testing.expectEqual(want, try rule.depth(greedy));
        try std.testing.expectEqual(@as(i64, 1), try rule.headDepth(&sampled, null, probe)); // sampled, either way
    }
}

test "depth floor: a model's floor_rates reaches the depth rule's constants, off unless the model sets it" {
    const gpa = std.testing.allocator;
    var off = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 15, .draft_confidence = 0.5 }, 16, 15);
    defer off.deinit(gpa);
    try std.testing.expect(!off.k.floor_rates);
    var on = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 15, .draft_confidence = 0.5, .floor_rates = true }, 16, 15);
    defer on.deinit(gpa);
    try std.testing.expect(on.k.floor_rates);
}

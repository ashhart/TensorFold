//! The depth rule against a round's noise, and the probe a lone window must keep for it to recover.
const std = @import("std");
const depth = @import("depth.zig");
const win = @import("windows.zig");
const Stream = @import("stream.zig").Stream;
const Table = @import("table.zig").Table;
const t = std.testing;

fn rule(gpa: std.mem.Allocator, family: *const Table, shared: *const Table, most: u32, prior: []const f64) !depth.Rule {
    return .{ .gpa = gpa, .prior = prior, .most_drafts = most, .family_costs = family, .shared_costs = shared, .mtp_step_ms = 0.5, .plain_guard = true, .node_probabilities = false, .batch_rows = 8 };
}

test "depth: a noisy round does not take a stream off the depth it holds" {
    const gpa = t.allocator;
    var family: Table = .{};
    defer family.deinit(gpa);
    for ([_]f64{ 16, 20, 24 }, 1..) |ms, rows| try family.put(gpa, @intCast(rows), ms); // forward ms by rows
    const shared: Table = .{};
    var r = try rule(gpa, &family, &shared, 2, &.{ 0.72, 0.54 });
    defer r.deinit();
    var st: ?depth.State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try r.observeCost(1, 18.1, false, who);
    try r.observeCost(2, 20.8, false, who);
    try t.expectEqual(@as(i64, 2), try r.depth(who)); // 1.72/18.1 against 2.1088/20.8: two drafts lead
    // a round inflated by the probe that followed it: two drafts now trail one, but by less than the margin
    for (0..12) |_| try r.observeCost(2, 22.65, false, who);
    try t.expectEqual(@as(i64, 2), try r.depth(who));
    // a depth that keeps losing by more than the margin still gives way
    try r.observeCost(2, 30.0, false, who);
    try t.expectEqual(@as(i64, 1), try r.depth(who));
}

test "depth: a probe asks one draft past the depth a stream holds" {
    const gpa = t.allocator;
    var family: Table = .{};
    defer family.deinit(gpa);
    for ([_]f64{ 16, 20, 24, 30 }, 1..) |ms, rows| try family.put(gpa, @intCast(rows), ms);
    const shared: Table = .{};
    var r = try rule(gpa, &family, &shared, 3, &.{ 0.72, 0.54, 0.5 });
    defer r.deinit();
    var st: ?depth.State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try r.observeCost(1, 18.1, false, who);
    try r.observeCost(2, 20.8, false, who);
    try r.observeCost(3, 25.0, false, who);
    for (1..8) |_| try t.expectEqual(@as(i64, 2), try r.depth(who));
    try t.expectEqual(@as(i64, 3), try r.depth(who)); // every eighth round probes one deeper
    try t.expectEqual(@as(?i64, 3), st.?.probe);
}

test "windows: a lone window keeps the probe a depth rule asked for" {
    var s: Stream = undefined;
    const probed: win.Plan = .{ .stream = &s, .position = 0, .kind = .head, .held = 2, .probe_rows = 2 };
    var counts = [_]u32{1};
    win.keepProbe(&probed, &counts);
    try t.expectEqual(@as(u32, 2), counts[0]);
    counts[0] = 3;
    win.keepProbe(&probed, &counts); // an allocator already deeper than the probe keeps its own choice
    try t.expectEqual(@as(u32, 3), counts[0]);
    const priced: win.Plan = .{ .stream = &s, .position = 0, .kind = .head, .held = 2 };
    counts[0] = 1;
    win.keepProbe(&priced, &counts);
    try t.expectEqual(@as(u32, 1), counts[0]);
}

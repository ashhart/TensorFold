//! copy_price.zig's estimates of a lone copy's cost, and the plain round it must beat, on hand-timed rounds.
const std = @import("std");
const depth = @import("depth.zig");
const price = @import("copy_price.zig");
const Table = @import("table.zig").Table;

const gpa = std.testing.allocator;

/// The forwards as timed at load on a short scratch context: 5 ms plus 0.2 ms a row.
fn startup() !Table {
    var t: Table = .{};
    for (1..17) |w| try t.put(gpa, @intCast(w), 5 + 0.2 * @as(f64, @floatFromInt(w)));
    return t;
}

fn rule(forwards: *const Table, plain_guard: bool) depth.Rule {
    return .{ .gpa = gpa, .prior = &.{0.8}, .most_drafts = 4, .family_costs = forwards, .shared_costs = forwards, .mtp_step_ms = 0.5, .plain_guard = plain_guard, .copy_guard = true, .node_probabilities = false, .batch_rows = 32 };
}

/// A plain round of `ms` into the stream's own times, as the round loop records it with or without plain_guard.
fn timePlain(r: *depth.Rule, ms: f64, who: depth.Who) !void {
    if (r.plain_guard) try r.observeCost(0, ms, false, who) else try price.observeOwn(r, 0, ms, false, who);
}

const eps = 1e-9;

test "copy price: before any copy is timed, each of its n + 1 rows costs a forward row plus plain's excess" {
    var forwards = try startup();
    defer forwards.deinit(gpa);
    for ([_]bool{ true, false }) |plain_guard| {
        var r = rule(&forwards, plain_guard);
        defer r.deinit();
        var st: ?depth.State = null;
        defer if (st) |*x| x.deinit(gpa);
        const who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
        // nothing timed: plain is its forward, and a copy is its own forward
        try std.testing.expectApproxEqAbs(5.2, (try price.plain(&r, who)).?, eps);
        try std.testing.expectApproxEqAbs(forwards.get(4).?, (try price.cost(&r, 3, who)).?, eps);
        // a 90 ms plain round: 84.8 ms past its forward, paid by each of the copy's n + 1 rows
        try timePlain(&r, 90, who);
        try std.testing.expectApproxEqAbs(90, (try price.plain(&r, who)).?, eps);
        try std.testing.expectApproxEqAbs(forwards.get(4).? + 4 * (90 - 5.2), (try price.cost(&r, 3, who)).?, eps);
        try std.testing.expectApproxEqAbs(forwards.get(16).? + 16 * (90 - 5.2), (try price.cost(&r, 15, who)).?, eps);
        try std.testing.expectEqual(@as(?f64, null), try price.cost(&r, 16, who)); // past the forwards timed at load
    }
}

test "copy price: a timed width as timed, others on the line from plain to the nearest timed copy" {
    var forwards = try startup();
    defer forwards.deinit(gpa);
    for ([_]bool{ true, false }) |plain_guard| {
        var r = rule(&forwards, plain_guard);
        defer r.deinit();
        var st: ?depth.State = null;
        defer if (st) |*x| x.deinit(gpa);
        const who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
        try timePlain(&r, 90, who);
        try price.observeCopy(&r, 3, 250, false, who);
        try price.observeCopy(&r, 15, 690, false, who);
        try price.observeCopy(&r, 7, 999, true, who); // a stream's first round stays out
        try std.testing.expectApproxEqAbs(250, (try price.cost(&r, 3, who)).?, eps);
        try std.testing.expectApproxEqAbs(690, (try price.cost(&r, 15, who)).?, eps);
        try std.testing.expectApproxEqAbs(90 + 5 * 160.0 / 3.0, (try price.cost(&r, 5, who)).?, eps); // nearest: 3
        try std.testing.expectApproxEqAbs(90 + 12 * 40.0, (try price.cost(&r, 12, who)).?, eps); // nearest: 15
        try std.testing.expectApproxEqAbs(90 + 9 * 160.0 / 3.0, (try price.cost(&r, 9, who)).?, eps); // a tie: the narrower
        // a timed copy repeats into a blend of its times (cost_rate), as head rounds do
        try price.observeCopy(&r, 3, 350, false, who);
        try std.testing.expectApproxEqAbs(250 + r.k.cost_rate * 100, (try price.cost(&r, 3, who)).?, eps);
    }
}

test "copy price: a timed copy cheaper than plain leaves no negative row cost" {
    var forwards = try startup();
    defer forwards.deinit(gpa);
    var r = rule(&forwards, true);
    defer r.deinit();
    var st: ?depth.State = null;
    defer if (st) |*x| x.deinit(gpa);
    const who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try timePlain(&r, 90, who);
    try price.observeCopy(&r, 2, 80, false, who);
    try std.testing.expectApproxEqAbs(90, (try price.cost(&r, 4, who)).?, eps);
}

test "copy price: without plain_guard, plain is roundCost's model on the stream's own head rounds" {
    var forwards = try startup();
    defer forwards.deinit(gpa);
    var r = rule(&forwards, false);
    defer r.deinit();
    var st: ?depth.State = null;
    defer if (st) |*x| x.deinit(gpa);
    const who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    // two rows of a 1-draft head round took 130 ms: 62 ms a row past its forward and head step
    try price.observeOwn(&r, 1, 130, false, who);
    try std.testing.expectApproxEqAbs(5.2 + (130 - 5.4 - 0.5) / 2.0, (try price.plain(&r, who)).?, eps);
    // the head's own prices do not read them: they stay on the rule's greedy rounds without plain_guard
    try std.testing.expectApproxEqAbs(forwards.get(2).? + 0.5, (try r.roundCost(1, who)).?, eps);
}

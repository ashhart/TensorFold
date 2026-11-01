//! Synthetic pass-cost fixtures: these pin pricing decisions, not device performance.
const std = @import("std");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const Drafted = @import("drafted.zig").Drafted;
const dt = @import("drafted_test.zig");

const gpa = std.testing.allocator;

fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

test "a lone stream's depth prices its hold at the calibrated pass, not at a linear step" {
    var windows: Table = .{};
    defer windows.deinit(gpa);
    for (1..5) |w| try windows.put(gpa, @intCast(w), 10); // every window costs 10 ms
    const shared: Table = .{};
    // one draft costs the pass's 8 ms floor and three only 9; a linear 1 ms step would price them 1 and 3
    const grid = [_]depth.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 8 }, .{ .streams = 1, .depth = 3, .ms = 9 } };
    for ([_][]const depth.HoldCost{ &.{}, &grid }) |costs| {
        const linear = costs.len == 0;
        var rule: depth.Rule = .{ .gpa = gpa, .k = .{ .hold_costs = costs }, .prior = &.{ 0.5, 0.5, 0.5 }, .most_drafts = 3, .family_costs = &windows, .shared_costs = &shared, .mtp_step_ms = 1.0, .plain_guard = true, .node_probabilities = false, .batch_rows = 8 };
        defer rule.deinit();
        try std.testing.expectApproxEqAbs(@as(f64, 10), (try rule.roundCost(0, null)).?, 1e-12); // no draft, no hold
        try std.testing.expectApproxEqAbs(@as(f64, if (linear) 11 else 18), (try rule.roundCost(1, null)).?, 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, if (linear) 13 else 19), (try rule.roundCost(3, null)).?, 1e-12);
        var st: ?depth.State = null;
        defer if (st) |*s| s.deinit(gpa);
        const who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
        // half the drafts land: two pay at 1 ms a step, none at the 8 ms floor (against plain rounds)
        try std.testing.expectEqual(@as(i64, if (linear) 2 else 0), try rule.depth(who));
    }
}

test "shared depth budgets price the pass their holds make: a floor cuts the drafts a linear step keeps" {
    var shared: Table = .{};
    defer shared.deinit(gpa);
    for (1..9) |rows| try shared.put(gpa, @intCast(rows), 10);
    const empty: Table = .{};
    // any hold costs a 6 ms pass, though nine drafts' 10.8 ms average a linear 1.2 ms apiece
    const grid = [_]depth.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 6.0 }, .{ .streams = 1, .depth = 9, .ms = 10.8 }, .{ .streams = 2, .depth = 1, .ms = 6.2 }, .{ .streams = 2, .depth = 9, .ms = 11.0 } };
    for ([_][]const depth.HoldCost{ &.{}, &grid }) |costs| {
        var rule: depth.Rule = .{ .gpa = gpa, .k = .{ .hold_costs = costs }, .prior = &.{0.5}, .most_drafts = 1, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 1.2, .plain_guard = true, .node_probabilities = false, .batched_heads = true, .batch_rows = 8 };
        defer rule.deinit();
        try rule.observeOverhead(2, 2, 10, false, 0); // measured: the round costs nothing past its forward
        var states: [2]?depth.State = @splat(null);
        defer for (&states) |*st| if (st.*) |*x| x.deinit(gpa);
        var whos: [2]depth.Who = undefined;
        for (&whos, &states) |*w, *st| w.* = .{ .state = st, .draft_room = 100, .finished = false, .forced = false };
        var unused: u8 = 0;
        var out: [2]i64 = undefined;
        try rule.budgets(&whos, .{ .ptr = &unused, .copy = noCopy }, false, &out);
        // 3.0 expected tokens in 11.2 ms beat 2 in 10 by a linear step, but lose in the 16.2 ms the pass takes
        const want: [2]i64 = if (costs.len == 0) .{ 1, 1 } else .{ 0, 0 };
        try std.testing.expectEqualSlices(i64, &want, &out);
    }
}

test "a hold past the pass's rows leaves a round's fixed overhead the host's own: its passes are priced as run" {
    var shared: Table = .{};
    defer shared.deinit(gpa);
    for (1..33) |rows| try shared.put(gpa, @intCast(rows), 10);
    const empty: Table = .{};
    // SYNTHETIC: every hold pass costs 10 ms whatever its shape
    const grid = [_]depth.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 10 }, .{ .streams = 1, .depth = 9, .ms = 10 }, .{ .streams = 32, .depth = 1, .ms = 10 }, .{ .streams = 32, .depth = 9, .ms = 10 } };
    const orders = [_][8]i64{ .{ 9, 1, 1, 1, 1, 1, 1, 1 }, .{ 1, 1, 1, 1, 1, 1, 1, 9 } };
    for (orders) |depths| for ([_]u32{ 0, 32 }) |hold_rows| {
        var rule: depth.Rule = .{ .gpa = gpa, .k = .{ .hold_costs = &grid, .hold_rows = hold_rows }, .prior = &.{0.5}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 1, .plain_guard = true, .node_probabilities = false, .batched_heads = true, .batch_rows = 32 };
        defer rule.deinit();
        // Greedy grouping makes two passes in either order, without padding early blocks to a later depth.
        const passes: f64 = if (hold_rows == 0) 10 else 20;
        try std.testing.expectEqual(passes, rule.headMs(&depths));
        try rule.observeOverhead(8, 24, 10 + 3 + passes, false, rule.headMs(&depths));
        try std.testing.expectApproxEqAbs(@as(f64, 3), rule.fixedMs(8, false), 1e-9);
    };
}

/// What the target alone decodes for `prompt`, one token a step, no drafter.
fn plainTokens(model: Model, prompt: []const u32, max_new: u32) ![]u32 {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var cfg = try Config.init(gpa, model, 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "plain", .prompt = prompt, .max_new = max_new, .drafts = false });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return gpa.dupe(u32, s.emitted());
}

// cheap passes: after the first round, drafts pay
const open_grid = [_]depth.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 0.5 }, .{ .streams = 2, .depth = 1, .ms = 0.6 } };

test "a head that prices its hold passes holds nothing at open: the whole prompt is absorbed and decoding is unchanged" {
    const plain = try plainTokens(dt.base(), &dt.p1, 64);
    defer gpa.free(plain);
    for ([_]bool{ false, true }) |grid| {
        const costs: []const depth.HoldCost = if (grid) &open_grid else &.{};
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var strict: dt.Strict = .{ .inner = target.backend() };
        defer strict.deinit();
        var head: dt.FakeDrafter = .{ .facts_ = .{ .depth = 4, .step_ms = 0.5, .batched = true, .limit_shared_drafts = true, .hold_costs = costs } };
        defer head.deinit();
        var wrapped = try Drafted.init(gpa, strict.backend(), head.drafter());
        defer wrapped.deinit();
        var cfg = try Config.init(gpa, wrapped.facts(dt.base()), 16, 15);
        defer cfg.deinit(gpa);
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
        defer engine.deinit();
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &dt.p1, .max_new = 64 });
        defer s.deinit(gpa);
        try engine.addStream(&s);
        // every prompt row reaches the drafter either way; only the hold after it waits for the depth rule's first round
        try std.testing.expectEqual(@as(u64, dt.p1.len), head.prompt_rows);
        try std.testing.expectEqual(@as(u32, if (grid) 0 else 1), head.holds);
        try std.testing.expectEqual(!grid, s.next.?.count > 0);
        while (engine.activeCount() > 0) try engine.step();
        try std.testing.expectEqualSlices(u32, plain, s.emitted());
        try std.testing.expect(head.holds > 0);
    }
}

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

test "copy edges: zero and non-finite measured costs keep the offered width and backoff" {
    const bad_costs = [_]f64{ 0, -0.0, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64) };
    for (bad_costs) |bad| for ([_]bool{ false, true }) |take| {
        var costs = try startup();
        defer costs.deinit(gpa);
        var r = rule(&costs);
        defer r.deinit();
        var state: ?depth.State = null;
        defer if (state) |*s| s.deinit(gpa);
        const who: depth.Who = .{ .state = &state, .draft_room = 16, .finished = false, .forced = false };
        try price.observeCopy(&r, 1, bad, false, who);
        state.?.copy_plain = 3;
        state.?.copy_wait = 16;
        const rounds = state.?.copy_rounds;
        try std.testing.expectEqual(@as(i64, 4), try r.copyRows(who, &.{0.75}, 4, take));
        try std.testing.expectEqual(@as(?u64, 3), state.?.copy_plain);
        try std.testing.expectEqual(@as(?u64, 16), state.?.copy_wait);
        try std.testing.expectEqual(rounds, state.?.copy_rounds);
    };
}

test "copy edges: a later invalid cost also falls back from a previously priced prefix" {
    var costs = try startup();
    defer costs.deinit(gpa);
    var r = rule(&costs);
    defer r.deinit();
    var state: ?depth.State = null;
    defer if (state) |*s| s.deinit(gpa);
    const who: depth.Who = .{ .state = &state, .draft_room = 16, .finished = false, .forced = false };
    try price.observeCopy(&r, 1, 5, false, who);
    try price.observeCopy(&r, 2, std.math.nan(f64), false, who);
    state.?.copy_plain = 3;
    state.?.copy_wait = 16;
    try std.testing.expectEqual(@as(i64, 4), try r.copyRows(who, &.{0.75}, 4, true));
    try std.testing.expectEqual(@as(?u64, 3), state.?.copy_plain);
    try std.testing.expectEqual(@as(?u64, 16), state.?.copy_wait);
}

test "copy edges: finite prices still select the hand-priced prefix and preserve read versus take" {
    var costs = try startup();
    defer costs.deinit(gpa);
    var r = rule(&costs);
    defer r.deinit();
    var state: ?depth.State = null;
    defer if (state) |*s| s.deinit(gpa);
    const who: depth.Who = .{ .state = &state, .draft_room = 16, .finished = false, .forced = false };
    for ([_]f64{ 5, 20, 50, 100 }, 1..) |ms, width| try price.observeCopy(&r, @intCast(width), ms, false, who);
    state.?.copy_plain = 3;
    state.?.copy_wait = 16;
    try std.testing.expectEqual(@as(i64, 1), try r.copyRows(who, &.{0.75}, 4, false));
    try std.testing.expectEqual(@as(?u64, 3), state.?.copy_plain);
    try std.testing.expectEqual(@as(?u64, 16), state.?.copy_wait);
    try std.testing.expectEqual(@as(i64, 1), try r.copyRows(who, &.{0.75}, 4, true));
    try std.testing.expectEqual(@as(?u64, 0), state.?.copy_plain);
    try std.testing.expectEqual(@as(?u64, r.k.depth_probe_every), state.?.copy_wait);
}

test "copy edges: copies-off performs no pricing clock read, allocation, trail or retention" {
    const cfgs = @import("config.zig");
    const Engine = @import("engine.zig").Engine;
    const sm = @import("stream.zig");
    const fake = @import("fake.zig");
    const lone = @import("lone_copy.zig");
    const ev = @import("events.zig");
    const forwards = [_]cfgs.Cost{ .{ .width = 1, .ms = 10 }, .{ .width = 2, .ms = 11 } };
    var cfg = try cfgs.Config.init(gpa, .{ .exact_width = 16, .copy_guard = true, .copy_drafts = false, .window_costs = &forwards }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var log = ev.Log.init(gpa);
    defer log.deinit();
    engine.log = &log;
    var stream = try sm.Stream.init(gpa, .{ .id = "copies-off", .prompt = &.{ 1, 2, 3 }, .max_new = 16 });
    defer stream.deinit(gpa);
    stream.rounds = 2;
    try std.testing.expect(cfg.copy_guard and cfg.max_copy == 0);
    try std.testing.expect(!lone.enabled(&engine, &stream));
    try lone.time(&engine, &stream, .none, 1, null);
    lone.keep(&engine, &stream);
    try std.testing.expectEqual(@as(u64, 0), clock.calls);
    try std.testing.expect(stream.depth == null and !stream.prices_seeded);
    try std.testing.expectEqual(@as(usize, 0), log.events.items.len);
    for (engine.lone_kept) |mode| for (mode) |kept| try std.testing.expect(kept == null);
    cfg.max_copy = 1;
    try std.testing.expect(lone.enabled(&engine, &stream));
}

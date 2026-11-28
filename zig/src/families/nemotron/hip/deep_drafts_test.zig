//! Nemotron's deeper MTP windows: without --mtp-drafts past three the facts and lane bytes keep the four-row ones.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const round = @import("round.zig");
const head = @import("head.zig");
const hip_lanes = @import("lanes.zig");
const native = @import("native.zig");
const costs = @import("costs.zig");

const depth = lanes.depth;
const Table = lanes.table.Table;

/// Representative forward ms by window rows and shared rows for depth-rule equivalence tests.
fn timed() costs.Costs {
    var c: costs.Costs = .{ .step_ms = 0.83 };
    for ([_]f64{ 13.9, 15.1, 16.0, 16.9 }, 0..) |ms, i| c.window[i] = .{ .width = @intCast(i + 1), .ms = ms };
    c.windows = 4;
    for ([_]u32{ 2, 4, 8, 12 }, [_]f64{ 15.0, 17.0, 21.5, 26.0 }, 0..) |w, ms, i| c.shared[i] = .{ .width = w, .ms = ms };
    c.shareds = 4;
    return c;
}

/// The facts as the engine reported them before deeper windows: four rows a window, three drafts, no shared cap.
fn before(streams: usize, c: *const costs.Costs) lanes.Model {
    return .{
        .exact_width = 4,
        .gpu_tokens = false,
        .mtp = true,
        .speculate = true,
        .speculate_early = false,
        .drafts = 3,
        .hidden_rows = true,
        .batch_rows = @intCast(4 * streams),
        .max_streams = @intCast(streams),
        .draft_streams = true,
        .plain_guard = c.windows > 0,
        .window_costs = c.window[0..c.windows],
        .shared_costs = c.shared[0..c.shareds],
        .mtp_step_ms = c.step_ms,
    };
}

test "without --mtp-drafts the window, lane bytes and facts are the four-row ones plus a shared cap of three" {
    const a = std.testing.allocator;
    const asked = (hip.Policy{}).mtp.drafts;
    const window = round.windowFor(asked);
    try std.testing.expectEqual(@as(usize, 4), window);
    try std.testing.expectEqual(native.laneBytesFor(32768, 4, 65536, 24_541_184, 2688, 2), native.laneBytesFor(32768, window, 65536, 24_541_184, 2688, 2));
    var c = timed();
    for ([_]usize{ 1, 3, 4 }) |streams| {
        const at: hip_lanes.Setup = .{ .window = window, .rows = round.planRows(streams, window), .slots = streams };
        try std.testing.expectEqual(streams * window, at.rows);
        var got = hip_lanes.factsFor(at, true, asked, &c);
        try std.testing.expectEqual(@as(u32, 3), got.shared_mtp_cap);
        got.shared_mtp_cap = 0;
        try std.testing.expectEqualDeep(before(streams, &c), got);
        var cfg = try lanes.Config.init(a, hip_lanes.factsFor(at, true, asked, &c), @intCast(window), @intCast(window - 1));
        defer cfg.deinit(a);
        try std.testing.expectEqual(@as(u32, 3), cfg.most_drafts);
        try std.testing.expectEqual(@as(u32, 4), cfg.family_width);
        try std.testing.expectEqual(@as(u32, 3), cfg.shared_mtp_cap);
        try std.testing.expectEqual(@as(u32, @intCast(4 * streams)), cfg.batch_rows);
    }
}

const NoCopy = struct {
    fn copy(_: *anyopaque, _: *const depth.Who) bool {
        return false;
    }
};

fn ruleOf(prior: []const f64, family: *const Table, shared: *const Table, cap: u32, guard: bool) depth.Rule {
    return .{
        .gpa = std.testing.allocator,
        .prior = prior,
        .most_drafts = 3,
        .shared_cap = cap,
        .family_costs = family,
        .shared_costs = shared,
        .mtp_step_ms = 0.83,
        .plain_guard = guard,
        .node_probabilities = false,
        .batch_rows = 16,
    };
}

fn whoOf(state: *?depth.State) depth.Who {
    return .{ .state = state, .draft_room = 100, .finished = false, .forced = false };
}

test "at three drafts the shared cap of three changes no lone depth and no shared budget" {
    const a = std.testing.allocator;
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = NoCopy.copy };
    var family: Table = .{};
    defer family.deinit(a);
    var shared: Table = .{};
    defer shared.deinit(a);
    const c = timed();
    for (c.window[0..c.windows]) |x| try family.put(a, x.width, x.ms);
    for (1..17) |w| try shared.put(a, @intCast(w), 13.0 + @as(f64, @floatFromInt(w)));
    var none: Table = .{};
    defer none.deinit(a);
    const priors = [_][3]f64{ .{ 0.95, 0.9, 0.85 }, .{ 0.85, 0.7, 0.6 }, .{ 0.5, 0.4, 0.3 } };
    for (&priors) |*prior| for ([_]*const Table{ &family, &none }) |fam| for ([_]bool{ false, true }) |guard| {
        var old = ruleOf(prior, fam, &shared, 0, guard);
        defer old.deinit();
        var new = ruleOf(prior, fam, &shared, 3, guard);
        defer new.deinit();
        var so: [3]?depth.State = @splat(null);
        var sn: [3]?depth.State = @splat(null);
        defer for (&so, &sn) |*x, *y| {
            if (x.*) |*s| s.deinit(a);
            if (y.*) |*s| s.deinit(a);
        };
        for (0..24) |r| {
            const lone_old = try old.headDepth(&whoOf(&so[0]), null, probe);
            try std.testing.expectEqual(lone_old, try new.headDepth(&whoOf(&sn[0]), null, probe));
            try old.observeDepth(whoOf(&so[0]), lone_old, @intCast(r % 4));
            try new.observeDepth(whoOf(&sn[0]), lone_old, @intCast(r % 4));
            var wo = [_]depth.Who{ whoOf(&so[1]), whoOf(&so[2]) };
            var wn = [_]depth.Who{ whoOf(&sn[1]), whoOf(&sn[2]) };
            var bo: [2]i64 = undefined;
            var bn: [2]i64 = undefined;
            try old.budgets(&wo, probe, false, &bo);
            try new.budgets(&wn, probe, false, &bn);
            try std.testing.expectEqualSlices(i64, &bo, &bn);
        }
    };
}

test "--mtp-drafts 4 to 15 widens the window to 16; shared rounds keep 3 drafts and at least default plan rows" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 15), head.max_depth);
    try std.testing.expectEqual(@as(u32, 15), native.mtp_max_drafts);
    var c = timed();
    for (4..16) |n| for ([_]usize{ 1, 2, 4, 8, 16 }) |streams| {
        const asked: u8 = @intCast(n);
        const window = round.windowFor(asked);
        try std.testing.expectEqual(n + 1, window);
        const at: hip_lanes.Setup = .{ .window = window, .rows = round.planRows(streams, window), .slots = streams };
        try std.testing.expectEqual(@max(streams * 4, @min(streams * window, 32)), at.rows);
        var cfg = try lanes.Config.init(a, hip_lanes.factsFor(at, true, asked, &c), @intCast(window), @intCast(window - 1));
        defer cfg.deinit(a);
        try std.testing.expectEqual(@as(u32, asked), cfg.most_drafts);
        try std.testing.expectEqual(@as(u32, @intCast(window)), cfg.family_width);
        try std.testing.expectEqual(@as(u32, 3), cfg.shared_mtp_cap);
        try std.testing.expectEqual(@as(u32, @intCast(at.rows)), cfg.batch_rows);
    };
}

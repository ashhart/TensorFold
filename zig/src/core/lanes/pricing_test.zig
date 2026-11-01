//! Window and hold pricing: sampled draws stay local, and paid holds stay out of fixed overhead.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const be = @import("backend.zig");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const Model = @import("config.zig").Model;
const dr = @import("drafter.zig");
const Drafted = @import("drafted.zig").Drafted;
const dt = @import("drafted_test.zig");

const gpa = std.testing.allocator;

/// A round's ms: the model's window cost for its rows, plus 3 ms a row of host draws when a stream in it samples.
const RowClock = struct {
    target: *const fake.Fake,

    fn clock(c: *RowClock) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *RowClock = @ptrCast(@alignCast(ptr));
        const rows: f64 = @floatFromInt(c.target.last_rows);
        return 5.0 + 0.8 * rows + (if (c.target.last_sampled) 3.0 * rows else 0.0);
    }
};

/// The greedy stream's rounds and drafted rows after `sampled` sampled streams ran on the same engine.
fn greedyAfter(node_probabilities: bool, sampled: usize) ![2]u64 {
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 8, .window_costs = &costs, .mtp_step_ms = 0.5, .draft_probabilities = node_probabilities }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: RowClock = .{ .target = &target };
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const prompts = [_][]const u32{ &.{ 2, 7, 1, 8, 2, 8, 1, 8 }, &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5 } };
    for (0..sampled + 1) |i| {
        const greedy = i == sampled;
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = prompts[@intFromBool(greedy)], .max_new = 96, .sampling = if (greedy) null else .{ .seed = 5 + i, .temperature = 0.8 } });
        defer s.deinit(gpa);
        try engine.addStream(&s);
        while (engine.activeCount() > 0) try engine.step();
        if (greedy) return .{ s.rounds, s.drafted };
    }
    unreachable;
}

test "a greedy stream after sampled streams drafts as on a fresh engine, by depth or by node chances" {
    for ([_]bool{ false, true }) |node_probabilities| {
        const fresh = try greedyAfter(node_probabilities, 0);
        const after = try greedyAfter(node_probabilities, 2);
        try std.testing.expectEqual(fresh, after);
        try std.testing.expect(fresh[1] > fresh[0]); // it drafted
    }
}

test "observed padded holds do not make unpaid holds look profitable" {
    var shared: Table = .{};
    defer shared.deinit(gpa);
    for (1..7) |rows| try shared.put(gpa, @intCast(rows), 10);
    const holds = [_]depth.HoldCost{
        .{ .streams = 1, .depth = 1, .ms = 4 },
        .{ .streams = 2, .depth = 1, .ms = 4 },
        .{ .streams = 2, .depth = 3, .ms = 6 },
        .{ .streams = 3, .depth = 3, .ms = 24 },
    };
    const depths = [_]i64{ 1, 3, 0 };
    for ([_]bool{ false, true }) |sampled| for ([_]f64{ 0, 2.5 }) |host_ms| {
        var rule: depth.Rule = .{
            .gpa = gpa,
            .k = .{ .hold_costs = &holds },
            .prior = &.{0.75},
            .most_drafts = 3,
            .family_costs = &shared,
            .shared_costs = &shared,
            .mtp_step_ms = 1,
            .batched_heads = true,
            .plain_guard = true,
            .node_probabilities = true,
            .granted_depth = true,
            .batch_rows = 32,
        };
        defer rule.deinit();
        var states: [3]?depth.State = @splat(null);
        var who: [3]depth.Who = undefined;
        for (&who, &states, 0..) |*w, *state, i| w.* = .{
            .state = state,
            .draft_room = 32,
            .finished = false,
            .forced = false,
            .granted = if (i < 2) 1 else 0,
            .gain = if (i < 2) 0.75 else 0,
        };
        // The actual padded pass costs 6 ms; its linear head-step estimate is only 3 ms.
        try rule.observeOverhead(3, 5, 10 + host_ms + 6, sampled, rule.headMs(&depths));
        var out = depths;
        try rule.payHolds(&who, sampled, &out);
        const want: [3]i64 = if (host_ms == 0) .{ 0, 0, 0 } else depths;
        try std.testing.expectEqualSlices(i64, &want, &out);
    };
}

// Synthetic tables and clocks test pricing, not device performance.

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
    const depths = [_]i64{ 9, 1, 1, 1, 1, 1, 1, 1 };
    for ([_]u32{ 0, 32 }) |hold_rows| {
        var rule: depth.Rule = .{ .gpa = gpa, .k = .{ .hold_costs = &grid, .hold_rows = hold_rows }, .prior = &.{0.5}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 1, .plain_guard = true, .node_probabilities = false, .batched_heads = true, .batch_rows = 32 };
        defer rule.deinit();
        // thirty-two rows hold three blocks of nine: eight blocks are two passes, one pass only without the bound
        const passes: f64 = if (hold_rows == 0) 10 else 20;
        try std.testing.expectEqual(passes, rule.headMs(&depths));
        try rule.observeOverhead(8, 24, 10 + 3 + passes, false, rule.headMs(&depths));
        try std.testing.expectApproxEqAbs(@as(f64, 3), rule.fixedMs(8, false), 1e-9);
    }
}

/// Fake holds report actual settled duration independently of the modeled grid.
const Timed = struct {
    head: dt.FakeDrafter,
    hold_ms: f64, // what a hold pass takes, whatever its grid models
    host_ms: f64, // host work a round spends beyond its forward and holds: the true fixed overhead
    vtable: dr.Drafter.VTable = undefined,
    inner_hold: *const fn (*anyopaque, []const dr.Hold) anyerror!void = undefined,
    window: f64 = 0, // ms settled since the round clock started
    settled: f64 = 0, // ms settled since the timing source was last taken
    holds: u32 = 0,

    fn of(ptr: *anyopaque) *Timed {
        return @fieldParentPtr("head", @as(*dt.FakeDrafter, @ptrCast(@alignCast(ptr))));
    }

    fn owner(ptr: *anyopaque) *Timed {
        return @ptrCast(@alignCast(ptr));
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = of(ptr);
        try x.inner_hold(ptr, items);
        x.holds += 1;
        x.window += x.hold_ms;
        x.settled += x.hold_ms;
    }

    fn take(ptr: *anyopaque) f64 {
        const x = owner(ptr);
        defer x.settled = 0;
        return x.settled;
    }

    fn drafter(x: *Timed) dr.Drafter {
        const inner = x.head.drafter();
        x.vtable = inner.vtable.*;
        x.inner_hold = inner.vtable.hold;
        x.vtable.hold = hold;
        return .{ .ptr = &x.head, .vtable = &x.vtable, .hold_timing = .{ .ptr = x, .take = take } };
    }

    fn clock(x: *Timed) be.Clock {
        return .{ .ptr = x, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(ptr: *anyopaque) void {
        owner(ptr).window = 0;
    }

    /// A round's wall ms: the 10 ms forward, the host's fixed work, and the holds it ran.
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const x = owner(ptr);
        return 10.0 + x.host_ms + x.window;
    }
};

// every target width costs 10 ms; a one-block hold is modeled at 2 ms and a two-block one at 2.5
const timed_grid = [_]depth.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 2 }, .{ .streams = 2, .depth = 1, .ms = 2.5 } };
var timed_costs: [8]Cost = undefined;

fn timedModel() Model {
    for (&timed_costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 10 };
    return .{ .exact_width = 8, .window_costs = &timed_costs, .hidden_rows = true };
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

fn timedLive(s: *const sm.Stream) bool {
    return !s.finished and s.emitted().len + 4 <= s.max_new;
}

fn timedRun(host_ms: f64) !void {
    const prompts = [2][]const u32{ &dt.p1, dt.long[0..11] };
    var plain: [2][]u32 = undefined;
    var planned: usize = 0;
    defer for (plain[0..planned]) |p| gpa.free(p);
    for (&plain, prompts) |*p, prompt| {
        p.* = try plainTokens(timedModel(), prompt, 48);
        planned += 1;
    }
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var fix: Timed = .{
        .head = .{ .facts_ = .{ .depth = 1, .step_ms = 0.5, .batched = true, .limit_shared_drafts = true, .hold_costs = &timed_grid } },
        .hold_ms = 6,
        .host_ms = host_ms,
    };
    defer fix.head.deinit();
    var wrapped = try Drafted.init(gpa, strict.backend(), fix.drafter());
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, wrapped.facts(timedModel()), 16, 15);
    defer cfg.deinit(gpa);
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), fix.clock());
    defer engine.deinit();
    var streams: [2]sm.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    for (&streams, prompts, [2][]const u8{ "a", "b" }) |*s, prompt, id| {
        s.* = try sm.Stream.init(gpa, .{ .id = id, .prompt = prompt, .max_new = 48 });
        made += 1;
        try engine.addStream(s);
    }
    var seen: usize = 0;
    while (engine.activeCount() > 0) {
        fix.settled += 15; // load or calibration work settled before this round: no part of its overhead
        try engine.step();
        if (!timedLive(&streams[0]) or !timedLive(&streams[1])) continue;
        seen += 1;
        // the 6 ms a hold really took, not the 2.5 its grid models, leaves the round's fixed overhead the host's own
        try std.testing.expectApproxEqAbs(host_ms, engine.rule.fixedMs(2, false), 1e-9);
    }
    try std.testing.expect(seen >= 8);
    try std.testing.expect(fix.holds > 0);
    for (plain, 0..) |p, i| try std.testing.expectEqualSlices(u32, p, streams[i].emitted());
}

test "a round's overhead excludes the hold time its backend settled, and work settled outside the round" {
    for ([_]f64{ 0, 2.5 }) |host_ms| try timedRun(host_ms);
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
        // Deferring holds must not drop prompt context.
        try std.testing.expectEqual(@as(u64, dt.p1.len), head.prompt_rows);
        try std.testing.expectEqual(@as(u32, if (grid) 0 else 1), head.holds);
        try std.testing.expectEqual(!grid, s.next.?.count > 0);
        while (engine.activeCount() > 0) try engine.step();
        try std.testing.expectEqualSlices(u32, plain, s.emitted());
        try std.testing.expect(head.holds > 0);
    }
}

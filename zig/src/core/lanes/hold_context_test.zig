//! Deterministic consumer funding transitions; synthetic prices are not device or performance receipts.
const std = @import("std");
const Fixture = @import("hold_prefix_test.zig").Fixture;
const sm = @import("stream.zig");
const win = @import("windows.zig");
const price = @import("hold_price.zig");
const alloc = @import("allocate.zig");
const gpa = std.testing.allocator;
const flat: [8]f64 = @splat(10);
const holds = [_]price.HoldCost{
    .{ .streams = 1, .depth = 1, .ms = 1 },
    .{ .streams = 4, .depth = 1, .ms = 1 },
};

fn head(id: []const u8, chances: []const f64) !sm.Stream {
    var s = try sm.Stream.init(gpa, .{ .id = id, .prompt = &.{ 2, 7, 13 }, .max_new = 64 });
    errdefer s.deinit(gpa);
    s.granted = @intCast(chances.len);
    try s.draft_chances.appendSlice(gpa, chances);
    return s;
}

test "context regimes round up and live quotes retain the strict funding boundary" {
    var contexts = [_]price.ContextCost{
        .{ .streams = 1, .rows = 4, .ms = 9 },
        .{ .streams = 1, .rows = 1, .ms = 1 },
        .{ .streams = 1, .rows = 8, .ms = 0 },
    };
    var f: Fixture = .{};
    try f.init(.{ .drafts = 1, .forward = flat, .holds = &holds, .prior = &.{1}, .contexts = &contexts });
    defer f.deinit();
    var s = try head("regime", &.{1});
    defer s.deinit(gpa);
    const plans = [_]win.Plan{.{ .stream = &s, .position = 0, .kind = .none }};
    for ([_]u32{ 0, 1, 2, 4, 5, 8, 9, 0 }, [_]i64{ 1, 1, 0, 0, 0, 0, 0, 1 }) |rows, want| {
        s.hold_context_rows = rows;
        try f.expectBudget(&plans, want);
    }
    s.hold_context_rows = 3;
    // One token/10ms ties exactly with two/(10 + clean 1 + context 9); only a strict win funds.
    for ([_]f64{ 8, 9, 10, 8 }, [_]i64{ 1, 0, 0, 1 }) |ms, want| {
        contexts[0].ms = ms;
        try f.expectBudget(&plans, want);
    }
    for ([_]f64{ 0, -1, std.math.inf(f64), std.math.nan(f64) }) |ms| {
        contexts[0].ms = ms;
        try f.expectBudget(&plans, 0);
    }
    f.engine.rule.k.hold_context_costs = &.{};
    try f.expectBudget(&plans, 1); // Explicit opt-out retains the original clean-context budget.
}

test "one context projection is shared across the complete selected group and its depths" {
    const contexts = [_]price.ContextCost{
        .{ .streams = 1, .rows = 4, .ms = 12 },
        .{ .streams = 2, .rows = 4, .ms = 12 },
    };
    var f: Fixture = .{};
    try f.init(.{ .drafts = 2, .forward = flat, .holds = &holds, .prior = &.{ 1, 1 }, .contexts = &contexts });
    defer f.deinit();
    var first = try head("first", &.{ 1, 1 });
    defer first.deinit(gpa);
    var second = try head("second", &.{ 1, 1 });
    defer second.deinit(gpa);
    first.hold_context_rows = 1;
    second.hold_context_rows = 4;
    const plans = [_]win.Plan{
        .{ .stream = &first, .position = 0, .kind = .none },
        .{ .stream = &second, .position = 0, .kind = .none },
    };
    var out: [2]i64 = undefined;
    for ([_]u32{ 0, 4, 3, 4 }) |rows| {
        f.engine.rule.k.hold_rows = rows;
        try f.budgetsFor(&plans, &.{ 0, 1 }, &out);
        // Joint 6/23 pays; a lone 4/23 or two physical groups at 6/36 cannot beat plain 2/10.
        try std.testing.expectEqualSlices(i64, if (rows == 3) &.{ 0, 0 } else &.{ 2, 2 }, &out);
    }
}

test "unselected and forced companions cannot contaminate the selected context debt" {
    for ([_]bool{ false, true }) |reversed| {
        const contexts = [_]price.ContextCost{.{ .streams = 1, .rows = 1, .ms = 1 }};
        var f: Fixture = .{};
        try f.init(.{ .drafts = 1, .forward = flat, .holds = &holds, .prior = &.{1}, .contexts = &contexts });
        defer f.deinit();
        var cheap = try head("cheap", &.{1});
        defer cheap.deinit(gpa);
        var expensive = try head("expensive", &.{0.8});
        defer expensive.deinit(gpa);
        cheap.hold_context_rows = 1;
        expensive.hold_context_rows = 1000;
        const streams = if (reversed) [2]*sm.Stream{ &expensive, &cheap } else [2]*sm.Stream{ &cheap, &expensive };
        const plans = [_]win.Plan{
            .{ .stream = streams[0], .position = 0, .kind = .none },
            .{ .stream = streams[1], .position = 0, .kind = .none },
        };
        var out: [2]i64 = undefined;
        for (0..2) |phase| {
            if (phase == 1) try expensive.force.append(gpa, 11);
            try f.budgetsFor(&plans, &.{ 0, 1 }, &out);
            try std.testing.expectEqualSlices(i64, if (reversed) &.{ 0, 1 } else &.{ 1, 0 }, &out);
        }
    }
}

test "stream coverage rounds up and extrapolation charges the selected count beyond the measured width" {
    var contexts = [_]price.ContextCost{
        .{ .streams = 1, .rows = 1, .ms = 0.1 },
        .{ .streams = 4, .rows = 1, .ms = 9 },
    };
    var f: Fixture = .{};
    try f.init(.{ .drafts = 1, .forward = flat, .holds = &holds, .prior = &.{1}, .contexts = &contexts });
    defer f.deinit();
    var streams: [3]sm.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    var plans: [3]win.Plan = undefined;
    for (&streams, &plans) |*s, *p| {
        s.* = try head("width", &.{1});
        made += 1;
        s.hold_context_rows = 1;
        p.* = .{ .stream = s, .position = 0, .kind = .none };
    }
    var out: [3]i64 = undefined;
    try f.budgetsFor(&plans, &.{ 0, 1, 2 }, &out);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0 }, &out);
    contexts[1].streams = 2;
    contexts[1].ms = 3;
    try f.budgetsFor(&plans, &.{ 0, 1, 2 }, &out);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1 }, &out);
}

test "a due context-unfunded recovery keeps its trial until debt becomes affordable" {
    const contexts = [_]price.ContextCost{.{ .streams = 1, .rows = 1, .ms = 9 }};
    var f: Fixture = .{};
    try f.init(.{ .drafts = 1, .forward = flat, .holds = &holds, .prior = &.{1}, .contexts = &contexts });
    defer f.deinit();
    f.engine.rule.k.depth_probe_every = 4;
    var s = try head("due", &.{1});
    defer s.deinit(gpa);
    s.granted = 0;
    s.hold_context_rows = 1;
    const plans = [_]win.Plan{.{ .stream = &s, .position = 0, .kind = .none }};
    for (0..20) |_| try f.expectBudget(&plans, 0);
    s.hold_context_rows = 0;
    try f.expectBudget(&plans, 1);
    // The funded trial, not any preceding due refusal, doubles the four-round wait to eight.
    for (0..7) |_| try f.expectBudget(&plans, 0);
    try f.expectBudget(&plans, 1);
}

test "copy allocations owe no neural context and serial depths pay their projection only once" {
    const contexts = [_]price.ContextCost{.{ .streams = 1, .rows = 4, .ms = 12 }};
    var slots = [_]?f64{ null, 10, 10, 10, 10, 10 };
    const costs: @import("table.zig").Table = .{ .slots = &slots };
    const got = try alloc.allocateDrafts(gpa, &.{ 1, 1 }, &.{ &.{1}, &.{ 1, 1 } }, &costs, 0, 5, .{
        .hold_costs = &holds,
        .hold_context_costs = &contexts,
        .hold_context_rows = &.{ 1000, 4 },
        .charged = &.{ false, true },
        .batched = false,
    });
    defer gpa.free(got);
    // The free copy wins at 3/10; the head cannot pay 5/23, despite costing context just once.
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, got);
    const alone = try alloc.allocateDrafts(gpa, &.{1}, &.{&.{ 1, 1 }}, &costs, 0, 3, .{
        .hold_costs = &holds,
        .hold_context_costs = &contexts,
        .hold_context_rows = &.{4},
        .batched = false,
    });
    defer gpa.free(alone);
    try std.testing.expectEqualSlices(u32, &.{2}, alone); // 3/23 beats 1/10; charging every depth would refuse.
}

test "an unaffordable due peer cannot veto or consume the cheap context trial" {
    for ([_]bool{ false, true }) |reversed| {
        const contexts = [_]price.ContextCost{
            .{ .streams = 1, .rows = 1, .ms = 1 },
            .{ .streams = 1, .rows = 8, .ms = 1000 },
            .{ .streams = 2, .rows = 1, .ms = 1 },
            .{ .streams = 2, .rows = 8, .ms = 1000 },
        };
        var f: Fixture = .{};
        try f.init(.{ .drafts = 1, .forward = flat, .holds = &holds, .prior = &.{1}, .contexts = &contexts });
        defer f.deinit();
        f.engine.rule.k.depth_probe_every = 1;
        var cheap = try head("cheap-trial", &.{1});
        defer cheap.deinit(gpa);
        var dear = try head("dear-trial", &.{1});
        defer dear.deinit(gpa);
        cheap.granted = 0;
        dear.granted = 0;
        cheap.hold_context_rows = 1;
        dear.hold_context_rows = 8;
        dear.verify_row_ms = 100;
        const streams = if (reversed) [2]*sm.Stream{ &dear, &cheap } else [2]*sm.Stream{ &cheap, &dear };
        var whos = [_]@import("depth.zig").Who{
            win.who(streams[0], f.engine.rule.k.hold_costs.len > 0),
            win.who(streams[1], f.engine.rule.k.hold_costs.len > 0),
        };
        var out: [2]i64 = undefined;
        try f.engine.rule.budgets(&whos, win.probe(&f.engine), true, &out);
        try std.testing.expectEqualSlices(i64, if (reversed) &.{ 0, 1 } else &.{ 1, 0 }, &out);
        dear.hold_context_rows = 1;
        dear.verify_row_ms = 0;
        whos = .{
            win.who(streams[0], f.engine.rule.k.hold_costs.len > 0),
            win.who(streams[1], f.engine.rule.k.hold_costs.len > 0),
        };
        try f.engine.rule.budgets(&whos, win.probe(&f.engine), true, &out);
        try std.testing.expectEqualSlices(i64, if (reversed) &.{ 1, 0 } else &.{ 0, 1 }, &out);
    }
}

test "context-aware recovery still reaches a profitable deeper clean shape" {
    const contexts = [_]price.ContextCost{.{ .streams = 1, .rows = 1, .ms = 1 }};
    var f: Fixture = .{};
    try f.init(.{
        .drafts = 2,
        .forward = flat,
        .holds = &.{ .{ .streams = 1, .depth = 1, .ms = 12 }, .{ .streams = 1, .depth = 2, .ms = 1 } },
        .prior = &.{ 1, 1 },
        .contexts = &contexts,
    });
    defer f.deinit();
    f.engine.rule.k.depth_probe_every = 1;
    f.engine.rule.k.draft_slack = 1;
    var s = try head("deeper-trial", &.{ 1, 1 });
    defer s.deinit(gpa);
    s.granted = 0;
    s.hold_context_rows = 1;
    var whos = [_]@import("depth.zig").Who{win.who(&s, f.engine.rule.k.hold_costs.len > 0)};
    var out: [1]i64 = undefined;
    // The reserved first trial cannot pay at 2/23, but its allowed depth two can pay at 3/12.
    try f.engine.rule.budgets(&whos, win.probe(&f.engine), false, &out);
    try std.testing.expectEqual(@as(i64, 1), out[0]);
}

const dt = @import("drafted_test.zig");
const dr = @import("drafter.zig");
const fake = @import("fake.zig");
const Drafted = @import("drafted.zig").Drafted;
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;

/// Real feature ingestion and exact fake-target drafts, with synthetic deferred-row accounting.
const Staged = struct {
    head: dt.FakeDrafter,
    table: dr.Drafter.VTable = undefined,
    absorb_inner: *const fn (*anyopaque, []const dr.Absorb) anyerror!void = undefined,
    hold_inner: *const fn (*anyopaque, []const dr.Hold) anyerror!void = undefined,

    fn of(ptr: *anyopaque) *Staged {
        return @fieldParentPtr("head", @as(*dt.FakeDrafter, @ptrCast(@alignCast(ptr))));
    }

    fn absorb(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const x = of(ptr);
        try x.absorb_inner(ptr, items);
        for (items) |it| if (!it.prompt) {
            it.stream.hold_context_rows += it.features.rows;
            const lane = x.head.lanes.getPtr(it.stream) orelse return error.UnknownStream;
            lane.held.clearRetainingCapacity(); // Staging retires the previous proposal before the next budget.
        };
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = of(ptr);
        try x.hold_inner(ptr, items);
        for (items) |it| {
            const lane = x.head.lanes.getPtr(it.stream) orelse return error.UnknownStream;
            try std.testing.expectEqualSlices(u32, it.stream.context.items[0..@intCast(it.position - 1)], lane.history.items);
            var guess: std.ArrayList(u32) = .empty;
            defer guess.deinit(gpa);
            try guess.appendSlice(gpa, lane.history.items);
            try guess.append(gpa, it.pending);
            for (lane.held.items, 0..) |*token, i| {
                token.* = fake.next(guess.items, it.stream.sampling, it.position + i);
                try guess.append(gpa, token.*);
            }
            it.stream.hold_context_rows = 0;
        }
    }

    fn chances(ptr: *anyopaque, s: *sm.Stream, out: []f64) anyerror!bool {
        const lane = of(ptr).head.lanes.getPtr(s) orelse return error.UnknownStream;
        if (out.len > lane.held.items.len) return error.DraftHeldFewer;
        @memset(out, 1);
        return true;
    }

    fn drafter(x: *Staged) dr.Drafter {
        const inner = x.head.drafter();
        x.table = inner.vtable.*;
        x.absorb_inner = inner.vtable.absorb;
        x.hold_inner = inner.vtable.hold;
        x.table.absorb = absorb;
        x.table.hold = hold;
        x.table.chances = chances;
        return .{ .ptr = &x.head, .vtable = &x.table };
    }
};

test "shared funding sees newly staged debt and zero-budget rounds retain every committed row" {
    for ([_]bool{ false, true }) |batched| {
        var contexts = [_]price.ContextCost{
            .{ .streams = 1, .rows = 1, .ms = 1 },
            .{ .streams = 1, .rows = 64, .ms = 1000 },
        };
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var strict: dt.Strict = .{ .inner = target.backend() };
        defer strict.deinit();
        var staged: Staged = .{ .head = .{ .facts_ = .{
            .depth = 1,
            .step_ms = 0.5,
            .batched = batched,
            .chances = true,
            .hold_costs = &holds,
            .hold_context_costs = &contexts,
        } } };
        defer staged.head.deinit();
        var wrapped = try Drafted.init(gpa, strict.backend(), staged.drafter());
        defer wrapped.deinit();
        var costs: [8]@import("config.zig").Cost = undefined;
        for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 10 };
        var cfg = try Config.init(gpa, wrapped.facts(.{
            .exact_width = 8,
            .window_costs = &costs,
            .hidden_rows = true,
            .batch_rows = 8,
            .lone_copies = false,
        }), 8, 7);
        defer cfg.deinit(gpa);
        cfg.timed = false;
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
        defer engine.deinit();
        engine.rule.k.depth_probe_every = 1;
        var s = try sm.Stream.init(gpa, .{ .id = "staged", .prompt = &.{ 2, 7, 13 }, .max_new = 32 });
        defer s.deinit(gpa);
        try engine.addStream(&s);
        defer engine.release(&s);
        try engine.step(); // The first live row funds the clean one-row regime.
        try std.testing.expectEqual(@as(u32, 1), s.next.?.count);
        const initial_holds = staged.head.holds;
        for (0..2) |_| {
            const before_cache = s.cache_len;
            const before_live = staged.head.live_rows;
            try engine.step();
            try std.testing.expectEqual(@as(u32, 0), s.next.?.count);
            try std.testing.expectEqual(initial_holds, staged.head.holds);
            try std.testing.expectEqual(s.cache_len - before_cache, staged.head.live_rows - before_live);
            const lane = staged.head.lanes.getPtr(&s).?;
            try std.testing.expectEqualSlices(u32, s.context.items[0..@intCast(s.cache_len)], lane.history.items);
        }
        // Debt crossed the one-row kernel regime before funding, then survived the next zero-budget round.
        try std.testing.expect(s.hold_context_rows >= 3);
        contexts[1].ms = 1;
        try engine.step();
        try std.testing.expectEqual(@as(u32, 1), s.next.?.count);
        try std.testing.expectEqual(initial_holds + 1, staged.head.holds);
        try engine.step();
        try std.testing.expect(s.accepted > 0);
        const emitted = s.emitted();
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, s.prompt());
        for (emitted, 0..) |token, i| {
            try std.testing.expectEqual(fake.next(history.items, null, s.prompt_len + i), token);
            try history.append(gpa, token);
        }
    }
}

test "a higher-gain costly head cannot hide a profitable lower-debt normal peer" {
    for ([_]bool{ false, true }) |reversed| {
        const contexts = [_]price.ContextCost{
            .{ .streams = 1, .rows = 1, .ms = 1 },
            .{ .streams = 1, .rows = 8, .ms = 1000 },
            .{ .streams = 2, .rows = 1, .ms = 1 },
            .{ .streams = 2, .rows = 8, .ms = 1000 },
        };
        var f: Fixture = .{};
        try f.init(.{ .drafts = 1, .forward = flat, .holds = &holds, .prior = &.{1}, .contexts = &contexts });
        defer f.deinit();
        var cheap = try head("cheap-normal", &.{0.8});
        defer cheap.deinit(gpa);
        var dear = try head("dear-normal", &.{1});
        defer dear.deinit(gpa);
        cheap.hold_context_rows = 1;
        dear.hold_context_rows = 8;
        const streams = if (reversed) [2]*sm.Stream{ &dear, &cheap } else [2]*sm.Stream{ &cheap, &dear };
        const plans = [_]win.Plan{
            .{ .stream = streams[0], .position = 0, .kind = .none },
            .{ .stream = streams[1], .position = 0, .kind = .none },
        };
        var out: [2]i64 = undefined;
        try f.budgetsFor(&plans, &.{ 0, 1 }, &out);
        try std.testing.expectEqualSlices(i64, if (reversed) &.{ 0, 1 } else &.{ 1, 0 }, &out);
    }
}

test "row allocation retains a profitable lower-debt peer before a costly higher-gain head" {
    for ([_]bool{ false, true }) |batched| for ([_]bool{ false, true }) |reversed| {
        const contexts = [_]price.ContextCost{
            .{ .streams = 1, .rows = 1, .ms = 1 },
            .{ .streams = 1, .rows = 8, .ms = 1000 },
            .{ .streams = 2, .rows = 1, .ms = 1 },
            .{ .streams = 2, .rows = 8, .ms = 1000 },
        };
        var slots = [_]?f64{ null, 10, 10, 10, 10 };
        const costs: @import("table.zig").Table = .{ .slots = &slots };
        const probabilities = if (reversed) [_][]const f64{ &.{1}, &.{0.8} } else [_][]const f64{ &.{0.8}, &.{1} };
        const debts = if (reversed) [_]u32{ 8, 1 } else [_]u32{ 1, 8 };
        const got = try alloc.allocateDrafts(gpa, &.{ 1, 1 }, &probabilities, &costs, 0, 4, .{
            .hold_costs = &holds,
            .hold_context_costs = &contexts,
            .hold_context_rows = &debts,
            .batched = batched,
        });
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, if (reversed) &.{ 0, 1 } else &.{ 1, 0 }, got);
    };
}

test "a lone no-confidence head prices newly staged context and retains whole-round observations" {
    var contexts = [_]price.ContextCost{.{ .streams = 1, .rows = 64, .ms = 1000 }};
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var staged: Staged = .{ .head = .{ .facts_ = .{
        .depth = 1,
        .step_ms = 0.5,
        .prior = &.{1},
        .chances = false,
        .hold_costs = &holds,
        .hold_context_costs = &contexts,
    } } };
    defer staged.head.deinit();
    var wrapped = try Drafted.init(gpa, strict.backend(), staged.drafter());
    defer wrapped.deinit();
    var costs: [8]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 10 };
    var cfg = try Config.init(gpa, wrapped.facts(.{
        .exact_width = 8,
        .window_costs = &costs,
        .hidden_rows = true,
        .batch_rows = 8,
        .lone_copies = false,
    }), 8, 7);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    engine.rule.k.depth_probe_every = 4;
    var s = try sm.Stream.init(gpa, .{ .id = "lone-context", .prompt = &.{ 2, 7, 13 }, .max_new = 32 });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    defer engine.release(&s);
    const initial_holds = staged.head.holds;
    for (0..3) |_| {
        const before_cache = s.cache_len;
        const before_live = staged.head.live_rows;
        const before_debt = s.hold_context_rows;
        try engine.step();
        try std.testing.expectEqual(@as(u32, 0), s.next.?.count);
        try std.testing.expectEqual(initial_holds, staged.head.holds);
        try std.testing.expectEqual(before_debt + @as(u32, @intCast(s.cache_len - before_cache)), s.hold_context_rows);
        try std.testing.expectEqual(s.cache_len - before_cache, staged.head.live_rows - before_live);
        try std.testing.expectEqualSlices(u32, s.context.items[0..@intCast(s.cache_len)], staged.head.lanes.getPtr(&s).?.history.items);
    }
    // The original no-confidence plain-depth probe still runs on its fourth round.
    try engine.step();
    try std.testing.expectEqual(@as(u32, 1), s.next.?.count);
    try std.testing.expectEqual(initial_holds + 1, staged.head.holds);
    try std.testing.expectEqual(@as(u32, 0), s.hold_context_rows);
    const projected_end = s.cache_len;
    try engine.step();
    try std.testing.expectEqual(@as(u32, 0), s.next.?.count);
    try std.testing.expectEqual(@as(u32, @intCast(s.cache_len - projected_end)), s.hold_context_rows);
    try std.testing.expect(s.accepted > 0);
    contexts[0].ms = 1;
    try engine.step();
    try std.testing.expectEqual(@as(u32, 1), s.next.?.count);
    try std.testing.expectEqual(@as(u32, 0), s.hold_context_rows);
    // Observed whole rounds already include context; an expensive quote must not be added again.
    contexts[0].ms = 1000;
    try engine.rule.observeCost(0, 10, false, win.who(&s, engine.rule.k.hold_costs.len > 0));
    try engine.rule.observeCost(1, 11, false, win.who(&s, engine.rule.k.hold_costs.len > 0));
    try engine.step();
    try std.testing.expectEqual(@as(u32, 1), s.next.?.count);
    try std.testing.expectEqual(@as(u32, 0), s.hold_context_rows);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, s.prompt());
    for (s.emitted(), 0..) |token, i| {
        try std.testing.expectEqual(fake.next(history.items, null, s.prompt_len + i), token);
        try history.append(gpa, token);
    }
}

test "fixed query blocks fund the real split instead of four cheap short heads" {
    const grid = [_]price.HoldCost{
        .{ .streams = 1, .depth = 1, .ms = 3 },
        .{ .streams = 2, .depth = 1, .ms = 4 },
        .{ .streams = 3, .depth = 1, .ms = 5 },
    };
    var f: Fixture = .{};
    try f.init(.{ .drafts = 1, .forward = flat, .holds = &grid, .prior = &.{0.8} });
    defer f.deinit();
    f.engine.rule.k.hold_rows = 32;
    var streams: [4]sm.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    while (made < streams.len) : (made += 1) streams[made] = try head("fixed-query", &.{0.8});
    var plans: [4]win.Plan = undefined;
    for (&plans, &streams) |*p, *s| p.* = .{ .stream = s, .position = 0, .kind = .none };
    var out: [4]i64 = undefined;
    for ([_]u32{ 0, 9, 0 }) |query_rows| {
        f.engine.rule.k.hold_block_rows = query_rows;
        try f.budgetsFor(&plans, &.{ 0, 1, 2, 3 }, &out);
        // Four fixed queries split 3+1 and tie plain; the first three still pay.
        try std.testing.expectEqualSlices(i64, if (query_rows == 0) &.{ 1, 1, 1, 1 } else &.{ 1, 1, 1, 0 }, &out);
    }
}

//! Context quotes feed the existing depth rules; synthetic prices are not device receipts.
const std = @import("std");
const price = @import("hold_price.zig");
const alloc = @import("allocate.zig");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const sm = @import("stream.zig");
const dt = @import("drafted_test.zig");
const dr = @import("drafter.zig");
const fake = @import("fake.zig");
const Drafted = @import("drafted.zig").Drafted;
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const gpa = std.testing.allocator;
const holds = [_]price.HoldCost{
    .{ .streams = 1, .depth = 1, .ms = 1 },
    .{ .streams = 4, .depth = 1, .ms = 1 },
};

test "context quotes round up rows and streams, reject invalid coverage and extrapolate count only" {
    var contexts = [_]price.ContextCost{
        .{ .streams = 1, .rows = 1, .ms = 1 },
        .{ .streams = 1, .rows = 4, .ms = 3 },
        .{ .streams = 4, .rows = 4, .ms = 8 },
    };
    try std.testing.expectEqual(@as(f64, 0), price.contextMs(&contexts, 4, 0));
    try std.testing.expectEqual(@as(f64, 0), price.contextMs(&.{}, 4, 2048));
    try std.testing.expectEqual(@as(f64, 3), price.contextMs(&contexts, 1, 2));
    try std.testing.expectEqual(@as(f64, 8), price.contextMs(&contexts, 2, 1));
    try std.testing.expectEqual(@as(f64, 16), price.contextMs(&contexts, 8, 4));
    try std.testing.expectEqual(std.math.inf(f64), price.contextMs(&contexts, 1, 5));
    for ([_]f64{ 0, -1, std.math.inf(f64), std.math.nan(f64) }) |ms| {
        contexts[1].ms = ms;
        try std.testing.expectEqual(std.math.inf(f64), price.contextMs(&contexts, 1, 2));
    }
}

test "physical context passes ignore zero, free and uncharged blocks" {
    const contexts = [_]price.ContextCost{
        .{ .streams = 1, .rows = 4, .ms = 3 },
        .{ .streams = 2, .rows = 4, .ms = 5 },
    };
    try std.testing.expectEqual(@as(f64, 6), price.countsMs(&holds, &contexts, 32, &.{ 10, 2 }, &.{}, 1, &.{ 1, 4 }));
    try std.testing.expectEqual(@as(f64, 8), price.countsMs(&holds, &contexts, 9, &.{ 10, 2 }, &.{}, 1, &.{ 1, 4 }));
    try std.testing.expectEqual(@as(f64, 4), price.countsMs(&holds, &contexts, 32, &.{ 9, 2, 1 }, &.{ false, true, true }, 1, &.{ 1000, 4, 1000 }));
    try std.testing.expectEqual(@as(f64, 0), price.countsMs(&holds, &contexts, 32, &.{ 1, 1 }, &.{}, 1, &.{ 1000, 1000 }));
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

test "lone depth reacts to staged context without double charging an observed whole round" {
    var slots = [_]?f64{ null, 10, 10, 10 };
    const costs: Table = .{ .slots = &slots };
    const contexts = [_]price.ContextCost{.{ .streams = 1, .rows = 4, .ms = 10 }};
    var rule: depth.Rule = .{
        .gpa = gpa,
        .prior = &.{1},
        .most_drafts = 1,
        .family_costs = &costs,
        .shared_costs = &costs,
        .mtp_step_ms = 1,
        .plain_guard = true,
        .node_probabilities = false,
        .batch_rows = 8,
        .k = .{ .hold_costs = &holds, .hold_context_costs = &contexts },
    };
    defer rule.deinit();
    var state: ?depth.State = null;
    defer if (state) |*st| st.deinit(gpa);
    var who: depth.Who = .{ .state = &state, .draft_room = 16, .finished = false, .forced = false };
    try std.testing.expectEqual(@as(i64, 1), try rule.depth(who));
    who.hold_context_rows = 3;
    try std.testing.expectEqual(@as(i64, 0), try rule.depth(who));
    // Measured whole rounds include their context already; only the clean modeled decomposition adds it.
    try rule.observeCost(1, 11, false, who);
    try std.testing.expectEqual(@as(i64, 1), try rule.depth(who));
}

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

    fn drafter(x: *Staged) dr.Drafter {
        const inner = x.head.drafter();
        x.table = inner.vtable.*;
        x.absorb_inner = inner.vtable.absorb;
        x.hold_inner = inner.vtable.hold;
        x.table.absorb = absorb;
        x.table.hold = hold;
        return .{ .ptr = &x.head, .vtable = &x.table };
    }
};

test "depth funding sees newly staged debt and zero-budget rounds retain every committed row" {
    for ([_]bool{ false, true }) |batched| for ([_]usize{ 1, 2 }) |n| {
        var contexts = [_]price.ContextCost{
            .{ .streams = 1, .rows = 64, .ms = 1 },
            .{ .streams = 2, .rows = 64, .ms = 1 },
        };
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var strict: dt.Strict = .{ .inner = target.backend() };
        defer strict.deinit();
        var staged: Staged = .{ .head = .{ .facts_ = .{
            .depth = 1,
            .step_ms = 0.5,
            .prior = &.{1},
            .batched = batched,
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
        engine.rule.k.depth_probe_every = 128;
        var streams: [2]sm.Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(gpa);
        while (made < n) : (made += 1) streams[made] = try sm.Stream.init(gpa, .{ .id = "staged", .prompt = &.{ 2, 7, 13 }, .max_new = 32 });
        var opened: usize = 0;
        defer for (streams[0..opened]) |*s| engine.release(s);
        while (opened < n) : (opened += 1) try engine.addStream(&streams[opened]);
        try engine.step();
        for (streams[0..n]) |s| try std.testing.expectEqual(@as(u32, 1), s.next.?.count);
        const initial_holds = staged.head.holds;
        for (&contexts) |*quote| quote.ms = 1000;
        for (0..2) |_| {
            const before_live = staged.head.live_rows;
            var before_cache: u64 = 0;
            for (streams[0..n]) |s| before_cache += s.cache_len;
            try engine.step();
            var after_cache: u64 = 0;
            for (streams[0..n]) |*s| {
                after_cache += s.cache_len;
                try std.testing.expectEqual(@as(u32, 0), s.next.?.count);
                const lane = staged.head.lanes.getPtr(s).?;
                try std.testing.expectEqualSlices(u32, s.context.items[0..@intCast(s.cache_len)], lane.history.items);
            }
            try std.testing.expectEqual(initial_holds, staged.head.holds);
            try std.testing.expectEqual(after_cache - before_cache, staged.head.live_rows - before_live);
        }
        for (streams[0..n]) |s| try std.testing.expect(s.hold_context_rows >= 3);
        for (&contexts) |*quote| quote.ms = 1;
        try engine.step();
        for (streams[0..n]) |s| try std.testing.expectEqual(@as(u32, 1), s.next.?.count);
        try std.testing.expect(staged.head.holds > initial_holds);
        try engine.step();
        for (streams[0..n]) |*s| {
            try std.testing.expect(s.accepted > 0);
            var history: std.ArrayList(u32) = .empty;
            defer history.deinit(gpa);
            try history.appendSlice(gpa, s.prompt());
            for (s.emitted(), 0..) |token, i| {
                try std.testing.expectEqual(fake.next(history.items, null, s.prompt_len + i), token);
                try history.append(gpa, token);
            }
        }
    };
}

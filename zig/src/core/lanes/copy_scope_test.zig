//! Copy-pricing admission is independent of plain_guard and admits opted-in model drafters.
const std = @import("std");
const cfgs = @import("config.zig");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const lone = @import("lone_copy.zig");
const Drafted = @import("drafted.zig").Drafted;
const FakeDrafter = @import("drafted_test.zig").FakeDrafter;
const gpa = std.testing.allocator;
const costs = [_]cfgs.Cost{ .{ .width = 1, .ms = 5 }, .{ .width = 2, .ms = 6 } };

test "lone copy price: family opt-in survives loaded and inactive model heads" {
    for ([_]bool{ false, true }) |guard| {
        for ([_]bool{ false, true }) |head| {
            for ([_]bool{ false, true }) |speculate| {
                var cfg = try cfgs.Config.init(gpa, .{ .exact_width = 16, .copy_guard = guard, .mtp = head, .speculate = speculate, .window_costs = &costs }, 16, 15);
                defer cfg.deinit(gpa);
                try std.testing.expectEqual(guard, cfg.copy_guard);
            }
        }
    }
}

test "lone copy price: an external drafter preserves the target family opt-in" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var drafter: FakeDrafter = .{};
    defer drafter.deinit();
    var wrapped = try Drafted.init(gpa, target.backend(), drafter.drafter());
    defer wrapped.deinit();
    const model = wrapped.facts(.{ .exact_width = 16, .copy_guard = true, .window_costs = &costs });
    try std.testing.expect(model.mtp);
    var cfg = try cfgs.Config.init(gpa, model, 16, 15);
    defer cfg.deinit(gpa);
    try std.testing.expect(cfg.copy_guard);
}

test "lone copy price: runtime exclusion leaves pricing state and copy history untouched" {
    var cfg = try cfgs.Config.init(gpa, .{ .exact_width = 16, .copy_guard = true, .window_costs = &costs }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "scope", .prompt = &.{ 1, 2, 3 }, .max_new = 16 });
    defer s.deinit(gpa);
    try std.testing.expect(lone.enabled(&e, &s));
    for (0..8) |kind| {
        cfg.copy_guard = true;
        cfg.family_mtp = true;
        cfg.max_copy = 15;
        cfg.timed = true;
        cfg.root_copies = false;
        cfg.context_copies = false;
        cfg.node_probabilities = false;
        cfg.lone_cost_tile = 0;
        e.alone = true;
        s.drafts = true;
        switch (kind) {
            0 => cfg.copy_guard = false,
            1 => cfg.max_copy = 0,
            2 => cfg.timed = false,
            3 => cfg.root_copies = true,
            4 => cfg.context_copies = true,
            5 => cfg.node_probabilities = true,
            6 => cfg.lone_cost_tile = 2,
            7 => e.alone = false,
            else => unreachable,
        }
        try std.testing.expect(!lone.enabled(&e, &s));
        try std.testing.expectEqual(@as(usize, 3), try lone.rows(&e, &s, .{ .tokens = &.{ 4, 5, 6 } }, true));
        try lone.time(&e, &s, .copy, 4, null);
        try std.testing.expect(s.depth == null and !s.prices_seeded);
        try std.testing.expectEqual([2]f64{ -1, -1 }, s.copied_landed);
    }
}

test "lone copy price: a pipelined model head is windowed while copies-only decode keeps its old exclusion" {
    for ([_]bool{ false, true }) |head| for ([_]bool{ false, true }) |pipelined| {
        var cfg = try cfgs.Config.init(gpa, .{ .exact_width = 16, .copy_guard = true, .mtp = head, .speculate = head, .gpu_tokens = pipelined, .window_costs = &costs }, 16, 15);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        var s = try sm.Stream.init(gpa, .{ .id = "windowed", .prompt = &.{ 1, 2, 3 }, .max_new = 16 });
        defer s.deinit(gpa);
        try std.testing.expectEqual(head or !pipelined, lone.enabled(&e, &s));
        s.drafts = false;
        try std.testing.expect(!lone.enabled(&e, &s));
    };
}

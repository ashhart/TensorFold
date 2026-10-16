//! Window pricing on the fake target: sampled streams' host draws never reach a later greedy stream's windows.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const be = @import("backend.zig");

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
